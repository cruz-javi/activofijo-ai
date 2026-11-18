"""Puente orquestador entre el servidor FastMCP institucional y el LLM local Ollama."""

import importlib.util
import json
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP

from activofijo_ai.core.config import Settings
from activofijo_ai.modules.llm.mcp_server import (
    construir_parrafo_respaldo_activo,
    mcp_server,
    normalizar_moneda_texto,
)
from activofijo_ai.modules.llm.provider import (
    LlmErrorExternoError,
    LlmNoConfiguradoError,
    OpenAICompatibleLlmClient,
    extraer_json_objeto,
)
from activofijo_ai.modules.llm.schemas import (
    AyudaSistemaRequest,
    AyudaSistemaResponse,
    ConsultaLenguajeNaturalRequest,
    ConsultaLenguajeNaturalResponse,
    EnlaceDirectoSistema,
    FiltrosDetectadosReporte,
    RedaccionAsistidaRequest,
    RedaccionAsistidaResponse,
)
from activofijo_ai.modules.system.schemas import EstadoMcp

MESES_ES = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)


def _fecha_bolivia_formal() -> str:
    hoy = datetime.now()
    return f"Santa Cruz de la Sierra, {hoy.day:02d} de {MESES_ES[hoy.month - 1]} de {hoy.year}"


def _resolver_motor(settings: Settings) -> tuple[str, str]:
    if settings.llm_provider.lower() == "ollama":
        return "ollama_personalizado", settings.llm_model
    return "llm_remoto", settings.llm_model


def _extraer_dict_tool_mcp(resultado: Any) -> dict[str, Any]:
    if isinstance(resultado, tuple) and len(resultado) == 2 and isinstance(resultado[1], dict):
        return dict(resultado[1])
    if isinstance(resultado, dict):
        return resultado
    if isinstance(resultado, list) and resultado:
        primero = resultado[0]
        texto = getattr(primero, "text", None)
        if isinstance(texto, str):
            data = json.loads(texto)
            if isinstance(data, dict):
                return data
    return {}


def _extraer_json_recurso_mcp(contenidos: Any) -> dict[str, Any]:
    lista = list(contenidos or [])
    if not lista:
        return {}
    raw = getattr(lista[0], "content", "")
    if isinstance(raw, str) and raw.strip():
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _extraer_texto_prompt_mcp(prompt_result: Any) -> str:
    mensajes = getattr(prompt_result, "messages", []) or []
    partes: list[str] = []
    for msg in mensajes:
        contenido = getattr(msg, "content", None)
        texto = getattr(contenido, "text", None)
        if isinstance(texto, str) and texto.strip():
            partes.append(texto.strip())
    return "\n\n".join(partes)


class McpBridgeService:
    """Orquesta Recursos, Herramientas y Prompts del servidor FastMCP con el LLM local."""

    def __init__(self, transport: Any = None, server: FastMCP | None = None) -> None:
        self._transport = transport
        self._mcp = server or mcp_server

    async def obtener_estado(self) -> EstadoMcp:
        """Lista herramientas, recursos y plantillas activas registradas en el servidor FastMCP."""
        sdk_disponible = importlib.util.find_spec("mcp") is not None
        tools = await self._mcp.list_tools()
        recursos = await self._mcp.list_resources()
        plantillas = await self._mcp.list_resource_templates()
        prompts = await self._mcp.list_prompts()

        uris_recursos = [str(r.uri) for r in recursos] + [str(t.uriTemplate) for t in plantillas]
        return EstadoMcp(
            habilitado=sdk_disponible and len(tools) > 0,
            transporte="fastmcp_streamable_http_y_stdio",
            herramientas_registradas=[t.name for t in tools],
            recursos_registrados=uris_recursos,
            prompts_registrados=[p.name for p in prompts],
        )

    async def asistir_redaccion(
        self,
        payload: RedaccionAsistidaRequest,
        settings: Settings,
    ) -> RedaccionAsistidaResponse:
        plantilla = _extraer_json_recurso_mcp(
            await self._mcp.read_resource(f"uagrm://normativa/plantilla/{payload.tipo_documento}")
        )
        normas = [str(n) for n in plantilla.get("normas", [])]
        referencia_base = str(
            plantilla.get("referencia_base") or "SOLICITUD ADMINISTRATIVA DE ACTIVO FIJO"
        )
        destinatario = (
            payload.destinatario_nombre
            or str(plantilla.get("destinatario_default") or "Jefatura del Departamento de Activos Fijos")
        ).strip()
        destinatario_cargo = (
            payload.destinatario_cargo
            or str(plantilla.get("cargo_destinatario_default") or "DUAF — UAGRM")
        ).strip()

        es_nuevo_sin_anexo = payload.tipo_documento == "SOLICITUD_NUEVO_ACTIVO"
        ctx_raw = payload.contexto_activo or {}
        cod_inicial = str(payload.cod_activo or ctx_raw.get("codigo") or "").strip()

        ficha_mcp: dict[str, Any] = {}
        if not es_nuevo_sin_anexo and (cod_inicial or ctx_raw):
            res_ficha = await self._mcp.call_tool(
                "consultar_ficha_activo",
                {
                    "cod_activo": cod_inicial,
                    "contexto_activo": ctx_raw,
                    "sanitizar_publico": payload.modo_acceso == "PUBLICO_HOME",
                },
            )
            ficha_mcp = _extraer_dict_tool_mcp(res_ficha)

        codigo_activo = str(ficha_mcp.get("codigo") or "").strip() or None
        ctx_verificado = (
            ficha_mcp.get("recurso_activo_mcp")
            if isinstance(ficha_mcp.get("recurso_activo_mcp"), dict)
            else {}
        )
        incluye_anexo = bool(codigo_activo and ctx_verificado and not es_nuevo_sin_anexo)
        lugar_fecha = _fecha_bolivia_formal()

        recurso_mcp = {
            "tipo_documento": payload.tipo_documento,
            "modo_acceso": payload.modo_acceso,
            "incluye_anexo_tecnico": incluye_anexo,
            "solicitante": {
                "nombre": payload.solicitante_nombre or "Responsable / Funcionario UAGRM",
                "cargo": payload.solicitante_cargo or "Funcionario Institucional",
                "unidad": payload.solicitante_unidad or "Unidad Universitaria UAGRM",
            },
            "normativa_aplicable": normas,
            "recurso_activo_mcp": ctx_verificado if not es_nuevo_sin_anexo else None,
        }

        prompt_res = await self._mcp.get_prompt(
            "prompt_redaccion_carta_uagrm",
            {
                "recurso_mcp_json": json.dumps(recurso_mcp, ensure_ascii=False),
                "instruccion_usuario": payload.instruccion_usuario,
            },
        )
        prompt_usuario = _extraer_texto_prompt_mcp(prompt_res)

        llm = OpenAICompatibleLlmClient(settings, transport=self._transport)
        advertencias: list[str] = []

        if settings.env != "test" or self._transport is not None:
            try:
                res = await llm.generar_chat_completado(
                    mensajes=[
                        {
                            "role": "system",
                            "content": (
                                "Redactor institucional UAGRM bajo Protocolo MCP. "
                                "Usa únicamente los códigos y valores del recurso MCP; nunca inventes cifras."
                            ),
                        },
                        {"role": "user", "content": prompt_usuario},
                    ],
                    response_format={"type": "json_object"},
                )
                datos = extraer_json_objeto(res.contenido)
                if datos and isinstance(datos.get("cuerpo_parrafos"), list):
                    validado = _extraer_dict_tool_mcp(
                        await self._mcp.call_tool(
                            "validar_integridad_texto",
                            {
                                "cuerpo_parrafos": [str(p) for p in datos["cuerpo_parrafos"]],
                                "contexto_activo": ctx_verificado,
                                "cod_activo": codigo_activo,
                                "es_nuevo_sin_anexo": es_nuevo_sin_anexo,
                            },
                        )
                    )
                    parrafos = [str(p) for p in validado.get("cuerpo_parrafos", []) if str(p).strip()]
                    advertencias.extend(str(a) for a in validado.get("advertencias", []))
                    if parrafos:
                        referencia = normalizar_moneda_texto(
                            str(datos.get("referencia") or referencia_base)
                        )
                        despedida = normalizar_moneda_texto(
                            str(
                                datos.get("despedida")
                                or "Con este motivo, saludo a usted con las consideraciones más distinguidas."
                            )
                        )
                        motor_id, _ = _resolver_motor(settings)
                        return RedaccionAsistidaResponse(
                            motor_utilizado=motor_id,  # type: ignore[arg-type]
                            modelo=res.modelo,
                            referencia=referencia,
                            lugar_fecha=lugar_fecha,
                            destinatario_nombre=destinatario,
                            destinatario_cargo=destinatario_cargo,
                            cuerpo_parrafos=parrafos,
                            despedida=despedida,
                            incluye_anexo_tecnico=incluye_anexo,
                            fundamento_normativo=normas,
                            advertencias=advertencias,
                            tokens_usados=res.tokens_usados,
                        )
            except (LlmNoConfiguradoError, LlmErrorExternoError):
                advertencias.append(
                    "Modelo Ollama local no disponible en este instante; borrador generado mediante motor normativo determinista MCP."
                )

        parrafos_det = self._redactar_determinista(
            payload, ctx_verificado, codigo_activo, incluye_anexo
        )
        return RedaccionAsistidaResponse(
            motor_utilizado="respaldo_determinista",
            modelo=f"{settings.llm_model} (respaldo-determinista)",
            referencia=(
                f"{referencia_base} — {codigo_activo}"
                if codigo_activo and not es_nuevo_sin_anexo
                else referencia_base
            ),
            lugar_fecha=lugar_fecha,
            destinatario_nombre=destinatario,
            destinatario_cargo=destinatario_cargo,
            cuerpo_parrafos=parrafos_det,
            despedida=(
                "Agradeciendo de antemano su atención favorable a la presente gestión, "
                "me despido con las consideraciones más distinguidas."
            ),
            incluye_anexo_tecnico=incluye_anexo,
            fundamento_normativo=normas,
            advertencias=advertencias,
            tokens_usados=0,
        )

    def _redactar_determinista(
        self,
        payload: RedaccionAsistidaRequest,
        ctx: dict[str, Any],
        codigo: str | None,
        incluye_anexo: bool,
    ) -> list[str]:
        solicitante_unidad = payload.solicitante_unidad or "la unidad universitaria correspondiente"
        instruccion = payload.instruccion_usuario.strip().rstrip(".")

        if payload.tipo_documento == "SOLICITUD_NUEVO_ACTIVO":
            return [
                (
                    f"De mi mayor consideración: Por intermedio de la presente nota, en representación de "
                    f"{solicitante_unidad}, solicito respetuosamente la dotación e incorporación de nuevo equipamiento "
                    f"institucional en el marco del D.S. N° 0181 (NB-SABS), conforme al siguiente detalle operativo: "
                    f"{instruccion}."
                ),
                (
                    "El requerimiento responde a la necesidad de garantizar la continuidad de las actividades académicas "
                    "y administrativas asignadas a nuestra dependencia, optimizando las condiciones operativas del personal."
                ),
                (
                    "Una vez autorizada la adquisición o asignación correspondiente, la unidad solicitante asumirá el "
                    "registro, verificación de inventario y custodia formal mediante acta oficial del Departamento de Activos Fijos."
                ),
            ]

        introduccion_map = {
            "SOLICITUD_CORRECCION_DATOS": (
                "solicito formalmente la corrección y actualización en el padrón patrimonial de los datos técnicos "
                "o de registro asignados al bien descrito a continuación"
            ),
            "SOLICITUD_BAJA": (
                "solicito respetuosamente la evaluación técnica y desincorporación (baja patrimonial) conforme al "
                "Manual de Procedimientos de Baja de Bienes (R.R. N° 449/2013)"
            ),
            "SOLICITUD_INVENTARIO": (
                "solicito la programación de levantamiento e inventario físico verificado de los bienes asignados "
                "a nuestra dependencia conforme al D.S. N° 0181"
            ),
            "SOLICITUD_REASIGNACION": (
                "solicito formalizar la reasignación y transferencia administrativa de custodia patrimonial hacia "
                "la dependencia o funcionario receptor correspondiente"
            ),
            "SOLICITUD_REVALUO": (
                "solicito la inspección y revalorización técnica pericial al amparo del Manual UAGRM-MRVT-AF "
                "(R.R. N° 567/2018 V2) considerando que el bien conserva condiciones operativas útiles"
            ),
            "DESCARGO_INCIDENTE": (
                "elevo el presente informe de descargo oportuno en cumplimiento de los Arts. 8 y 12 del Manual "
                "R.R. N° 449/2013 y la Ley N° 1178 para fines de constancia administrativa y verificación técnica"
            ),
            "INFORME_TECNICO": (
                "remito el presente Dictamen Técnico de evaluación patrimonial sustentado en los registros del "
                "padrón institucional y la normativa universitaria vigente"
            ),
        }
        verbo = introduccion_map.get(
            payload.tipo_documento,
            "presento formalmente la siguiente solicitud ante el Departamento de Activos Fijos",
        )

        parrafos = [
            (
                f"De mi mayor consideración: En cumplimiento de la normativa patrimonial vigente de la UAGRM "
                f"y en representación de {solicitante_unidad}, {verbo}, conforme al siguiente fundamento: {instruccion}."
            )
        ]

        if codigo and ctx:
            parrafos.append(construir_parrafo_respaldo_activo(ctx, codigo))
        elif codigo:
            parrafos.append(f"El trámite comprende al bien patrimonial identificado con el código {codigo}.")

        cierre = (
            "Se adjunta como Anexo en la página siguiente el Informe Técnico oficial extraído del padrón institucional "
            "para su verificación y trámite reglamentario."
            if incluye_anexo
            else "Quedo a entera disposición para coordinar la verificación documental o en sitio que Jefatura estime pertinente."
        )
        parrafos.append(cierre)
        return parrafos

    async def interpretar_consulta_reporte(
        self,
        payload: ConsultaLenguajeNaturalRequest,
        settings: Settings,
    ) -> ConsultaLenguajeNaturalResponse:
        filtros_dict = _extraer_dict_tool_mcp(
            await self._mcp.call_tool(
                "filtrar_catalogo_reportes", {"consulta_usuario": payload.consulta_usuario}
            )
        )
        filtros_det = FiltrosDetectadosReporte(**filtros_dict)

        lote = payload.resumen_lote
        resumen_ej: str | None = None
        hallazgos: list[str] = []
        advertencias: list[str] = []

        if lote is not None:
            lote_res = _extraer_dict_tool_mcp(
                await self._mcp.call_tool(
                    "resumir_lote_patrimonial",
                    {
                        "titulo_reporte": lote.titulo_reporte,
                        "total_registros": lote.total_registros,
                        "valor_total_bs": lote.valor_total_bs,
                        "estados": lote.estados,
                        "grupos_top": lote.grupos_top,
                        "criterios_aplicados": lote.criterios_aplicados,
                    },
                )
            )
            resumen_ej = lote_res.get("resumen_ejecutivo")
            hallazgos = [str(h) for h in lote_res.get("hallazgos_clave", [])]

        llm = OpenAICompatibleLlmClient(settings, transport=self._transport)
        if (payload.consulta_usuario.strip() or lote is not None) and (
            settings.env != "test" or self._transport is not None
        ):
            try:
                prompt_res = await self._mcp.get_prompt(
                    "prompt_interpretar_reporte_uagrm",
                    {
                        "contexto_modulo": payload.contexto_modulo,
                        "consulta_usuario": payload.consulta_usuario,
                        "resumen_lote_json": json.dumps(lote.model_dump(), ensure_ascii=False)
                        if lote
                        else "null",
                    },
                )
                prompt = _extraer_texto_prompt_mcp(prompt_res)
                res = await llm.generar_chat_completado(
                    mensajes=[
                        {
                            "role": "system",
                            "content": "Traductor MCP de consultas a JSON estricto para reportes patrimoniales UAGRM.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                parsed = extraer_json_objeto(res.contenido)
                if parsed:
                    raw_f = parsed.get("filtros")
                    if isinstance(raw_f, dict):
                        filtros_det = self._combinar_filtros(filtros_det, raw_f)
                    if isinstance(parsed.get("resumen_ejecutivo"), str) and parsed["resumen_ejecutivo"].strip():
                        texto_res = normalizar_moneda_texto(parsed["resumen_ejecutivo"])
                        if lote is None or str(lote.total_registros) in texto_res:
                            resumen_ej = texto_res
                    if isinstance(parsed.get("hallazgos_clave"), list) and parsed["hallazgos_clave"]:
                        hallazgos = [
                            normalizar_moneda_texto(str(h))
                            for h in parsed["hallazgos_clave"][:4]
                            if str(h).strip()
                        ]
                    explicacion = (
                        normalizar_moneda_texto(str(parsed.get("explicacion") or ""))
                        or self._explicar_filtros(filtros_det)
                    )
                    motor_id, _ = _resolver_motor(settings)
                    return ConsultaLenguajeNaturalResponse(
                        motor_utilizado=motor_id,  # type: ignore[arg-type]
                        modelo=res.modelo,
                        filtros_detectados=filtros_det,
                        explicacion=explicacion,
                        resumen_ejecutivo=resumen_ej,
                        hallazgos_clave=hallazgos,
                        advertencias=advertencias,
                        tokens_usados=res.tokens_usados,
                    )
            except (LlmNoConfiguradoError, LlmErrorExternoError):
                advertencias.append("Intérprete ejecutado en modo determinista local MCP.")

        return ConsultaLenguajeNaturalResponse(
            motor_utilizado="respaldo_determinista",
            modelo=f"{settings.llm_model} (respaldo-determinista)",
            filtros_detectados=filtros_det,
            explicacion=self._explicar_filtros(filtros_det),
            resumen_ejecutivo=resumen_ej,
            hallazgos_clave=hallazgos,
            advertencias=advertencias,
            tokens_usados=0,
        )

    def _combinar_filtros(
        self, base: FiltrosDetectadosReporte, raw: dict[str, Any]
    ) -> FiltrosDetectadosReporte:
        return FiltrosDetectadosReporte(
            search=str(raw["search"]).strip() if raw.get("search") else base.search,
            estado=str(raw["estado"]).upper() if raw.get("estado") in ("ACTIVO", "ASIGNADO", "BAJA") else base.estado,
            condicion=str(raw["condicion"]).upper() if raw.get("condicion") in ("BUENO", "REGULAR", "MALO") else base.condicion,
            grupo_contable=str(raw["grupo_contable"]).strip() if raw.get("grupo_contable") else base.grupo_contable,
            ubicacion=str(raw["ubicacion"]).strip() if raw.get("ubicacion") else base.ubicacion,
            valor_min=float(raw["valor_min"]) if isinstance(raw.get("valor_min"), (int, float)) else base.valor_min,
            valor_max=float(raw["valor_max"]) if isinstance(raw.get("valor_max"), (int, float)) else base.valor_max,
            gestion=int(raw["gestion"]) if isinstance(raw.get("gestion"), int) else base.gestion,
        )

    def _explicar_filtros(self, f: FiltrosDetectadosReporte) -> str:
        partes: list[str] = []
        if f.search:
            partes.append(f"código/texto '{f.search}'")
        if f.grupo_contable:
            partes.append(f"grupo '{f.grupo_contable}'")
        if f.estado:
            partes.append(f"estado {f.estado}")
        if f.condicion:
            partes.append(f"condición física {f.condicion}")
        if f.ubicacion:
            partes.append(f"unidad '{f.ubicacion}'")
        if f.valor_min is not None:
            partes.append(f"valor >= Bs. {f.valor_min:,.2f}")
        if f.valor_max is not None:
            partes.append(f"valor <= Bs. {f.valor_max:,.2f}")
        if f.gestion is not None:
            partes.append(f"gestión {f.gestion}")

        if not partes:
            return "Se analizará el lote actual completo sin restricciones adicionales."
        return "Filtros MCP interpretados: " + ", ".join(partes) + "."

    async def responder_ayuda_sistema(
        self,
        payload: AyudaSistemaRequest,
        settings: Settings,
    ) -> AyudaSistemaResponse:
        guia_mcp = _extraer_dict_tool_mcp(
            await self._mcp.call_tool(
                "consultar_guia_sistema",
                {
                    "pregunta": payload.pregunta,
                    "rol_usuario": payload.rol_usuario,
                    "permisos_usuario": payload.permisos_usuario,
                },
            )
        )

        if guia_mcp.get("fuera_de_ambito"):
            return AyudaSistemaResponse(
                motor_utilizado="respaldo_determinista",
                modelo=settings.llm_model,
                respuesta=str(guia_mcp.get("respuesta_base") or ""),
                pasos=[str(p) for p in guia_mcp.get("pasos_base", [])],
                enlaces_directos=[],
                fuera_de_ambito=True,
                tokens_usados=0,
            )

        enlaces = [
            EnlaceDirectoSistema(**item)
            for item in guia_mcp.get("enlaces_directos", [])
            if isinstance(item, dict)
        ]
        respuesta_det = str(guia_mcp.get("respuesta_base") or "")
        pasos_det = [str(p) for p in guia_mcp.get("pasos_base", [])]
        catalogo_permitido = guia_mcp.get("catalogo_permitido", [])

        llm = OpenAICompatibleLlmClient(settings, transport=self._transport)
        if settings.env != "test" or self._transport is not None:
            try:
                prompt_res = await self._mcp.get_prompt(
                    "prompt_asistente_sistema_uagrm",
                    {
                        "rol_usuario": payload.rol_usuario,
                        "ruta_actual": payload.ruta_actual,
                        "catalogo_permitido_json": json.dumps(catalogo_permitido, ensure_ascii=False),
                        "pregunta": payload.pregunta,
                    },
                )
                prompt = _extraer_texto_prompt_mcp(prompt_res)
                res = await llm.generar_chat_completado(
                    mensajes=[
                        {
                            "role": "system",
                            "content": (
                                "Asistente guiado UAGRM. Nunca inventes rutas ni funciones inexistentes; "
                                "usa solo los módulos permitidos recibidos."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                parsed = extraer_json_objeto(res.contenido)
                if parsed and isinstance(parsed.get("respuesta"), str) and parsed["respuesta"].strip():
                    pasos_llm = [
                        normalizar_moneda_texto(str(p))
                        for p in (parsed.get("pasos") or [])
                        if str(p).strip()
                    ][:4]
                    motor_id, _ = _resolver_motor(settings)
                    return AyudaSistemaResponse(
                        motor_utilizado=motor_id,  # type: ignore[arg-type]
                        modelo=res.modelo,
                        respuesta=normalizar_moneda_texto(parsed["respuesta"]),
                        pasos=pasos_llm or pasos_det,
                        enlaces_directos=enlaces,
                        fuera_de_ambito=bool(parsed.get("fuera_de_ambito", False)),
                        tokens_usados=res.tokens_usados,
                    )
            except (LlmNoConfiguradoError, LlmErrorExternoError):
                pass

        return AyudaSistemaResponse(
            motor_utilizado="respaldo_determinista",
            modelo=f"{settings.llm_model} (guia-mcp)",
            respuesta=respuesta_det,
            pasos=pasos_det,
            enlaces_directos=enlaces,
            fuera_de_ambito=False,
            tokens_usados=0,
        )
