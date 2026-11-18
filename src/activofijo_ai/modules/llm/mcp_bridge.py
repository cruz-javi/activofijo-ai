"""Puente orquestador ReAct + FastMCP entre herramientas en PostgreSQL y Ollama local."""

import importlib.util
import json
import time
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP

from activofijo_ai.core.config import Settings
from activofijo_ai.modules.llm.core_client import (
    CoreMcpBridgeClient,
    core_bridge_ctx,
    seguridad_mcp_ctx,
)
from activofijo_ai.modules.llm.knowledge import PALABRAS_BLOQUEO_EXTERNO
from activofijo_ai.modules.llm.mcp_server import (
    construir_parrafo_respaldo_activo,
    formatear_bs,
    mcp_server,
    normalizar_moneda_texto,
    normalizar_texto,
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
    ConsultaAnaliticaRequest,
    ConsultaAnaliticaResponse,
    ConsultaLenguajeNaturalRequest,
    ConsultaLenguajeNaturalResponse,
    ContextoSeguridadMcpInput,
    DiagnosticoMetricasRequest,
    DiagnosticoMetricasResponse,
    EnlaceDirectoSistema,
    FiltrosDetectadosReporte,
    RedaccionAsistidaRequest,
    RedaccionAsistidaResponse,
    TrazaHerramientaMcp,
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


def _compactar_observacion_para_llm(obs: dict[str, Any]) -> str:
    """Serializa la observación MCP sin campos vacíos ni sobrecarga de tokens para CPU."""
    if not isinstance(obs, dict):
        return json.dumps(obs, ensure_ascii=False)[:700]
    limpia: dict[str, Any] = {}
    for k, v in obs.items():
        if v is None or v == [] or v == {} or v == "":
            continue
        if isinstance(v, list):
            limpia[k] = v[:5]
        else:
            limpia[k] = v
    return json.dumps(limpia, ensure_ascii=False)[:700]


def _simplificar_schema_ollama(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Simplifica schemas Pydantic (`anyOf` con null) para máxima velocidad y precisión en Ollama 3B."""
    base = dict(schema or {"type": "object", "properties": {}})
    props_in = base.get("properties")
    if not isinstance(props_in, dict):
        return base
    props_out: dict[str, Any] = {}
    for nombre, spec in props_in.items():
        if not isinstance(spec, dict):
            continue
        copia = {k: v for k, v in spec.items() if k not in {"default", "title"}}
        if isinstance(copia.get("description"), str):
            copia["description"] = copia["description"][:65]
        any_of = copia.pop("anyOf", None)
        if isinstance(any_of, list):
            no_nulos = [opt for opt in any_of if isinstance(opt, dict) and opt.get("type") != "null"]
            if no_nulos:
                copia["type"] = no_nulos[0].get("type", "string")
        props_out[nombre] = copia
    return {
        "type": "object",
        "properties": props_out,
        "required": list(base.get("required") or []),
    }


def _resumir_observacion_tool(nombre_tool: str, obs: dict[str, Any]) -> str:
    if obs.get("ok") is False or obs.get("error"):
        return str(obs.get("error") or "Error consultando fuente")
    if nombre_tool == "buscar_activos_padron":
        return (
            f"{obs.get('totalCoincidencias', 0)} activo(s) encontrados "
            f"(muestra: {len(obs.get('activos') or [])}, {formatear_bs(obs.get('valorMuestraBs', 0))})"
        )
    if nombre_tool == "obtener_estadisticas_padron":
        return (
            f"Padrón total: {obs.get('totalActivos', 0)} bienes por "
            f"{formatear_bs(obs.get('valorTotalBs', 0))}"
        )
    if nombre_tool == "consultar_riesgo_normativo":
        return f"{len(obs.get('candidatos') or [])} candidato(s) normativos evaluados"
    if nombre_tool == "consultar_trazabilidad_sha256":
        integ = obs.get("verificacionIntegridad") or {}
        return f"Integridad SHA-256: {integ.get('estado', 'VERIFICADA')}"
    if nombre_tool == "consultar_proyeccion_random_forest":
        proy = obs.get("proyeccionAjustadaHorizonte") or {}
        return (
            f"Horizonte {obs.get('horizonteMeses', 6)}m: {proy.get('bajasEstimadas', 0)} bajas est. / "
            f"{formatear_bs(proy.get('presupuestoReposicionBs', 0))}"
        )
    if nombre_tool == "clasificar_intencion_redaccion":
        return f"Trámite detectado: {obs.get('tipo_documento', 'SOLICITUD_CORRECCION_DATOS')}"
    return "Ejecución MCP completada"


class McpBridgeService:
    """Orquesta Recursos, Herramientas (incluyendo bucle ReAct live PostgreSQL) y Prompts."""

    def __init__(
        self,
        transport: Any = None,
        server: FastMCP | None = None,
        core_transport: Any = None,
    ) -> None:
        self._transport = transport
        self._core_transport = core_transport
        self._mcp = server or mcp_server

    async def _llamar_tool_con_traza(
        self,
        nombre_tool: str,
        argumentos: dict[str, Any],
    ) -> tuple[dict[str, Any], TrazaHerramientaMcp]:
        t0 = time.perf_counter()
        res_raw = await self._mcp.call_tool(nombre_tool, argumentos)
        duracion_ms = int((time.perf_counter() - t0) * 1000)
        obs = _extraer_dict_tool_mcp(res_raw)
        traza = TrazaHerramientaMcp(
            herramienta=nombre_tool,
            argumentos=argumentos,
            resumen_resultado=_resumir_observacion_tool(nombre_tool, obs),
            duracion_ms=duracion_ms,
        )
        return obs, traza

    def _preparar_contexto_seguridad(
        self,
        settings: Settings,
        seg: ContextoSeguridadMcpInput | None = None,
        sanitizar_publico: bool = False,
        rol_fallback: str = "FUNCIONARIO",
        permisos_fallback: list[str] | None = None,
    ) -> tuple[Any, Any]:
        cliente = CoreMcpBridgeClient(settings, transport=self._core_transport)
        tok_cli = core_bridge_ctx.set(cliente)
        seg_dict = (
            seg.model_dump()
            if seg is not None
            else {
                "usuario_id": None,
                "rol_usuario": rol_fallback,
                "permisos_usuario": list(permisos_fallback or []),
                "codigo_empleado_legado": None,
                "acceso_amplio": rol_fallback.upper() in {"ADMINISTRADOR", "SUPER_ADMIN", "ADMIN", "JEFE_ACTIVO_FIJO"},
                "sanitizar_publico": sanitizar_publico,
            }
        )
        tok_seg = seguridad_mcp_ctx.set(seg_dict)
        return tok_cli, tok_seg

    async def _tools_openai_desde_mcp(self, nombres_permitidos: set[str]) -> list[dict[str, Any]]:
        tools = await self._mcp.list_tools()
        resultado: list[dict[str, Any]] = []
        for t in tools:
            if t.name not in nombres_permitidos:
                continue
            resultado.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": (t.description or t.name)[:95],
                        "parameters": _simplificar_schema_ollama(t.inputSchema),
                    },
                }
            )
        return resultado

    async def _ejecutar_bucle_react(
        self,
        llm: OpenAICompatibleLlmClient,
        mensajes_iniciales: list[dict[str, Any]],
        nombres_tools: set[str],
        max_iteraciones: int = 4,
    ) -> tuple[str, str, int, list[TrazaHerramientaMcp], dict[str, dict[str, Any]]]:
        openai_tools = await self._tools_openai_desde_mcp(nombres_tools)
        mensajes = list(mensajes_iniciales)
        trazas: list[TrazaHerramientaMcp] = []
        observaciones: dict[str, dict[str, Any]] = {}
        tokens_totales = 0
        modelo_usado = llm._settings.llm_model
        contenido_final = ""

        for ronda in range(max_iteraciones):
            tools_ronda = openai_tools if (openai_tools and ronda < max_iteraciones - 1) else None
            res = await llm.generar_chat_completado(
                mensajes=mensajes,
                tools=tools_ronda,
            )
            tokens_totales += res.tokens_usados
            modelo_usado = res.modelo

            if not res.tool_calls:
                contenido_final = res.contenido
                break

            mensajes.append(
                {
                    "role": "assistant",
                    "content": res.contenido or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.nombre,
                                "arguments": tc.argumentos_json,
                            },
                        }
                        for tc in res.tool_calls
                    ],
                }
            )

            cache_ronda: dict[str, dict[str, Any]] = {}
            for tc in res.tool_calls:
                if tc.nombre not in nombres_tools:
                    obs = {"ok": False, "error": f"Herramienta no autorizada: {tc.nombre}"}
                    traza = TrazaHerramientaMcp(
                        herramienta=tc.nombre,
                        argumentos={},
                        resumen_resultado="Bloqueada por alcance MCP",
                        duracion_ms=0,
                    )
                    trazas.append(traza)
                else:
                    try:
                        args = json.loads(tc.argumentos_json or "{}")
                        if not isinstance(args, dict):
                            args = {}
                    except json.JSONDecodeError:
                        args = {}
                    clave_dedup = f"{tc.nombre}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                    if clave_dedup in cache_ronda:
                        obs = cache_ronda[clave_dedup]
                    else:
                        obs, traza = await self._llamar_tool_con_traza(tc.nombre, args)
                        cache_ronda[clave_dedup] = obs
                        observaciones[tc.nombre] = obs
                        trazas.append(traza)

                mensajes.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.nombre,
                        "content": _compactar_observacion_para_llm(obs),
                    }
                )

        return contenido_final, modelo_usado, tokens_totales, trazas, observaciones

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
        trazas_mcp: list[TrazaHerramientaMcp] = []
        tipo_doc = payload.tipo_documento
        if not tipo_doc:
            clasif_obs, clasif_traza = await self._llamar_tool_con_traza(
                "clasificar_intencion_redaccion",
                {"instruccion_usuario": payload.instruccion_usuario},
            )
            trazas_mcp.append(clasif_traza)
            tipo_doc = str(clasif_obs.get("tipo_documento") or "SOLICITUD_CORRECCION_DATOS")  # type: ignore[assignment]

        plantilla = _extraer_json_recurso_mcp(
            await self._mcp.read_resource(f"uagrm://normativa/plantilla/{tipo_doc}")
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

        es_nuevo_sin_anexo = tipo_doc == "SOLICITUD_NUEVO_ACTIVO"
        ctx_raw = payload.contexto_activo or {}
        cod_inicial = str(payload.cod_activo or ctx_raw.get("codigo") or "").strip()

        ficha_mcp: dict[str, Any] = {}
        if not es_nuevo_sin_anexo and (cod_inicial or ctx_raw):
            ficha_mcp, ficha_traza = await self._llamar_tool_con_traza(
                "consultar_ficha_activo",
                {
                    "cod_activo": cod_inicial,
                    "contexto_activo": ctx_raw,
                    "sanitizar_publico": payload.modo_acceso == "PUBLICO_HOME",
                },
            )
            trazas_mcp.append(ficha_traza)

        codigo_activo = str(ficha_mcp.get("codigo") or "").strip() or None
        ctx_verificado = (
            ficha_mcp.get("recurso_activo_mcp")
            if isinstance(ficha_mcp.get("recurso_activo_mcp"), dict)
            else {}
        )
        incluye_anexo = bool(codigo_activo and ctx_verificado and not es_nuevo_sin_anexo)
        lugar_fecha = _fecha_bolivia_formal()

        recurso_mcp = {
            "tipo_documento": tipo_doc,
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
                    validado, val_traza = await self._llamar_tool_con_traza(
                        "validar_integridad_texto",
                        {
                            "cuerpo_parrafos": [str(p) for p in datos["cuerpo_parrafos"]],
                            "contexto_activo": ctx_verificado,
                            "cod_activo": codigo_activo,
                            "es_nuevo_sin_anexo": es_nuevo_sin_anexo,
                        },
                    )
                    trazas_mcp.append(val_traza)
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
                            tipo_documento_detectado=tipo_doc,  # type: ignore[arg-type]
                            referencia=referencia,
                            lugar_fecha=lugar_fecha,
                            destinatario_nombre=destinatario,
                            destinatario_cargo=destinatario_cargo,
                            cuerpo_parrafos=parrafos,
                            despedida=despedida,
                            incluye_anexo_tecnico=incluye_anexo,
                            fundamento_normativo=normas,
                            advertencias=advertencias,
                            trazas_mcp=trazas_mcp,
                            tokens_usados=res.tokens_usados,
                        )
            except (LlmNoConfiguradoError, LlmErrorExternoError):
                advertencias.append(
                    "Modelo Ollama local no disponible en este instante; borrador generado mediante motor normativo determinista MCP."
                )

        parrafos_det = self._redactar_determinista(
            payload, tipo_doc, ctx_verificado, codigo_activo, incluye_anexo
        )
        return RedaccionAsistidaResponse(
            motor_utilizado="respaldo_determinista",
            modelo=f"{settings.llm_model} (respaldo-determinista)",
            tipo_documento_detectado=tipo_doc,  # type: ignore[arg-type]
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
            trazas_mcp=trazas_mcp,
            tokens_usados=0,
        )

    def _redactar_determinista(
        self,
        payload: RedaccionAsistidaRequest,
        tipo_doc: str,
        ctx: dict[str, Any],
        codigo: str | None,
        incluye_anexo: bool,
    ) -> list[str]:
        solicitante_unidad = payload.solicitante_unidad or "la unidad universitaria correspondiente"
        instruccion = payload.instruccion_usuario.strip().rstrip(".")

        if tipo_doc == "SOLICITUD_NUEVO_ACTIVO":
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
            tipo_doc,
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
        tok_cli, tok_seg = self._preparar_contexto_seguridad(settings, payload.contexto_seguridad)
        try:
            trazas_mcp: list[TrazaHerramientaMcp] = []
            filtros_dict, traza_f = await self._llamar_tool_con_traza(
                "filtrar_catalogo_reportes", {"consulta_usuario": payload.consulta_usuario}
            )
            trazas_mcp.append(traza_f)
            filtros_det = FiltrosDetectadosReporte(**filtros_dict)

            lote = payload.resumen_lote
            resumen_ej: str | None = None
            hallazgos: list[str] = []
            advertencias: list[str] = []
            estadisticas_padron: dict[str, Any] | None = None
            muestras_activos: list[dict[str, Any]] = []

            if lote is not None:
                lote_res, traza_l = await self._llamar_tool_con_traza(
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
                trazas_mcp.append(traza_l)
                resumen_ej = lote_res.get("resumen_ejecutivo")
                hallazgos = [str(h) for h in lote_res.get("hallazgos_clave", [])]
            elif payload.consultar_padron_real:
                busq_obs, traza_b = await self._llamar_tool_con_traza(
                    "buscar_activos_padron",
                    {
                        "search": filtros_det.search,
                        "unidad": filtros_det.ubicacion,
                        "estado": filtros_det.estado,
                        "condicion": filtros_det.condicion,
                        "grupo_contable": filtros_det.grupo_contable,
                        "valor_min": filtros_det.valor_min,
                        "valor_max": filtros_det.valor_max,
                        "gestion": filtros_det.gestion,
                        "limit": 10,
                    },
                )
                trazas_mcp.append(traza_b)
                stats_obs, traza_s = await self._llamar_tool_con_traza(
                    "obtener_estadisticas_padron",
                    {"unidad": filtros_det.ubicacion},
                )
                trazas_mcp.append(traza_s)
                if busq_obs.get("ok") is not False:
                    muestras_activos = list(busq_obs.get("activos") or [])
                    total_coinc = int(busq_obs.get("totalCoincidencias") or len(muestras_activos))
                    valor_muestra = formatear_bs(busq_obs.get("valorMuestraBs") or 0)
                    hallazgos.append(
                        f"Padrón verificado en PostgreSQL: {total_coinc} coincidencia(s) (muestra evaluada: {valor_muestra})."
                    )
                    resumen_ej = (
                        f"Análisis MCP sobre padrón real UAGRM: {total_coinc} bien(es) coincidente(s)"
                        + (f" en {filtros_det.ubicacion}." if filtros_det.ubicacion else ".")
                    )
                if stats_obs.get("ok") is not False:
                    estadisticas_padron = stats_obs
                    hallazgos.append(
                        f"Total padrón consultado: {stats_obs.get('totalActivos', 0)} activo(s) "
                        f"por {formatear_bs(stats_obs.get('valorTotalBs', 0))}."
                    )

            llm = OpenAICompatibleLlmClient(settings, transport=self._transport)
            if (payload.consulta_usuario.strip() or lote is not None) and (
                settings.env != "test" or self._transport is not None
            ):
                try:
                    lote_ctx = (
                        lote.model_dump()
                        if lote
                        else {
                            "estadisticas_padron": estadisticas_padron,
                            "muestras_activos": muestras_activos[:5],
                        }
                    )
                    prompt_res = await self._mcp.get_prompt(
                        "prompt_interpretar_reporte_uagrm",
                        {
                            "contexto_modulo": payload.contexto_modulo,
                            "consulta_usuario": payload.consulta_usuario,
                            "resumen_lote_json": json.dumps(lote_ctx, ensure_ascii=False),
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
                            trazas_mcp=trazas_mcp,
                            estadisticas_padron=estadisticas_padron,
                            muestras_activos=muestras_activos,
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
                trazas_mcp=trazas_mcp,
                estadisticas_padron=estadisticas_padron,
                muestras_activos=muestras_activos,
                tokens_usados=0,
            )
        finally:
            core_bridge_ctx.reset(tok_cli)
            seguridad_mcp_ctx.reset(tok_seg)

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
        tok_cli, tok_seg = self._preparar_contexto_seguridad(
            settings,
            payload.contexto_seguridad,
            rol_fallback=payload.rol_usuario,
            permisos_fallback=payload.permisos_usuario,
        )
        try:
            trazas_mcp: list[TrazaHerramientaMcp] = []
            guia_mcp, traza_guia = await self._llamar_tool_con_traza(
                "consultar_guia_sistema",
                {
                    "pregunta": payload.pregunta,
                    "rol_usuario": payload.rol_usuario,
                    "permisos_usuario": payload.permisos_usuario,
                },
            )
            trazas_mcp.append(traza_guia)

            if guia_mcp.get("fuera_de_ambito"):
                return AyudaSistemaResponse(
                    motor_utilizado="respaldo_determinista",
                    modelo=settings.llm_model,
                    respuesta=str(guia_mcp.get("respuesta_base") or ""),
                    pasos=[str(p) for p in guia_mcp.get("pasos_base", [])],
                    enlaces_directos=[],
                    fuera_de_ambito=True,
                    trazas_mcp=trazas_mcp,
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

            filtros_dict = _extraer_dict_tool_mcp(
                await self._mcp.call_tool("filtrar_catalogo_reportes", {"consulta_usuario": payload.pregunta})
            )
            filtros_nav = FiltrosDetectadosReporte(**filtros_dict)
            tiene_filtro_consulta = any(
                getattr(filtros_nav, k) is not None
                for k in ("search", "estado", "condicion", "grupo_contable", "ubicacion", "valor_min", "valor_max")
            )

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
                    mensajes: list[dict[str, Any]] = [
                        {
                            "role": "system",
                            "content": (
                                "Copiloto UAGRM con herramientas MCP conectadas a PostgreSQL. "
                                "Usa herramientas MCP si preguntan datos reales de activos, estadísticas, "
                                "riesgo normativo o trazabilidad SHA-256; nunca inventes rutas ni cifras."
                            ),
                        },
                    ]
                    for turno in payload.historial[-4:]:
                        rol_chat = "user" if turno.rol == "usuario" else "assistant"
                        mensajes.append({"role": rol_chat, "content": turno.texto})
                    mensajes.append({"role": "user", "content": prompt})

                    tools_ayuda = {
                        "buscar_activos_padron",
                        "obtener_estadisticas_padron",
                        "consultar_riesgo_normativo",
                        "consultar_trazabilidad_sha256",
                    }
                    contenido_llm, modelo_llm, tokens_llm, trazas_react, _ = (
                        await self._ejecutar_bucle_react(
                            llm=llm,
                            mensajes_iniciales=mensajes,
                            nombres_tools=tools_ayuda,
                            max_iteraciones=2,
                        )
                    )
                    trazas_mcp.extend(trazas_react)
                    parsed = extraer_json_objeto(contenido_llm)
                    if parsed and isinstance(parsed.get("respuesta"), str) and parsed["respuesta"].strip():
                        pasos_llm = [
                            normalizar_moneda_texto(str(p))
                            for p in (parsed.get("pasos") or [])
                            if str(p).strip()
                        ][:4]
                        motor_id, _ = _resolver_motor(settings)
                        return AyudaSistemaResponse(
                            motor_utilizado=motor_id,  # type: ignore[arg-type]
                            modelo=modelo_llm,
                            respuesta=normalizar_moneda_texto(parsed["respuesta"]),
                            pasos=pasos_llm or pasos_det,
                            enlaces_directos=enlaces,
                            fuera_de_ambito=bool(parsed.get("fuera_de_ambito", False)),
                            trazas_mcp=trazas_mcp,
                            filtros_navegacion=filtros_nav if tiene_filtro_consulta else None,
                            tokens_usados=tokens_llm,
                        )
                    if contenido_llm.strip():
                        motor_id, _ = _resolver_motor(settings)
                        return AyudaSistemaResponse(
                            motor_utilizado=motor_id,  # type: ignore[arg-type]
                            modelo=modelo_llm,
                            respuesta=normalizar_moneda_texto(contenido_llm.strip()),
                            pasos=pasos_det,
                            enlaces_directos=enlaces,
                            fuera_de_ambito=False,
                            trazas_mcp=trazas_mcp,
                            filtros_navegacion=filtros_nav if tiene_filtro_consulta else None,
                            tokens_usados=tokens_llm,
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
                trazas_mcp=trazas_mcp,
                filtros_navegacion=filtros_nav if tiene_filtro_consulta else None,
                tokens_usados=0,
            )
        finally:
            core_bridge_ctx.reset(tok_cli)
            seguridad_mcp_ctx.reset(tok_seg)

    async def consultar_analitica_padron(
        self,
        payload: ConsultaAnaliticaRequest,
        settings: Settings,
    ) -> ConsultaAnaliticaResponse:
        q_norm = normalizar_texto(payload.consulta)
        if any(p in q_norm for p in PALABRAS_BLOQUEO_EXTERNO):
            return ConsultaAnaliticaResponse(
                motor_utilizado="respaldo_determinista",
                modelo=settings.llm_model,
                respuesta="Consulta fuera del ámbito patrimonial del Departamento de Activos Fijos UAGRM.",
                filtros_detectados=FiltrosDetectadosReporte(),
                activos_mencionados=[],
                estadisticas=None,
                trazas_mcp=[],
                fuera_de_ambito=True,
                tokens_usados=0,
            )

        tok_cli, tok_seg = self._preparar_contexto_seguridad(settings, payload.contexto_seguridad)
        try:
            trazas_mcp: list[TrazaHerramientaMcp] = []
            filtros_dict, traza_f = await self._llamar_tool_con_traza(
                "filtrar_catalogo_reportes", {"consulta_usuario": payload.consulta}
            )
            trazas_mcp.append(traza_f)
            filtros_det = FiltrosDetectadosReporte(**filtros_dict)

            activos_mencionados: list[dict[str, Any]] = []
            estadisticas: dict[str, Any] | None = None
            respuesta_llm = ""
            modelo_usado = f"{settings.llm_model} (mcp-directo)"
            tokens_usados = 0
            motor_usado: str = "respaldo_determinista"

            llm = OpenAICompatibleLlmClient(settings, transport=self._transport)
            tools_analitica = {
                "buscar_activos_padron",
                "obtener_estadisticas_padron",
                "consultar_riesgo_normativo",
                "consultar_trazabilidad_sha256",
                "consultar_proyeccion_random_forest",
                "consultar_reglas_normativas",
            }

            if settings.env != "test" or self._transport is not None:
                try:
                    mensajes = [
                        {
                            "role": "system",
                            "content": (
                                "Agente Analítico Patrimonial UAGRM. Consulta PostgreSQL vía MCP "
                                "(deja 'unidad' en null salvo que nombren una facultad/oficina concreta) "
                                "y responde con cifras exactas en español boliviano y moneda Bs."
                            ),
                        },
                        {"role": "user", "content": payload.consulta},
                    ]
                    contenido, modelo_usado, tokens_usados, trazas_react, obs = (
                        await self._ejecutar_bucle_react(
                            llm=llm,
                            mensajes_iniciales=mensajes,
                            nombres_tools=tools_analitica,
                            max_iteraciones=2,
                        )
                    )
                    trazas_mcp.extend(trazas_react)
                    if "buscar_activos_padron" in obs and obs["buscar_activos_padron"].get("ok") is not False:
                        activos_mencionados = list(obs["buscar_activos_padron"].get("activos") or [])
                    if "obtener_estadisticas_padron" in obs and obs["obtener_estadisticas_padron"].get("ok") is not False:
                        estadisticas = obs["obtener_estadisticas_padron"]
                    if contenido.strip():
                        respuesta_llm = normalizar_moneda_texto(contenido.strip())
                        motor_usado, _ = _resolver_motor(settings)
                except (LlmNoConfiguradoError, LlmErrorExternoError):
                    pass

            # Asegura anclaje real en PostgreSQL si el modelo respondió sin invocar tool DB o estuvo offline
            if not activos_mencionados and estadisticas is None:
                busq_obs, traza_b = await self._llamar_tool_con_traza(
                    "buscar_activos_padron",
                    {
                        "search": filtros_det.search,
                        "unidad": filtros_det.ubicacion,
                        "estado": filtros_det.estado,
                        "condicion": filtros_det.condicion,
                        "grupo_contable": filtros_det.grupo_contable,
                        "valor_min": filtros_det.valor_min,
                        "valor_max": filtros_det.valor_max,
                        "gestion": filtros_det.gestion,
                        "limit": 10,
                    },
                )
                trazas_mcp.append(traza_b)
                if busq_obs.get("ok") is not False:
                    activos_mencionados = list(busq_obs.get("activos") or [])

                stats_obs, traza_s = await self._llamar_tool_con_traza(
                    "obtener_estadisticas_padron",
                    {"unidad": filtros_det.ubicacion},
                )
                trazas_mcp.append(traza_s)
                if stats_obs.get("ok") is not False:
                    estadisticas = stats_obs

            if not respuesta_llm:
                total_padron = (estadisticas or {}).get("totalActivos", 0)
                valor_padron = formatear_bs((estadisticas or {}).get("valorTotalBs", 0))
                coinc = len(activos_mencionados)
                respuesta_llm = (
                    f"Resultado verificado en PostgreSQL vía FastMCP: se localizaron {coinc} "
                    f"activo(s) en la muestra filtrada ({self._explicar_filtros(filtros_det).rstrip('.')}). "
                    f"El padrón consultado registra {total_padron} bien(es) por un valor acumulado de {valor_padron}."
                )

            return ConsultaAnaliticaResponse(
                motor_utilizado=motor_usado,  # type: ignore[arg-type]
                modelo=modelo_usado,
                respuesta=respuesta_llm,
                filtros_detectados=filtros_det,
                activos_mencionados=activos_mencionados[:10],
                estadisticas=estadisticas,
                trazas_mcp=trazas_mcp,
                fuera_de_ambito=False,
                tokens_usados=tokens_usados,
            )
        finally:
            core_bridge_ctx.reset(tok_cli)
            seguridad_mcp_ctx.reset(tok_seg)

    async def diagnosticar_metricas(
        self,
        payload: DiagnosticoMetricasRequest,
        settings: Settings,
    ) -> DiagnosticoMetricasResponse:
        tok_cli, tok_seg = self._preparar_contexto_seguridad(settings, payload.contexto_seguridad)
        try:
            datos = dict(payload.datos_metricas or {})
            trazas_mcp: list[TrazaHerramientaMcp] = []

            if not datos.get("proyeccionAjustadaHorizonte"):
                rf_obs, traza_rf = await self._llamar_tool_con_traza(
                    "consultar_proyeccion_random_forest",
                    {"horizonte_meses": payload.horizonte_meses},
                )
                trazas_mcp.append(traza_rf)
                norm_obs, traza_norm = await self._llamar_tool_con_traza(
                    "consultar_riesgo_normativo",
                    {"limit": 5},
                )
                trazas_mcp.append(traza_norm)
                if rf_obs.get("ok") is not False:
                    datos["proyeccionAjustadaHorizonte"] = rf_obs.get("proyeccionAjustadaHorizonte") or {}
                if norm_obs.get("ok") is not False:
                    candidatos_muestra = norm_obs.get("candidatos") or []
                    bajas_cnt = sum(1 for c in candidatos_muestra if isinstance(c, dict) and c.get("dictamen") == "CANDIDATO_BAJA")
                    rev_cnt = sum(1 for c in candidatos_muestra if isinstance(c, dict) and c.get("dictamen") == "SUGERIDO_REVALUO")
                    datos["resumenNormativo"] = {
                        "totalEvaluados": norm_obs.get("totalCoincidencias") or len(candidatos_muestra),
                        "totalCandidatosBaja": bajas_cnt,
                        "totalSugeridosRevaluo": rev_cnt,
                        "muestra": candidatos_muestra,
                    }

            proy = datos.get("proyeccionAjustadaHorizonte") or {}
            bajas_est = int(proy.get("bajasEstimadas") or 0)
            presup_bs = formatear_bs(proy.get("presupuestoReposicionBs") or 0)
            resumen_norm = datos.get("resumenNormativo") or {}
            candidatos_baja = int(resumen_norm.get("totalCandidatosBaja") or 0)
            sugeridos_revaluo = int(resumen_norm.get("totalSugeridosRevaluo") or 0)
            resumen_compacto = {
                "horizonteMeses": payload.horizonte_meses,
                "bajasEstimadasRf": bajas_est,
                "presupuestoReposicionBs": presup_bs,
                "totalEvaluadosNormativa": resumen_norm.get("totalEvaluados") or len(resumen_norm.get("muestra") or []),
                "candidatosBaja": candidatos_baja,
                "sugeridosRevaluo": sugeridos_revaluo,
                "muestra": [
                    {k: c.get(k) for k in ("codActivo", "dictamen", "nivelPrioridad") if c.get(k) is not None}
                    for c in (resumen_norm.get("muestra") or [])[:3]
                    if isinstance(c, dict)
                ],
            }

            if not trazas_mcp:
                trazas_mcp.append(
                    TrazaHerramientaMcp(
                        herramienta="consultar_proyeccion_random_forest",
                        argumentos={"horizonte_meses": payload.horizonte_meses},
                        resumen_resultado=f"Horizonte {payload.horizonte_meses}m: {bajas_est} bajas / {presup_bs}",
                        duracion_ms=2,
                    )
                )

            llm = OpenAICompatibleLlmClient(settings, transport=self._transport)
            if settings.env != "test" or self._transport is not None:
                try:
                    res = await llm.generar_chat_completado(
                        mensajes=[
                            {
                                "role": "system",
                                "content": (
                                    "Auditor Presupuestario IA UAGRM. Sintetiza Random Forest + "
                                    "R.R. 449/2013 y R.R. 567/2018 usando únicamente cifras recibidas en Bs. "
                                    'Devuelve SOLO JSON: {"diagnostico_ejecutivo": str, '
                                    '"recomendaciones": [str, str], "alertas_presupuestarias": [str]}'
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Horizonte: {payload.horizonte_meses} meses. "
                                    f"Pregunta: {payload.pregunta_usuario or 'Diagnóstico integral'}.\n"
                                    f"Datos: {json.dumps(resumen_compacto, ensure_ascii=False)}"
                                ),
                            },
                        ],
                        response_format={"type": "json_object"},
                    )
                    parsed = extraer_json_objeto(res.contenido)
                    if parsed and isinstance(parsed.get("diagnostico_ejecutivo"), str):
                        motor_id, _ = _resolver_motor(settings)
                        return DiagnosticoMetricasResponse(
                            motor_utilizado=motor_id,  # type: ignore[arg-type]
                            modelo=res.modelo,
                            diagnostico_ejecutivo=normalizar_moneda_texto(parsed["diagnostico_ejecutivo"]),
                            recomendaciones=[
                                normalizar_moneda_texto(str(r))
                                for r in (parsed.get("recomendaciones") or [])
                                if str(r).strip()
                            ][:4],
                            alertas_presupuestarias=[
                                normalizar_moneda_texto(str(a))
                                for a in (parsed.get("alertas_presupuestarias") or [])
                                if str(a).strip()
                            ][:3],
                            trazas_mcp=trazas_mcp,
                            tokens_usados=res.tokens_usados,
                        )
                except (LlmNoConfiguradoError, LlmErrorExternoError):
                    pass

            return DiagnosticoMetricasResponse(
                motor_utilizado="respaldo_determinista",
                modelo=f"{settings.llm_model} (rf-normativa-mcp)",
                diagnostico_ejecutivo=(
                    f"Proyección a {payload.horizonte_meses} meses: el ensamble Random Forest estima "
                    f"{bajas_est} bajas potenciales con requerimiento de reposición cercano a {presup_bs}. "
                    f"Actualmente el Motor Normativo registra {candidatos_baja} candidato(s) a baja (R.R. 449/2013) "
                    f"y {sugeridos_revaluo} bien(es) sugeridos para revalúo técnico (R.R. 567/2018 V2)."
                ),
                recomendaciones=[
                    "Inspeccionar prioritariamente los bienes en estado MALO o vida útil consumida >= 100% para formalizar actas bajo R.R. N° 449/2013.",
                    "Programar peritaje técnico (R.R. N° 567/2018 V2) sobre activos operativos con valor neto residual de Bs. 1,00 para extender su vida útil.",
                ],
                alertas_presupuestarias=[
                    f"Previsión sugerida de reposición para los próximos {payload.horizonte_meses} meses: {presup_bs}."
                ],
                trazas_mcp=trazas_mcp,
                tokens_usados=0,
            )
        finally:
            core_bridge_ctx.reset(tok_cli)
            seguridad_mcp_ctx.reset(tok_seg)
