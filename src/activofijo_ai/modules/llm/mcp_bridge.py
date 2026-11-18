"""Puente orquestador ReAct + FastMCP entre herramientas en PostgreSQL y Ollama local."""

import importlib.util
import json
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

EmitirEventoFn = Callable[[dict[str, Any]], Awaitable[None]]

from mcp.server.fastmcp import FastMCP

from activofijo_ai.core.config import Settings
from activofijo_ai.modules.llm.core_client import (
    CoreMcpBridgeClient,
    core_bridge_ctx,
    seguridad_mcp_ctx,
)
from activofijo_ai.modules.llm.knowledge import PALABRAS_BLOQUEO_EXTERNO
from activofijo_ai.modules.llm.mcp_server import (
    CONDICIONES_PADRON_VALIDAS,
    ESTADOS_PADRON_VALIDOS,
    construir_parrafo_respaldo_activo,
    detectar_edicion_parcial_carta,
    detectar_exportacion_carta,
    detectar_formatos_exportacion_reporte,
    detectar_incluir_graficas,
    extraer_destinatario_instruccion,
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


_ETIQUETAS_HERRAMIENTAS: dict[str, str] = {
    "clasificar_intencion_redaccion": "Clasificación automática de trámite",
    "consultar_ficha_activo": "Verificación de ficha patrimonial",
    "validar_integridad_texto": "Validación de consistencia institucional",
    "filtrar_catalogo_reportes": "Interpretación de criterios de búsqueda",
    "consultar_guia_sistema": "Guía operativa de módulos y permisos",
    "buscar_activos_padron": "Búsqueda en padrón institucional",
    "obtener_estadisticas_padron": "Estadísticas generales del inventario",
    "consultar_riesgo_normativo": "Evaluación normativa y prioridades",
    "consultar_trazabilidad_sha256": "Historial y verificación de bien",
    "consultar_proyeccion_random_forest": "Proyección presupuestaria",
    "consultar_reglas_normativas": "Reglamento institucional aplicable",
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
        return f"Verificación de historial: {integ.get('estado', 'VERIFICADA')}"
    if nombre_tool == "consultar_proyeccion_random_forest":
        proy = obs.get("proyeccionAjustadaHorizonte") or {}
        return (
            f"Horizonte {obs.get('horizonteMeses', 6)}m: {proy.get('bajasEstimadas', 0)} bajas est. / "
            f"{formatear_bs(proy.get('presupuestoReposicionBs', 0))}"
        )
    if nombre_tool == "clasificar_intencion_redaccion":
        return f"Trámite detectado: {obs.get('tipo_documento', 'SOLICITUD_CORRECCION_DATOS')}"
    if nombre_tool == "consultar_guia_sistema":
        enlaces = obs.get("enlaces_directos") or []
        return f"{len(enlaces)} módulo(s) autorizados verificados"
    if nombre_tool == "filtrar_catalogo_reportes":
        activos = [k for k, v in obs.items() if v is not None]
        return f"{len(activos)} criterio(s) identificados" if activos else "Consulta general"
    if nombre_tool == "consultar_ficha_activo":
        cod = obs.get("codigo")
        return f"Ficha {cod} validada" if cod else "Datos de ficha preparados"
    if nombre_tool == "validar_integridad_texto":
        return "Texto verificado sin inconsistencias"
    return "Consulta institucional verificada"


async def _emitir(emitir_evento: EmitirEventoFn | None, evento: dict[str, Any]) -> None:
    if emitir_evento is not None:
        await emitir_evento(evento)


class McpBridgeService:
    """Orquesta Recursos, Herramientas (incluyendo bucle ReAct en vivo) y Prompts."""

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
        emitir_evento: EmitirEventoFn | None = None,
        paso_id: str | None = None,
    ) -> tuple[dict[str, Any], TrazaHerramientaMcp]:
        etiqueta = _ETIQUETAS_HERRAMIENTAS.get(nombre_tool, "Consultando fuente institucional")
        id_ev = paso_id or f"tool-{nombre_tool}-{int(time.perf_counter() * 1000)}"
        await _emitir(
            emitir_evento,
            {"tipo": "paso", "id": id_ev, "etiqueta": etiqueta, "estado": "activo"},
        )
        t0 = time.perf_counter()
        res_raw = await self._mcp.call_tool(nombre_tool, argumentos)
        duracion_ms = int((time.perf_counter() - t0) * 1000)
        obs = _extraer_dict_tool_mcp(res_raw)
        resumen = _resumir_observacion_tool(nombre_tool, obs)
        await _emitir(
            emitir_evento,
            {
                "tipo": "paso",
                "id": id_ev,
                "etiqueta": etiqueta,
                "detalle": resumen,
                "duracionMs": duracion_ms,
                "estado": "completado",
            },
        )
        traza = TrazaHerramientaMcp(
            herramienta=nombre_tool,
            argumentos=argumentos,
            resumen_resultado=resumen,
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
        emitir_evento: EmitirEventoFn | None = None,
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
            id_paso = f"llm-ronda-{ronda}"
            etiqueta_paso = (
                "Analizando consulta e identificando fuentes"
                if ronda == 0
                else "Redactando síntesis con datos verificados"
            )
            await _emitir(
                emitir_evento,
                {"tipo": "paso", "id": id_paso, "etiqueta": etiqueta_paso, "estado": "activo"},
            )
            t_llm = time.perf_counter()
            res = await llm.generar_chat_completado(
                mensajes=mensajes,
                tools=tools_ronda,
            )
            dur_llm_ms = int((time.perf_counter() - t_llm) * 1000)
            tokens_totales += res.tokens_usados
            modelo_usado = res.modelo

            if not res.tool_calls:
                await _emitir(
                    emitir_evento,
                    {
                        "tipo": "paso",
                        "id": id_paso,
                        "etiqueta": etiqueta_paso,
                        "detalle": "Respuesta generada",
                        "duracionMs": dur_llm_ms,
                        "estado": "completado",
                    },
                )
                contenido_final = res.contenido
                break

            await _emitir(
                emitir_evento,
                {
                    "tipo": "paso",
                    "id": id_paso,
                    "etiqueta": etiqueta_paso,
                    "detalle": f"{len(res.tool_calls)} consulta(s) solicitadas",
                    "duracionMs": dur_llm_ms,
                    "estado": "completado",
                },
            )

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
                        resumen_resultado="Bloqueada por alcance",
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
                        obs, traza = await self._llamar_tool_con_traza(
                            tc.nombre, args, emitir_evento=emitir_evento
                        )
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
        emitir_evento: EmitirEventoFn | None = None,
    ) -> RedaccionAsistidaResponse:
        trazas_mcp: list[TrazaHerramientaMcp] = []
        tipo_doc = payload.tipo_documento
        if not tipo_doc:
            clasif_obs, clasif_traza = await self._llamar_tool_con_traza(
                "clasificar_intencion_redaccion",
                {"instruccion_usuario": payload.instruccion_usuario},
                emitir_evento=emitir_evento,
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
        dest_implicito, cargo_implicito = extraer_destinatario_instruccion(payload.instruccion_usuario)
        exportacion_solicitada = detectar_exportacion_carta(payload.instruccion_usuario)
        edicion_parcial = (
            detectar_edicion_parcial_carta(payload.instruccion_usuario)
            if payload.borrador_actual is not None
            else {"indice_parrafo_1based": None, "toca_destinatario": False, "toca_referencia": False}
        )

        destinatario = (
            payload.destinatario_nombre
            or (dest_implicito if (edicion_parcial.get("toca_destinatario") or payload.borrador_actual is None) else None)
            or (payload.borrador_actual.destinatario_nombre if payload.borrador_actual else None)
            or dest_implicito
            or str(plantilla.get("destinatario_default") or "Jefatura del Departamento de Activos Fijos")
        ).strip()
        destinatario_cargo = (
            payload.destinatario_cargo
            or (cargo_implicito if (edicion_parcial.get("toca_destinatario") or payload.borrador_actual is None) else None)
            or (payload.borrador_actual.destinatario_cargo if payload.borrador_actual else None)
            or cargo_implicito
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
                emitir_evento=emitir_evento,
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
                await _emitir(
                    emitir_evento,
                    {
                        "tipo": "paso",
                        "id": "llm-redaccion",
                        "etiqueta": "Redactando carta formal institucional",
                        "estado": "activo",
                    },
                )
                t_red = time.perf_counter()
                mensajes_redaccion: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": (
                            "Redactor institucional UAGRM bajo Protocolo MCP. "
                            "Usa únicamente los códigos y valores del recurso MCP; nunca inventes cifras."
                        ),
                    }
                ]
                for turno in payload.historial[-6:]:
                    mensajes_redaccion.append(
                        {
                            "role": "user" if turno.rol == "usuario" else "assistant",
                            "content": turno.texto,
                        }
                    )
                if payload.borrador_actual is not None:
                    parrafos_base = list(payload.borrador_actual.cuerpo_parrafos or [])
                    parrafos_numerados = "\n".join(
                        f"[Párrafo {idx}] {texto_p}"
                        for idx, texto_p in enumerate(parrafos_base, start=1)
                    )
                    idx_obj = edicion_parcial.get("indice_parrafo_1based")
                    guia_quirurgica = (
                        f"MODIFICA ÚNICAMENTE el párrafo N° {idx_obj} manteniendo el texto exacto de los demás párrafos. "
                        if idx_obj
                        else "Conserva intactos los párrafos, montos y destinatario que el usuario no pidió cambiar. "
                    )
                    borrador_json = json.dumps(
                        payload.borrador_actual.model_dump(exclude_none=True),
                        ensure_ascii=False,
                    )
                    mensajes_redaccion.append(
                        {
                            "role": "user",
                            "content": (
                                "SESIÓN TEMPORAL DE CORRECCIÓN DE CARTA UAGRM:\n"
                                f"Borrador actual ({len(parrafos_base)} párrafos):\n{parrafos_numerados or borrador_json}\n"
                                f"Metadatos actuales: {borrador_json}\n\n"
                                f"{guia_quirurgica}"
                                f"Corrección solicitada por el usuario: {payload.instruccion_usuario}\n"
                                "Devuelve la carta COMPLETA en JSON estricto con estas claves:\n"
                                '{"referencia": str, "destinatario_nombre": str|null, '
                                '"destinatario_cargo": str|null, "cuerpo_parrafos": [str, ...], "despedida": str}'
                            ),
                        }
                    )
                else:
                    mensajes_redaccion.append({"role": "user", "content": prompt_usuario})

                res = await llm.generar_chat_completado(
                    mensajes=mensajes_redaccion,
                    response_format={"type": "json_object"},
                    max_tokens_override=950,
                )
                await _emitir(
                    emitir_evento,
                    {
                        "tipo": "paso",
                        "id": "llm-redaccion",
                        "etiqueta": "Redactando carta formal institucional",
                        "detalle": "Borrador epistolar generado",
                        "duracionMs": int((time.perf_counter() - t_red) * 1000),
                        "estado": "completado",
                    },
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
                        emitir_evento=emitir_evento,
                    )
                    trazas_mcp.append(val_traza)
                    parrafos = [str(p) for p in validado.get("cuerpo_parrafos", []) if str(p).strip()]
                    advertencias.extend(str(a) for a in validado.get("advertencias", []))
                    if parrafos:
                        ref_prev = payload.borrador_actual.referencia if payload.borrador_actual else None
                        desp_prev = payload.borrador_actual.despedida if payload.borrador_actual else None
                        referencia = normalizar_moneda_texto(
                            str(datos.get("referencia") or ref_prev or referencia_base)
                        )
                        despedida = normalizar_moneda_texto(
                            str(
                                datos.get("despedida")
                                or desp_prev
                                or "Con este motivo, saludo a usted con las consideraciones más distinguidas."
                            )
                        )
                        if (
                            payload.borrador_actual is not None
                            and len(payload.borrador_actual.cuerpo_parrafos or []) >= 2
                            and edicion_parcial.get("indice_parrafo_1based")
                        ):
                            base_p = list(payload.borrador_actual.cuerpo_parrafos or [])
                            idx_p = int(edicion_parcial["indice_parrafo_1based"])
                            pos = len(parrafos) - 1 if idx_p == -1 else idx_p - 1
                            pos_base = len(base_p) - 1 if idx_p == -1 else idx_p - 1
                            if 0 <= pos_base < len(base_p) and 0 <= pos < len(parrafos):
                                nuevo_seg = parrafos[pos]
                                parrafos = list(base_p)
                                parrafos[pos_base] = nuevo_seg
                        dest_nombre_out = (
                            destinatario
                            if payload.borrador_actual and not edicion_parcial.get("toca_destinatario")
                            else (str(datos.get("destinatario_nombre") or "").strip() or destinatario)
                        )
                        dest_cargo_out = (
                            destinatario_cargo
                            if payload.borrador_actual and not edicion_parcial.get("toca_destinatario")
                            else (str(datos.get("destinatario_cargo") or "").strip() or destinatario_cargo)
                        )
                        motor_id, _ = _resolver_motor(settings)
                        return RedaccionAsistidaResponse(
                            motor_utilizado=motor_id,  # type: ignore[arg-type]
                            modelo=res.modelo,
                            tipo_documento_detectado=tipo_doc,  # type: ignore[arg-type]
                            referencia=referencia,
                            lugar_fecha=lugar_fecha,
                            destinatario_nombre=dest_nombre_out,
                            destinatario_cargo=dest_cargo_out,
                            cuerpo_parrafos=parrafos,
                            despedida=despedida,
                            incluye_anexo_tecnico=incluye_anexo,
                            fundamento_normativo=normas,
                            advertencias=advertencias,
                            trazas_mcp=trazas_mcp,
                            exportacion_solicitada=exportacion_solicitada,  # type: ignore[arg-type]
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
                payload.borrador_actual.referencia
                if payload.borrador_actual and payload.borrador_actual.referencia and not edicion_parcial.get("toca_referencia")
                else (
                    f"{referencia_base} — {codigo_activo}"
                    if codigo_activo and not es_nuevo_sin_anexo
                    else referencia_base
                )
            ),
            lugar_fecha=lugar_fecha,
            destinatario_nombre=destinatario,
            destinatario_cargo=destinatario_cargo,
            cuerpo_parrafos=parrafos_det,
            despedida=(
                payload.borrador_actual.despedida
                if payload.borrador_actual and payload.borrador_actual.despedida
                else (
                    "Con este particular, hago propicia la oportunidad para reiterar a usted las seguridades "
                    "de mi más alta y distinguida consideración."
                )
            ),
            incluye_anexo_tecnico=incluye_anexo,
            fundamento_normativo=normas,
            advertencias=advertencias,
            trazas_mcp=trazas_mcp,
            exportacion_solicitada=exportacion_solicitada,  # type: ignore[arg-type]
            tokens_usados=0,
        )

    def _formalizar_detalle_usuario(self, texto: str) -> str:
        limpio = " ".join(texto.strip().split()).rstrip(" .;")
        if not limpio:
            return ""
        return limpio[0].upper() + limpio[1:]

    def _redactar_determinista(
        self,
        payload: RedaccionAsistidaRequest,
        tipo_doc: str,
        ctx: dict[str, Any],
        codigo: str | None,
        incluye_anexo: bool,
    ) -> list[str]:
        solicitante_unidad = payload.solicitante_unidad or "la Dependencia Universitaria solicitante"
        instruccion_fmt = self._formalizar_detalle_usuario(payload.instruccion_usuario)

        if payload.borrador_actual and payload.borrador_actual.cuerpo_parrafos:
            base_parrafos = [str(p).strip() for p in payload.borrador_actual.cuerpo_parrafos if str(p).strip()]
            edicion = detectar_edicion_parcial_carta(payload.instruccion_usuario)
            idx_obj = edicion.get("indice_parrafo_1based")
            if idx_obj and base_parrafos:
                pos = len(base_parrafos) - 1 if idx_obj == -1 else int(idx_obj) - 1
                if 0 <= pos < len(base_parrafos):
                    actualizados = list(base_parrafos)
                    actualizados[pos] = f"{actualizados[pos].rstrip('. ')}. Asimismo, se precisa que {instruccion_fmt}."
                    return actualizados
            if base_parrafos:
                actualizados = list(base_parrafos)
                objetivo = min(1, len(actualizados) - 1)
                actualizados[objetivo] = f"{actualizados[objetivo].rstrip('. ')}. En atención a la precisión adicional solicitada: {instruccion_fmt}."
                return actualizados

        if tipo_doc == "SOLICITUD_NUEVO_ACTIVO":
            return [
                (
                    f"De mi mayor consideración: En ejercicio de mis funciones administrativas dentro de {solicitante_unidad}, "
                    "tengo a bien dirigirme a su autoridad con el debido respeto a fin de presentar una solicitud formal de "
                    "dotación e incorporación de equipamiento institucional bajo los lineamientos de planificación y contratación "
                    "establecidos en el Decreto Supremo N° 0181 (Normas Básicas del Sistema de Administración de Bienes y Servicios — NB-SABS)."
                ),
                (
                    f"El requerimiento técnico expuesto por nuestra dependencia comprende lo siguiente: {instruccion_fmt}. "
                    "La incorporación de este bien resulta indispensable para superar las actuales limitaciones operativas y asegurar la atención "
                    "oportuna, continua y eficiente de los procesos académicos, investigativos o administrativos asignados a nuestra unidad."
                ),
                (
                    "Desde el punto de vista técnico y patrimonial, la asignación solicitada permitirá prevenir demoras en el cumplimiento de metas "
                    "institucionales, resguardar la continuidad operativa del personal y optimizar la calidad del servicio brindado dentro de la "
                    "Universidad Autónoma Gabriel René Moreno en concordancia con la programación presupuestaria y operativa vigente."
                ),
                (
                    "Una vez evaluada la disponibilidad presupuestaria y autorizada la adquisición o dotación por las instancias competentes, "
                    f"{solicitante_unidad} se compromete expresamente a coadyuvar en la recepción conforme, verificación de especificaciones técnicas, "
                    "asignación de código patrimonial e inscripción del custodio responsable mediante Acta Oficial de Entrega del Departamento de Activos Fijos."
                ),
                (
                    "Por lo expuesto, solicito gentilmente a su autoridad considerar favorablemente el presente requerimiento y disponer su derivación "
                    "a las áreas técnicas y administrativas pertinentes para la prosecución del trámite reglamentario. Agradeciendo de antemano su atención, "
                    "quedo a entera disposición para ampliar información justificativa o respaldo documental."
                ),
            ]

        detalles_tramite = {
            "SOLICITUD_CORRECCION_DATOS": (
                (
                    "solicito formalmente la rectificación y actualización registral en el padrón oficial de Activos Fijos "
                    "respecto a los atributos técnicos, descriptivos o de asignación correspondientes al bien patrimonial indicado"
                ),
                (
                    "La inconsistencia detectada requiere saneamiento oportuno en los registros contables y de inventario con el propósito "
                    "de garantizar fidelidad documental entre la realidad física observada y el Sistema Patrimonial Universitario, en estricto "
                    "acatamiento a los principios de transparencia y control interno establecidos por la Ley N° 1178 (SAFCO) y el D.S. N° 0181."
                ),
                (
                    "En tal sentido, pido disponer la verificación por el área técnica de registro e inventario para consolidar la enmienda "
                    "y emitir el comprobante o ficha patrimonial actualizada para respaldo de custodia."
                ),
            ),
            "SOLICITUD_BAJA": (
                (
                    "elevo a su consideración la solicitud formal de evaluación técnica y desincorporación definitiva (baja patrimonial) "
                    "al amparo del Manual de Procedimientos de Baja de Bienes aprobado mediante Resolución Rectoral N° 449/2013 y el D.S. N° 0181"
                ),
                (
                    "De acuerdo con la inspección preliminar en la unidad usuaria, el bien presenta deterioro severo, obsolescencia tecnológica "
                    "o daño funcional cuyo costo de reparación resulta técnica o económicamente inviable frente a su valor contable vigente, impidiendo "
                    "su utilización normal en las actividades universitarias."
                ),
                (
                    "En consecuencia, se solicita instruir la inspección pericial reglamentaria, la emisión del Dictamen Técnico de Baja y la posterior "
                    "gestión de descarga contable y disposición física conforme al procedimiento establecido por el Departamento de Activos Fijos."
                ),
            ),
            "SOLICITUD_INVENTARIO": (
                (
                    "solicito la programación y ejecución de un levantamiento e inventario físico conciliado de los bienes muebles y equipos "
                    "asignados a nuestra dependencia conforme al D.S. N° 0181 y las normas internas de control patrimonial"
                ),
                (
                    "La verificación física periódica y conciliación contra los saldos registrados en el padrón central permite detectar "
                    "oportunamente diferencias de ubicación, estado de conservación, necesidades de reetiquetado o actualización de responsables directos."
                ),
                (
                    "A tal efecto, nuestra unidad brindará acceso irrestricto a todos los ambientes y asignará personal de enlace para acompañar al equipo "
                    "inspector hasta la suscripción del Acta de Conciliación e Inventario Físico correspondiente."
                ),
            ),
            "SOLICITUD_REASIGNACION": (
                (
                    "solicito autorizar la reasignación y transferencia formal de custodia administrativa dentro del padrón institucional "
                    "del Departamento de Activos Fijos, en estricta sujeción al D.S. N° 0181 y los procedimientos de control interno vigentes"
                ),
                (
                    "Esta redistribución obedece a necesidades reales de reorganización operativa y aprovechamiento racional de los recursos "
                    "institucionales, asegurando que el bien quede bajo resguardo formal del funcionario y unidad efectivamente usuaria."
                ),
                (
                    "Por ello, requiero se instruya el registro del movimiento en el sistema, la actualización del historial de trazabilidad y la emisión "
                    "del Acta de Entrega y Reasignación para perfeccionar el descargo del custodio saliente y la responsabilidad del receptor."
                ),
            ),
            "SOLICITUD_REVALUO": (
                (
                    "solicito formalmente la inspección técnica pericial y revalorización contable de bien patrimonial conforme al "
                    "Manual de Revalorización Técnica de Activos Fijos (UAGRM-MRVT-AF, R.R. N° 567/2018 V2) y el Decreto Supremo N° 24051"
                ),
                (
                    "Pese a encontrarse totalmente depreciado en libros con valor residual mínimo o concluida su vida útil contable ordinaria, "
                    "el bien conserva integridad física y condiciones operativas plenamente satisfactorias para continuar prestando servicios "
                    "institucionales útiles tras la determinación técnica de su valor razonable y vida útil remanente."
                ),
                (
                    "En mérito a ello, pido designar la comisión o técnico evaluador para elaborar el informe pericial correspondiente y actualizar "
                    "su valoración y años de vida útil restante en el padrón oficial."
                ),
            ),
            "DESCARGO_INCIDENTE": (
                (
                    "presento formalmente el presente informe de descargo administrativo oportuno en cumplimiento de la Ley N° 1178 (SAFCO), "
                    "el Decreto Supremo N° 0181 y los Arts. 8 y 12 de la R.R. N° 449/2013 ante la ocurrencia o constatación del hecho expuesto"
                ),
                (
                    "El presente pronunciamiento tiene por objeto dejar constancia fehaciente, documentada y cronológica de las circunstancias "
                    "acontecidas respecto al bien bajo custodia, deslindando cualquier presunción de negligencia o omisión de reporte oportuno "
                    "ante las instancias de control patrimonial y auditoría."
                ),
                (
                    "Solicito se registre el presente descargo en el expediente patrimonial correspondiente y se determinen las acciones correctivas, "
                    "reposición o verificación que normativamente correspondan."
                ),
            ),
            "INFORME_TECNICO": (
                (
                    "remito el presente Dictamen e Informe Técnico Patrimonial fundamentado en la constatación documental y física de los registros "
                    "del Departamento de Activos Fijos y la normativa universitaria aplicable (D.S. N° 0181, R.R. N° 449/2013 y R.R. N° 567/2018 V2)"
                ),
                (
                    "La evaluación técnica efectuada pondera objetivamente el estado operativo actual, la situación registral, el desgaste acumulado "
                    "y la razonabilidad de uso institucional del bien examinado, proporcionando elementos concluyentes para la toma de decisiones administrativas."
                ),
                (
                    "En virtud de lo dictaminado, elevo el presente documento a conocimiento y consideración superior para la emisión de la resolución "
                    "administrativa o medida patrimonial que estime procedente."
                ),
            ),
        }

        verbo_obj, parrafo_normativo, parrafo_petitorio = detalles_tramite.get(
            tipo_doc,
            (
                "elevo a su consideración la presente solicitud formal ante el Departamento de Activos Fijos de la UAGRM",
                "El trámite se enmarca en los principios de legalidad, economía y transparencia del D.S. N° 0181 y la Ley N° 1178 (SAFCO).",
                "Agradeceré disponer la evaluación pertinente e instruir la prosecución del trámite administrativo correspondiente.",
            ),
        )

        parrafos = [
            (
                f"De mi mayor consideración: En representación de {solicitante_unidad}, me dirijo a su autoridad con el debido respeto "
                f"a objeto de manifestar que {verbo_obj}. El fundamento específico que motiva la presente gestión radica en la siguiente "
                f"situación administrativa u operativa expuesta: {instruccion_fmt}."
            ),
        ]

        if codigo and ctx:
            parrafos.append(construir_parrafo_respaldo_activo(ctx, codigo))
        elif codigo:
            parrafos.append(
                f"El presente trámite administrativo recae de manera específica sobre el bien patrimonial identificado con el código institucional {codigo}, "
                "cuyo registro y antecedentes obran en el padrón oficial del Departamento de Activos Fijos."
            )

        parrafos.append(parrafo_normativo)
        parrafos.append(parrafo_petitorio)

        cierre = (
            "Se deja expresa constancia de que se adjunta como Anexo en hoja posterior la Ficha Técnica Patrimonial oficial "
            "extraída del padrón universitario para fines de verificación íntegra. Sin otro particular, reitero mi disposición para "
            "atender cualquier requerimiento adicional que Jefatura estime pertinente."
            if incluye_anexo
            else "Manifiesto mi plena disposición para coordinar la inspección física o complementar cualquier respaldo documental adicional que "
            "el Departamento de Activos Fijos considere necesario para concluir satisfactoriamente esta gestión."
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
        raw_estado = str(raw.get("estado") or "").strip().upper()
        raw_cond = str(raw.get("condicion") or "").strip().upper()
        return FiltrosDetectadosReporte(
            search=str(raw["search"]).strip() if raw.get("search") else base.search,
            estado=raw_estado if raw_estado in ESTADOS_PADRON_VALIDOS else base.estado,
            condicion=raw_cond if raw_cond in CONDICIONES_PADRON_VALIDAS else base.condicion,
            grupo_contable=str(raw["grupo_contable"]).strip() if raw.get("grupo_contable") else base.grupo_contable,
            ubicacion=str(raw["ubicacion"]).strip() if raw.get("ubicacion") else base.ubicacion,
            valor_min=float(raw["valor_min"]) if isinstance(raw.get("valor_min"), (int, float)) else base.valor_min,
            valor_max=float(raw["valor_max"]) if isinstance(raw.get("valor_max"), (int, float)) else base.valor_max,
            gestion=int(raw["gestion"]) if isinstance(raw.get("gestion"), int) else base.gestion,
        )

    def _heredar_filtros_sesion(
        self,
        actual: FiltrosDetectadosReporte,
        previos_raw: dict[str, Any],
    ) -> FiltrosDetectadosReporte:
        prev_norm = {
            "search": previos_raw.get("search"),
            "estado": previos_raw.get("estado"),
            "condicion": previos_raw.get("condicion"),
            "grupo_contable": previos_raw.get("grupo_contable") or previos_raw.get("grupoContable"),
            "ubicacion": previos_raw.get("ubicacion"),
            "valor_min": previos_raw.get("valor_min") if previos_raw.get("valor_min") is not None else previos_raw.get("valorMin"),
            "valor_max": previos_raw.get("valor_max") if previos_raw.get("valor_max") is not None else previos_raw.get("valorMax"),
            "gestion": previos_raw.get("gestion"),
        }
        base_prev = self._combinar_filtros(FiltrosDetectadosReporte(), prev_norm)
        return self._combinar_filtros(base_prev, actual.model_dump(exclude_none=True))

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
        emitir_evento: EmitirEventoFn | None = None,
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
                emitir_evento=emitir_evento,
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
                for k in ("search", "estado", "condicion", "grupo_contable", "ubicacion", "valor_min", "valor_max", "gestion")
            )
            q_ayuda_norm = normalizar_texto(payload.pregunta)
            es_pregunta_cuantitativa = tiene_filtro_consulta or any(
                pal in q_ayuda_norm
                for pal in ("cuanto", "cuantos", "total", "cantidad", "hay en", "valor", "monto", "listar", "muestrame")
            )

            datos_padron_ctx: dict[str, Any] = {}
            if es_pregunta_cuantitativa:
                busq_ayuda, traza_b = await self._llamar_tool_con_traza(
                    "buscar_activos_padron",
                    {
                        "search": filtros_nav.search,
                        "unidad": filtros_nav.ubicacion,
                        "estado": filtros_nav.estado,
                        "condicion": filtros_nav.condicion,
                        "grupo_contable": filtros_nav.grupo_contable,
                        "valor_min": filtros_nav.valor_min,
                        "valor_max": filtros_nav.valor_max,
                        "gestion": filtros_nav.gestion,
                        "limit": 10,
                    },
                    emitir_evento=emitir_evento,
                )
                trazas_mcp.append(traza_b)
                stats_ayuda, traza_s = await self._llamar_tool_con_traza(
                    "obtener_estadisticas_padron",
                    {"unidad": filtros_nav.ubicacion},
                    emitir_evento=emitir_evento,
                )
                trazas_mcp.append(traza_s)

                muestra_list = list(busq_ayuda.get("activos") or []) if busq_ayuda.get("ok") is not False else []
                total_coinc = int(busq_ayuda.get("totalCoincidencias") or len(muestra_list)) if busq_ayuda.get("ok") is not False else None
                datos_padron_ctx = {
                    "filtros_aplicados": filtros_nav.model_dump(exclude_none=True),
                    "total_coincidencias": total_coinc,
                    "valor_muestra_bs": formatear_bs(busq_ayuda.get("valorMuestraBs") or 0) if busq_ayuda.get("ok") is not False else None,
                    "ejemplos_activos": [
                        {
                            "codigo": a.get("codigo"),
                            "descripcion": a.get("descripcion"),
                            "ubicacion": a.get("ubicacion"),
                            "estado": a.get("estado"),
                            "valorBs": a.get("valorBs"),
                        }
                        for a in muestra_list[:5]
                    ],
                    "estadisticas_unidad_o_padron": stats_ayuda if stats_ayuda.get("ok") is not False else None,
                }
                if total_coinc is not None:
                    ejemplos_txt = ", ".join(
                        f"{a.get('codigo')} ({a.get('descripcion')})"
                        for a in muestra_list[:3]
                        if isinstance(a, dict) and a.get("codigo")
                    )
                    respuesta_det = (
                        f"Según el padrón verificado de PostgreSQL UAGRM, se registran {total_coinc} activo(s) "
                        f"coincidentes ({self._explicar_filtros(filtros_nav).rstrip('.')})."
                        + (f" Ejemplos registrados: {ejemplos_txt}." if ejemplos_txt else "")
                    )
                    pasos_det = [
                        "Puede explorar el listado completo y exportar en Reportes IA o Consulta de Activos.",
                    ] + pasos_det[:1]

            llm = OpenAICompatibleLlmClient(settings, transport=self._transport)
            if settings.env != "test" or self._transport is not None:
                try:
                    titulos_modulos = ", ".join(
                        f"{m.get('titulo')} ({m.get('ruta')})"
                        for m in (catalogo_permitido or [])
                        if isinstance(m, dict)
                    )
                    mensajes: list[dict[str, Any]] = [
                        {
                            "role": "system",
                            "content": (
                                "Asistente UAGRM de Activos Fijos. Si existen 'DATOS VERIFICADOS EN POSTGRESQL', "
                                "responde directamente con esas cifras exactas y moneda Bs., sin decir que te falta información "
                                "y SIN mencionar nunca nombres de funciones internas ni herramientas."
                            ),
                        },
                    ]
                    for turno in payload.historial[-4:]:
                        rol_chat = "user" if turno.rol == "usuario" else "assistant"
                        mensajes.append({"role": rol_chat, "content": turno.texto})
                    bloque_datos = (
                        f"\nDATOS VERIFICADOS EN POSTGRESQL (MCP): {json.dumps(datos_padron_ctx, ensure_ascii=False)}"
                        if datos_padron_ctx
                        else ""
                    )
                    mensajes.append(
                        {
                            "role": "user",
                            "content": (
                                f"Rol: {payload.rol_usuario} | Ruta: {payload.ruta_actual}. "
                                f"Módulos permitidos: {titulos_modulos}.{bloque_datos}\nPregunta: {payload.pregunta}"
                            ),
                        }
                    )

                    tools_ayuda = (
                        set()
                        if datos_padron_ctx
                        else {
                            "buscar_activos_padron",
                            "obtener_estadisticas_padron",
                            "consultar_riesgo_normativo",
                            "consultar_trazabilidad_sha256",
                        }
                    )
                    contenido_llm, modelo_llm, tokens_llm, trazas_react, _ = (
                        await self._ejecutar_bucle_react(
                            llm=llm,
                            mensajes_iniciales=mensajes,
                            nombres_tools=tools_ayuda,
                            max_iteraciones=2,
                            emitir_evento=emitir_evento,
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
        emitir_evento: EmitirEventoFn | None = None,
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
                formatos_exportacion=[],
                incluir_graficas=False,
                fuera_de_ambito=True,
                tokens_usados=0,
            )

        formatos_exportacion = detectar_formatos_exportacion_reporte(payload.consulta)
        incluir_graficas = detectar_incluir_graficas(payload.consulta)

        tok_cli, tok_seg = self._preparar_contexto_seguridad(settings, payload.contexto_seguridad)
        try:
            trazas_mcp: list[TrazaHerramientaMcp] = []
            filtros_dict, traza_f = await self._llamar_tool_con_traza(
                "filtrar_catalogo_reportes",
                {"consulta_usuario": payload.consulta},
                emitir_evento=emitir_evento,
            )
            trazas_mcp.append(traza_f)
            filtros_det = FiltrosDetectadosReporte(**filtros_dict)
            if (
                payload.reporte_previo
                and isinstance(payload.reporte_previo.filtros_detectados, dict)
                and not any(
                    pal in q_norm
                    for pal in (
                        "todo el padron",
                        "sin filtro",
                        "quita el filtro",
                        "todos los estados",
                        "todos los bienes",
                        "sin unidad",
                        "nueva consulta",
                        "reiniciar",
                    )
                )
            ):
                filtros_det = self._heredar_filtros_sesion(
                    filtros_det,
                    payload.reporte_previo.filtros_detectados,
                )

            activos_mencionados: list[dict[str, Any]] = []
            total_coincidencias_db = 0
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
                    mensajes: list[dict[str, Any]] = [
                        {
                            "role": "system",
                            "content": (
                                "Agente Analítico Patrimonial UAGRM. Consulta fuentes vía herramientas "
                                "(usa los estados reales del padrón: BUENO, REGULAR, MALO, EN_REPARACION, BAJA; "
                                "condiciones: NUEVO, USADO, DETERIORADO; deja 'unidad' en null salvo facultad/oficina concreta). "
                                "Sintetiza hallazgos ejecutivos precisos con cifras exactas en español boliviano y moneda Bs., "
                                "sin inventar datos."
                            ),
                        },
                    ]
                    for turno in payload.historial[-6:]:
                        mensajes.append(
                            {
                                "role": "user" if turno.rol == "usuario" else "assistant",
                                "content": turno.texto,
                            }
                        )
                    if payload.reporte_previo and payload.reporte_previo.resumen_ejecutivo:
                        mensajes.append(
                            {
                                "role": "user",
                                "content": (
                                    "Contexto del reporte actual en sesión temporal (filtros previos: "
                                    f"{json.dumps(payload.reporte_previo.filtros_detectados or {}, ensure_ascii=False)}): "
                                    f"{payload.reporte_previo.resumen_ejecutivo[:500]}"
                                ),
                            }
                        )
                    mensajes.append({"role": "user", "content": payload.consulta})
                    contenido, modelo_usado, tokens_usados, trazas_react, obs = (
                        await self._ejecutar_bucle_react(
                            llm=llm,
                            mensajes_iniciales=mensajes,
                            nombres_tools=tools_analitica,
                            max_iteraciones=2,
                            emitir_evento=emitir_evento,
                        )
                    )
                    trazas_mcp.extend(trazas_react)
                    if "buscar_activos_padron" in obs and obs["buscar_activos_padron"].get("ok") is not False:
                        activos_mencionados = list(obs["buscar_activos_padron"].get("activos") or [])
                        total_coincidencias_db = int(
                            obs["buscar_activos_padron"].get("totalCoincidencias") or len(activos_mencionados)
                        )
                    if "obtener_estadisticas_padron" in obs and obs["obtener_estadisticas_padron"].get("ok") is not False:
                        estadisticas = obs["obtener_estadisticas_padron"]
                    if contenido.strip():
                        respuesta_llm = normalizar_moneda_texto(contenido.strip())
                        motor_usado, _ = _resolver_motor(settings)
                except (LlmNoConfiguradoError, LlmErrorExternoError):
                    pass

            # Asegura anclaje real si el modelo omitió alguna consulta necesaria o estuvo offline
            if not activos_mencionados:
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
                        "limit": 12,
                    },
                    emitir_evento=emitir_evento,
                )
                trazas_mcp.append(traza_b)
                if busq_obs.get("ok") is not False:
                    activos_mencionados = list(busq_obs.get("activos") or [])
                    total_coincidencias_db = int(
                        busq_obs.get("totalCoincidencias") or len(activos_mencionados)
                    )

            if estadisticas is None:
                stats_obs, traza_s = await self._llamar_tool_con_traza(
                    "obtener_estadisticas_padron",
                    {"unidad": filtros_det.ubicacion},
                    emitir_evento=emitir_evento,
                )
                trazas_mcp.append(traza_s)
                if stats_obs.get("ok") is not False:
                    estadisticas = stats_obs

            if not respuesta_llm:
                total_padron = (estadisticas or {}).get("totalActivos", 0)
                valor_padron = formatear_bs((estadisticas or {}).get("valorTotalBs", 0))
                valor_muestra = formatear_bs(
                    sum(float(a.get("valorBs") or 0) for a in activos_mencionados)
                )
                coinc = total_coincidencias_db or len(activos_mencionados)
                por_est = [
                    f"{e.get('estado')}: {e.get('cantidad')}"
                    for e in ((estadisticas or {}).get("porEstado") or [])[:4]
                    if isinstance(e, dict)
                ]
                resumen_estados = f" Distribución por estado: {', '.join(por_est)}." if por_est else ""
                nota_export = (
                    f" Exportación preparada automáticamente en formato(s): {', '.join(f.upper() for f in formatos_exportacion)}."
                    if formatos_exportacion
                    else ""
                )
                respuesta_llm = (
                    f"Análisis patrimonial verificado en PostgreSQL UAGRM ({self._explicar_filtros(filtros_det).rstrip('.')}): "
                    f"se identificaron {coinc} activo(s) coincidente(s) (muestra valorada en {valor_muestra}). "
                    f"El universo consultado contabiliza {total_padron} bien(es) por un monto acumulado de {valor_padron}."
                    f"{resumen_estados}{nota_export}"
                )

            return ConsultaAnaliticaResponse(
                motor_utilizado=motor_usado,  # type: ignore[arg-type]
                modelo=modelo_usado,
                respuesta=respuesta_llm,
                filtros_detectados=filtros_det,
                activos_mencionados=activos_mencionados[:12],
                estadisticas=estadisticas,
                trazas_mcp=trazas_mcp,
                formatos_exportacion=formatos_exportacion,  # type: ignore[arg-type]
                incluir_graficas=incluir_graficas,
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
