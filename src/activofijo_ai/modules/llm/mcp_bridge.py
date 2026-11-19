"""Puente orquestador ReAct + FastMCP entre herramientas en PostgreSQL y Ollama local."""

import importlib.util
import json
import re
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
from activofijo_ai.modules.llm.corrector_institucional import (
    corregir_texto_institucional,
    listar_facultades_mencionadas,
    resolver_facultad_oficial_uagrm,
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
    extraer_unidad_solicitante,
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
        etiquetas_campos = (
            ("search", "Búsqueda"),
            ("estado", "Estado"),
            ("condicion", "Condición"),
            ("grupo_contable", "Grupo"),
            ("ubicacion", "Unidad"),
            ("valor_min", "Valor mín."),
            ("valor_max", "Valor máx."),
            ("gestion", "Gestión"),
        )
        partes: list[str] = []
        for campo, titulo in etiquetas_campos:
            val = obs.get(campo)
            if val is None or val == "":
                continue
            if campo in ("valor_min", "valor_max"):
                partes.append(f"{titulo}: {formatear_bs(val)}")
            else:
                partes.append(f"{titulo}: {val}")
        if not partes:
            return "Sin criterios restrictivos (consulta general)"
        pref = "1 criterio" if len(partes) == 1 else f"{len(partes)} criterios"
        return f"{pref}: " + " · ".join(partes)
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
        tool_args_override: dict[str, dict[str, Any]] | None = None,
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
                    if tool_args_override and tc.nombre in tool_args_override:
                        for k_ov, v_ov in tool_args_override[tc.nombre].items():
                            if v_ov is not None:
                                args[k_ov] = v_ov
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
        instruccion_limpia = corregir_texto_institucional(payload.instruccion_usuario)
        tipo_doc = payload.tipo_documento
        if not tipo_doc:
            clasif_obs, clasif_traza = await self._llamar_tool_con_traza(
                "clasificar_intencion_redaccion",
                {"instruccion_usuario": instruccion_limpia},
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
        titulo_tramite = str(plantilla.get("titulo_documento") or "Solicitud Administrativa")
        ctx_raw = payload.contexto_activo or {}
        unidad_inferida = (
            corregir_texto_institucional(payload.solicitante_unidad)
            if payload.solicitante_unidad
            else (
                extraer_unidad_solicitante(instruccion_limpia)
                or str(ctx_raw.get("ubicacion") or "").strip()
                or None
            )
        )
        dest_implicito, cargo_implicito = extraer_destinatario_instruccion(instruccion_limpia)
        exportacion_solicitada = detectar_exportacion_carta(instruccion_limpia)
        edicion_parcial = (
            detectar_edicion_parcial_carta(instruccion_limpia)
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

        normas_legibles = "; ".join(
            n.replace("R.R. N°", "Resolución Rectoral N°").replace("R.R.", "Resolución Rectoral")
            for n in normas
        )
        lineas_antecedentes = [
            f"- Tipo de gestión: {titulo_tramite}",
            f"- Solicitante: {payload.solicitante_nombre or 'Autoridad / Representante Universitario'} ({payload.solicitante_cargo or 'Representante Institucional'})",
            f"- Unidad o Facultad solicitante: {unidad_inferida or 'Unidad Universitaria UAGRM'}",
            f"- Destinatario: {destinatario} ({destinatario_cargo})",
            f"- Normativa vigente aplicable: {normas_legibles or 'Decreto Supremo N° 0181 (NB-SABS) y Ley N° 1178 (SAFCO)'}",
        ]
        if incluye_anexo and ctx_verificado:
            lineas_antecedentes.append(
                "- Bien patrimonial verificado en padrón UAGRM: "
                f"Código {codigo_activo}, "
                f"Descripción: {ctx_verificado.get('descripcion') or 'No especificada'}, "
                f"Ubicación actual: {ctx_verificado.get('ubicacion') or 'No especificada'}, "
                f"Estado: {ctx_verificado.get('estado') or 'Registrado'}, "
                f"Valor contabilizado: {ctx_verificado.get('costo_bs') or 'Sin monto detallado'} "
                "(se adjunta ficha técnica oficial como Anexo en página 2)."
            )
        elif es_nuevo_sin_anexo:
            lineas_antecedentes.append(
                "- Modalidad: Nueva compra, adquisición o dotación de equipamiento. "
                "No existe activo fijo previamente codificado ni precio/valor contable asignado aún."
            )
        else:
            lineas_antecedentes.append(
                "- Nota documental: Sin código patrimonial específico adjunto; redactar con base únicamente en los datos descritos por el solicitante."
            )

        antecedentes_legibles = "\n".join(lineas_antecedentes)

        prompt_res = await self._mcp.get_prompt(
            "prompt_redaccion_carta_uagrm",
            {
                "recurso_mcp_json": antecedentes_legibles,
                "instruccion_usuario": instruccion_limpia,
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
                            "Eres Secretario Administrativo Senior de la UAGRM (Santa Cruz, Bolivia). "
                            "Redactas cartas administrativas oficiales impecables, serias y coherentes en español boliviano. "
                            "Jamás menciones términos internos de sistema ('MCP', 'JSON', 'true', 'false') ni inventes precios en Bs."
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
                                f"Corrección solicitada por el usuario: {instruccion_limpia}\n"
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
                    max_tokens_override=650 if payload.borrador_actual is not None else 360,
                    temperature_override=0.2,
                )
                await _emitir(
                    emitir_evento,
                    {
                        "tipo": "paso",
                        "id": "llm-redaccion",
                        "etiqueta": "Redactando carta formal institucional",
                        "detalle": "Análisis semántico y redacción completados",
                        "duracionMs": int((time.perf_counter() - t_red) * 1000),
                        "estado": "completado",
                    },
                )
                datos = extraer_json_objeto(res.contenido)
                if datos and payload.borrador_actual is None and not isinstance(datos.get("cuerpo_parrafos"), list):
                    referencia_comp, parrafos_comp, _unidad_comp = self._componer_carta_desde_semantica(
                        datos_llm=datos,
                        tipo_doc=tipo_doc,
                        referencia_base=referencia_base,
                        instruccion_limpia=instruccion_limpia,
                        unidad_inferida=unidad_inferida,
                        ctx_verificado=ctx_verificado,
                        ficha_mcp=ficha_mcp,
                        codigo_activo=codigo_activo,
                        incluye_anexo=incluye_anexo,
                        es_nuevo_sin_anexo=es_nuevo_sin_anexo,
                    )
                    if parrafos_comp:
                        validado, val_traza = await self._llamar_tool_con_traza(
                            "validar_integridad_texto",
                            {
                                "cuerpo_parrafos": parrafos_comp,
                                "contexto_activo": ctx_verificado,
                                "cod_activo": codigo_activo,
                                "es_nuevo_sin_anexo": es_nuevo_sin_anexo,
                            },
                            emitir_evento=emitir_evento,
                        )
                        trazas_mcp.append(val_traza)
                        parrafos_out = [str(p) for p in validado.get("cuerpo_parrafos", []) if str(p).strip()]
                        advertencias.extend(str(a) for a in validado.get("advertencias", []))
                        motor_id, _ = _resolver_motor(settings)
                        return RedaccionAsistidaResponse(
                            motor_utilizado=motor_id,  # type: ignore[arg-type]
                            modelo=res.modelo,
                            tipo_documento_detectado=tipo_doc,  # type: ignore[arg-type]
                            referencia=normalizar_moneda_texto(referencia_comp),
                            lugar_fecha=lugar_fecha,
                            destinatario_nombre=destinatario,
                            destinatario_cargo=destinatario_cargo,
                            cuerpo_parrafos=parrafos_out,
                            despedida=(
                                "Con este particular, hago propicia la oportunidad para reiterar a usted las seguridades "
                                "de mi más alta y distinguida consideración."
                            ),
                            incluye_anexo_tecnico=incluye_anexo,
                            fundamento_normativo=normas,
                            advertencias=advertencias,
                            trazas_mcp=trazas_mcp,
                            exportacion_solicitada=exportacion_solicitada,  # type: ignore[arg-type]
                            tokens_usados=res.tokens_usados,
                        )
                if datos and isinstance(datos.get("cuerpo_parrafos"), list):
                    parrafos_crudos_llm = [
                        corregir_texto_institucional(str(p))
                        for p in datos["cuerpo_parrafos"]
                        if str(p).strip()
                    ]
                    motivo_rechazo = self._respuesta_carta_es_deficiente(
                        referencia=corregir_texto_institucional(str(datos.get("referencia") or "")),
                        parrafos=parrafos_crudos_llm,
                        instruccion_usuario=instruccion_limpia,
                        ctx_verificado=ctx_verificado,
                        ficha_mcp=ficha_mcp,
                        unidad_inferida=unidad_inferida,
                    )
                    if motivo_rechazo is not None:
                        advertencias.append(motivo_rechazo)
                    else:
                        validado, val_traza = await self._llamar_tool_con_traza(
                            "validar_integridad_texto",
                            {
                                "cuerpo_parrafos": parrafos_crudos_llm,
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
                    "Modelo Ollama local no disponible en este instante; se aplicó redacción formal institucional."
                )

        parrafos_det = self._redactar_determinista(
            payload,
            tipo_doc,
            ctx_verificado,
            codigo_activo,
            incluye_anexo,
            unidad_inferida,
            instruccion_limpia,
        )
        referencia_det = self._construir_referencia_especifica(
            referencia_base=referencia_base,
            tipo_doc=tipo_doc,
            codigo_activo=codigo_activo,
            unidad_inferida=unidad_inferida,
            instruccion=instruccion_limpia,
            es_nuevo_sin_anexo=es_nuevo_sin_anexo,
        )
        return RedaccionAsistidaResponse(
            motor_utilizado="respaldo_determinista",
            modelo=f"{settings.llm_model} (redaccion-normativa)",
            tipo_documento_detectado=tipo_doc,  # type: ignore[arg-type]
            referencia=(
                payload.borrador_actual.referencia
                if payload.borrador_actual and payload.borrador_actual.referencia and not edicion_parcial.get("toca_referencia")
                else referencia_det
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

    def _sanitizar_bloque_semantico(
        self,
        texto: Any,
        facultades_permitidas: set[str],
        permite_bs: bool,
    ) -> str:
        bruto = corregir_texto_institucional(" ".join(str(texto or "").split()).strip())
        if not bruto:
            return ""
        bruto = re.sub(
            r"^(?:se[ñn]or(?:a|\(a\))?\s+[^,.;:]+[,.;:]\s*|de\s+mi\s+mayor\s+consideraci[oó]n[,.;:]\s*)+",
            "",
            bruto,
            flags=re.IGNORECASE,
        ).strip()
        norm = normalizar_texto(bruto)
        if re.search(
            r"\b(?:mcp|json|true|false|recurso activo|reglamentos relativos|reglamento relativo|interno sistema|publico home)\b",
            norm,
        ) or any(k in bruto.lower() for k in ("incluye_anexo", "recurso_activo", "tipo_documento", "solicitud_")):
            return ""
        if not permite_bs and re.search(r"\bbs\.?\s*\d", norm):
            return ""
        facs_campo = listar_facultades_mencionadas(bruto)
        if facultades_permitidas and (facs_campo - facultades_permitidas):
            return ""
        if not facultades_permitidas and facs_campo:
            return ""
        return bruto

    def _componer_carta_desde_semantica(
        self,
        datos_llm: dict[str, Any],
        tipo_doc: str,
        referencia_base: str,
        instruccion_limpia: str,
        unidad_inferida: str | None,
        ctx_verificado: dict[str, Any],
        ficha_mcp: dict[str, Any],
        codigo_activo: str | None,
        incluye_anexo: bool,
        es_nuevo_sin_anexo: bool,
    ) -> tuple[str, list[str], str]:
        facultades_permitidas = listar_facultades_mencionadas(instruccion_limpia)
        if unidad_inferida:
            facultades_permitidas |= listar_facultades_mencionadas(unidad_inferida)
        fac_oficial_usuario = resolver_facultad_oficial_uagrm(instruccion_limpia)
        if fac_oficial_usuario:
            facultades_permitidas.add(fac_oficial_usuario)

        tiene_costo_real = bool(
            (ficha_mcp and ficha_mcp.get("costo_bs"))
            or (
                ctx_verificado
                and (
                    ctx_verificado.get("costo_bs")
                    or ctx_verificado.get("valor_bs")
                    or ctx_verificado.get("costo")
                    or ctx_verificado.get("valor")
                )
            )
        )
        menciona_precio_usuario = bool(
            re.search(
                r"\b(?:bs|bolivian|usd|dolar|precio|costo|valor|presupuesto|cotizacion)\b|\d+\s*(?:bs|\$)",
                normalizar_texto(instruccion_limpia),
            )
        )
        permite_bs = tiene_costo_real or menciona_precio_usuario

        unidad_llm = self._sanitizar_bloque_semantico(
            datos_llm.get("unidad_solicitante"), facultades_permitidas, permite_bs=False
        )
        unidad_final = (
            unidad_inferida
            or (resolver_facultad_oficial_uagrm(unidad_llm) if unidad_llm else None)
            or unidad_llm
            or "la Dependencia Universitaria solicitante"
        )
        if fac_oficial_usuario and not unidad_inferida:
            unidad_final = fac_oficial_usuario

        objeto = self._sanitizar_bloque_semantico(
            datos_llm.get("objeto_concreto"), facultades_permitidas, permite_bs=permite_bs
        )
        if not objeto or len(objeto) < 12:
            objeto = self._extraer_nucleo_solicitud(instruccion_limpia)
        objeto = objeto.rstrip(" .;:")
        if objeto and objeto[0].isupper() and not objeto.startswith(("UAGRM", "DUAF", "FICCT", "POA")):
            objeto = objeto[0].lower() + objeto[1:]

        justificacion = self._sanitizar_bloque_semantico(
            datos_llm.get("justificacion_tecnica"), facultades_permitidas, permite_bs=permite_bs
        )
        impacto = self._sanitizar_bloque_semantico(
            datos_llm.get("impacto_institucional"), facultades_permitidas, permite_bs=permite_bs
        )
        detalle = self._sanitizar_bloque_semantico(
            datos_llm.get("detalle_complementario"), facultades_permitidas, permite_bs=permite_bs
        )
        asunto_llm = self._sanitizar_bloque_semantico(
            datos_llm.get("asunto_especifico"), facultades_permitidas, permite_bs=False
        )

        if asunto_llm and len(asunto_llm) >= 12:
            ref_limpia = asunto_llm.upper().rstrip(" .:")
            if unidad_final != "la Dependencia Universitaria solicitante" and normalizar_texto(unidad_final) not in normalizar_texto(ref_limpia):
                referencia = f"{ref_limpia} — {unidad_final.upper()}"
            else:
                referencia = ref_limpia
        else:
            referencia = self._construir_referencia_especifica(
                referencia_base=referencia_base,
                tipo_doc=tipo_doc,
                codigo_activo=codigo_activo,
                unidad_inferida=unidad_final if unidad_final != "la Dependencia Universitaria solicitante" else None,
                instruccion=instruccion_limpia,
                es_nuevo_sin_anexo=es_nuevo_sin_anexo,
            )

        if tipo_doc == "SOLICITUD_NUEVO_ACTIVO":
            p1 = (
                f"De mi mayor consideración: En representación de {unidad_final}, tengo a bien dirigirme a su autoridad "
                f"con el debido respeto a objeto de elevar formalmente la solicitud correspondiente a {objeto}, en el marco "
                "de la planificación institucional vigente y de las Normas Básicas del Sistema de Administración de Bienes y "
                "Servicios aprobadas mediante Decreto Supremo N° 0181 (NB-SABS)."
            )
            p2 = (
                (f"{justificacion.rstrip('.')}. " if justificacion else "")
                + (
                    f"{detalle.rstrip('.')}. "
                    if detalle and normalizar_texto(detalle) not in normalizar_texto(justificacion)
                    else ""
                )
                + (
                    f"El requerimiento expuesto por {unidad_final} responde a necesidades académicas, técnicas y operativas "
                    "debidamente verificadas, cuya atención resulta indispensable para fortalecer el desarrollo continuo de "
                    "las actividades docentes, prácticas y de gestión universitaria en condiciones acordes a los estándares institucionales."
                    if len(justificacion.split()) < 22
                    else ""
                )
            ).strip()
            p3 = (
                (f"{impacto.rstrip('.')}. " if impacto else "")
                + "Desde la perspectiva administrativa y de control interno establecida en la Ley N° 1178 (SAFCO), la "
                "incorporación oportuna de este equipamiento coadyuvará al cumplimiento de las metas institucionales programadas "
                "en el Plan Operativo Anual (POA), evitando brechas operativas y asegurando la calidad del servicio brindado por la UAGRM."
            ).strip()
            p4 = (
                "Una vez evaluada la disponibilidad presupuestaria y autorizada la adquisición o dotación por las instancias "
                f"competentes, {unidad_final} asume el compromiso expreso de participar en la verificación técnica de especificaciones, "
                "recepción conforme, asignación de código patrimonial e inscripción formal del custodio responsable mediante "
                "Acta Oficial de Entrega ante el Departamento de Activos Fijos."
            )
            p5 = (
                "Por lo anteriormente expuesto y fundamentado, solicito respetuosamente a su autoridad considerar favorablemente "
                "el presente requerimiento y disponer su derivación a las áreas financieras y administrativas pertinentes para la "
                "prosecución del trámite reglamentario, quedando a entera disposición para remitir especificaciones técnicas "
                "detalladas o cotizaciones referenciales cuando sean requeridas."
            )
            return referencia, [p1, p2, p3, p4, p5], unidad_final

        marcos_tramite: dict[str, tuple[str, str, str]] = {
            "SOLICITUD_BAJA": (
                "elevar a su consideración la solicitud formal de evaluación técnica y desincorporación definitiva (baja patrimonial)",
                "al amparo del Manual de Procedimientos de Baja de Bienes aprobado mediante Resolución Rectoral N° 449/2013, el Decreto Supremo N° 0181 (NB-SABS) y la Ley N° 1178 (SAFCO)",
                "instruir la inspección técnica pericial reglamentaria, la emisión del Dictamen Técnico de Baja y la consiguiente descarga contable y disposición física conforme al procedimiento vigente del Departamento de Activos Fijos",
            ),
            "SOLICITUD_REVALUO": (
                "solicitar formalmente la inspección técnica pericial y revalorización contable del bien patrimonial",
                "conforme al Manual de Revalorización Técnica de Activos Fijos (UAGRM-MRVT-AF, Resolución Rectoral N° 567/2018 V2), las Normas Contables aplicables y el Decreto Supremo N° 24051",
                "designar a la instancia o comisión técnica evaluadora para determinar el valor razonable y la vida útil remanente del bien, actualizando su registro en el padrón oficial",
            ),
            "SOLICITUD_REASIGNACION": (
                "solicitar autorización para la reasignación y transferencia formal de custodia administrativa",
                "en estricto cumplimiento del Decreto Supremo N° 0181 (NB-SABS) y la Ley N° 1178 (SAFCO) sobre responsabilidad por la custodia de bienes públicos",
                "instruir el registro del traspaso en el sistema patrimonial, la actualización del historial de trazabilidad y la emisión del Acta Oficial de Entrega y Reasignación para perfeccionar el descargo correspondiente",
            ),
            "SOLICITUD_INVENTARIO": (
                "solicitar la programación y ejecución de un levantamiento e inventario físico conciliado de bienes",
                "en cumplimiento de las disposiciones de control y conciliación física establecidas en el Decreto Supremo N° 0181 (NB-SABS) y la Ley N° 1178 (SAFCO)",
                "disponer el cronograma de verificación física con el equipo técnico de Control Patrimonial hasta la suscripción del Acta de Conciliación e Inventario correspondiente",
            ),
            "DESCARGO_INCIDENTE": (
                "presentar el presente informe de descargo y notificación administrativa oportuna",
                "en acatamiento de los Artículos 8 y 12 de la Resolución Rectoral N° 449/2013, el Decreto Supremo N° 0181 y la Ley N° 1178 (SAFCO)",
                "registrar el presente descargo documentado en el expediente patrimonial correspondiente y disponer las verificaciones o acciones administrativas que normativamente procedan",
            ),
            "INFORME_TECNICO": (
                "remitir formalmente el presente Dictamen e Informe Técnico Patrimonial",
                "sustentado en la verificación física y documental conforme al Decreto Supremo N° 0181 y las Resoluciones Rectorales N° 449/2013 y N° 567/2018 V2",
                "elevar el presente dictamen a consideración superior para la adopción de la determinación administrativa y patrimonial que corresponda",
            ),
            "SOLICITUD_CORRECCION_DATOS": (
                "solicitar formalmente la rectificación y actualización registral en el padrón institucional de Activos Fijos",
                "en observancia de los principios de veracidad registral y control interno establecidos por la Ley N° 1178 (SAFCO) y el Decreto Supremo N° 0181 (NB-SABS)",
                "disponer la verificación por el área técnica de registro e inventario para consolidar la enmienda y emitir la ficha patrimonial actualizada para respaldo de custodia",
            ),
        }
        verbo, marco_legal, petitorio = marcos_tramite.get(
            tipo_doc, marcos_tramite["SOLICITUD_CORRECCION_DATOS"]
        )

        p1 = (
            f"De mi mayor consideración: En representación de {unidad_final}, me dirijo a su autoridad con el debido respeto "
            f"a objeto de {verbo}, vinculada específicamente a {objeto}, {marco_legal}."
        )
        parrafos_comp: list[str] = [p1]
        if codigo_activo and ctx_verificado:
            parrafos_comp.append(construir_parrafo_respaldo_activo(ctx_verificado, codigo_activo))
        elif codigo_activo:
            parrafos_comp.append(
                f"El presente trámite administrativo recae sobre el bien patrimonial identificado con el código institucional "
                f"{codigo_activo}, cuyos antecedentes registrales y de custodia obran en el padrón oficial del Departamento de Activos Fijos."
            )

        p_just = (
            (f"{justificacion.rstrip('.')}. " if justificacion else "")
            + (
                f"{detalle.rstrip('.')}. "
                if detalle and normalizar_texto(detalle) not in normalizar_texto(justificacion)
                else ""
            )
            + (
                "Esta actuación resulta indispensable para resguardar la integridad documental, la trazabilidad histórica "
                "y la correcta administración del patrimonio universitario bajo responsabilidad de nuestra unidad."
                if len(justificacion.split()) < 18
                else ""
            )
        ).strip()
        parrafos_comp.append(p_just)

        p_imp = (
            (f"{impacto.rstrip('.')}. " if impacto else "")
            + (
                "Asimismo, se deja expresa constancia de que se adjunta como Anexo en hoja posterior la Ficha Técnica "
                "Patrimonial verificada del padrón institucional para respaldo íntegro del expediente."
                if incluye_anexo
                else "La presente gestión se ajusta estrictamente a los procedimientos internos de control patrimonial y "
                "transparencia administrativa vigentes en la Universidad Autónoma Gabriel René Moreno."
            )
        ).strip()
        parrafos_comp.append(p_imp)

        parrafos_comp.append(
            f"En virtud de lo expuesto, solicito respetuosamente a su autoridad {petitorio}, manifestando nuestra plena "
            "disposición para coordinar inspecciones técnicas o presentar la documentación complementaria que Jefatura estime pertinente."
        )
        return referencia, parrafos_comp, unidad_final

    def _respuesta_carta_es_deficiente(
        self,
        referencia: str,
        parrafos: list[str],
        instruccion_usuario: str,
        ctx_verificado: dict[str, Any],
        ficha_mcp: dict[str, Any] | None = None,
        unidad_inferida: str | None = None,
    ) -> str | None:
        if not parrafos:
            return "Respuesta del modelo sin párrafos redactados."
        unido = f"{referencia} {' '.join(parrafos)}"
        norm = normalizar_texto(unido)
        if re.search(
            r"\b(?:mcp|json|true|false|recurso activo|reglamentos relativos|reglamento relativo|interno sistema|publico home)\b",
            norm,
        ) or any(
            k in unido.lower()
            for k in ("incluye_anexo", "recurso_activo", "tipo_documento", "solicitud_")
        ):
            return (
                "Se aplicó redacción formal institucional verificada al detectar términos técnicos internos en la salida del modelo."
            )

        facs_permitidas = listar_facultades_mencionadas(instruccion_usuario)
        if unidad_inferida:
            facs_permitidas |= listar_facultades_mencionadas(unidad_inferida)
        facs_salida = listar_facultades_mencionadas(unido)
        if facs_permitidas and (facs_salida - facs_permitidas):
            return (
                "Se ajustó el borrador para preservar exactamente la facultad o unidad indicada en la solicitud."
            )

        tiene_costo_real = bool(
            (ficha_mcp and ficha_mcp.get("costo_bs"))
            or (
                ctx_verificado
                and (
                    ctx_verificado.get("costo_bs")
                    or ctx_verificado.get("valor_bs")
                    or ctx_verificado.get("costo")
                    or ctx_verificado.get("valor")
                )
            )
        )
        menciona_precio_usuario = bool(
            re.search(
                r"\b(?:bs|bolivian|usd|dolar|precio|costo|valor|presupuesto|cotizacion)\b|\d+\s*(?:bs|\$)",
                normalizar_texto(instruccion_usuario),
            )
        )
        if not tiene_costo_real and not menciona_precio_usuario and re.search(r"\bbs\.?\s*\d", norm):
            return (
                "Se sustituyó el borrador para evitar montos estimados en Bs. no respaldados en la solicitud ni en el padrón."
            )
        return None

    def _construir_referencia_especifica(
        self,
        referencia_base: str,
        tipo_doc: str,
        codigo_activo: str | None,
        unidad_inferida: str | None,
        instruccion: str,
        es_nuevo_sin_anexo: bool,
    ) -> str:
        if tipo_doc == "SOLICITUD_NUEVO_ACTIVO":
            base = "SOLICITUD DE ADQUISICIÓN Y DOTACIÓN DE NUEVO EQUIPAMIENTO INSTITUCIONAL"
            if unidad_inferida:
                return f"{base} — {unidad_inferida.upper()}"
            return base
        if codigo_activo and not es_nuevo_sin_anexo:
            return f"{referencia_base} — {codigo_activo}"
        if unidad_inferida:
            return f"{referencia_base} — {unidad_inferida.upper()}"
        return referencia_base

    def _extraer_nucleo_solicitud(self, texto: str) -> str:
        limpio = " ".join(texto.strip().split()).rstrip(" .;")
        if not limpio:
            return "el requerimiento institucional descrito"
        sin_intro = re.sub(
            r"^(?:como\s+representant?e?s?(?:\s+de\s+[^,;]+)?[,;\s]+|en\s+representaci[oó]n\s+de\s+[^,;]+[,;\s]+|por\s+medio\s+de\s+la\s+presente[,;\s]+)+",
            "",
            limpio,
            flags=re.IGNORECASE,
        ).strip()
        sin_verbo = re.sub(
            r"^(?:solicito|solicitamos|requiero|requerimos|necesito|necesitamos|quiero|pedimos)\s+",
            "",
            sin_intro,
            flags=re.IGNORECASE,
        ).strip()
        nucleo = sin_verbo or sin_intro or limpio
        nucleo = re.sub(r"\bnuestras?\s+", lambda m: "las " if m.group(0).lower().endswith("s ") else "la ", nucleo, flags=re.IGNORECASE)
        return nucleo

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
        unidad_inferida: str | None = None,
        instruccion_corregida: str | None = None,
    ) -> list[str]:
        solicitante_unidad = (
            unidad_inferida
            or payload.solicitante_unidad
            or "la Dependencia Universitaria solicitante"
        )
        texto_base = instruccion_corregida or corregir_texto_institucional(payload.instruccion_usuario)
        nucleo = self._extraer_nucleo_solicitud(texto_base)

        if payload.borrador_actual and payload.borrador_actual.cuerpo_parrafos:
            base_parrafos = [str(p).strip() for p in payload.borrador_actual.cuerpo_parrafos if str(p).strip()]
            edicion = detectar_edicion_parcial_carta(payload.instruccion_usuario)
            idx_obj = edicion.get("indice_parrafo_1based")
            if idx_obj and base_parrafos:
                pos = len(base_parrafos) - 1 if idx_obj == -1 else int(idx_obj) - 1
                if 0 <= pos < len(base_parrafos):
                    actualizados = list(base_parrafos)
                    actualizados[pos] = f"{actualizados[pos].rstrip('. ')}. Asimismo, se precisa que {nucleo}."
                    return actualizados
            if base_parrafos:
                actualizados = list(base_parrafos)
                objetivo = min(1, len(actualizados) - 1)
                actualizados[objetivo] = f"{actualizados[objetivo].rstrip('. ')}. En atención a la precisión adicional solicitada: {nucleo}."
                return actualizados

        if tipo_doc == "SOLICITUD_NUEVO_ACTIVO":
            return [
                (
                    f"De mi mayor consideración: En representación de {solicitante_unidad}, "
                    "me dirijo a su autoridad con el debido respeto a objeto de presentar la solicitud formal correspondiente a "
                    f"{nucleo}, en estricta sujeción a la planificación institucional y a las Normas Básicas del Sistema de "
                    "Administración de Bienes y Servicios aprobadas mediante Decreto Supremo N° 0181 (NB-SABS)."
                ),
                (
                    f"El requerimiento expuesto por {solicitante_unidad} responde a necesidades académicas y operativas debidamente constatadas, "
                    "cuya atención resulta indispensable para fortalecer el desarrollo de actividades formativas, prácticas técnicas, "
                    "exposiciones académicas y labores administrativas regulares en condiciones adecuadas de calidad universitaria."
                ),
                (
                    "Desde el ámbito administrativo y de control interno establecido en la Ley N° 1178 (SAFCO), la incorporación oportuna de "
                    "estos bienes permitirá cubrir las brechas actuales de equipamiento, prevenir interrupciones en el dictado de clases o "
                    "prácticas y garantizar la ejecución eficiente de los objetivos trazados en el Plan Operativo Anual (POA) de la unidad."
                ),
                (
                    "Una vez evaluada la disponibilidad presupuestaria y autorizada la adquisición o dotación por las instancias competentes, "
                    f"{solicitante_unidad} asume el compromiso expreso de coadyuvar en la verificación técnica de especificaciones, "
                    "recepción conforme, asignación de código patrimonial e inscripción formal del custodio responsable mediante Acta de Entrega del Departamento de Activos Fijos."
                ),
                (
                    "Por lo anteriormente fundamentado, solicito respetuosamente a su autoridad considerar favorablemente esta gestión y disponer "
                    "su derivación a las áreas financieras y de contrataciones pertinentes para la prosecución del trámite reglamentario, quedando "
                    "a plena disposición para remitir especificaciones técnicas o cotizaciones referenciales cuando sean requeridas."
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
                f"a objeto de manifestar que {verbo_obj}. La presente gestión institucional atiende específicamente {nucleo}."
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
                    self._args_padron_desde_filtros(filtros_det, limit=10),
                )
                trazas_mcp.append(traza_b)
                stats_obs, traza_s = await self._llamar_tool_con_traza(
                    "obtener_estadisticas_padron",
                    self._args_padron_desde_filtros(filtros_det),
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

    def _args_padron_desde_filtros(
        self, f: FiltrosDetectadosReporte, limit: int | None = None
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "search": f.search,
            "unidad": f.ubicacion,
            "estado": f.estado,
            "condicion": f.condicion,
            "grupo_contable": f.grupo_contable,
            "valor_min": f.valor_min,
            "valor_max": f.valor_max,
            "gestion": f.gestion,
        }
        if limit is not None:
            args["limit"] = limit
        return args

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
                    self._args_padron_desde_filtros(filtros_nav),
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
        consulta_corregida = corregir_texto_institucional(payload.consulta)
        q_norm = normalizar_texto(consulta_corregida)
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

        formatos_exportacion = detectar_formatos_exportacion_reporte(consulta_corregida)
        incluir_graficas = detectar_incluir_graficas(consulta_corregida)

        tok_cli, tok_seg = self._preparar_contexto_seguridad(settings, payload.contexto_seguridad)
        try:
            trazas_mcp: list[TrazaHerramientaMcp] = []
            filtros_dict, traza_f = await self._llamar_tool_con_traza(
                "filtrar_catalogo_reportes",
                {"consulta_usuario": consulta_corregida},
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

            args_stats_det = self._args_padron_desde_filtros(filtros_det)
            args_busqueda_det = self._args_padron_desde_filtros(filtros_det, limit=12)
            enforced_tool_args = {
                "buscar_activos_padron": args_busqueda_det,
                "obtener_estadisticas_padron": args_stats_det,
            }

            obs: dict[str, dict[str, Any]] = {}
            es_consulta_especializada = any(
                k in q_norm
                for k in (
                    "riesgo",
                    "candidato",
                    "random forest",
                    "proyeccion",
                    "trazabilidad",
                    "sha",
                    "normativa",
                    "reglamento",
                    "vida util",
                )
            )
            usar_react = self._transport is not None or es_consulta_especializada

            if (settings.env != "test" or self._transport is not None) and usar_react:
                try:
                    criterios_obligatorios = {
                        k: v for k, v in args_stats_det.items() if v is not None
                    }
                    mensajes: list[dict[str, Any]] = [
                        {
                            "role": "system",
                            "content": (
                                "Agente Analítico Patrimonial UAGRM. Consulta fuentes vía herramientas "
                                "aplicando ESTRICTAMENTE los criterios solicitados "
                                f"({json.dumps(criterios_obligatorios, ensure_ascii=False)}). "
                                "Usa los estados reales (BUENO, REGULAR, MALO, EN_REPARACION, BAJA) y "
                                "condiciones (NUEVO, USADO, DETERIORADO). Sintetiza únicamente información "
                                "de los bienes coincidentes en español boliviano y moneda Bs.; si hay 0 coincidencias, indícalo claramente."
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
                            tool_args_override=enforced_tool_args,
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
                    obs = {}

            # Garantiza anclaje real filtrado incluso si el LLM omitió consultar o en vía rápida directa
            busco_db = "buscar_activos_padron" in obs and obs["buscar_activos_padron"].get("ok") is not False
            if not busco_db:
                busq_obs, traza_b = await self._llamar_tool_con_traza(
                    "buscar_activos_padron",
                    args_busqueda_det,
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
                    args_stats_det,
                    emitir_evento=emitir_evento,
                )
                trazas_mcp.append(traza_s)
                if stats_obs.get("ok") is not False:
                    estadisticas = stats_obs

            if not respuesta_llm:
                respuesta_llm = self._construir_sintesis_ejecutiva_padron(
                    filtros_det=filtros_det,
                    estadisticas=estadisticas,
                    activos_mencionados=activos_mencionados,
                    total_coincidencias_db=total_coincidencias_db,
                    formatos_exportacion=formatos_exportacion,
                )
                if not usar_react and settings.env != "test":
                    await _emitir(
                        emitir_evento,
                        {
                            "tipo": "paso",
                            "id": "sintesis-padron",
                            "etiqueta": "Consolidando síntesis ejecutiva del informe",
                            "detalle": "Resumen verificado con datos del padrón",
                            "duracionMs": 5,
                            "estado": "completado",
                        },
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

    def _construir_sintesis_ejecutiva_padron(
        self,
        filtros_det: FiltrosDetectadosReporte,
        estadisticas: dict[str, Any] | None,
        activos_mencionados: list[dict[str, Any]],
        total_coincidencias_db: int,
        formatos_exportacion: list[str],
    ) -> str:
        criterios_txt = self._explicar_filtros(filtros_det).rstrip(".")
        total_filtrados = int(
            (estadisticas or {}).get("totalActivos")
            or total_coincidencias_db
            or len(activos_mencionados)
        )
        if total_filtrados == 0:
            return (
                f"No se identificaron activos fijos que cumplan exactamente con los criterios aplicados "
                f"({criterios_txt}). Verifique la dependencia, estado o palabra clave e intente ajustar la consulta."
            )

        valor_total = float(
            (estadisticas or {}).get("valorTotalBs")
            if estadisticas and estadisticas.get("valorTotalBs") is not None
            else sum(float(a.get("valorBs") or 0) for a in activos_mencionados)
        )
        valor_fmt = formatear_bs(valor_total)

        partes: list[str] = [
            (
                f"El presente informe consolida {total_filtrados} bien(es) patrimoniales bajo los criterios aplicados "
                f"({criterios_txt}), con un valor contable acumulado de {valor_fmt}."
            )
        ]

        por_estado = [
            e for e in ((estadisticas or {}).get("porEstado") or [])[:4] if isinstance(e, dict)
        ]
        if por_estado:
            items_est = [
                f"{int(e.get('cantidad') or 0)} en estado {e.get('estado')} ({formatear_bs(float(e.get('valorBs') or 0))})"
                for e in por_estado
                if int(e.get("cantidad") or 0) > 0
            ]
            if items_est:
                partes.append(f"Distribución por estado: {'; '.join(items_est)}.")

        por_grupo = [
            g
            for g in ((estadisticas or {}).get("porGrupoTop") or (estadisticas or {}).get("porGrupo") or [])[:2]
            if isinstance(g, dict)
        ]
        if por_grupo:
            g0 = por_grupo[0]
            nombre_g = g0.get("grupo") or g0.get("nombre")
            cant_g = int(g0.get("cantidad") or 0)
            val_g = formatear_bs(float(g0.get("valorBs") or 0))
            if nombre_g and cant_g > 0:
                partes.append(
                    f"El principal grupo contable corresponde a {nombre_g} ({cant_g} bien(es), {val_g})."
                )

        ordenados = sorted(
            [a for a in activos_mencionados if isinstance(a, dict) and a.get("codigo")],
            key=lambda x: float(x.get("valorBs") or 0),
            reverse=True,
        )[:3]
        if ordenados:
            ejemplos = [
                f"{a.get('codigo')} — {str(a.get('descripcion') or 'Activo')[:42].rstrip()} ({a.get('estado') or 'ACTIVO'}, {formatear_bs(float(a.get('valorBs') or 0))})"
                for a in ordenados
            ]
            partes.append(f"Activos destacados del lote: {'; '.join(ejemplos)}.")

        if formatos_exportacion:
            partes.append(
                f"Documento listo para emisión en formato(s): {', '.join(f.upper() for f in formatos_exportacion)}."
            )

        return " ".join(partes)

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
