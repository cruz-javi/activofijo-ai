import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from activofijo_ai.core.config import Settings, get_settings
from activofijo_ai.core.security import verificar_clave_interna
from activofijo_ai.modules.llm.mcp_bridge import McpBridgeService
from activofijo_ai.modules.llm.provider import LlmErrorExternoError, LlmNoConfiguradoError
from activofijo_ai.modules.llm.schemas import (
    AyudaSistemaRequest,
    AyudaSistemaResponse,
    ConsultaAnaliticaRequest,
    ConsultaAnaliticaResponse,
    ConsultaLenguajeNaturalRequest,
    ConsultaLenguajeNaturalResponse,
    DiagnosticoMetricasRequest,
    DiagnosticoMetricasResponse,
    RedaccionAsistidaRequest,
    RedaccionAsistidaResponse,
)

router = APIRouter(
    prefix="/v1/llm",
    tags=["Agente LLM y MCP"],
    dependencies=[Depends(verificar_clave_interna)],
)


def get_mcp_bridge_service() -> McpBridgeService:
    """Provee el servicio orquestador MCP + LLM."""
    return McpBridgeService()


async def _generar_stream_sse(
    request: Request,
    ejecutor: Callable[[Callable[[dict[str, Any]], Awaitable[None]]], Awaitable[BaseModel]],
) -> AsyncIterator[str]:
    cola: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def emitir(ev: dict[str, Any]) -> None:
        await cola.put(ev)

    async def correr() -> None:
        try:
            resultado = await ejecutor(emitir)
            await cola.put({"tipo": "resultado", "data": resultado.model_dump(mode="json")})
        except asyncio.CancelledError:
            raise
        except (LlmNoConfiguradoError, LlmErrorExternoError, ValueError, RuntimeError, httpx.HTTPError) as exc:
            await cola.put({"tipo": "error", "mensaje": str(exc)})
        finally:
            await cola.put(None)

    tarea = asyncio.create_task(correr())
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                item = await asyncio.wait_for(cola.get(), timeout=0.35)
            except TimeoutError:
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    finally:
        if not tarea.done():
            tarea.cancel()


@router.post("/asistir-redaccion", response_model=RedaccionAsistidaResponse)
async def asistir_redaccion(
    payload: RedaccionAsistidaRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    mcp: Annotated[McpBridgeService, Depends(get_mcp_bridge_service)],
) -> RedaccionAsistidaResponse:
    """Redacta cartas, descargos e informes técnicos UAGRM con anclaje MCP sin alucinaciones."""
    return await mcp.asistir_redaccion(payload, settings)


@router.post("/asistir-redaccion/stream")
async def asistir_redaccion_stream(
    request: Request,
    payload: RedaccionAsistidaRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    mcp: Annotated[McpBridgeService, Depends(get_mcp_bridge_service)],
) -> StreamingResponse:
    """Emite progreso en tiempo real (SSE) durante la redacción formal."""
    return StreamingResponse(
        _generar_stream_sse(request, lambda emit: mcp.asistir_redaccion(payload, settings, emit)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/interpretar-consulta", response_model=ConsultaLenguajeNaturalResponse)
async def interpretar_consulta(
    payload: ConsultaLenguajeNaturalRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    mcp: Annotated[McpBridgeService, Depends(get_mcp_bridge_service)],
) -> ConsultaLenguajeNaturalResponse:
    """Traduce peticiones en lenguaje natural a filtros estructurados y resume lotes de reporte."""
    return await mcp.interpretar_consulta_reporte(payload, settings)


@router.post("/asistente-sistema", response_model=AyudaSistemaResponse)
async def asistente_sistema(
    payload: AyudaSistemaRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    mcp: Annotated[McpBridgeService, Depends(get_mcp_bridge_service)],
) -> AyudaSistemaResponse:
    """Asistente guiado del sistema UAGRM con enlaces directos filtrados por permisos RBAC."""
    return await mcp.responder_ayuda_sistema(payload, settings)


@router.post("/asistente-sistema/stream")
async def asistente_sistema_stream(
    request: Request,
    payload: AyudaSistemaRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    mcp: Annotated[McpBridgeService, Depends(get_mcp_bridge_service)],
) -> StreamingResponse:
    """Emite pasos en tiempo real (SSE) mientras el asistente consulta fuentes institucionales."""
    return StreamingResponse(
        _generar_stream_sse(request, lambda emit: mcp.responder_ayuda_sistema(payload, settings, emit)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/consulta-analitica", response_model=ConsultaAnaliticaResponse)
async def consulta_analitica(
    payload: ConsultaAnaliticaRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    mcp: Annotated[McpBridgeService, Depends(get_mcp_bridge_service)],
) -> ConsultaAnaliticaResponse:
    """Agente ReAct + FastMCP conectado a PostgreSQL para análisis patrimonial en lenguaje natural."""
    return await mcp.consultar_analitica_padron(payload, settings)


@router.post("/consulta-analitica/stream")
async def consulta_analitica_stream(
    request: Request,
    payload: ConsultaAnaliticaRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    mcp: Annotated[McpBridgeService, Depends(get_mcp_bridge_service)],
) -> StreamingResponse:
    """Emite progreso en tiempo real (SSE) durante consultas analíticas del padrón."""
    return StreamingResponse(
        _generar_stream_sse(request, lambda emit: mcp.consultar_analitica_padron(payload, settings, emit)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/diagnostico-metricas", response_model=DiagnosticoMetricasResponse)
async def diagnostico_metricas(
    payload: DiagnosticoMetricasRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    mcp: Annotated[McpBridgeService, Depends(get_mcp_bridge_service)],
) -> DiagnosticoMetricasResponse:
    """Diagnóstico presupuestario IA integrando proyección Random Forest y priorización normativa."""
    return await mcp.diagnosticar_metricas(payload, settings)
