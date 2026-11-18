from typing import Annotated

from fastapi import APIRouter, Depends

from activofijo_ai.core.config import Settings, get_settings
from activofijo_ai.core.security import verificar_clave_interna
from activofijo_ai.modules.llm.mcp_bridge import McpBridgeService
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


@router.post("/asistir-redaccion", response_model=RedaccionAsistidaResponse)
async def asistir_redaccion(
    payload: RedaccionAsistidaRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    mcp: Annotated[McpBridgeService, Depends(get_mcp_bridge_service)],
) -> RedaccionAsistidaResponse:
    """Redacta cartas, descargos e informes técnicos UAGRM con anclaje MCP sin alucinaciones."""
    return await mcp.asistir_redaccion(payload, settings)


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


@router.post("/consulta-analitica", response_model=ConsultaAnaliticaResponse)
async def consulta_analitica(
    payload: ConsultaAnaliticaRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    mcp: Annotated[McpBridgeService, Depends(get_mcp_bridge_service)],
) -> ConsultaAnaliticaResponse:
    """Agente ReAct + FastMCP conectado a PostgreSQL para análisis patrimonial en lenguaje natural."""
    return await mcp.consultar_analitica_padron(payload, settings)


@router.post("/diagnostico-metricas", response_model=DiagnosticoMetricasResponse)
async def diagnostico_metricas(
    payload: DiagnosticoMetricasRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    mcp: Annotated[McpBridgeService, Depends(get_mcp_bridge_service)],
) -> DiagnosticoMetricasResponse:
    """Diagnóstico presupuestario IA integrando proyección Random Forest y priorización normativa."""
    return await mcp.diagnosticar_metricas(payload, settings)
