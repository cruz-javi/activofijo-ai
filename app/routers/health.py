from typing import Annotated

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas.common import EstadoServicioIaResponse, HealthResponse
from app.security import verificar_clave_interna
from app.services.llm_provider import OpenAICompatibleLlmClient
from app.services.mcp_bridge import McpBridgeService
from app.services.model_registry import inspeccionar_modelos_en_disco

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Salud"])
def health_check(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Liveness check ligero en memoria para Docker/Kubernetes sin despertar APIs externas."""
    return HealthResponse(
        status="ok",
        servicio="activofijo-ai",
        version="0.1.0",
        entorno=settings.env,
    )


@router.get(
    "/v1/estado",
    response_model=EstadoServicioIaResponse,
    dependencies=[Depends(verificar_clave_interna)],
    tags=["Estado"],
)
def estado_servicio(settings: Annotated[Settings, Depends(get_settings)]) -> EstadoServicioIaResponse:
    """Reporta la disponibilidad de modelos en disco, configuración LLM y herramientas MCP."""
    modelos = inspeccionar_modelos_en_disco(settings.models_dir)
    todos_listos = all(modelo.cargado for modelo in modelos)

    llm_estado = OpenAICompatibleLlmClient(settings).obtener_estado()
    mcp_estado = McpBridgeService().obtener_estado()

    return EstadoServicioIaResponse(
        estado="operativo" if todos_listos else "operativo_sin_modelos",
        modelos_ml=modelos,
        llm=llm_estado,
        mcp=mcp_estado,
    )
