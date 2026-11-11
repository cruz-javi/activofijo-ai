from typing import Annotated

from fastapi import APIRouter, Depends

from activofijo_ai.core.config import Settings, get_settings
from activofijo_ai.core.security import verificar_clave_interna
from activofijo_ai.modules.llm.mcp_bridge import McpBridgeService
from activofijo_ai.modules.llm.provider import OpenAICompatibleLlmClient
from activofijo_ai.modules.system.registry import inspeccionar_modelos_en_disco
from activofijo_ai.modules.system.schemas import EstadoServicioIaResponse, HealthResponse

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
def estado_servicio(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EstadoServicioIaResponse:
    """Reporta la disponibilidad de modelos en disco, configuración LLM y herramientas MCP."""
    modelos = inspeccionar_modelos_en_disco(settings.models_dir)
    todos_listos = all(modelo.cargado for modelo in modelos)

    return EstadoServicioIaResponse(
        estado="operativo" if todos_listos else "operativo_sin_modelos",
        modelos_ml=modelos,
        llm=OpenAICompatibleLlmClient(settings).obtener_estado(),
        mcp=McpBridgeService().obtener_estado(),
    )

