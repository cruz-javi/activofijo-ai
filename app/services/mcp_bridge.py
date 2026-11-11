import importlib.util

from app.schemas.common import EstadoMcp

HERRAMIENTAS_MCP_PLANIFICADAS: list[str] = [
    "consultar_ficha_activo",
    "consultar_reglas_normativas",
    "filtrar_catalogo_reportes",
]


class McpBridgeService:
    """Puente MCP de solo lectura hacia herramientas autorizadas en activofijo-core."""

    def obtener_estado(self) -> EstadoMcp:
        """Indica si el SDK MCP está instalado y qué herramientas de solo lectura están registradas."""
        sdk_disponible = importlib.util.find_spec("mcp") is not None
        return EstadoMcp(
            habilitado=sdk_disponible,
            herramientas_registradas=list(HERRAMIENTAS_MCP_PLANIFICADAS),
        )

