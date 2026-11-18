"""Cliente HTTP interno seguro para que FastMCP consulte PostgreSQL vía activofijo-core."""

from contextvars import ContextVar
from typing import Any

import httpx

from activofijo_ai.core.config import Settings

DEFAULT_SEGURIDAD_MCP: dict[str, Any] = {
    "usuario_id": None,
    "rol_usuario": "FUNCIONARIO",
    "permisos_usuario": [],
    "codigo_empleado_legado": None,
    "acceso_amplio": False,
    "sanitizar_publico": False,
}

seguridad_mcp_ctx: ContextVar[dict[str, Any] | None] = ContextVar(
    "seguridad_mcp_ctx",
    default=None,
)
core_bridge_ctx: ContextVar["CoreMcpBridgeClient | None"] = ContextVar(
    "core_bridge_ctx",
    default=None,
)


class CoreMcpBridgeClient:
    """Cliente asíncrono hacia /internal/mcp/* en activofijo-core autenticado con X-AI-Internal-Key."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def post_internal(
        self,
        ruta: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Ejecuta una consulta read-only contra el puente MCP interno en activofijo-core."""
        base = self._settings.core_internal_url.rstrip("/")
        url = f"{base}/internal/mcp/{ruta.lstrip('/')}"
        headers = {
            "Content-Type": "application/json",
            "X-AI-Internal-Key": self._settings.ai_internal_api_key,
        }
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(12.0, connect=3.0),
                transport=self._transport,
            ) as client:
                response = await client.post(url, json=payload, headers=headers)
            if response.status_code >= 400:
                return {
                    "ok": False,
                    "error": f"activofijo-core HTTP {response.status_code}",
                }
            data = response.json()
            return data if isinstance(data, dict) else {"ok": False, "error": "Respuesta inválida"}
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"Puente MCP Core no disponible: {exc.__class__.__name__}"}
