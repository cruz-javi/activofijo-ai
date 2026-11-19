import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from activofijo_ai.core.config import get_settings
from activofijo_ai.core.security import validar_clave_interna_valor
from activofijo_ai.modules.audio.engine import obtener_modelo_whisper_compartido
from activofijo_ai.modules.audio.router import router as audio_router
from activofijo_ai.modules.llm.mcp_server import crear_servidor_mcp
from activofijo_ai.modules.llm.provider import OpenAICompatibleLlmClient
from activofijo_ai.modules.llm.router import router as llm_router
from activofijo_ai.modules.ml.router import router as ml_router
from activofijo_ai.modules.system.router import router as system_router
from activofijo_ai.modules.vision.engine import obtener_motor_rapidocr_compartido
from activofijo_ai.modules.vision.router import router as vision_router


class _ProteccionClaveInternaMcpMiddleware:
    """Exige X-AI-Internal-Key también en el endpoint ASGI StreamableHTTP /mcp."""

    def __init__(self, app_destino: ASGIApp, fastapi_app: FastAPI) -> None:
        self._app_destino = app_destino
        self._fastapi_app = fastapi_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = {
                k.decode("latin-1").lower(): v.decode("latin-1")
                for k, v in scope.get("headers", [])
            }
            proveedor_settings = self._fastapi_app.dependency_overrides.get(
                get_settings, get_settings
            )
            settings = proveedor_settings()
            error = validar_clave_interna_valor(settings, headers.get("x-ai-internal-key"))
            if error is not None:
                codigo_http, detalle = error
                respuesta = JSONResponse(status_code=codigo_http, content={"detail": detalle})
                await respuesta(scope, receive, send)
                return

        await self._app_destino(scope, receive, send)


def create_app() -> FastAPI:
    """Construye la aplicación FastAPI modular de activofijo-ai con servidor FastMCP integrado."""
    servidor_mcp = crear_servidor_mcp()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Precalienta ONNX, Whisper local y Ollama en RAM en segundo plano."""
        obtener_motor_rapidocr_compartido()
        settings = _app.dependency_overrides.get(get_settings, get_settings)()
        if settings.env != "test" and settings.llm_provider.lower() == "ollama":
            asyncio.create_task(
                asyncio.to_thread(
                    obtener_modelo_whisper_compartido,
                    settings.whisper_model,
                    str(settings.models_dir),
                )
            )
        warmup_task: asyncio.Task[bool] | None = None
        if settings.env != "test" and settings.llm_provider.lower() == "ollama":
            warmup_task = asyncio.create_task(
                OpenAICompatibleLlmClient(settings).precalentar_ollama()
            )
        async with servidor_mcp.session_manager.run():
            yield
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()

    app = FastAPI(
        title="UAGRM Activo Fijo AI Service",
        description="Microservicio interno privado de ML, Visión Artificial y orquestación LLM/MCP.",
        version="0.1.0",
        lifespan=_lifespan,
    )

    app.include_router(system_router)
    app.include_router(ml_router)
    app.include_router(vision_router)
    app.include_router(audio_router)
    app.include_router(llm_router)
    app.mount("/mcp", _ProteccionClaveInternaMcpMiddleware(servidor_mcp.streamable_http_app(), app))
    return app


app = create_app()
