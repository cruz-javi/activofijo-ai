from fastapi import FastAPI

from activofijo_ai.modules.llm.router import router as llm_router
from activofijo_ai.modules.ml.router import router as ml_router
from activofijo_ai.modules.system.router import router as system_router
from activofijo_ai.modules.vision.router import router as vision_router


def create_app() -> FastAPI:
    """Construye la aplicación FastAPI modular de activofijo-ai."""
    app = FastAPI(
        title="UAGRM Activo Fijo AI Service",
        description="Microservicio interno privado de ML, Visión Artificial y orquestación LLM/MCP.",
        version="0.1.0",
    )

    app.include_router(system_router)
    app.include_router(ml_router)
    app.include_router(vision_router)
    app.include_router(llm_router)
    return app


app = create_app()

