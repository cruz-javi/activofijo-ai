from fastapi import FastAPI

from app.routers import health, llm, ml, vision


def create_app() -> FastAPI:
    """Construye la aplicación FastAPI interna de activofijo-ai."""
    app = FastAPI(
        title="UAGRM Activo Fijo AI Service",
        description="Microservicio interno privado de ML, Visión Artificial y orquestación LLM/MCP.",
        version="0.1.0",
    )

    app.include_router(health.router)
    app.include_router(ml.router)
    app.include_router(vision.router)
    app.include_router(llm.router)
    return app


app = create_app()

