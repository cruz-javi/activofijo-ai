# `activofijo-ai` — Servicio Analítico y Orquestación IA (UAGRM Activo Fijo)

Microservicio privado en **Python + FastAPI** encargado del entrenamiento/inferencia de modelos de Machine Learning (`CART`, `LightGBM`, `Random Forest`), extracción óptica documental Web y orquestación LLM + MCP (usando **Groq API** compatible con OpenAI por defecto, u **Ollama** local).

> **Seguridad:** No se expone directamente al navegador ni a la aplicación móvil. Únicamente `activofijo-core` se comunica con este servicio enviando el encabezado secreto `X-AI-Internal-Key`.

## Desarrollo Local Rápido (Sin Docker)

```bash
cd activofijo-ai
uv venv .venv
.venv\Scripts\activate
uv pip install -e ".[dev]"
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Pruebas Unitarias

```bash
cd activofijo-ai
.venv\Scripts\python -m pytest
```

## Contenedor Docker (Producción / `activofijo-net`)

```bash
docker compose build
docker compose up -d
```

