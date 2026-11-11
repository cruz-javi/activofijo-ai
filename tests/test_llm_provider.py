import httpx
import pytest

from activofijo_ai.core.config import Settings
from activofijo_ai.modules.llm.provider import (
    LlmErrorExternoError,
    LlmNoConfiguradoError,
    OpenAICompatibleLlmClient,
)


async def test_llm_falla_controladamente_si_falta_api_key() -> None:
    settings = Settings(llm_provider="groq", llm_api_key="")
    client = OpenAICompatibleLlmClient(settings)

    with pytest.raises(LlmNoConfiguradoError):
        await client.generar_chat_completado([{"role": "user", "content": "hola"}])


async def test_llm_llama_correctamente_endpoint_compatible_groq() -> None:
    captura: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captura["url"] = str(request.url)
        captura["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={
                "model": "llama-3.3-70b-versatile",
                "choices": [{"message": {"content": '{"resumen":"ok"}'}}],
                "usage": {"total_tokens": 42},
            },
        )

    settings = Settings(
        llm_provider="groq",
        llm_base_url="https://api.groq.com/openai/v1",
        llm_api_key="gsk_test_key",
        llm_model="llama-3.3-70b-versatile",
    )
    client = OpenAICompatibleLlmClient(settings, transport=httpx.MockTransport(handler))

    resultado = await client.generar_chat_completado(
        [{"role": "user", "content": "Devuelve JSON"}],
        response_format={"type": "json_object"},
    )

    assert captura["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captura["auth"] == "Bearer gsk_test_key"
    assert resultado.contenido == '{"resumen":"ok"}'
    assert resultado.tokens_usados == 42


async def test_llm_mapea_rate_limit_429_como_error_externo() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate_limit_exceeded"}})

    settings = Settings(llm_provider="groq", llm_api_key="gsk_test_key")
    client = OpenAICompatibleLlmClient(settings, transport=httpx.MockTransport(handler))

    with pytest.raises(LlmErrorExternoError) as exc_info:
        await client.generar_chat_completado([{"role": "user", "content": "hola"}])

    assert exc_info.value.status_code == 429
