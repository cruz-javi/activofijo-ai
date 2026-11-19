import httpx
import pytest

from activofijo_ai.core.config import Settings
from activofijo_ai.modules.llm.provider import (
    LlmErrorExternoError,
    LlmNoConfiguradoError,
    OpenAICompatibleLlmClient,
    ResultadoChatLlm,
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


async def test_llm_soporta_tool_calls_nativos_openai() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "uagrm-activo-fijo",
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "obtener_estadisticas_padron",
                                        "arguments": {"unidad": "FICCT"},
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"total_tokens": 33},
            },
        )

    settings = Settings(llm_provider="ollama", llm_model="uagrm-activo-fijo")
    client = OpenAICompatibleLlmClient(settings, transport=httpx.MockTransport(handler))
    res = await client.generar_chat_completado(
        [{"role": "user", "content": "Estadísticas FICCT"}],
        tools=[{"type": "function", "function": {"name": "obtener_estadisticas_padron"}}],
    )
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].nombre == "obtener_estadisticas_padron"
    assert '"FICCT"' in res.tool_calls[0].argumentos_json


async def test_llm_activa_respaldo_remoto_si_ollama_falla(monkeypatch: pytest.MonkeyPatch) -> None:
    llamadas: list[tuple[str, list[str]]] = []

    async def fake_invocar(**kwargs: object) -> ResultadoChatLlm:
        prov = str(kwargs["proveedor"])
        modelos = list(kwargs["modelos"])  # type: ignore[arg-type]
        llamadas.append((prov, modelos))
        if prov == "ollama":
            raise LlmErrorExternoError("Ollama sin respuesta")
        return ResultadoChatLlm(
            contenido='{"ok":true}',
            modelo=modelos[0],
            tokens_usados=55,
            proveedor=prov,
        )

    settings = Settings(
        llm_provider="ollama",
        llm_model="uagrm-activo-fijo",
        llm_fallback_provider="groq",
        llm_fallback_base_url="https://api.groq.com/openai/v1",
        llm_fallback_api_key="gsk_fb_key",
        llm_fallback_model="qwen/qwen3.8-27b",
        llm_fallback_timeout_seconds=25.0,
    )
    client = OpenAICompatibleLlmClient(settings)
    monkeypatch.setattr(client, "_invocar_proveedor", fake_invocar)

    res = await client.generar_chat_completado([{"role": "user", "content": "hola"}])

    assert res.proveedor == "groq"
    assert res.modelo == "qwen/qwen3.8-27b"
    assert llamadas == [
        ("ollama", ["uagrm-activo-fijo", "qwen2.5:3b"]),
        ("groq", ["qwen/qwen3.8-27b"]),
    ]
