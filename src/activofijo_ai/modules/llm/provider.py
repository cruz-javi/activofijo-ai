from dataclasses import dataclass
from typing import Any

import httpx

from activofijo_ai.core.config import Settings
from activofijo_ai.modules.system.schemas import EstadoProveedorLlm


class LlmNoConfiguradoError(RuntimeError):
    """Se lanza cuando falta LLM_API_KEY o configuración obligatoria del proveedor."""


class LlmErrorExternoError(RuntimeError):
    """Se lanza ante timeout, límite de tasa (429) o fallo HTTP del proveedor LLM."""

    def __init__(self, mensaje: str, status_code: int | None = None) -> None:
        super().__init__(mensaje)
        self.status_code = status_code


@dataclass(frozen=True)
class ResultadoChatLlm:
    contenido: str
    modelo: str
    tokens_usados: int


class OpenAICompatibleLlmClient:
    """Adaptador asíncrono compatible con Groq (https://api.groq.com/openai/v1) y Ollama."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    def obtener_estado(self) -> EstadoProveedorLlm:
        """Devuelve el estado de configuración del proveedor LLM sin exponer credenciales."""
        return EstadoProveedorLlm(
            proveedor=self._settings.llm_provider,
            modelo=self._settings.llm_model,
            base_url=self._settings.llm_base_url,
            api_key_configurada=bool(self._settings.llm_api_key.strip()),
        )

    async def generar_chat_completado(
        self,
        mensajes: list[dict[str, str]],
        response_format: dict[str, str] | None = None,
    ) -> ResultadoChatLlm:
        """Envía una solicitud Chat Completions al proveedor OpenAI-compatible configurado."""
        api_key = self._settings.llm_api_key.strip()
        if not api_key and self._settings.llm_provider.lower() != "ollama":
            raise LlmNoConfiguradoError(
                "LLM_API_KEY no configurada. Configure su clave de Groq o proveedor compatible."
            )

        url = f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": mensajes,
            "max_tokens": self._settings.llm_max_tokens,
            "temperature": 0.2,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            async with httpx.AsyncClient(
                timeout=self._settings.llm_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise LlmErrorExternoError(
                "Tiempo de espera agotado al consultar el proveedor LLM.", status_code=504
            ) from exc
        except httpx.HTTPError as exc:
            raise LlmErrorExternoError(
                f"Error de red al comunicarse con el proveedor LLM: {exc}", status_code=502
            ) from exc

        if response.status_code == 429:
            raise LlmErrorExternoError(
                "Límite de tasa del proveedor LLM alcanzado (HTTP 429). Reintente en unos segundos.",
                status_code=429,
            )

        if response.status_code >= 400:
            raise LlmErrorExternoError(
                f"El proveedor LLM respondió HTTP {response.status_code}.",
                status_code=response.status_code,
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise LlmErrorExternoError("Respuesta del proveedor LLM sin opciones válidas.")

        mensaje = choices[0].get("message") or {}
        contenido = str(mensaje.get("content") or "")
        usage = data.get("usage") or {}
        tokens = int(usage.get("total_tokens") or 0)

        return ResultadoChatLlm(
            contenido=contenido,
            modelo=str(data.get("model") or self._settings.llm_model),
            tokens_usados=tokens,
        )

