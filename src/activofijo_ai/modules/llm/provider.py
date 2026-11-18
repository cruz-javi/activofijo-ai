import json
import re
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
class ToolCallLlm:
    id: str
    nombre: str
    argumentos_json: str


@dataclass(frozen=True)
class ResultadoChatLlm:
    contenido: str
    modelo: str
    tokens_usados: int
    tool_calls: tuple[ToolCallLlm, ...] = ()


def extraer_json_objeto(texto: str) -> dict[str, Any] | None:
    """Extrae el primer objeto JSON válido incluso si viene envuelto en cercas markdown."""
    limpio = texto.strip()
    if limpio.startswith("```"):
        limpio = re.sub(r"^```(?:json)?\s*", "", limpio, flags=re.IGNORECASE)
        limpio = re.sub(r"\s*```$", "", limpio)

    try:
        parsed = json.loads(limpio)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", limpio)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


class OpenAICompatibleLlmClient:
    """Adaptador asíncrono compatible con Ollama local (`uagrm-activo-fijo`) y Groq."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    def obtener_estado(self) -> EstadoProveedorLlm:
        """Devuelve el estado de configuración del proveedor LLM sin exponer credenciales."""
        es_local = self._settings.llm_provider.lower() == "ollama"
        return EstadoProveedorLlm(
            proveedor=self._settings.llm_provider,
            modelo=self._settings.llm_model,
            base_url=self._settings.llm_base_url,
            api_key_configurada=es_local or bool(self._settings.llm_api_key.strip()),
        )

    async def generar_chat_completado(
        self,
        mensajes: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        timeout_override: float | None = None,
        max_tokens_override: int | None = None,
        temperature_override: float | None = None,
    ) -> ResultadoChatLlm:
        """Envía una solicitud Chat Completions (con soporte tool_calls ReAct) al proveedor LLM."""
        api_key = self._settings.llm_api_key.strip()
        es_ollama = self._settings.llm_provider.lower() == "ollama"
        if not api_key and not es_ollama:
            raise LlmNoConfiguradoError(
                "LLM_API_KEY no configurada. Configure Ollama local o una clave compatible."
            )

        url = f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        modelos_a_intentar = [self._settings.llm_model]
        if es_ollama and self._settings.llm_model != "qwen2.5:3b":
            modelos_a_intentar.append("qwen2.5:3b")

        ultimo_error: LlmErrorExternoError | None = None

        for modelo in modelos_a_intentar:
            payload: dict[str, Any] = {
                "model": modelo,
                "messages": mensajes,
                "max_tokens": max_tokens_override or self._settings.llm_max_tokens,
                "temperature": 0.1 if temperature_override is None else temperature_override,
            }
            if response_format is not None and not tools:
                payload["response_format"] = response_format
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice or "auto"

            try:
                async with httpx.AsyncClient(
                    timeout=timeout_override or self._settings.llm_timeout_seconds,
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

            if response.status_code == 404 and es_ollama and modelo != modelos_a_intentar[-1]:
                continue

            if response.status_code == 429:
                raise LlmErrorExternoError(
                    "Límite de tasa del proveedor LLM alcanzado (HTTP 429). Reintente en unos segundos.",
                    status_code=429,
                )

            if response.status_code >= 400:
                ultimo_error = LlmErrorExternoError(
                    f"El proveedor LLM respondió HTTP {response.status_code}.",
                    status_code=response.status_code,
                )
                break

            data = response.json()
            choices = data.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                raise LlmErrorExternoError("Respuesta del proveedor LLM sin opciones válidas.")

            mensaje = choices[0].get("message") or {}
            contenido = str(mensaje.get("content") or "")
            raw_calls = mensaje.get("tool_calls") or []
            tool_calls_norm: list[ToolCallLlm] = []
            for idx, tc in enumerate(raw_calls):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                nombre = str(fn.get("name") or "").strip()
                if not nombre:
                    continue
                args_raw = fn.get("arguments")
                args_str = (
                    json.dumps(args_raw, ensure_ascii=False)
                    if isinstance(args_raw, dict)
                    else str(args_raw or "{}")
                )
                tool_calls_norm.append(
                    ToolCallLlm(
                        id=str(tc.get("id") or f"call_{idx}"),
                        nombre=nombre,
                        argumentos_json=args_str,
                    )
                )

            usage = data.get("usage") or {}
            tokens = int(usage.get("total_tokens") or 0)

            return ResultadoChatLlm(
                contenido=contenido,
                modelo=str(data.get("model") or modelo),
                tokens_usados=tokens,
                tool_calls=tuple(tool_calls_norm),
            )

        assert ultimo_error is not None
        raise ultimo_error

    async def precalentar_ollama(self) -> bool:
        """Precarga el modelo personalizado en RAM de Ollama con keep_alive."""
        if self._settings.llm_provider.lower() != "ollama":
            return False
        base = self._settings.llm_base_url.rstrip("/")
        base = base.removesuffix("/v1")
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(12.0, connect=2.0),
                transport=self._transport,
            ) as client:
                res = await client.post(
                    f"{base}/api/generate",
                    json={
                        "model": self._settings.llm_model,
                        "prompt": "",
                        "keep_alive": "25m",
                    },
                )
            return res.status_code < 400
        except httpx.HTTPError:
            return False
