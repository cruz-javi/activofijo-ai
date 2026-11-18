import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from activofijo_ai.core.config import Settings, get_settings


def validar_clave_interna_valor(
    settings: Settings,
    x_ai_internal_key: str | None,
) -> tuple[int, dict[str, str]] | None:
    """Valida la clave interna compartida en tiempo constante devolviendo error o None."""
    clave_esperada = settings.ai_internal_api_key.strip()
    if len(clave_esperada) < 32:
        return (
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {
                "codigo": "IA_CLAVE_INTERNA_NO_CONFIGURADA",
                "mensaje": (
                    "AI_INTERNAL_API_KEY no está configurada correctamente en activofijo-ai "
                    "(mínimo 32 caracteres)."
                ),
            },
        )

    if not x_ai_internal_key or not hmac.compare_digest(x_ai_internal_key, clave_esperada):
        return (
            status.HTTP_401_UNAUTHORIZED,
            {
                "codigo": "CLAVE_INTERNA_INVALIDA",
                "mensaje": "Encabezado X-AI-Internal-Key ausente o inválido.",
            },
        )
    return None


def verificar_clave_interna(
    settings: Annotated[Settings, Depends(get_settings)],
    x_ai_internal_key: Annotated[str | None, Header(alias="X-AI-Internal-Key")] = None,
) -> None:
    """Verifica que la petición interna provenga de activofijo-core usando comparación en tiempo constante."""
    error = validar_clave_interna_valor(settings, x_ai_internal_key)
    if error is not None:
        codigo_http, detalle = error
        raise HTTPException(status_code=codigo_http, detail=detalle)

