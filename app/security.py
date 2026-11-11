import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


def verificar_clave_interna(
    settings: Annotated[Settings, Depends(get_settings)],
    x_ai_internal_key: Annotated[str | None, Header(alias="X-AI-Internal-Key")] = None,
) -> None:
    """Verifica que la petición interna provenga de activofijo-core usando comparación en tiempo constante."""
    clave_esperada = settings.ai_internal_api_key.strip()
    if len(clave_esperada) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "codigo": "IA_CLAVE_INTERNA_NO_CONFIGURADA",
                "mensaje": "AI_INTERNAL_API_KEY no está configurada correctamente en activofijo-ai (mínimo 32 caracteres).",
            },
        )

    if not x_ai_internal_key or not hmac.compare_digest(x_ai_internal_key, clave_esperada):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "codigo": "CLAVE_INTERNA_INVALIDA",
                "mensaje": "Encabezado X-AI-Internal-Key ausente o inválido.",
            },
        )
