from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from activofijo_ai.core.config import Settings, get_settings
from activofijo_ai.core.security import verificar_clave_interna
from activofijo_ai.modules.audio.engine import (
    ServicioTranscripcionVoz,
    decodificar_audio_en_ram,
)
from activofijo_ai.modules.audio.schemas import (
    TranscribirAudioRequest,
    TranscribirAudioResponse,
)

router = APIRouter(
    prefix="/v1/audio",
    tags=["Transcripción de Voz"],
    dependencies=[Depends(verificar_clave_interna)],
)


def get_servicio_transcripcion_voz(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ServicioTranscripcionVoz:
    """Provee el servicio de transcripción local Faster-Whisper."""
    return ServicioTranscripcionVoz(
        nombre_modelo=settings.whisper_model,
        models_dir=settings.models_dir,
    )


@router.post("/transcribir", response_model=TranscribirAudioResponse)
def transcribir_audio(
    payload: TranscribirAudioRequest,
    response: Response,
    servicio: Annotated[ServicioTranscripcionVoz, Depends(get_servicio_transcripcion_voz)],
) -> TranscribirAudioResponse:
    """Transcribe dictado por voz en español boliviano 100% en local y en RAM."""
    response.headers["Cache-Control"] = "no-store"
    try:
        audio_pcm = decodificar_audio_en_ram(payload.audio_base64, payload.mime_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "codigo": "AUDIO_INVALIDO",
                "mensaje": str(exc),
            },
        ) from exc

    resultado = servicio.transcribir(audio_pcm, payload.idioma)
    return TranscribirAudioResponse(
        texto=resultado.texto,
        idioma_detectado=resultado.idioma_detectado,
        duracion_audio_seg=resultado.duracion_audio_seg,
        modelo=resultado.modelo,
        motor_utilizado=resultado.motor_utilizado,
        tiempo_ms=resultado.tiempo_ms,
    )
