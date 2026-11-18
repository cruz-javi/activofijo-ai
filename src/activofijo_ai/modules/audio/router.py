import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from activofijo_ai.core.config import Settings, get_settings
from activofijo_ai.core.security import verificar_clave_interna
from activofijo_ai.modules.audio.engine import (
    ServicioTranscripcionVoz,
    decodificar_audio_en_ram,
)
from activofijo_ai.modules.audio.inspeccion_nlp import (
    interpretar_inspeccion_con_respaldo_ia,
)
from activofijo_ai.modules.audio.schemas import (
    InterpretarInspeccionVozResponse,
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


@router.post("/interpretar-inspeccion", response_model=InterpretarInspeccionVozResponse)
async def interpretar_inspeccion_voz(
    payload: TranscribirAudioRequest,
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    servicio: Annotated[ServicioTranscripcionVoz, Depends(get_servicio_transcripcion_voz)],
) -> InterpretarInspeccionVozResponse:
    """
    Endpoint móvil sin manos de baja latencia: transcribe audio en RAM e interpreta en un solo salto
    el estado físico (BUENO/REGULAR/MALO) y observación limpia para autocompletar inspecciones.
    """
    response.headers["Cache-Control"] = "no-store"
    t0 = time.perf_counter()
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

    trans = servicio.transcribir(audio_pcm, payload.idioma)
    interpretacion = await interpretar_inspeccion_con_respaldo_ia(trans.texto, settings)
    return InterpretarInspeccionVozResponse(
        texto_transcrito=trans.texto,
        estado_fisico=interpretacion.estado_fisico,
        observacion=interpretacion.observacion,
        accion=interpretacion.accion,
        confirma_directo=interpretacion.confirma_directo,
        motor_transcripcion=trans.motor_utilizado,
        motor_interpretacion=interpretacion.motor_interpretacion,
        tiempo_ms=int((time.perf_counter() - t0) * 1000),
    )
