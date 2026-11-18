from pydantic import BaseModel, Field


class TranscribirAudioRequest(BaseModel):
    audio_base64: str = Field(..., min_length=64, max_length=6_000_000)
    mime_type: str = Field(default="audio/webm", max_length=80)
    idioma: str = Field(default="es", min_length=2, max_length=12)


class TranscribirAudioResponse(BaseModel):
    texto: str = ""
    idioma_detectado: str = "es"
    duracion_audio_seg: float = Field(default=0.0, ge=0.0)
    modelo: str = "whisper-base-int8"
    motor_utilizado: str = "faster_whisper_local"
    tiempo_ms: int = Field(default=0, ge=0)
