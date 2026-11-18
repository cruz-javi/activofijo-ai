from typing import Literal

from pydantic import BaseModel, Field


class TranscribirAudioRequest(BaseModel):
    audio_base64: str = Field(..., min_length=64, max_length=6_000_000)
    mime_type: str = Field(default="audio/webm", max_length=80)
    idioma: str = Field(default="es", min_length=2, max_length=12)
    etapa_modal: Literal["dictado_inicial", "esperando_confirmacion"] = "dictado_inicial"
    activo_codigo: str | None = Field(default=None, max_length=40)
    activo_descripcion: str | None = Field(default=None, max_length=160)
    estado_actual: Literal["BUENO", "REGULAR", "MALO"] | None = None
    observacion_actual: str | None = Field(default=None, max_length=240)


class TranscribirAudioResponse(BaseModel):
    texto: str = ""
    idioma_detectado: str = "es"
    duracion_audio_seg: float = Field(default=0.0, ge=0.0)
    modelo: str = "whisper-base-int8"
    motor_utilizado: str = "faster_whisper_local"
    tiempo_ms: int = Field(default=0, ge=0)


class InterpretarInspeccionVozResponse(BaseModel):
    texto_transcrito: str = ""
    estado_fisico: Literal["BUENO", "REGULAR", "MALO"] | None = None
    observacion: str = ""
    accion: Literal["completar", "confirmar", "cancelar", "ignorar"] = "completar"
    confirma_directo: bool = False
    motor_transcripcion: str = "faster_whisper_local"
    motor_interpretacion: str = "reglas_rapidas"
    tiempo_ms: int = Field(default=0, ge=0)
