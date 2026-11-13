from typing import Literal

from pydantic import BaseModel, Field


class OcrDocumentoRequest(BaseModel):
    nombre_archivo: str = Field(..., min_length=1, max_length=255)
    mime_type: str = Field(..., min_length=3, max_length=100)
    contenido_base64: str = Field(..., min_length=16)


class OcrDocumentoResponse(BaseModel):
    modelo_version: str
    texto_detectado: str
    codigos_candidatos: list[str] = Field(default_factory=list)
    confianza_promedio: float = Field(..., ge=0, le=1)


class LeerCodigoEtiquetaRequest(BaseModel):
    contenido_base64: str = Field(..., min_length=64, max_length=950_000)
    mime_type: Literal["image/jpeg", "image/png"] = "image/jpeg"
    prefijos_conocidos: list[str] = Field(default_factory=list, max_length=10)


class LeerCodigoEtiquetaResponse(BaseModel):
    codigo_principal: str | None = None
    candidatos: list[str] = Field(default_factory=list)
    texto_crudo: str = ""
    motor_utilizado: str = "ppocr_v4_onnx"
    pasada_exitosa: str | None = None
    confianza: float = Field(default=0.0, ge=0.0, le=1.0)
    tiempo_ms: int = Field(default=0, ge=0)
