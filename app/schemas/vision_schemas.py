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

