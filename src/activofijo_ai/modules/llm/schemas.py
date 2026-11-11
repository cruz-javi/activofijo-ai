from typing import Literal

from pydantic import BaseModel, Field


class RedaccionAsistidaRequest(BaseModel):
    tipo_documento: Literal["ACTA_BAJA", "ACTA_ASIGNACION", "INFORME_TECNICO", "NOTA_INTERNA"]
    cod_activo: str | None = Field(default=None, max_length=50)
    instruccion_usuario: str = Field(..., min_length=5, max_length=1000)
    contexto_institucional: dict[str, object] = Field(default_factory=dict)


class RedaccionAsistidaResponse(BaseModel):
    proveedor: str
    modelo: str
    borrador_texto: str
    advertencias: list[str] = Field(default_factory=list)
    tokens_usados: int = 0


class ConsultaLenguajeNaturalRequest(BaseModel):
    consulta_usuario: str = Field(..., min_length=4, max_length=500)


class ConsultaLenguajeNaturalResponse(BaseModel):
    filtros_detectados: dict[str, object] = Field(default_factory=dict)
    explicacion: str
    advertencias: list[str] = Field(default_factory=list)

