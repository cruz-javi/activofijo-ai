from typing import Literal

from pydantic import BaseModel, Field


class ActivoEvaluacionInput(BaseModel):
    cod_activo: str = Field(..., min_length=1, max_length=50)
    antiguedad_meses: int = Field(..., ge=0)
    porcentaje_vida_util: float = Field(..., ge=0)
    ratio_valor_neto: float = Field(..., ge=0)
    estado_fisico: Literal["BUENO", "REGULAR", "MALO", "INUTILIZADO"]
    cod_grupo: int = Field(..., ge=1)
    cantidad_revaluos: int = Field(default=0, ge=0)
    cantidad_hallazgos_malos: int = Field(default=0, ge=0)


class PriorizacionRequest(BaseModel):
    activos: list[ActivoEvaluacionInput] = Field(..., min_length=1, max_length=500)


class PrediccionPrioridadItem(BaseModel):
    cod_activo: str
    categoria: Literal["REVALUO_PRIORITARIO", "BAJA_PRIORITARIA", "OPERATIVIDAD_NORMAL"]
    score_prioridad: float = Field(..., ge=0, le=100)
    regla_explicativa: str


class PriorizacionResponse(BaseModel):
    modelo_version: str
    resultados: list[PrediccionPrioridadItem]


class PuntoSerieHistorica(BaseModel):
    gestion: int = Field(..., ge=2000, le=2100)
    mes: int = Field(..., ge=1, le=12)
    cod_grupo: int | None = None
    bajas_registradas: int = Field(..., ge=0)
    monto_reposicion_bs: float = Field(..., ge=0)


class PronosticoSerieRequest(BaseModel):
    horizontes_meses: int = Field(default=6, ge=1, le=24)
    historico: list[PuntoSerieHistorica] = Field(..., min_length=3)


class PronosticoResponse(BaseModel):
    modelo_version: str
    rmse: float
    r2: float
    bajas_esperadas_total: int
    presupuesto_estimado_bs: float

