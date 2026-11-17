from pydantic import BaseModel, Field


class PuntoSerieHistorica(BaseModel):
    gestion: int = Field(..., ge=2000, le=2100)
    mes: int = Field(..., ge=1, le=12)
    cod_grupo: int | None = None
    bajas_registradas: int = Field(..., ge=0)
    monto_reposicion_bs: float = Field(..., ge=0)
    altas_registradas: int = Field(default=0, ge=0)
    altas_monto_bs: float = Field(default=0.0, ge=0)
    reasignaciones_registradas: int = Field(default=0, ge=0)
    revaluos_registrados: int = Field(default=0, ge=0)
    revaluos_monto_neto_bs: float = Field(default=0.0)


class GrupoDesgasteInput(BaseModel):
    cod_grupo: int = Field(..., ge=0)
    des_grupo: str = Field(..., min_length=1, max_length=120)
    cantidad: int = Field(..., ge=0)
    monto_total: float = Field(..., ge=0)
    pct_desgaste: float = Field(default=25.0, ge=0, le=100)


class PronosticoSerieRequest(BaseModel):
    horizontes_meses: int = Field(default=6, ge=1, le=24)
    historico: list[PuntoSerieHistorica] = Field(..., min_length=3)
    ratio_desgaste_catalogo: float = Field(default=0.28, ge=0.0, le=1.0)
    grupos: list[GrupoDesgasteInput] = Field(default_factory=list)


class SiguientePeriodoPronostico(BaseModel):
    altas_proyectadas: int
    altas_monto_bs: float
    bajas_proyectadas: int
    bajas_valor_bs: float
    reasignaciones_proyectadas: int
    revaluos_proyectados: int
    revaluos_monto_neto_bs: float
    balance_neto_bs: float


class IntervaloConfianzaBs(BaseModel):
    min_bs: float
    max_bs: float


class ImportanciaVariableRF(BaseModel):
    variable: str
    etiqueta: str
    importancia_pct: float


class ProyeccionGrupoRF(BaseModel):
    cod_grupo: int
    des_grupo: str
    activos_evaluados: int
    pct_desgaste: float
    bajas_estimadas: int
    presupuesto_reposicion_bs: float


class PronosticoResponse(BaseModel):
    modelo_version: str
    arboles: int
    rmse: float
    mae: float
    r2: float
    nivel_confianza_pct: int
    bajas_esperadas_total: int
    presupuesto_estimado_bs: float
    siguiente_periodo: SiguientePeriodoPronostico
    intervalo_confianza_90: IntervaloConfianzaBs
    importancia_variables: list[ImportanciaVariableRF]
    proyeccion_por_grupo: list[ProyeccionGrupoRF]
