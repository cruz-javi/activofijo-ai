from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from activofijo_ai.core.config import Settings, get_settings
from activofijo_ai.core.security import verificar_clave_interna
from activofijo_ai.modules.ml.random_forest_engine import ejecutar_pronostico_serie
from activofijo_ai.modules.ml.schemas import (
    PriorizacionRequest,
    PronosticoResponse,
    PronosticoSerieRequest,
)

router = APIRouter(
    prefix="/v1/ml",
    tags=["Machine Learning"],
    dependencies=[Depends(verificar_clave_interna)],
)


@router.post("/normativa/priorizar")
def priorizar_normativa(_payload: PriorizacionRequest) -> None:
    """Endpoint preparado para clasificación explicable CART + LightGBM."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "codigo": "MODELO_NO_CARGADO",
            "mensaje": "El modelo CART + LightGBM (priorizador_v1.joblib) aún no ha sido entrenado.",
        },
    )


@router.post("/metricas/pronosticar", response_model=PronosticoResponse)
def pronosticar_metricas(
    payload: PronosticoSerieRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PronosticoResponse:
    """Ejecuta regresión predictiva con Random Forest Regressor sobre series patrimoniales."""
    resultado = ejecutar_pronostico_serie(
        historico=[p.model_dump() for p in payload.historico],
        horizontes_meses=payload.horizontes_meses,
        ratio_desgaste_catalogo=payload.ratio_desgaste_catalogo,
        models_dir=settings.models_dir,
        grupos=[g.model_dump() for g in payload.grupos],
    )
    return PronosticoResponse.model_validate(resultado)
