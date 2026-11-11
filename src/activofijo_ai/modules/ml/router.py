from fastapi import APIRouter, Depends, HTTPException, status

from activofijo_ai.core.security import verificar_clave_interna
from activofijo_ai.modules.ml.schemas import PriorizacionRequest, PronosticoSerieRequest

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


@router.post("/metricas/pronosticar")
def pronosticar_metricas(_payload: PronosticoSerieRequest) -> None:
    """Endpoint preparado para regresión predictiva Random Forest."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "codigo": "MODELO_NO_CARGADO",
            "mensaje": "El modelo Random Forest Regressor (pronostico_rf_v1.joblib) aún no ha sido entrenado.",
        },
    )

