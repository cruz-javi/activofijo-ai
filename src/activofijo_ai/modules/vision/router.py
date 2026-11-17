from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from activofijo_ai.core.security import verificar_clave_interna
from activofijo_ai.modules.vision.engine import (
    ServicioVisionEtiqueta,
    get_servicio_vision_etiqueta,
)
from activofijo_ai.modules.vision.preprocessor import decodificar_imagen_en_ram
from activofijo_ai.modules.vision.schemas import (
    LeerCodigoEtiquetaRequest,
    LeerCodigoEtiquetaResponse,
)

router = APIRouter(
    prefix="/v1/vision",
    tags=["Visión Artificial"],
    dependencies=[Depends(verificar_clave_interna)],
)


@router.post("/leer-codigo-etiqueta", response_model=LeerCodigoEtiquetaResponse)
def leer_codigo_etiqueta(
    payload: LeerCodigoEtiquetaRequest,
    response: Response,
    servicio: Annotated[ServicioVisionEtiqueta, Depends(get_servicio_vision_etiqueta)],
) -> LeerCodigoEtiquetaResponse:
    """Extrae en RAM códigos de etiquetas (impresos, marcador o lápiz) con PP-OCRv4 ONNX + CLAHE."""
    response.headers["Cache-Control"] = "no-store"
    try:
        imagen = decodificar_imagen_en_ram(payload.contenido_base64)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "codigo": "IMAGEN_INVALIDA",
                "mensaje": str(exc),
            },
        ) from exc

    resultado = servicio.leer_codigo(imagen, payload.prefijos_conocidos)
    return LeerCodigoEtiquetaResponse(
        codigo_principal=resultado.codigo_principal,
        candidatos=resultado.candidatos,
        texto_crudo=resultado.texto_crudo,
        motor_utilizado=resultado.motor_utilizado,
        pasada_exitosa=resultado.pasada_exitosa,
        confianza=resultado.confianza,
        tiempo_ms=resultado.tiempo_ms,
    )
