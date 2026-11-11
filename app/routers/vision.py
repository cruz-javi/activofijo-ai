from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.vision_schemas import OcrDocumentoRequest
from app.security import verificar_clave_interna

router = APIRouter(
    prefix="/v1/vision",
    tags=["Visión Artificial"],
    dependencies=[Depends(verificar_clave_interna)],
)


@router.post("/ocr-documento")
def ocr_documento(_payload: OcrDocumentoRequest) -> None:
    """Endpoint preparado para extracción óptica en documentos de respaldo Web (§3.5 / HU2-14)."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "codigo": "MODELO_NO_CARGADO",
            "mensaje": "El modelo CRNN-CTC (crnn_ctc_v1.onnx) aún no ha sido entrenado ni exportado.",
        },
    )

