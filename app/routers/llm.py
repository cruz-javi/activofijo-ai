from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.llm_schemas import (
    ConsultaLenguajeNaturalRequest,
    RedaccionAsistidaRequest,
)
from app.security import verificar_clave_interna

router = APIRouter(
    prefix="/v1/llm",
    tags=["Agente LLM y MCP"],
    dependencies=[Depends(verificar_clave_interna)],
)


@router.post("/asistir-redaccion")
def asistir_redaccion(_payload: RedaccionAsistidaRequest) -> None:
    """Endpoint preparado para asistencia PLN en actas e informes (§3.6 / HU3-19)."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "codigo": "ASISTENTE_EN_CONSTRUCCION",
            "mensaje": "La redacción asistida con Groq + MCP se habilitará en la siguiente fase de Sprint 3.",
        },
    )


@router.post("/interpretar-consulta")
def interpretar_consulta(_payload: ConsultaLenguajeNaturalRequest) -> None:
    """Endpoint preparado para traducción de lenguaje natural a filtros estructurados (HU1-06)."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "codigo": "ASISTENTE_EN_CONSTRUCCION",
            "mensaje": "El intérprete de consultas en lenguaje natural se habilitará en la siguiente fase de Sprint 3.",
        },
    )

