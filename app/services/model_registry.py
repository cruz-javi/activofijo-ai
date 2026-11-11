from dataclasses import dataclass
from pathlib import Path

from app.schemas.common import EstadoModelo


@dataclass(frozen=True)
class EspecificacionModelo:
    identificador: str
    algoritmo: str
    nombre_archivo: str


CATALOGO_MODELOS_ESPERADOS: tuple[EspecificacionModelo, ...] = (
    EspecificacionModelo(
        identificador="priorizador_normativo",
        algoritmo="CART + LightGBM",
        nombre_archivo="priorizador_v1.joblib",
    ),
    EspecificacionModelo(
        identificador="pronostico_patrimonial",
        algoritmo="Random Forest Regressor",
        nombre_archivo="pronostico_rf_v1.joblib",
    ),
    EspecificacionModelo(
        identificador="ocr_respaldo_documental",
        algoritmo="CRNN-CTC (ONNX Runtime)",
        nombre_archivo="crnn_ctc_v1.onnx",
    ),
)


def inspeccionar_modelos_en_disco(models_dir: Path) -> list[EstadoModelo]:
    """Verifica la disponibilidad física de los artefactos ML/Vision sin cargarlos en memoria aún."""
    resultados: list[EstadoModelo] = []
    for spec in CATALOGO_MODELOS_ESPERADOS:
        ruta = models_dir / spec.nombre_archivo
        existe = ruta.is_file()
        resultados.append(
            EstadoModelo(
                identificador=spec.identificador,
                algoritmo=spec.algoritmo,
                cargado=existe,
                estado="listo" if existe else "pendiente_entrenamiento",
                ruta_esperada=f"models/{spec.nombre_archivo}",
            )
        )
    return resultados

