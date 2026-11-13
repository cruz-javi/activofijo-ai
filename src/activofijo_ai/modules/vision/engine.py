import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np
from rapidocr_onnxruntime import RapidOCR

from activofijo_ai.modules.vision.extractor import extraer_codigos_de_bloques
from activofijo_ai.modules.vision.preprocessor import generar_pasadas_contraste

UMBRAL_SALIDA_TEMPRANA = 0.78


class OcrEjecutor(Protocol):
    def __call__(self, imagen: np.ndarray, **kwargs: Any) -> tuple[list[Any] | None, Any]: ...


@dataclass(frozen=True)
class ResultadoVisionEtiqueta:
    codigo_principal: str | None
    candidatos: list[str]
    texto_crudo: str
    motor_utilizado: str
    pasada_exitosa: str | None
    confianza: float
    tiempo_ms: int


@lru_cache(maxsize=1)
def obtener_motor_rapidocr_compartido() -> RapidOCR:
    """
    Inicializa y precalienta una sola vez en CPU las sesiones ONNX PP-OCRv4 (~15 MB),
    evitando sobrecarga por petición en FastAPI.
    """
    motor = RapidOCR(
        use_cls=False,
        det_limit_side_len=480,
        max_side_len=640,
        min_side_len=16,
    )
    # Warmup de grafos ONNX Runtime para que la primera lectura real desde el móvil sea veloz.
    imagen_warmup = np.full((64, 192, 3), 240, dtype=np.uint8)
    motor(imagen_warmup, use_cls=False)
    return motor


class ServicioVisionEtiqueta:
    """Orquesta pasadas adaptativas en RAM sobre un motor OCR ONNX liviano."""

    def __init__(self, ejecutor: OcrEjecutor | None = None) -> None:
        self._ejecutor = ejecutor

    def _motor(self) -> OcrEjecutor:
        if self._ejecutor is not None:
            return self._ejecutor
        return obtener_motor_rapidocr_compartido()

    def leer_codigo(
        self,
        imagen_bgr: np.ndarray,
        prefijos_conocidos: list[str] | None = None,
    ) -> ResultadoVisionEtiqueta:
        inicio = time.perf_counter()
        motor = self._motor()

        mejor_codigo: str | None = None
        mejores_candidatos: list[str] = []
        mejor_texto = ""
        mejor_pasada: str | None = None
        mejor_confianza = 0.0
        mejor_ranking = -1.0

        for nombre_pasada, vista in generar_pasadas_contraste(imagen_bgr):
            bloques, _ = motor(vista, use_cls=False)
            extraido = extraer_codigos_de_bloques(bloques, prefijos_conocidos)

            if extraido.codigo_principal and extraido.puntaje_ranking > mejor_ranking:
                mejor_codigo = extraido.codigo_principal
                mejores_candidatos = extraido.candidatos
                mejor_texto = extraido.texto_crudo
                mejor_pasada = nombre_pasada
                mejor_confianza = extraido.confianza
                mejor_ranking = extraido.puntaje_ranking

            lectura_completa = (
                bool(extraido.codigo_principal)
                and extraido.confianza >= UMBRAL_SALIDA_TEMPRANA
                and len(extraido.codigo_principal or "") >= 7
                and (not prefijos_conocidos or extraido.coincide_prefijo)
            )
            if lectura_completa:
                break

        tiempo_ms = round((time.perf_counter() - inicio) * 1000)
        return ResultadoVisionEtiqueta(
            codigo_principal=mejor_codigo,
            candidatos=mejores_candidatos,
            texto_crudo=mejor_texto,
            motor_utilizado="ppocr_v4_onnx",
            pasada_exitosa=mejor_pasada,
            confianza=mejor_confianza,
            tiempo_ms=tiempo_ms,
        )


def get_servicio_vision_etiqueta() -> ServicioVisionEtiqueta:
    return ServicioVisionEtiqueta()
