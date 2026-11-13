import base64
import binascii
from collections.abc import Iterator

import cv2
import numpy as np

MAX_BYTES_IMAGEN = 1_200_000
MAX_LADO_IMAGEN = 640
MIN_LADO_UPSCALING = 180


def decodificar_imagen_en_ram(contenido_base64: str) -> np.ndarray:
    """Decodifica Base64 estrictamente en memoria y normaliza la escala para OCR rápido."""
    limpio = contenido_base64.strip()
    if "," in limpio and limpio.lower().startswith("data:image/"):
        limpio = limpio.split(",", 1)[1]

    try:
        raw = base64.b64decode(limpio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("El contenido de imagen no es Base64 válido.") from exc

    if len(raw) == 0 or len(raw) > MAX_BYTES_IMAGEN:
        raise ValueError("La imagen está vacía o supera el límite permitido en RAM.")

    arreglo = np.frombuffer(raw, dtype=np.uint8)
    imagen = cv2.imdecode(arreglo, cv2.IMREAD_COLOR)
    if imagen is None or imagen.size == 0:
        raise ValueError("No se pudo decodificar el archivo de imagen JPEG/PNG.")

    return _normalizar_dimensiones(imagen)


def _normalizar_dimensiones(imagen: np.ndarray) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    lado_mayor = max(alto, ancho)

    if lado_mayor > MAX_LADO_IMAGEN:
        escala = MAX_LADO_IMAGEN / float(lado_mayor)
        return cv2.resize(
            imagen,
            (max(1, round(ancho * escala)), max(1, round(alto * escala))),
            interpolation=cv2.INTER_AREA,
        )

    if lado_mayor < MIN_LADO_UPSCALING:
        escala = min(2.0, MIN_LADO_UPSCALING / float(max(1, lado_mayor)))
        return cv2.resize(
            imagen,
            (max(1, round(ancho * escala)), max(1, round(alto * escala))),
            interpolation=cv2.INTER_CUBIC,
        )

    return imagen


def generar_pasadas_contraste(imagen_bgr: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
    """
    Generador perezoso en RAM con 3 pasadas progresivas para código impreso, marcador o lápiz tenue.
    Al detenerse apenas aparece un código confiable, evita cálculos innecesarios.
    """
    # Pasada 1: imagen normalizada tal como llega del recorte del visor.
    yield ("original", imagen_bgr)

    # Pasada 2: realce adaptativo CLAHE en canal de luminancia + nitidez para grafito/marcador tenue.
    lab = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l_realzada = clahe.apply(l_chan)
    bgr_clahe = cv2.cvtColor(cv2.merge((l_realzada, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    suavizada = cv2.GaussianBlur(bgr_clahe, (0, 0), sigmaX=1.1)
    nitida = cv2.addWeighted(bgr_clahe, 1.35, suavizada, -0.35, 0)
    yield ("clahe_realce", nitida)

    # Pasada 3: binarización adaptativa suave (C=5) para grafito muy tenue o reflejos metálicos.
    gris_clahe = cv2.GaussianBlur(l_realzada, (3, 3), 0)
    binaria = cv2.adaptiveThreshold(
        gris_clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    yield ("umbral_adaptativo", cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR))
