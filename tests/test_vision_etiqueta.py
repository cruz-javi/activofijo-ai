import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi.testclient import TestClient

from activofijo_ai.core.config import Settings, get_settings
from activofijo_ai.main import create_app
from activofijo_ai.modules.vision.engine import ServicioVisionEtiqueta, get_servicio_vision_etiqueta
from activofijo_ai.modules.vision.extractor import extraer_codigos_de_bloques
from activofijo_ai.modules.vision.preprocessor import (
    decodificar_imagen_en_ram,
    generar_pasadas_contraste,
)

CLAVE_TEST = "test_internal_ai_key_super_segura_32_caracteres!"


def _jpeg_base64_sintetico(
    texto: str = "U01000045",
    fondo: int = 235,
    color_trazo: int = 185,
    grosor: int = 2,
) -> str:
    lienzo = np.full((180, 480, 3), fondo, dtype=np.uint8)
    cv2.putText(
        lienzo,
        texto,
        (42, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (color_trazo, color_trazo, color_trazo),
        grosor,
        cv2.LINE_AA,
    )
    ok, buffer = cv2.imencode(".jpg", lienzo, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    assert ok
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _crear_cliente(tmp_path: Path) -> TestClient:
    app = create_app()
    settings = Settings(env="test", ai_internal_api_key=CLAVE_TEST, models_dir=tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_decodifica_en_ram_y_genera_pasadas_clahe() -> None:
    b64 = _jpeg_base64_sintetico("ACT-2026", fondo=225, color_trazo=205)
    img = decodificar_imagen_en_ram(f"data:image/jpeg;base64,{b64}")
    assert img.ndim == 3
    assert max(img.shape[:2]) <= 640

    pasadas = list(generar_pasadas_contraste(img))
    assert [nombre for nombre, _ in pasadas] == [
        "original",
        "clahe_realce",
        "umbral_adaptativo",
    ]


def test_extractor_filtra_ruido_y_une_fragmentos() -> None:
    bloques = [
        ([[10, 10], [180, 10], [180, 35], [10, 35]], "UAGRM ACTIVO FIJO", 0.99),
        ([[10, 50], [210, 50], [210, 85], [10, 85]], "U01 000045", 0.89),
    ]
    resultado = extraer_codigos_de_bloques(bloques, prefijos_conocidos=["U01"])
    assert resultado.codigo_principal == "U01000045"
    assert "UAGRM" not in resultado.candidatos
    assert resultado.confianza > 0.8


def test_motor_escala_a_clahe_cuando_original_no_basta() -> None:
    llamadas = 0

    def ejecutor_simulado(_img: np.ndarray, **_kwargs: Any):
        nonlocal llamadas
        llamadas += 1
        if llamadas == 1:
            return [], []
        return [([[10, 20], [220, 20], [220, 60], [10, 60]], "U03000120", 0.94)], []

    servicio = ServicioVisionEtiqueta(ejecutor=ejecutor_simulado)
    img = np.full((120, 320, 3), 230, dtype=np.uint8)
    resultado = servicio.leer_codigo(img, prefijos_conocidos=["U03"])

    assert llamadas == 2
    assert resultado.codigo_principal == "U03000120"
    assert resultado.pasada_exitosa == "clahe_realce"


def test_endpoint_vision_real_onnx_y_seguridad(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path)
    headers = {"X-AI-Internal-Key": CLAVE_TEST}

    # Sin clave -> 401
    resp_unauth = client.post(
        "/v1/vision/leer-codigo-etiqueta",
        json={"contenido_base64": "A" * 80, "mime_type": "image/jpeg"},
    )
    assert resp_unauth.status_code == 401

    # Imagen inválida -> 422
    resp_invalida = client.post(
        "/v1/vision/leer-codigo-etiqueta",
        headers=headers,
        json={"contenido_base64": "no-es-base64!!!", "mime_type": "image/jpeg"},
    )
    assert resp_invalida.status_code == 422

    # Lectura real end-to-end con RapidOCR PP-OCRv4 ONNX en trazos de bajo contraste (tipo lápiz)
    b64_lapiz = _jpeg_base64_sintetico("U01000045", fondo=235, color_trazo=195, grosor=2)
    resp_ok = client.post(
        "/v1/vision/leer-codigo-etiqueta",
        headers=headers,
        json={
            "contenido_base64": b64_lapiz,
            "mime_type": "image/jpeg",
            "prefijos_conocidos": ["U01"],
        },
    )
    assert resp_ok.status_code == 200
    assert resp_ok.headers.get("cache-control") == "no-store"
    body = resp_ok.json()
    assert body["codigo_principal"] == "U01000045"
    assert body["motor_utilizado"] == "ppocr_v4_onnx"
    assert body["confianza"] > 0.7

    # Caso extremo: grafito muy tenue sobre fondo claro (solo 14 niveles de contraste) recuperado por CLAHE
    b64_tenue = _jpeg_base64_sintetico("U02000107", fondo=235, color_trazo=221, grosor=2)
    resp_tenue = client.post(
        "/v1/vision/leer-codigo-etiqueta",
        headers=headers,
        json={
            "contenido_base64": b64_tenue,
            "mime_type": "image/jpeg",
            "prefijos_conocidos": ["U02"],
        },
    )
    assert resp_tenue.status_code == 200
    body_tenue = resp_tenue.json()
    assert body_tenue["codigo_principal"] == "U02000107"


def test_endpoint_con_override_de_servicio_mock(tmp_path: Path) -> None:
    app = create_app()
    settings = Settings(env="test", ai_internal_api_key=CLAVE_TEST, models_dir=tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_servicio_vision_etiqueta] = lambda: ServicioVisionEtiqueta(
        ejecutor=lambda _img, **_kw: (
            [([[5, 5], [150, 5], [150, 40], [5, 40]], "ACT-001", 0.91)],
            [],
        )
    )
    client = TestClient(app)
    b64 = _jpeg_base64_sintetico("ACT-001")
    resp = client.post(
        "/v1/vision/leer-codigo-etiqueta",
        headers={"X-AI-Internal-Key": CLAVE_TEST},
        json={"contenido_base64": b64, "mime_type": "image/jpeg"},
    )
    assert resp.status_code == 200
    assert resp.json()["codigo_principal"] == "ACT-001"
