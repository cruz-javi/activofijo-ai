from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app

CLAVE_TEST = "test_internal_ai_key_super_segura_32_caracteres!"


def _crear_cliente(tmp_path: Path) -> TestClient:
    app = create_app()
    settings = Settings(
        env="test",
        ai_internal_api_key=CLAVE_TEST,
        models_dir=tmp_path,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_stubs_ml_y_llm_responden_501_controlado_con_clave_valida(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path)
    headers = {"X-AI-Internal-Key": CLAVE_TEST}

    resp_ml = client.post(
        "/v1/ml/normativa/priorizar",
        headers=headers,
        json={
            "activos": [
                {
                    "cod_activo": "U03000001",
                    "antiguedad_meses": 60,
                    "porcentaje_vida_util": 95.0,
                    "ratio_valor_neto": 0.01,
                    "estado_fisico": "REGULAR",
                    "cod_grupo": 3,
                }
            ]
        },
    )
    assert resp_ml.status_code == 501
    assert resp_ml.json()["detail"]["codigo"] == "MODELO_NO_CARGADO"

    resp_llm = client.post(
        "/v1/llm/asistir-redaccion",
        headers=headers,
        json={
            "tipo_documento": "INFORME_TECNICO",
            "cod_activo": "U03000001",
            "instruccion_usuario": "Redactar justificación técnica para revisión.",
            "contexto_institucional": {},
        },
    )
    assert resp_llm.status_code == 501
    assert resp_llm.json()["detail"]["codigo"] == "ASISTENTE_EN_CONSTRUCCION"

