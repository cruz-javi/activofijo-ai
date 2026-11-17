from pathlib import Path

from fastapi.testclient import TestClient

from activofijo_ai.core.config import Settings, get_settings
from activofijo_ai.main import create_app

CLAVE_TEST = "test_internal_ai_key_super_segura_32_caracteres!"
MODELS_REALES = Path(__file__).resolve().parents[1] / "models"


def _crear_cliente(models_dir: Path) -> TestClient:
    app = create_app()
    settings = Settings(
        env="test",
        ai_internal_api_key=CLAVE_TEST,
        models_dir=models_dir,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_pronostico_rf_con_artefacto_preentrenado() -> None:
    assert (MODELS_REALES / "pronostico_rf_v1.joblib").is_file()

    client = _crear_cliente(MODELS_REALES)
    headers = {"X-AI-Internal-Key": CLAVE_TEST}

    payload = {
        "horizontes_meses": 6,
        "ratio_desgaste_catalogo": 0.34,
        "historico": [
            {
                "gestion": 2026,
                "mes": m,
                "bajas_registradas": 8 + (m % 3),
                "monto_reposicion_bs": 42000.0 + m * 2500.0,
                "altas_registradas": 15 + (m % 4),
                "altas_monto_bs": 95000.0 + m * 4000.0,
                "reasignaciones_registradas": 22 + m,
                "revaluos_registrados": 4 + (m % 2),
                "revaluos_monto_neto_bs": 14000.0 + m * 800.0,
            }
            for m in range(1, 11)
        ],
        "grupos": [
            {
                "cod_grupo": 4,
                "des_grupo": "Equipos de Computación",
                "cantidad": 480,
                "monto_total": 3120000.0,
                "pct_desgaste": 42.5,
            },
            {
                "cod_grupo": 2,
                "des_grupo": "Muebles y Enseres",
                "cantidad": 350,
                "monto_total": 890000.0,
                "pct_desgaste": 26.0,
            },
        ],
    }

    resp = client.post("/v1/ml/metricas/pronosticar", headers=headers, json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert body["modelo_version"] == "rf-patrimonio-v1.0"
    assert body["arboles"] == 100
    assert 0.0 <= body["r2"] <= 1.0
    assert body["rmse"] >= 0.0
    assert body["bajas_esperadas_total"] > 0
    assert body["presupuesto_estimado_bs"] > 0

    sig = body["siguiente_periodo"]
    assert sig["bajas_proyectadas"] >= 0
    assert sig["altas_proyectadas"] >= 0

    ic = body["intervalo_confianza_90"]
    assert ic["min_bs"] <= sig["bajas_valor_bs"] <= ic["max_bs"]

    assert len(body["importancia_variables"]) == 8
    suma_imp = sum(item["importancia_pct"] for item in body["importancia_variables"])
    assert 98.0 <= suma_imp <= 102.0

    assert len(body["proyeccion_por_grupo"]) == 2
    assert body["proyeccion_por_grupo"][0]["cod_grupo"] == 4


def test_pronostico_rf_funciona_offline_si_no_hay_artefacto_en_disco(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path)
    headers = {"X-AI-Internal-Key": CLAVE_TEST}

    resp = client.post(
        "/v1/ml/metricas/pronosticar",
        headers=headers,
        json={
            "horizontes_meses": 3,
            "historico": [
                {"gestion": 2026, "mes": 7, "bajas_registradas": 5, "monto_reposicion_bs": 21000.0},
                {"gestion": 2026, "mes": 8, "bajas_registradas": 7, "monto_reposicion_bs": 29000.0},
                {"gestion": 2026, "mes": 9, "bajas_registradas": 6, "monto_reposicion_bs": 25500.0},
                {"gestion": 2026, "mes": 10, "bajas_registradas": 8, "monto_reposicion_bs": 34000.0},
                {"gestion": 2026, "mes": 11, "bajas_registradas": 9, "monto_reposicion_bs": 39500.0},
                {"gestion": 2026, "mes": 12, "bajas_registradas": 10, "monto_reposicion_bs": 43000.0},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["bajas_esperadas_total"] > 0
    assert body["presupuesto_estimado_bs"] > 0
