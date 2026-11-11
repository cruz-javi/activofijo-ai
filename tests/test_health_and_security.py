from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app

CLAVE_TEST = "test_internal_ai_key_super_segura_32_caracteres!"


def _crear_cliente(tmp_path: Path, ai_key: str = CLAVE_TEST, llm_key: str = "") -> TestClient:
    app = create_app()
    settings = Settings(
        env="test",
        ai_internal_api_key=ai_key,
        llm_api_key=llm_key,
        models_dir=tmp_path,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_health_responde_ok_sin_autenticacion(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["servicio"] == "activofijo-ai"


def test_estado_rechaza_sin_clave_interna(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path)
    resp = client.get("/v1/estado")
    assert resp.status_code == 401


def test_estado_rechaza_si_clave_interna_en_servidor_no_configurada(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path, ai_key="corta")
    resp = client.get("/v1/estado", headers={"X-AI-Internal-Key": CLAVE_TEST})
    assert resp.status_code == 503
    assert resp.json()["detail"]["codigo"] == "IA_CLAVE_INTERNA_NO_CONFIGURADA"


def test_estado_reporta_modelos_pendientes_sin_exponer_secretos(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path, llm_key="gsk_secreto_no_exponer")
    resp = client.get("/v1/estado", headers={"X-AI-Internal-Key": CLAVE_TEST})
    assert resp.status_code == 200
    body = resp.json()
    assert body["estado"] == "operativo_sin_modelos"
    assert body["llm"]["proveedor"] == "groq"
    assert body["llm"]["api_key_configurada"] is True
    assert "gsk_secreto_no_exponer" not in resp.text
    assert all(modelo["cargado"] is False for modelo in body["modelos_ml"])
    assert "consultar_ficha_activo" in body["mcp"]["herramientas_registradas"]

