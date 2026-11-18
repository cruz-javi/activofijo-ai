import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from mcp.shared.memory import create_connected_server_and_client_session

from activofijo_ai.core.config import Settings, get_settings
from activofijo_ai.main import create_app
from activofijo_ai.modules.llm.mcp_bridge import McpBridgeService
from activofijo_ai.modules.llm.mcp_server import mcp_server
from activofijo_ai.modules.llm.router import get_mcp_bridge_service

CLAVE_TEST = "test_internal_ai_key_super_segura_32_caracteres!"


def _crear_cliente(
    tmp_path: Path,
    transport: httpx.AsyncBaseTransport | None = None,
) -> TestClient:
    app = create_app()
    settings = Settings(
        env="test",
        llm_provider="ollama",
        llm_model="uagrm-activo-fijo",
        ai_internal_api_key=CLAVE_TEST,
        models_dir=tmp_path,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_mcp_bridge_service] = lambda: McpBridgeService(
        transport=transport
    )
    return TestClient(app)


def test_endpoints_descartados_ya_no_existen(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path)
    headers = {"X-AI-Internal-Key": CLAVE_TEST}

    assert client.post("/v1/ml/normativa/priorizar", headers=headers, json={}).status_code == 404
    assert client.post("/v1/vision/ocr-documento", headers=headers, json={}).status_code == 404


def test_asistir_redaccion_baja_con_activo_y_nuevo_sin_anexo(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path)
    headers = {"X-AI-Internal-Key": CLAVE_TEST}

    resp_baja = client.post(
        "/v1/llm/asistir-redaccion",
        headers=headers,
        json={
            "tipo_documento": "SOLICITUD_BAJA",
            "modo_acceso": "PUBLICO_HOME",
            "cod_activo": "U04000018",
            "instruccion_usuario": "Equipo quedó inservible por falla eléctrica irreversible.",
            "solicitante_nombre": "Ing. Mario Ruiz",
            "solicitante_unidad": "Biblioteca Central",
            "contexto_activo": {
                "codigo": "U04000018",
                "descripcion": "Servidor blade 64 cores",
                "estado": "ACTIVO",
                "costo": 42500.5,
                "asignacion": {"ubicacion": {"oficina": "Biblioteca Central"}},
            },
        },
    )
    assert resp_baja.status_code == 200
    data_baja = resp_baja.json()
    assert data_baja["incluye_anexo_tecnico"] is True
    assert any("U04000018" in p for p in data_baja["cuerpo_parrafos"])
    assert any("449/2013" in n for n in data_baja["fundamento_normativo"])

    resp_nuevo = client.post(
        "/v1/llm/asistir-redaccion",
        headers=headers,
        json={
            "tipo_documento": "SOLICITUD_NUEVO_ACTIVO",
            "modo_acceso": "PUBLICO_HOME",
            "instruccion_usuario": "Solicitamos 4 proyectores láser para las aulas del módulo 236.",
            "solicitante_unidad": "Decanato FICCT",
        },
    )
    assert resp_nuevo.status_code == 200
    data_nuevo = resp_nuevo.json()
    assert data_nuevo["incluye_anexo_tecnico"] is False
    assert "4 proyectores" in " ".join(data_nuevo["cuerpo_parrafos"])


def test_validador_anti_alucinacion_ollama_inserta_codigo_faltante(tmp_path: Path) -> None:
    async def mock_ollama(_request: httpx.Request) -> httpx.Response:
        contenido = json.dumps(
            {
                "referencia": "SOLICITUD DE CORRECCIÓN DE DATOS",
                "cuerpo_parrafos": ["Solicito corregir el número de serie registrado."],
                "despedida": "Atentamente.",
            }
        )
        return httpx.Response(
            200,
            json={
                "model": "uagrm-activo-fijo",
                "choices": [{"message": {"content": contenido}}],
                "usage": {"total_tokens": 90},
            },
        )

    client = _crear_cliente(tmp_path, transport=httpx.MockTransport(mock_ollama))
    resp = client.post(
        "/v1/llm/asistir-redaccion",
        headers={"X-AI-Internal-Key": CLAVE_TEST},
        json={
            "tipo_documento": "SOLICITUD_CORRECCION_DATOS",
            "cod_activo": "U03000999",
            "instruccion_usuario": "Corregir serie mal transcrita en inventario.",
            "contexto_activo": {
                "codigo": "U03000999",
                "descripcion": "Computadora portátil Dell",
                "estado": "ASIGNADO",
                "costo": 7200,
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["motor_utilizado"] == "ollama_personalizado"
    assert body["modelo"] == "uagrm-activo-fijo"
    assert body["incluye_anexo_tecnico"] is True
    assert any("U03000999" in p for p in body["cuerpo_parrafos"])
    assert len(body["advertencias"]) >= 1


def test_interpretar_consulta_y_resumen_ejecutivo_lote(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path)
    resp = client.post(
        "/v1/llm/interpretar-consulta",
        headers={"X-AI-Internal-Key": CLAVE_TEST},
        json={
            "consulta_usuario": "computadoras en estado malo de la FICCT desde 2022 mayores a 3000 bs",
            "contexto_modulo": "activos",
            "resumen_lote": {
                "titulo_reporte": "REPORTE DE COMPUTADORAS FICCT",
                "total_registros": 14,
                "valor_total_bs": 68400.0,
                "estados": {"ACTIVO": 10, "BAJA": 4},
                "grupos_top": [{"nombre": "EQUIPOS DE COMPUTACIÓN", "cantidad": 14}],
                "criterios_aplicados": ["Unidad: FICCT"],
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    filtros = body["filtros_detectados"]
    assert filtros["grupo_contable"] == "EQUIPOS DE COMPUTACIÓN"
    assert filtros["condicion"] == "MALO"
    assert filtros["ubicacion"] == "FICCT"
    assert filtros["valor_min"] == 3000.0
    assert filtros["gestion"] == 2022
    assert "14 registro(s)" in (body["resumen_ejecutivo"] or "")


def test_asistente_sistema_respeta_rbac_y_bloquea_fuera_de_dominio(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path)
    headers = {"X-AI-Internal-Key": CLAVE_TEST}

    # Funcionario sin activos:baja preguntando por baja -> redirigido a Documentos y Actas
    resp_func = client.post(
        "/v1/llm/asistente-sistema",
        headers=headers,
        json={
            "pregunta": "¿Dónde está el formulario de baja de un activo?",
            "ruta_actual": "/dashboard",
            "rol_usuario": "FUNCIONARIO",
            "permisos_usuario": ["activos:consultar"],
        },
    )
    assert resp_func.status_code == 200
    body_func = resp_func.json()
    assert body_func["fuera_de_ambito"] is False
    assert all(link["ruta"] != "/formularios/baja" for link in body_func["enlaces_directos"])
    assert any(link["ruta"] == "/gestion-documental" for link in body_func["enlaces_directos"])
    assert any("Documentos y Actas" in paso for paso in body_func["pasos"])

    # Jefe de Activo Fijo con activos:baja -> enlace directo a /formularios/baja
    resp_jefe = client.post(
        "/v1/llm/asistente-sistema",
        headers=headers,
        json={
            "pregunta": "¿Dónde está el formulario de baja?",
            "ruta_actual": "/dashboard",
            "rol_usuario": "JEFE_ACTIVO_FIJO",
            "permisos_usuario": ["activos:consultar", "activos:baja"],
        },
    )
    assert resp_jefe.status_code == 200
    assert any(
        link["ruta"] == "/formularios/baja" for link in resp_jefe.json()["enlaces_directos"]
    )

    # Pregunta fuera de ámbito -> bloqueada controladamente
    resp_off = client.post(
        "/v1/llm/asistente-sistema",
        headers=headers,
        json={
            "pregunta": "Dame una receta de cocina para preparar pastel de chocolate",
            "ruta_actual": "/dashboard",
            "rol_usuario": "FUNCIONARIO",
            "permisos_usuario": ["activos:consultar"],
        },
    )
    assert resp_off.status_code == 200
    assert resp_off.json()["fuera_de_ambito"] is True
    assert resp_off.json()["enlaces_directos"] == []


async def test_servidor_fastmcp_nativo_expone_recursos_tools_y_prompts() -> None:
    async with create_connected_server_and_client_session(mcp_server._mcp_server) as session:
        tools_res = await session.list_tools()
        tool_names = {t.name for t in tools_res.tools}
        assert {
            "consultar_ficha_activo",
            "consultar_reglas_normativas",
            "filtrar_catalogo_reportes",
            "resumir_lote_patrimonial",
            "consultar_guia_sistema",
            "validar_integridad_texto",
        }.issubset(tool_names)

        resources_res = await session.list_resources()
        assert "uagrm://normativa/catalogo" in {str(r.uri) for r in resources_res.resources}

        prompts_res = await session.list_prompts()
        assert {
            "prompt_redaccion_carta_uagrm",
            "prompt_interpretar_reporte_uagrm",
            "prompt_asistente_sistema_uagrm",
        }.issubset({p.name for p in prompts_res.prompts})

        ficha_call = await session.call_tool(
            "consultar_ficha_activo",
            {
                "cod_activo": "U04000018",
                "sanitizar_publico": True,
                "contexto_activo": {
                    "codigo": "U04000018",
                    "descripcion": "Servidor blade",
                    "costo": 1500,
                    "ci": "9999999",
                    "nit": "1234567",
                },
            },
        )
        assert ficha_call.structuredContent is not None
        assert ficha_call.structuredContent["costo_bs"] == "Bs. 1.500,00"
        recurso = ficha_call.structuredContent["recurso_activo_mcp"]
        assert "ci" not in recurso and "nit" not in recurso


def test_endpoint_jsonrpc_streamable_http_mcp_protegido(tmp_path: Path) -> None:
    app = create_app()
    settings = Settings(env="test", ai_internal_api_key=CLAVE_TEST, models_dir=tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings

    mcp_headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    with TestClient(app) as client:
        unauth = client.post(
            "/mcp/",
            headers=mcp_headers,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert unauth.status_code == 401

        auth_list = client.post(
            "/mcp/",
            headers={**mcp_headers, "X-AI-Internal-Key": CLAVE_TEST},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert auth_list.status_code == 200
        rpc_body = auth_list.json()
        assert any(t["name"] == "filtrar_catalogo_reportes" for t in rpc_body["result"]["tools"])

        auth_call = client.post(
            "/mcp/",
            headers={**mcp_headers, "X-AI-Internal-Key": CLAVE_TEST},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "filtrar_catalogo_reportes",
                    "arguments": {"consulta_usuario": "vehiculos en estado malo de la FICCT"},
                },
            },
        )
        assert auth_call.status_code == 200
        call_result = auth_call.json()["result"]
        assert call_result["structuredContent"]["grupo_contable"] == "VEHÍCULOS"
        assert call_result["structuredContent"]["condicion"] == "MALO"
        assert call_result["structuredContent"]["ubicacion"] == "FICCT"
