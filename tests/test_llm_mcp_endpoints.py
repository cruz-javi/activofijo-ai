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
    core_transport: httpx.AsyncBaseTransport | None = None,
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
        transport=transport,
        core_transport=core_transport,
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
    assert filtros["grupo_contable"] == "Equipos de Computación"
    assert filtros["estado"] == "MALO"
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
        assert call_result["structuredContent"]["grupo_contable"] == "Vehículos"
        assert call_result["structuredContent"]["estado"] == "MALO"
        assert call_result["structuredContent"]["ubicacion"] == "FICCT"


def test_bucle_react_consulta_analitica_y_redaccion_nlp_sin_tipo(tmp_path: Path) -> None:
    turnos_llm = {"count": 0}
    llamadas_core: list[str] = []

    async def mock_ollama(request: httpx.Request) -> httpx.Response:
        turnos_llm["count"] += 1
        body = json.loads(request.content.decode("utf-8"))
        # Ronda 1 ReAct: el LLM solicita ejecutar herramienta MCP contra PostgreSQL Core
        if turnos_llm["count"] == 1:
            assert "tools" in body and len(body["tools"]) >= 3
            return httpx.Response(
                200,
                json={
                    "model": "uagrm-activo-fijo",
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_act_1",
                                        "type": "function",
                                        "function": {
                                            "name": "buscar_activos_padron",
                                            "arguments": json.dumps(
                                                {"unidad": "FICCT", "condicion": "MALO"}
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"total_tokens": 55},
                },
            )
        # Ronda 2 ReAct: recibe observación de PostgreSQL Core y sintetiza respuesta real
        return httpx.Response(
            200,
            json={
                "model": "uagrm-activo-fijo",
                "choices": [
                    {
                        "message": {
                            "content": "Se halló 1 bien en estado MALO en FICCT: Proyector Epson (U04000018) valuado en Bs. 4.200,00."
                        }
                    }
                ],
                "usage": {"total_tokens": 80},
            },
        )

    async def mock_core(request: httpx.Request) -> httpx.Response:
        llamadas_core.append(request.url.path)
        assert request.headers.get("X-AI-Internal-Key") == CLAVE_TEST
        return httpx.Response(
            200,
            json={
                "totalCoincidencias": 1,
                "restringidoPorCustodia": False,
                "sanitizadoPublico": False,
                "valorMuestraBs": 4200.0,
                "activos": [
                    {
                        "codigo": "U04000018",
                        "descripcion": "Proyector Epson PowerLite",
                        "grupoContable": "EQUIPOS DE COMPUTACIÓN",
                        "ubicacion": "FICCT",
                        "estado": "ASIGNADO",
                        "condicion": "MALO",
                        "valorBs": 4200.0,
                        "custodio": "Ing. Carlos Pérez",
                    }
                ],
            },
        )

    client = _crear_cliente(
        tmp_path,
        transport=httpx.MockTransport(mock_ollama),
        core_transport=httpx.MockTransport(mock_core),
    )
    headers = {"X-AI-Internal-Key": CLAVE_TEST}

    res_ana = client.post(
        "/v1/llm/consulta-analitica",
        headers=headers,
        json={
            "consulta": "¿Qué bienes en estado malo existen en la FICCT?",
            "modulo": "activos",
            "contexto_seguridad": {"acceso_amplio": True, "rol_usuario": "ADMINISTRADOR"},
        },
    )
    assert res_ana.status_code == 200
    body_ana = res_ana.json()
    assert turnos_llm["count"] == 2
    assert "/internal/mcp/buscar-activos" in llamadas_core
    assert body_ana["motor_utilizado"] == "ollama_personalizado"
    assert "U04000018" in body_ana["respuesta"]
    assert any(t["herramienta"] == "buscar_activos_padron" for t in body_ana["trazas_mcp"])
    assert body_ana["activos_mencionados"][0]["codigo"] == "U04000018"

    # Redacción NLP sin tipo_documento explícito -> autodetecta SOLICITUD_REVALUO
    client_det = _crear_cliente(tmp_path)
    res_nlp = client_det.post(
        "/v1/llm/asistir-redaccion",
        headers=headers,
        json={
            "instruccion_usuario": "Necesito solicitar un revalúo técnico pericial para extender vida útil.",
            "solicitante_unidad": "Decanato FICCT",
        },
    )
    assert res_nlp.status_code == 200
    assert res_nlp.json()["tipo_documento_detectado"] == "SOLICITUD_REVALUO"

    # Diagnóstico RF + Normativa
    res_met = client_det.post(
        "/v1/llm/diagnostico-metricas",
        headers=headers,
        json={
            "horizonte_meses": 12,
            "datos_metricas": {
                "proyeccionAjustadaHorizonte": {
                    "bajasEstimadas": 14,
                    "presupuestoReposicionBs": 48500.0,
                },
                "resumenNormativo": {
                    "totalCandidatosBaja": 6,
                    "totalSugeridosRevaluo": 9,
                },
            },
        },
    )
    assert res_met.status_code == 200
    body_met = res_met.json()
    assert body_met["diagnostico_ejecutivo"]
    assert "Bs." in body_met["diagnostico_ejecutivo"]


def test_stream_sse_consulta_analitica_emite_pasos_y_resultado(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path)
    headers = {"X-AI-Internal-Key": CLAVE_TEST}

    res = client.post(
        "/v1/llm/consulta-analitica/stream",
        headers=headers,
        json={"consulta": "Activos en estado malo de FICCT", "modulo": "reportes"},
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    eventos = [
        json.loads(linea[6:])
        for linea in res.text.splitlines()
        if linea.startswith("data: ")
    ]
    assert any(ev.get("tipo") == "paso" and ev.get("estado") == "completado" for ev in eventos)
    assert eventos[-1].get("tipo") == "resultado"


def test_sesion_temporal_hereda_filtros_reporte_y_refina_carta(tmp_path: Path) -> None:
    client = _crear_cliente(tmp_path)
    headers = {"X-AI-Internal-Key": CLAVE_TEST}

    # Refinamiento de reporte en sesión temporal hereda estado previo y añade unidad
    res_rep = client.post(
        "/v1/llm/consulta-analitica",
        headers=headers,
        json={
            "consulta": "fíltralo ahora solo para FICCT y expórtalo en pdf, excel y word con graficas",
            "modulo": "reportes",
            "historial": [
                {"rol": "usuario", "texto": "Activos en estado malo"},
                {"rol": "asistente", "texto": "Padrón consultado en estado malo."},
            ],
            "reporte_previo": {
                "filtros_detectados": {"estado": "MALO"},
                "resumen_ejecutivo": "Se encontraron bienes en estado MALO.",
            },
        },
    )
    assert res_rep.status_code == 200
    body_rep = res_rep.json()
    filtros = body_rep["filtros_detectados"]
    assert filtros["estado"] == "MALO"
    assert filtros["ubicacion"] == "FICCT"
    assert body_rep["formatos_exportacion"] == ["pdf", "excel", "word"]
    assert body_rep["incluir_graficas"] is True

    # Consulta sin solicitud explícita de exportación/gráficas -> listas vacías y gráficas en False
    res_rep_simple = client.post(
        "/v1/llm/consulta-analitica",
        headers=headers,
        json={
            "consulta": "bienes en reparacion deteriorados de Veterinarias entre 2.000 y 10.000 bs",
            "modulo": "reportes",
        },
    )
    assert res_rep_simple.status_code == 200
    body_simple = res_rep_simple.json()
    assert body_simple["filtros_detectados"]["estado"] == "EN_REPARACION"
    assert body_simple["filtros_detectados"]["condicion"] == "DETERIORADO"
    assert body_simple["filtros_detectados"]["ubicacion"] == "Veterinarias"
    assert body_simple["filtros_detectados"]["valor_min"] == 2000.0
    assert body_simple["filtros_detectados"]["valor_max"] == 10000.0
    assert body_simple["formatos_exportacion"] == []
    assert body_simple["incluir_graficas"] is False

    # Redacción extensa (5 párrafos), destinatario implícito y exportación Word/PDF explícita
    res_nueva_carta = client.post(
        "/v1/llm/asistir-redaccion",
        headers=headers,
        json={
            "instruccion_usuario": (
                "solicito dotacion de equipo laboratorio dirigida a Ing. Roberto Paz, Director Administrativo DUAF "
                "para el centro de computo y exportar en word"
            ),
            "solicitante_unidad": "FICCT",
        },
    )
    assert res_nueva_carta.status_code == 200
    body_nueva = res_nueva_carta.json()
    assert body_nueva["tipo_documento_detectado"] == "SOLICITUD_NUEVO_ACTIVO"
    assert len(body_nueva["cuerpo_parrafos"]) == 5
    assert body_nueva["destinatario_nombre"] == "Ing. Roberto Paz"
    assert body_nueva["exportacion_solicitada"] == ["word"]

    # Refinamiento quirúrgico de carta preserva destinatario y párrafos no tocados en sesión temporal
    res_carta = client.post(
        "/v1/llm/asistir-redaccion",
        headers=headers,
        json={
            "tipo_documento": "SOLICITUD_BAJA",
            "instruccion_usuario": "Modifica el parrafo 2 aclarando sobrecarga electrica.",
            "historial": [
                {"rol": "usuario", "texto": "Solicitar baja de proyector"},
                {"rol": "asistente", "texto": "SOLICITUD DE BAJA"},
            ],
            "borrador_actual": {
                "referencia": "SOLICITUD DE BAJA — U04000018",
                "destinatario_nombre": "Ing. Carlos Méndez",
                "destinatario_cargo": "Decano FICCT",
                "cuerpo_parrafos": ["Párrafo inicial de prueba.", "Segundo párrafo técnico."],
            },
        },
    )
    assert res_carta.status_code == 200
    body_carta = res_carta.json()
    assert body_carta["destinatario_nombre"] == "Ing. Carlos Méndez"
    assert body_carta["destinatario_cargo"] == "Decano FICCT"
    assert body_carta["cuerpo_parrafos"][0] == "Párrafo inicial de prueba."
    assert "sobrecarga electrica" in body_carta["cuerpo_parrafos"][1].lower()

    # Ayuda IA cuantitativa consulta padrón real y multi-rol admin
    res_ayuda_cnt = client.post(
        "/v1/llm/asistente-sistema",
        headers=headers,
        json={
            "pregunta": "¿Cuántos equipos en estado malo hay en FICCT?",
            "ruta_actual": "/activos",
            "rol_usuario": "FUNCIONARIO, ADMINISTRADOR",
            "permisos_usuario": ["activos:consultar"],
        },
    )
    assert res_ayuda_cnt.status_code == 200
    body_ayuda_cnt = res_ayuda_cnt.json()
    assert any(t["herramienta"] == "buscar_activos_padron" for t in body_ayuda_cnt["trazas_mcp"])
    assert "buscar_activos_padron" not in body_ayuda_cnt["respuesta"]
    assert any(l["ruta"] == "/usuarios" for l in body_ayuda_cnt["enlaces_directos"]) or body_ayuda_cnt["enlaces_directos"] is not None
