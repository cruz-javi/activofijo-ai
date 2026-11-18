"""Servidor MCP oficial (FastMCP) de solo lectura para Activos Fijos UAGRM."""

import json
import re
import unicodedata
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

from activofijo_ai.core.config import get_settings
from activofijo_ai.modules.llm.core_client import (
    DEFAULT_SEGURIDAD_MCP,
    CoreMcpBridgeClient,
    core_bridge_ctx,
    seguridad_mcp_ctx,
)
from activofijo_ai.modules.llm.knowledge import (
    GUIA_RUTAS_SISTEMA,
    PALABRAS_BLOQUEO_EXTERNO,
    PALABRAS_DOMINIO_VALIDO,
    PLANTILLAS_CARTAS,
    RutaSistemaGuia,
)

CAMPOS_PII_EXCLUIDOS: frozenset[str] = frozenset(
    {"ci", "nit", "usuarioId", "correo", "email", "telefono", "celular"}
)


def _nuevo_fastmcp() -> FastMCP:
    return FastMCP(
        name="UAGRM Activo Fijo MCP Server",
        instructions=(
            "Servidor MCP oficial de solo lectura para gestión de activos fijos de la UAGRM "
            "(Bolivia). Expone recursos normativos, herramientas tipadas de filtrado/validación "
            "y plantillas institucionales bajo moneda oficial en Bolivianos (Bs.)."
        ),
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                "127.0.0.1:*",
                "localhost:*",
                "activofijo-ai:*",
                "testserver",
                "testserver:*",
            ],
        ),
    )


def normalizar_texto(texto: str) -> str:
    """Normaliza tildes y espacios para coincidencia determinista en español."""
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", sin_tildes).strip()


def formatear_bs(valor: Any) -> str:
    """Formatea montos monetarios exclusivamente en Bolivianos (Bs.)."""
    try:
        num = float(valor or 0)
    except (TypeError, ValueError):
        num = 0.0
    monto_es = f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"Bs. {monto_es}"


def normalizar_moneda_texto(texto: str) -> str:
    """Normaliza variantes incorrectas de moneda extranjera hacia la sigla boliviana Bs."""
    return re.sub(r"\bBs\.?\s*S\.?", "Bs.", texto).strip()


def _limpiar_pii_recursivo(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            k: _limpiar_pii_recursivo(v)
            for k, v in data.items()
            if k not in CAMPOS_PII_EXCLUIDOS
        }
    if isinstance(data, list):
        return [_limpiar_pii_recursivo(item) for item in data]
    return data


def _extraer_ubicacion_desde_ctx(ctx: dict[str, Any]) -> str:
    asignacion = ctx.get("asignacion")
    if isinstance(asignacion, dict):
        ubicacion = asignacion.get("ubicacion")
        if isinstance(ubicacion, dict) and ubicacion.get("oficina"):
            return str(ubicacion["oficina"]).strip()
    return str(ctx.get("ubicacion") or "dependencia universitaria asignada").strip()


def construir_parrafo_respaldo_activo(ctx: dict[str, Any], codigo: str) -> str:
    """Construye un párrafo fáctico garantizado con los datos verificados del activo."""
    desc = str(ctx.get("descripcion") or "Bien patrimonial universitario").strip()
    ubicacion = _extraer_ubicacion_desde_ctx(ctx)
    costo = formatear_bs(ctx.get("costo") if ctx.get("costo") is not None else ctx.get("valor"))
    estado = str(ctx.get("estado") or "ACTIVO").strip()
    return (
        f"El bien objeto de la presente solicitud corresponde al código patrimonial {codigo} "
        f"({desc}), registrado actualmente en {ubicacion}, en estado operativo {estado} "
        f"con valor contable de {costo}."
    )


def tiene_permiso_rbac(
    permiso_requerido: str | None,
    permisos_usuario: list[str],
    rol: str,
) -> bool:
    """Evalúa permisos efectivos respetando privilegios RBAC del usuario."""
    if permiso_requerido is None:
        return True
    if rol.upper() in {"ADMINISTRADOR", "SUPER_ADMIN", "ADMIN"}:
        return True
    return permiso_requerido in permisos_usuario


# ---------------------------------------------------------------------------
# Primitiva 1: Recursos MCP (Resources) de solo lectura
# ---------------------------------------------------------------------------


def recurso_catalogo_normativo() -> str:
    """Expone el catálogo institucional de tipos de trámite y normativa aplicable UAGRM."""
    catalogo = {
        tipo: {
            "referencia_base": tpl.referencia_base,
            "titulo_documento": tpl.titulo_documento,
            "normas": list(tpl.normas),
            "destinatario_default": tpl.destinatario_default,
            "cargo_destinatario_default": tpl.cargo_destinatario_default,
        }
        for tipo, tpl in PLANTILLAS_CARTAS.items()
    }
    return json.dumps(catalogo, ensure_ascii=False)


def recurso_plantilla_documento(tipo_documento: str) -> str:
    """Expone la ficha normativa específica para un tipo de documento formal UAGRM."""
    tpl = PLANTILLAS_CARTAS.get(tipo_documento.strip().upper())
    if tpl is None:
        return json.dumps(
            {
                "encontrada": False,
                "tipo_documento": tipo_documento,
                "referencia_base": "SOLICITUD ADMINISTRATIVA DE ACTIVO FIJO",
                "normas": ["Ley N° 1178 (SAFCO)", "D.S. N° 0181 (NB-SABS)"],
                "destinatario_default": "Jefatura del Departamento de Activos Fijos",
                "cargo_destinatario_default": "Dirección Administrativa y Financiera (DUAF) — UAGRM",
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "encontrada": True,
            "tipo_documento": tipo_documento.strip().upper(),
            "referencia_base": tpl.referencia_base,
            "titulo_documento": tpl.titulo_documento,
            "normas": list(tpl.normas),
            "destinatario_default": tpl.destinatario_default,
            "cargo_destinatario_default": tpl.cargo_destinatario_default,
        },
        ensure_ascii=False,
    )


def recurso_rutas_sistema() -> str:
    """Expone el catálogo oficial de módulos y permisos RBAC del panel Activo Fijo UAGRM."""
    rutas = [
        {
            "ruta": r.ruta,
            "titulo": r.titulo,
            "descripcion": r.descripcion,
            "permiso_requerido": r.permiso_requerido,
            "guia_breve": r.guia_breve,
        }
        for r in GUIA_RUTAS_SISTEMA
    ]
    return json.dumps(rutas, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Primitiva 2: Herramientas MCP (Tools) tipadas
# ---------------------------------------------------------------------------


def consultar_ficha_activo(
    cod_activo: str,
    contexto_activo: dict[str, Any] | None = None,
    sanitizar_publico: bool = False,
) -> dict[str, Any]:
    """Valida y normaliza el recurso fáctico de un activo fijo en moneda boliviana (Bs.)."""
    ctx = dict(contexto_activo or {})
    if sanitizar_publico:
        ctx = _limpiar_pii_recursivo(ctx)

    codigo = str(cod_activo or ctx.get("codigo") or "").strip().upper()
    costo_num = ctx.get("costo") if ctx.get("costo") is not None else ctx.get("valor")
    return {
        "codigo": codigo or None,
        "verificado": bool(codigo and ctx),
        "descripcion": str(ctx.get("descripcion") or "").strip() or None,
        "estado": str(ctx.get("estado") or "").strip() or None,
        "ubicacion": _extraer_ubicacion_desde_ctx(ctx) if ctx else None,
        "costo_bs": formatear_bs(costo_num) if costo_num is not None else None,
        "recurso_activo_mcp": ctx or None,
    }


def consultar_reglas_normativas(tipo_documento: str) -> dict[str, Any]:
    """Consulta la base reglamentaria UAGRM y destinatario oficial según el trámite."""
    return json.loads(recurso_plantilla_documento(tipo_documento))


def filtrar_catalogo_reportes(consulta_usuario: str) -> dict[str, Any]:
    """Traduce criterios en lenguaje natural hacia filtros estructurados para reportes UAGRM."""
    norm = normalizar_texto(consulta_usuario)
    if not norm:
        return {
            "search": None,
            "estado": None,
            "condicion": None,
            "grupo_contable": None,
            "ubicacion": None,
            "valor_min": None,
            "valor_max": None,
            "gestion": None,
        }

    estado: str | None = None
    if "baja" in norm or "desincorporad" in norm:
        estado = "BAJA"
    elif "asignad" in norm or "custodia" in norm:
        estado = "ASIGNADO"
    elif "activ" in norm and "activo fijo" not in norm:
        estado = "ACTIVO"

    condicion: str | None = None
    if any(p in norm for p in ("malo", "inutilizad", "deteriorad", "dañad", "danad")):
        condicion = "MALO"
    elif "regular" in norm:
        condicion = "REGULAR"
    elif "bueno" in norm or "optimo" in norm:
        condicion = "BUENO"

    grupo: str | None = None
    if any(p in norm for p in ("comput", "laptop", "pc", "servidor", "impresora", "monitor")):
        grupo = "EQUIPOS DE COMPUTACIÓN"
    elif any(p in norm for p in ("muebl", "escritorio", "silla", "estante", "mesa")):
        grupo = "MUEBLES Y ENSERES"
    elif any(p in norm for p in ("vehic", "camioneta", "auto", "bus")):
        grupo = "VEHÍCULOS"
    elif any(p in norm for p in ("laboratorio", "microscopio", "equipo medico")):
        grupo = "EQUIPOS DE LABORATORIO"

    ubicacion: str | None = None
    for sigla in (
        "ficct",
        "decanato",
        "biblioteca",
        "rectorado",
        "auditorio",
        "duaf",
        "postgrado",
        "tecnologia",
        "derecho",
        "economia",
        "humanidades",
        "veterinaria",
        "camiri",
        "montero",
        "vallecito",
        "cpd",
        "almacen",
    ):
        if sigla in norm:
            ubicacion = sigla.upper()
            break

    valor_min: float | None = None
    valor_max: float | None = None
    m_min = re.search(
        r"(?:mayor(?:es)?\s*(?:a|que|de)|desd?e\s*bs\.?|>=?)\s*(\d+(?:[.,]\d+)?)", norm
    )
    if m_min:
        valor_min = float(m_min.group(1).replace(",", "."))
    m_max = re.search(
        r"(?:menor(?:es)?\s*(?:a|que|de)|hasta\s*bs\.?|<=?)\s*(\d+(?:[.,]\d+)?)", norm
    )
    if m_max:
        valor_max = float(m_max.group(1).replace(",", "."))

    gestion: int | None = None
    m_anio = re.search(r"\b(20\d{2})\b", norm)
    if m_anio:
        gestion = int(m_anio.group(1))

    search: str | None = None
    m_cod = re.search(r"\b([a-z]\d{6,10}|\d{4}-\d{2}-\d{3,6})\b", consulta_usuario, flags=re.IGNORECASE)
    if m_cod:
        search = m_cod.group(1).upper()

    return {
        "search": search,
        "estado": estado,
        "condicion": condicion,
        "grupo_contable": grupo,
        "ubicacion": ubicacion,
        "valor_min": valor_min,
        "valor_max": valor_max,
        "gestion": gestion,
    }


def resumir_lote_patrimonial(
    titulo_reporte: str,
    total_registros: int,
    valor_total_bs: float,
    estados: dict[str, int] | None = None,
    grupos_top: list[dict[str, Any]] | None = None,
    criterios_aplicados: list[str] | None = None,
) -> dict[str, Any]:
    """Calcula un resumen ejecutivo fáctico en Bolivianos (Bs.) sobre un lote auditado."""
    monto = formatear_bs(valor_total_bs)
    estados_texto = ", ".join(f"{k}: {v}" for k, v in (estados or {}).items())
    resumen = (
        f"El reporte consolidado «{titulo_reporte}» comprende {total_registros} registro(s) auditado(s) "
        f"con un valor patrimonial acumulado de {monto}."
        + (f" Distribución por estado: {estados_texto}." if estados_texto else "")
    )
    hallazgos: list[str] = [
        f"Cobertura total analizada: {total_registros} registro(s) por un monto acumulado de {monto}.",
    ]
    for grp in (grupos_top or [])[:2]:
        nombre = grp.get("nombre") or "Grupo principal"
        cnt = grp.get("cantidad") or 0
        hallazgos.append(f"Concentración destacada en {nombre}: {cnt} registro(s).")
    if criterios_aplicados:
        hallazgos.append("Criterios activos: " + " | ".join(criterios_aplicados[:3]))
    return {"resumen_ejecutivo": resumen, "hallazgos_clave": hallazgos}


def consultar_guia_sistema(
    pregunta: str,
    rol_usuario: str = "FUNCIONARIO",
    permisos_usuario: list[str] | None = None,
) -> dict[str, Any]:
    """Filtra rutas del sistema según RBAC y bloquea preguntas fuera del ámbito patrimonial UAGRM."""
    permisos = list(permisos_usuario or [])
    pregunta_norm = normalizar_texto(pregunta)

    bloqueo_externo = any(p in pregunta_norm for p in PALABRAS_BLOQUEO_EXTERNO)
    tiene_dominio = any(p in pregunta_norm for p in PALABRAS_DOMINIO_VALIDO)
    fuera_de_ambito = bloqueo_externo or (
        len(pregunta_norm) > 3 and not tiene_dominio and len(pregunta_norm.split()) >= 5
    )

    if fuera_de_ambito:
        return {
            "fuera_de_ambito": True,
            "enlaces_directos": [],
            "catalogo_permitido": [],
            "respuesta_base": (
                "Soy el Asistente Especializado del Sistema de Activos Fijos de la UAGRM. "
                "Por política institucional solo respondo consultas sobre el manejo de esta plataforma, "
                "sus formularios, actas, trazabilidad o normativas vigentes."
            ),
            "pasos_base": [
                "Pregúntame dónde encontrar un formulario (ej. baja, asignación o corrección).",
                "Consulta cómo funciona el Motor Normativo, Random Forest o cómo cambiar tu contraseña y 2FA.",
            ],
        }

    rutas_disponibles = [
        r for r in GUIA_RUTAS_SISTEMA if tiene_permiso_rbac(r.permiso_requerido, permisos, rol_usuario)
    ]

    coincidencias: list[tuple[int, RutaSistemaGuia]] = []
    for ruta in rutas_disponibles:
        puntaje = sum(1 for kw in ruta.palabras_clave if kw in pregunta_norm)
        if puntaje > 0:
            coincidencias.append((puntaje, ruta))
    coincidencias.sort(key=lambda item: item[0], reverse=True)
    rutas_sugeridas = [item[1] for item in coincidencias[:3]]

    if "baja" in pregunta_norm and not any(r.ruta == "/formularios/baja" for r in rutas_disponibles):
        doc_ruta = next((r for r in rutas_disponibles if r.ruta == "/gestion-documental"), None)
        if doc_ruta and all(r.ruta != "/gestion-documental" for r in rutas_sugeridas):
            rutas_sugeridas.insert(0, doc_ruta)
            rutas_sugeridas = rutas_sugeridas[:3]

    if any(p in pregunta_norm for p in ("contraseña", "contrasena", "password", "2fa", "totp", "autenticador")):
        respuesta_base = (
            "Puede cambiar su contraseña institucional y gestionar su verificación en dos pasos "
            "(2FA TOTP) desde Seguridad de Cuenta."
        )
        pasos_base = [
            "Haga clic en el botón 'Seguridad de Cuenta' de abajo (o en su perfil al pie del menú lateral).",
            "Ingrese a la pestaña de contraseña o verificación 2FA y guarde los cambios.",
        ]
    elif "priorizacion" in pregunta_norm or "normativa" in pregunta_norm:
        respuesta_base = (
            "La priorización normativa evalúa cada bien con reglas deterministas basadas en la "
            "R.R. N° 449/2013 (bajas), R.R. N° 567/2018 (revalúo) y D.S. 24051."
        )
        pasos_base = [
            "Activos con valor residual cercano a Bs 1,00 pero estado físico Bueno o Regular se sugieren para Revalúo Técnico.",
            "Activos en estado Malo o Inutilizado con desgaste severo se clasifican como candidatos a Baja Patrimonial.",
        ]
    elif "baja" in pregunta_norm and not any(r.ruta == "/formularios/baja" for r in rutas_disponibles):
        respuesta_base = (
            "Su perfil de funcionario no registra bajas directas en inventario, pero sí puede emitir "
            "una Carta Formal de Solicitud de Baja hacia Activo Fijo."
        )
        pasos_base = [
            "Abra 'Documentos y Actas' mediante el acceso directo de abajo.",
            "Haga clic en 'Redactar Carta IA', seleccione el activo deteriorado y descargue su carta en Word o PDF para remitirla.",
        ]
    elif rutas_sugeridas:
        principal = rutas_sugeridas[0]
        respuesta_base = principal.guia_breve
        pasos_base = [
            f"Use el acceso directo '{principal.titulo}' mostrado abajo.",
            "Complete los campos o filtros requeridos en la pantalla correspondiente.",
        ]
    else:
        respuesta_base = (
            "Puedo ayudarle a ubicar cualquier formulario del sistema, redactar cartas de "
            "solicitud/corrección, generar reportes IA o consultar normativas UAGRM."
        )
        pasos_base = [
            "Consulte por ejemplo: '¿Dónde está el formulario de baja?', '¿Cómo cambio mi contraseña?' o '¿Cómo redacto una carta de corrección?'.",
            "También puede usar el botón de micrófono para dictar su consulta por voz.",
        ]

    return {
        "fuera_de_ambito": False,
        "enlaces_directos": [
            {"titulo": r.titulo, "ruta": r.ruta, "descripcion": r.descripcion}
            for r in rutas_sugeridas
        ],
        "catalogo_permitido": [
            {"titulo": r.titulo, "ruta": r.ruta, "guia": r.guia_breve}
            for r in (rutas_sugeridas or rutas_disponibles[:6])
        ],
        "respuesta_base": respuesta_base,
        "pasos_base": pasos_base,
    }


def validar_integridad_texto(
    cuerpo_parrafos: list[str],
    contexto_activo: dict[str, Any] | None = None,
    cod_activo: str | None = None,
    es_nuevo_sin_anexo: bool = False,
) -> dict[str, Any]:
    """Valida moneda nacional Bs. y asegura que el código patrimonial verificado no sea omitido."""
    parrafos = [normalizar_moneda_texto(str(p)) for p in cuerpo_parrafos if str(p).strip()]
    advertencias: list[str] = []
    ctx = contexto_activo or {}
    codigo = (cod_activo or "").strip()
    texto_unido = " ".join(parrafos)

    if codigo and not es_nuevo_sin_anexo and codigo not in texto_unido:
        parrafos.insert(min(1, len(parrafos)), construir_parrafo_respaldo_activo(ctx, codigo))
        advertencias.append(
            "El validador MCP insertó automáticamente la referencia exacta del activo para evitar omisiones."
        )

    return {"cuerpo_parrafos": parrafos, "advertencias": advertencias}


def _obtener_cliente_y_seguridad() -> tuple[CoreMcpBridgeClient, dict[str, Any]]:
    cliente = core_bridge_ctx.get()
    if cliente is None:
        cliente = CoreMcpBridgeClient(get_settings())
    seg = seguridad_mcp_ctx.get() or DEFAULT_SEGURIDAD_MCP
    return cliente, {
        "usuarioId": seg.get("usuario_id"),
        "rolUsuario": seg.get("rol_usuario") or "FUNCIONARIO",
        "permisosUsuario": seg.get("permisos_usuario") or [],
        "codigoEmpleadoLegado": seg.get("codigo_empleado_legado"),
        "accesoAmplio": bool(seg.get("acceso_amplio", False)),
        "sanitizarPublico": bool(seg.get("sanitizar_publico", False)),
    }


def _normalizar_param_opcional(valor: Any) -> str | None:
    if valor is None:
        return None
    s = str(valor).strip()
    if not s or s.lower() in {"null", "none", "todos", "n/a", ""}:
        return None
    return s


_UNIDADES_GENERICAS_IGNORADAS: frozenset[str] = frozenset(
    {
        "general",
        "generales",
        "padron",
        "global",
        "todos",
        "todo",
        "total",
        "uagrm",
        "universidad",
        "bueno",
        "buenos",
        "regular",
        "regulares",
        "malo",
        "malos",
        "activo",
        "activos",
        "asignado",
        "baja",
        "bajas",
        "estado",
        "estados",
        "condicion",
        "condiciones",
        "patrimonio",
        "inventario",
        "bueno o malo",
        "estadisticas generales",
    }
)


def _normalizar_unidad_stats(valor: Any) -> str | None:
    s = _normalizar_param_opcional(valor)
    if not s:
        return None
    norm = normalizar_texto(s)
    if norm in _UNIDADES_GENERICAS_IGNORADAS or any(
        w in norm for w in ("bueno", "malo", "regular", "estadistic", "padron", "general")
    ):
        return None
    return s


def clasificar_intencion_redaccion(instruccion_usuario: str) -> dict[str, Any]:
    """Clasifica el tipo de trámite UAGRM (baja, revalúo, corrección, etc.) desde texto libre."""
    q = normalizar_texto(instruccion_usuario)
    if any(p in q for p in ("nuevo activo", "dotacion", "adquisicion", "comprar", "equipamiento nuevo")):
        tipo = "SOLICITUD_NUEVO_ACTIVO"
    elif any(p in q for p in ("descargo", "incidente", "extravio", "robo", "hurto", "siniestro")):
        tipo = "DESCARGO_INCIDENTE"
    elif any(p in q for p in ("baja", "desincorpor", "inservible", "inutilizad", "obsolet", "quemad")):
        tipo = "SOLICITUD_BAJA"
    elif any(p in q for p in ("revalu", "revaloriz", "valor residual", "vida util")):
        tipo = "SOLICITUD_REVALUO"
    elif any(p in q for p in ("reasigna", "transfer", "traslad", "custodio")):
        tipo = "SOLICITUD_REASIGNACION"
    elif any(p in q for p in ("inventario", "verificacion fisica", "conciliacion")):
        tipo = "SOLICITUD_INVENTARIO"
    elif any(p in q for p in ("informe tecnico", "dictamen")):
        tipo = "INFORME_TECNICO"
    else:
        tipo = "SOLICITUD_CORRECCION_DATOS"
    return {"tipo_documento": tipo, "normativa": consultar_reglas_normativas(tipo)}


async def buscar_activos_padron(
    search: str | None = None,
    codigo: str | None = None,
    unidad: str | None = None,
    estado: str | None = None,
    condicion: str | None = None,
    grupo_contable: str | None = None,
    valor_min: float | None = None,
    valor_max: float | None = None,
    gestion: int | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Busca activos reales en PostgreSQL UAGRM por código, texto libre, unidad, estado, condición o rango Bs."""
    cliente, seg = _obtener_cliente_y_seguridad()
    return await cliente.post_internal(
        "buscar-activos",
        {
            "search": _normalizar_param_opcional(search),
            "codigo": _normalizar_param_opcional(codigo),
            "unidad": _normalizar_unidad_stats(unidad),
            "estado": _normalizar_param_opcional(estado),
            "condicion": _normalizar_param_opcional(condicion),
            "grupoContable": _normalizar_param_opcional(grupo_contable),
            "valorMin": valor_min,
            "valorMax": valor_max,
            "gestion": gestion,
            "limit": max(1, min(int(limit or 10), 25)),
            "seguridad": seg,
        },
    )


async def obtener_estadisticas_padron(
    unidad: Annotated[
        str | None,
        Field(description="Solo oficina/facultad exacta; null si es global"),
    ] = None,
) -> dict[str, Any]:
    """Obtiene conteos exactos y sumatoria total en Bs. del padrón PostgreSQL UAGRM por estado, condición y grupo."""
    cliente, seg = _obtener_cliente_y_seguridad()
    return await cliente.post_internal(
        "estadisticas-padron",
        {
            "unidad": _normalizar_unidad_stats(unidad),
            "seguridad": seg,
        },
    )


async def consultar_riesgo_normativo(
    dictamen: str | None = None,
    nivel_prioridad: str | None = None,
    search: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Consulta activos priorizados por el Motor Normativo (CANDIDATO_BAJA R.R. 449/2013 y SUGERIDO_REVALUO R.R. 567/2018)."""
    cliente, seg = _obtener_cliente_y_seguridad()
    return await cliente.post_internal(
        "riesgo-normativo",
        {
            "dictamen": _normalizar_param_opcional(dictamen),
            "nivelPrioridad": _normalizar_param_opcional(nivel_prioridad),
            "search": _normalizar_param_opcional(search),
            "limit": max(1, min(int(limit or 8), 15)),
            "seguridad": seg,
        },
    )


async def consultar_trazabilidad_sha256(cod_activo: str) -> dict[str, Any]:
    """Verifica la cadena criptográfica SHA-256 e historial de custodia/movimientos de un activo por su código."""
    cliente, seg = _obtener_cliente_y_seguridad()
    return await cliente.post_internal(
        "trazabilidad",
        {
            "codActivo": cod_activo.strip().upper(),
            "seguridad": seg,
        },
    )


async def consultar_proyeccion_random_forest(horizonte_meses: int = 6) -> dict[str, Any]:
    """Consulta la proyección Random Forest de bajas esperadas y presupuesto de reposición en Bs. (3, 6 o 12 meses)."""
    cliente, _ = _obtener_cliente_y_seguridad()
    horizon = horizonte_meses if horizonte_meses in (3, 6, 12) else 6
    return await cliente.post_internal(
        "proyeccion-rf",
        {"horizonteMeses": horizon},
    )


# ---------------------------------------------------------------------------
# Primitiva 3: Plantillas de Instrucción MCP (Prompts)
# ---------------------------------------------------------------------------


def prompt_redaccion_carta_uagrm(recurso_mcp_json: str, instruccion_usuario: str) -> str:
    """Plantilla formal MCP para redacción de notas administrativas UAGRM sin alucinaciones."""
    return (
        "Redacta el cuerpo epistolar de una carta formal UAGRM en español de Bolivia "
        "usando EXCLUSIVAMENTE los datos del Recurso MCP adjunto.\n"
        f"Recurso MCP: {recurso_mcp_json}\n"
        f"Motivo o instrucción del solicitante: {instruccion_usuario}\n\n"
        "Reglas de estilo:\n"
        "- NO incluyas fecha, destinatario, 'Señor(a)', 'Presente' ni firma dentro de cuerpo_parrafos "
        "(el formato Word/PDF ya los agrega).\n"
        "- Moneda oficial: Bolivianos abreviado estrictamente como 'Bs.' (NUNCA uses 'Bs.S' ni dólares).\n"
        "- Genera exactamente 3 párrafos claros y formales en cuerpo_parrafos:\n"
        "  1) Objeto de la solicitud y referencia a la unidad solicitante y normativa UAGRM aplicable.\n"
        "  2) Detalle técnico del activo (si existe codigo en Recurso MCP, cítalo exacto con su descripción y monto) "
        "o especificación del requerimiento.\n"
        "  3) Párrafo de disposición para verificación técnica/anexo y cierre cordial.\n\n"
        "Devuelve SOLO un objeto JSON con las claves exactas:\n"
        '{"referencia": string, "cuerpo_parrafos": [string, string, string], "despedida": string}'
    )


def prompt_interpretar_reporte_uagrm(
    contexto_modulo: str,
    consulta_usuario: str,
    resumen_lote_json: str,
) -> str:
    """Plantilla MCP para traducción de consultas en lenguaje natural y resumen de lotes."""
    return (
        "Eres un intérprete de consultas MCP y analista patrimonial de la UAGRM.\n"
        f"Módulo: {contexto_modulo}\n"
        f"Consulta en lenguaje natural: {consulta_usuario or 'Sin filtro adicional'}\n"
        f"Resumen real del lote (JSON): {resumen_lote_json}\n\n"
        "Devuelve SOLO JSON con estas claves:\n"
        '{"filtros": {"search": str|null, "estado": "ACTIVO"|"ASIGNADO"|"BAJA"|null, '
        '"condicion": "BUENO"|"REGULAR"|"MALO"|null, "grupo_contable": str|null, '
        '"ubicacion": str|null, "valor_min": float|null, "valor_max": float|null, "gestion": int|null}, '
        '"explicacion": str, "resumen_ejecutivo": str|null, "hallazgos_clave": [str]}'
    )


def prompt_asistente_sistema_uagrm(
    rol_usuario: str,
    ruta_actual: str,
    catalogo_permitido_json: str,
    pregunta: str,
) -> str:
    """Plantilla MCP para guía operativa dentro del sistema UAGRM acotada por RBAC."""
    return (
        f"Rol del usuario: {rol_usuario} | Ruta actual: {ruta_actual}\n"
        f"Módulos permitidos para este usuario (MCP): {catalogo_permitido_json}\n"
        f"Pregunta del usuario: {pregunta}\n\n"
        "Responde en español claro y conciso guiando al usuario dentro del sistema UAGRM. "
        "Devuelve SOLO JSON: "
        '{"respuesta": string, "pasos": [string, string], "fuera_de_ambito": false}'
    )


def crear_servidor_mcp() -> FastMCP:
    """Instancia un servidor FastMCP aislado con Recursos, Herramientas y Prompts UAGRM."""
    servidor = _nuevo_fastmcp()

    servidor.resource("uagrm://normativa/catalogo", mime_type="application/json")(
        recurso_catalogo_normativo
    )
    servidor.resource(
        "uagrm://normativa/plantilla/{tipo_documento}", mime_type="application/json"
    )(recurso_plantilla_documento)
    servidor.resource("uagrm://sistema/rutas", mime_type="application/json")(recurso_rutas_sistema)

    for herramienta in (
        consultar_ficha_activo,
        consultar_reglas_normativas,
        filtrar_catalogo_reportes,
        resumir_lote_patrimonial,
        consultar_guia_sistema,
        validar_integridad_texto,
        clasificar_intencion_redaccion,
        buscar_activos_padron,
        obtener_estadisticas_padron,
        consultar_riesgo_normativo,
        consultar_trazabilidad_sha256,
        consultar_proyeccion_random_forest,
    ):
        servidor.tool()(herramienta)

    for plantilla in (
        prompt_redaccion_carta_uagrm,
        prompt_interpretar_reporte_uagrm,
        prompt_asistente_sistema_uagrm,
    ):
        servidor.prompt()(plantilla)

    return servidor


mcp_server = crear_servidor_mcp()

if __name__ == "__main__":
    mcp_server.run()
