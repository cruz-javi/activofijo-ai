"""Servidor MCP oficial (FastMCP) de solo lectura para Activos Fijos UAGRM."""

import difflib
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
from activofijo_ai.modules.llm.corrector_institucional import (
    corregir_texto_institucional,
    resolver_facultad_oficial_uagrm,
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
    """Evalúa permisos efectivos respetando privilegios RBAC del usuario (multi-rol)."""
    if permiso_requerido is None:
        return True
    roles_tokens = [
        tok.strip().upper()
        for tok in re.split(r"[,;\s]+", rol or "")
        if tok.strip()
    ]
    if any(r in {"ADMINISTRADOR", "SUPER_ADMIN", "ADMIN"} for r in roles_tokens):
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


ESTADOS_PADRON_VALIDOS: frozenset[str] = frozenset(
    {"BUENO", "REGULAR", "MALO", "EN_REPARACION", "BAJA", "ACTIVO", "ASIGNADO"}
)
CONDICIONES_PADRON_VALIDAS: frozenset[str] = frozenset(
    {"NUEVO", "USADO", "DETERIORADO", "BUENO", "REGULAR", "MALO"}
)

_PALABRAS_NO_SEARCH: frozenset[str] = frozenset(
    {
        "de", "del", "la", "las", "el", "los", "lo", "un", "una", "unos", "unas",
        "en", "por", "para", "con", "sin", "sobre", "entre", "desde", "hasta", "hacia",
        "y", "o", "u", "a", "al", "se", "su", "sus", "me", "nos", "le", "les",
        "que", "cual", "cuales", "cuanto", "cuantos", "cuanta", "quien", "donde",
        "cuyo", "cuya", "cuyos", "cuyas", "como", "aqui", "alla", "ahi", "solo", "solamente",
        "hay", "son", "estan", "esta", "este", "estos", "estas", "ese", "esa", "esos", "esas",
        "estar", "estaban", "fueron", "seran", "tener", "puede", "pueden", "puedes", "podria",
        "debe", "deben", "encuentra", "encuentran", "ubicado", "ubicada", "ubicados", "ubicadas",
        "ubica", "ubican", "ubiquen", "conoce", "conocido", "conocida", "conocidos", "conocidas",
        "llama", "llamado", "llamada", "llamados", "llamadas", "pertenece", "pertenecen",
        "perteneciente", "pertenecientes", "corresponde", "corresponden", "asignado", "asignada",
        "asignados", "asignadas", "registrado", "registrada", "registrados", "registradas",
        "inventariado", "inventariados", "todos", "todas", "todo", "toda", "tengan", "tenga",
        "tiene", "tienen", "sea", "sean", "algun", "algunos", "algunas", "algo", "ningun",
        "ninguno", "mismo", "misma", "tambien", "ademas", "aproximadamente", "decir", "alias",
        "sigla", "siglas", "quiero", "queremos", "necesito", "necesitamos", "dame", "muestrame",
        "mostrar", "ver", "lista", "listado", "genera", "generar", "haz", "hazme", "saca",
        "sacar", "busca", "buscar", "filtralo", "filtra", "filtrar", "detalla", "detallar",
        "indica", "indicar", "favor", "hola", "buenos", "dias", "tardes", "gracias",
        "reporte", "reportes", "informe", "informes", "consulta", "padron", "resumen",
        "conteo", "cantidad", "total", "inventario", "activo", "activos", "bien", "bienes",
        "equipo", "equipos", "item", "items",
        "vehiculo", "vehiculos", "mueble", "muebles", "enseres", "computadora", "computadoras",
        "laptop", "laptops", "portatil", "portatiles", "pc", "maquinaria", "herramienta",
        "herramientas", "edificacion", "edificaciones", "inmueble", "inmuebles", "terreno",
        "terrenos", "semoviente", "semovientes", "animal", "animales",
        "estado", "estados", "condicion", "condiciones", "bueno", "buen", "buena",
        "buenas", "regular", "regulares", "malo", "malos", "mal", "mala", "malas",
        "reparacion", "mantenimiento", "baja", "bajas", "dado", "dados", "desincorporado",
        "nuevo", "nuevos", "nueva", "nuevas", "usado", "usados", "usada", "usadas",
        "deteriorado", "deteriorados", "deteriorada", "deterioradas", "inservible", "inservibles",
        "inutilizado", "inutilizados", "danado", "danados", "averiado", "averiados",
        "unidad", "unidades", "oficina", "oficinas", "facultad", "facultades", "departamento",
        "dependencia", "sede", "sedes", "area", "campus", "centro", "experimental", "central",
        "universitaria", "universitario", "carrera", "programa", "pabellon", "modulo", "bloque",
        "aula", "ciencias", "exactas", "tecnologia", "salud", "humana", "administrativa",
        "financiera", "veterinaria", "veterinarias", "computacion", "telecomunicaciones", "cs",
        "valor", "monto", "precio", "costo", "bs", "boliviano", "bolivianos", "mayor", "mayores",
        "menor", "menores", "superior", "superiores", "inferior", "inferiores", "mas", "menos",
        "gestion", "ano", "anio", "marca", "tipo", "exporta", "exportar", "exportalo", "descarga",
        "descargar", "descargalo", "imprime", "imprimir", "formato", "formatos", "pdf", "excel",
        "xlsx", "word", "docx", "csv", "grafica", "graficas", "grafico", "graficos", "tres",
        "ahora", "fic", "ficct", "fcet", "fcv", "daf", "duaf", "rectorado", "vicerrectorado",
        "biblioteca", "medicina", "prado", "yabare", "decanato", "auditorio", "postgrado",
        "derecho", "economia", "humanidades", "contaduria", "contables", "auditoria",
        "agricolas", "agronomia", "juridicas", "politicas", "camiri", "montero",
        "vallecito", "cpd", "almacen",
    }
)


def _extraer_termino_busqueda_residual(norm: str) -> str | None:
    for tk in re.findall(r"[a-z0-9]{3,}", norm):
        if tk.isdigit() or tk in _PALABRAS_NO_SEARCH:
            continue
        if len(tk) >= 5 and tk.endswith("s") and not tk.endswith("ss"):
            base = tk[:-1]
            if base in _PALABRAS_NO_SEARCH:
                continue
            return base.upper()
        return tk.upper()
    return None


def _resolver_search_consulta(consulta_usuario: str, norm: str) -> str | None:
    m_cod = re.search(
        r"\b([a-z]\d{6,10}|\d{4}-\d{2}-\d{3,6})\b",
        consulta_usuario,
        flags=re.IGNORECASE,
    )
    if m_cod:
        return m_cod.group(1).upper()

    for palabra_clave in (
        "aire acondicionado",
        "proyector",
        "impresora",
        "microscopio",
        "servidor",
        "camioneta",
        "escritorio",
        "estante",
        "archivador",
        "fotocopiadora",
        "scanner",
        "silla",
        "mesa",
        "monitor",
        "ups",
        "pizarra",
    ):
        if re.search(rf"\b{re.escape(palabra_clave)}(?:s|es)?\b", norm):
            return palabra_clave.upper()

    return _extraer_termino_busqueda_residual(norm)


_ALIAS_UBICACIONES: tuple[tuple[str, str], ...] = (
    ("facultad de ciencias de la computacion y telecomunicaciones", "FICCT"),
    ("facultad de cs de la computacion y telecomunicaciones", "FICCT"),
    ("facultad de cs. de la computacion y telecomunicaciones", "FICCT"),
    ("facultad de ciencias de la computacion", "FICCT"),
    ("facultad de cs de la computacion", "FICCT"),
    ("facultad de computacion", "FICCT"),
    ("ciencias de la computacion", "FICCT"),
    ("cs de la computacion", "FICCT"),
    ("computacion y telecomunicaciones", "FICCT"),
    ("ingenieria en sistemas", "FICCT"),
    ("ingenieria de sistemas", "FICCT"),
    ("ingenieria informatica", "FICCT"),
    ("ficct", "FICCT"),
    ("la fic", "FICCT"),
    ("fic", "FICCT"),
    ("facultad de ciencias veterinarias", "Veterinarias"),
    ("ciencias veterinarias", "Veterinarias"),
    ("medicina veterinaria", "Veterinarias"),
    ("veterinarias", "Veterinarias"),
    ("veterinaria", "Veterinarias"),
    ("zootecnia", "Veterinarias"),
    ("fcv", "Veterinarias"),
    ("centro experimental el prado", "El Prado"),
    ("el prado", "El Prado"),
    ("prado", "El Prado"),
    ("centro experimental yabare", "Yabaré"),
    ("yabare", "Yabaré"),
    ("facultad de ciencias exactas y tecnologia", "Exactas y Tecnología"),
    ("ciencias exactas y tecnologia", "Exactas y Tecnología"),
    ("exactas y tecnologia", "Exactas y Tecnología"),
    ("ciencias exactas", "Exactas y Tecnología"),
    ("facultad de tecnologia", "Exactas y Tecnología"),
    ("fcet", "Exactas y Tecnología"),
    ("facultad de ciencias de la salud humana", "Salud Humana"),
    ("ciencias de la salud humana", "Salud Humana"),
    ("ciencias de la salud", "Salud Humana"),
    ("salud humana", "Salud Humana"),
    ("facultad de medicina", "Salud Humana"),
    ("medicina", "Salud Humana"),
    ("direccion administrativa y financiera", "DAF"),
    ("administrativa y financiera", "DAF"),
    ("duaf", "DAF"),
    ("daf", "DAF"),
    ("biblioteca central universitaria", "Biblioteca"),
    ("biblioteca central", "Biblioteca"),
    ("biblioteca", "Biblioteca"),
    ("rectorado y vicerrectorado", "Rectorado"),
    ("vicerrectorado", "Rectorado"),
    ("rectorado", "Rectorado"),
    ("decanato", "DECANATO"),
    ("auditorio", "AUDITORIO"),
    ("postgrado", "POSTGRADO"),
    ("departamento de activo fijo", "Activo Fijo"),
    ("activo fijo y control patrimonial", "Activo Fijo"),
    ("tecnologias de la informacion", "DTIC"),
    ("dtic", "DTIC"),
    ("cpd", "CPD"),
    ("facultad de ciencias contables", "Ciencias Contables"),
    ("ciencias contables", "Ciencias Contables"),
    ("facultad de contaduria", "Ciencias Contables"),
    ("contaduria publica", "Ciencias Contables"),
    ("contaduria", "Ciencias Contables"),
    ("auditoria financiera", "Ciencias Contables"),
    ("facultad de ciencias economicas", "Económicas"),
    ("ciencias economicas", "Económicas"),
    ("economia y empresariales", "Económicas"),
    ("economia", "Económicas"),
    ("facultad de ciencias juridicas", "Jurídicas"),
    ("ciencias juridicas", "Jurídicas"),
    ("facultad de derecho", "Jurídicas"),
    ("derecho", "Jurídicas"),
    ("facultad de humanidades", "Humanidades"),
    ("humanidades", "Humanidades"),
    ("facultad de ciencias agricolas", "Agrícolas"),
    ("ciencias agricolas", "Agrícolas"),
    ("agronomia", "Agrícolas"),
    ("facultad de ciencias del habitat", "Hábitat"),
    ("ciencias del habitat", "Hábitat"),
    ("arquitectura", "Hábitat"),
    ("facultad de ciencias farmaceuticas", "Farmacéuticas"),
    ("farmaceuticas y bioquimicas", "Farmacéuticas"),
    ("bioquimica y farmacia", "Farmacéuticas"),
    ("farmacia", "Farmacéuticas"),
    ("facultad politecnica", "Politécnica"),
    ("politecnica", "Politécnica"),
    ("facultad integral del norte", "FINOR"),
    ("finor", "FINOR"),
    ("montero", "Montero"),
    ("facultad integral del chaco", "FICH"),
    ("fich", "FICH"),
    ("camiri", "Camiri"),
    ("facultad integral de los valles", "Valles Cruceños"),
    ("valles crucenos", "Valles Cruceños"),
    ("fivc", "FIVC"),
    ("tecnologia", "Exactas y Tecnología"),
    ("vallecito", "Agrícolas"),
    ("almacen", "Almacén"),
)

_STOP_UBICACION_DINAMICA = (
    r"\b(?:en\s+(?:mal|buen|regular)\s+estado|estado\s+\w+|condicion\s+\w+|"
    r"o\s+(?:mas\s+)?conocid[oa]\s+como|tambien\s+(?:conocid[oa]|llamad[oa])\s+como|"
    r"es\s+decir|que\s+queda|con\s+valor|mayor(?:es)?|menor(?:es)?|entre\s+\d|gestion\s+20\d{2}|"
    r"de\s+la\s+gestion|exporta\w*|en\s+pdf|en\s+excel|en\s+word|con\s+grafic\w*)\b"
)

_REFERENCIAS_FUZZY_UNIDAD: tuple[tuple[str, str], ...] = (
    ("computacion telecomunicaciones sistemas informatica ficct fic", "FICCT"),
    ("exactas tecnologia ingenieria civil quimica fcet", "Exactas y Tecnología"),
    ("veterinaria veterinarias zootecnia fcv", "Veterinarias"),
    ("el prado centro experimental prado", "El Prado"),
    ("yabare centro experimental yabare", "Yabaré"),
    ("salud humana medicina enfermeria", "Salud Humana"),
    ("farmaceuticas bioquimicas farmacia bioquimica", "Farmacéuticas"),
    ("direccion administrativa financiera daf duaf", "DAF"),
    ("departamento activo fijo control patrimonial", "Activo Fijo"),
    ("tecnologias informacion comunicacion dtic cpd", "DTIC"),
    ("biblioteca central universitaria", "Biblioteca"),
    ("rectorado vicerrectorado", "Rectorado"),
    ("ciencias contables contaduria auditoria control gestion", "Ciencias Contables"),
    ("ciencias economicas economia empresariales administracion", "Económicas"),
    ("ciencias juridicas derecho politicas sociales relaciones", "Jurídicas"),
    ("ciencias agricolas agronomia forestal vallecito", "Agrícolas"),
    ("ciencias habitat diseno arte arquitectura", "Hábitat"),
    ("facultad politecnica", "Politécnica"),
    ("humanidades educacion psicologia", "Humanidades"),
    ("integral norte finor montero", "FINOR"),
    ("integral chaco fich camiri", "FICH"),
    ("integral valles crucenos fivc", "Valles Cruceños"),
)


def _parse_monto_bo(raw: str) -> float | None:
    limpio = raw.strip().replace(" ", "")
    if not limpio:
        return None
    if "," in limpio and "." in limpio:
        if limpio.rfind(",") > limpio.rfind("."):
            limpio = limpio.replace(".", "").replace(",", ".")
        else:
            limpio = limpio.replace(",", "")
    elif "." in limpio:
        partes = limpio.split(".")
        if (len(partes) == 2 and len(partes[1]) == 3) or len(partes) > 2:
            limpio = "".join(partes)
    elif "," in limpio:
        partes = limpio.split(",")
        if (len(partes) == 2 and len(partes[1]) == 3) or len(partes) > 2:
            limpio = "".join(partes)
        else:
            limpio = limpio.replace(",", ".")
    try:
        return float(limpio)
    except ValueError:
        return None


def _mejor_unidad_cercana(fragmento: str) -> str | None:
    """Empareja una mención libre o con errores fonéticos con la unidad institucional UAGRM más cercana."""
    tokens = [
        tk
        for tk in re.findall(r"[a-z0-9]{3,}", normalizar_texto(fragmento))
        if tk not in {"de", "del", "la", "el", "en", "y", "los", "las", "facultad", "unidad", "oficina", "centro", "experimental", "direccion", "carrera"}
    ]
    if not tokens:
        return None

    mejor_unidad: str | None = None
    mejor_puntaje = 0.0
    for ref_claves, unidad_canon in _REFERENCIAS_FUZZY_UNIDAD:
        palabras_ref = ref_claves.split()
        coincidencias = 0.0
        for tk in tokens:
            mejor_tk = max(
                difflib.SequenceMatcher(None, tk, pal).ratio()
                for pal in palabras_ref
            )
            if mejor_tk >= 0.76:
                coincidencias += mejor_tk
        puntaje = coincidencias / max(len(tokens), 1)
        if puntaje > mejor_puntaje:
            mejor_puntaje = puntaje
            mejor_unidad = unidad_canon

    if mejor_unidad and mejor_puntaje >= 0.72:
        return mejor_unidad
    return None


def _suprimir_clausula_ubicacion(norm: str, inicio: int, fin: int) -> str:
    """Remueve de la consulta normalizada el sintagma locativo y su posible explicación de alias."""
    prefijo = norm[:inicio]
    m_pre = re.search(
        r"(?:(?:se\s+)?(?:encuentr[ae]n?|ubicad[oa]s?|pertenec\w+|asignad[oa]s?)\s+)?"
        r"(?:en|de|a|para)?\s*(?:l[ao]s?|el)?\s*"
        r"(?:facultad(?:es)?|unidad(?:es)?|oficina(?:s)?|departamento|dependencia|direccion|carrera|centro(?:\s+experimental)?|sede)?\s*"
        r"(?:de\s+(?:l[ao]s?|el)\s+|de\s+|del\s+)?$",
        prefijo,
    )
    ini_real = m_pre.start() if m_pre and m_pre.group(0).strip() else inicio

    sufijo = norm[fin:]
    m_suf = re.match(
        r"^(?:\s*(?:,|;)?\s*(?:o\s+(?:mas\s+)?conocid[oa]\s+como|tambien\s+(?:conocid[oa]|llamad[oa])\s+como|"
        r"es\s+decir|alias|siglas?|o\s+(?:la|el)\s+facultad(?:\s+de)?)\s+"
        r"(?:(?:la|el|las|los)\s+)?[a-z0-9\.\-]{2,25}(?:\s+[a-z0-9\.\-]{2,20}){0,4})",
        sufijo,
    )
    fin_real = fin + (m_suf.end() if m_suf else 0)
    limpio = f"{norm[:ini_real]} {norm[fin_real:]}"
    return " ".join(limpio.split())


def _resolver_ubicacion_con_fragmento(norm: str) -> tuple[str | None, str]:
    if any(
        p in norm
        for p in ("todas las unidades", "todas las oficinas", "todo el padron", "toda la universidad")
    ):
        return None, norm

    for clave, valor_canon in _ALIAS_UBICACIONES:
        m_alias = re.search(rf"\b{re.escape(clave)}\b", norm)
        if m_alias:
            norm_restante = _suprimir_clausula_ubicacion(norm, m_alias.start(), m_alias.end())
            return valor_canon, norm_restante

    m_dyn = re.search(
        r"(?:en\s+(?:la|el)\s+|de\s+(?:la|el)\s+)?(?:facultad|unidad|oficina|departamento|direccion|carrera|centro|sede)\s+(?:de\s+(?:la|el)\s+|de\s+|del\s+)?([a-z0-9\s\.\-]{3,55})",
        norm,
    )
    if m_dyn:
        frag = re.split(_STOP_UBICACION_DINAMICA, m_dyn.group(1))[0].strip(" .,;:-")
        cercana = _mejor_unidad_cercana(frag)
        norm_restante = _suprimir_clausula_ubicacion(norm, m_dyn.start(1), m_dyn.start(1) + len(frag))
        if cercana:
            return cercana, norm_restante
        palabras_frag = [p for p in frag.split() if p not in {"la", "el", "de", "del", "los", "las"}]
        if frag and frag not in _UNIDADES_GENERICAS_IGNORADAS and 3 <= len(frag) <= 32 and len(palabras_frag) <= 3:
            return (frag.title() if len(frag) > 5 else frag.upper()), norm_restante
    return None, norm


def _resolver_ubicacion_consulta(consulta_usuario: str, norm: str) -> str | None:
    ubicacion, _ = _resolver_ubicacion_con_fragmento(norm)
    return ubicacion


def filtrar_catalogo_reportes(consulta_usuario: str) -> dict[str, Any]:
    """Traduce criterios complejos en lenguaje natural hacia filtros exactos del padrón PostgreSQL UAGRM."""
    consulta_corregida = corregir_texto_institucional(consulta_usuario)
    norm = normalizar_texto(consulta_corregida)
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

    ubicacion, norm_sin_ubicacion = _resolver_ubicacion_con_fragmento(norm)

    estado: str | None = None
    if any(p in norm for p in ("en reparacion", "reparacion", "mantenimiento", "por reparar", "en taller")):
        estado = "EN_REPARACION"
    elif re.search(r"\b(?:de\s+baja|dados?\s+de\s+baja|baja\s+patrimonial|desincorporad\w*|\bbajas?\b)", norm):
        estado = "BAJA"
    elif any(
        p in norm
        for p in ("mal estado", "estado malo", "danad", "averiad", "inutilizad", "inservible", "defectuos")
    ) or re.search(r"\bmalos?\b|\bmalas?\b", norm):
        estado = "MALO"
    elif "estado regular" in norm or re.search(r"\bregular(?:es)?\b", norm):
        estado = "REGULAR"
    elif any(p in norm for p in ("buen estado", "estado bueno", "en buen uso", "optimo", "optimas")) or re.search(
        r"\bbuenos?\b|\bbuenas?\b", norm
    ):
        estado = "BUENO"
    elif "asignad" in norm and "no asignad" not in norm:
        estado = "ASIGNADO"

    condicion: str | None = None
    if "deteriorad" in norm or "desgastad" in norm:
        condicion = "DETERIORADO"
    elif re.search(r"\busad[oa]s?\b|\bsegunda mano\b", norm):
        condicion = "USADO"
    elif (
        any(
            p in norm
            for p in ("condicion nuevo", "condicion nueva", "en nuevo estado", "activos nuevos", "bienes nuevos", "equipos nuevos")
        )
        or re.search(r"\bnuev[oa]s?\b", norm)
    ) and "nueva sesion" not in norm and "nueva consulta" not in norm and "nuevo reporte" not in norm:
        condicion = "NUEVO"

    grupo: str | None = None
    if any(
        p in norm_sin_ubicacion
        for p in ("comput", "laptop", "portatil", "pc ", "servidor", "impresora", "monitor", "comunicacion", "telecom", "red ")
    ):
        grupo = "Equipos de Computación"
    elif any(p in norm_sin_ubicacion for p in ("muebl", "enseres", "escritorio", "silla", "estante", "mesa", "archivador", "armario", "butaca")):
        grupo = "Muebles y Enseres"
    elif any(p in norm_sin_ubicacion for p in ("vehic", "automotor", "camioneta", "vagoneta", "bus ", "automovil", "camion")):
        grupo = "Vehículos"
    elif any(p in norm_sin_ubicacion for p in ("maquinaria", "laboratorio", "microscopio", "equipo medico", "generador", "centrifuga", "compresor")):
        grupo = "Maquinaria y Equipo"
    elif any(p in norm_sin_ubicacion for p in ("herramienta", "equipo tecnico", "taladro", "soldador")):
        grupo = "Herramientas"
    elif any(p in norm_sin_ubicacion for p in ("edificacion", "edificio", "inmueble", "infraestructura")):
        grupo = "Edificaciones"
    elif any(p in norm_sin_ubicacion for p in ("terreno", "lote", "predio")):
        grupo = "Terrenos"
    elif any(p in norm_sin_ubicacion for p in ("semoviente", "pedigree", "ganado", "reproductor")):
        grupo = "Semovientes"
    elif "animal" in norm_sin_ubicacion and "trabajo" in norm_sin_ubicacion:
        grupo = "Animales de Trabajo"
    elif any(p in norm_sin_ubicacion for p in ("patrimonio historico", "historico", "artistico", "cultural")):
        grupo = "Patrimonio Histórico"

    valor_min: float | None = None
    valor_max: float | None = None
    m_rango = re.search(
        r"entre\s*(?:bs\.?\s*)?(\d+(?:[.,]\d+)?)\s*(?:bs\.?)?\s*(?:y|a|hasta)\s*(?:bs\.?\s*)?(\d+(?:[.,]\d+)?)",
        norm,
    )
    if m_rango:
        v1 = _parse_monto_bo(m_rango.group(1))
        v2 = _parse_monto_bo(m_rango.group(2))
        if v1 is not None and v2 is not None:
            valor_min = min(v1, v2)
            valor_max = max(v1, v2)
    else:
        for pat_min in (
            r"(?:mayor(?:es)?\s*(?:a|que|de)|superior(?:es)?\s*(?:a|que)|mas\s+de|a\s+partir\s+de|>=?|>)\s*(?:bs\.?\s*)?(\d+(?:[.,]\d+)?)",
            r"desde\s+bs\.?\s*(\d+(?:[.,]\d+)?)",
        ):
            m_min = re.search(pat_min, norm)
            if m_min:
                valor_min = _parse_monto_bo(m_min.group(1))
                break
        m_max = re.search(
            r"(?:menor(?:es)?\s*(?:a|que|de)|inferior(?:es)?\s*(?:a|que)|menos\s+de|hasta|como\s+maximo|<=?|<)\s*(?:bs\.?\s*)?(\d+(?:[.,]\d+)?)",
            norm,
        )
        if m_max:
            valor_max = _parse_monto_bo(m_max.group(1))

    gestion: int | None = None
    m_anio = re.search(r"\b(199\d|20\d{2})\b", norm)
    if m_anio:
        gestion = int(m_anio.group(1))

    search = _resolver_search_consulta(consulta_usuario, norm_sin_ubicacion)

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
        "en reparacion",
        "reparacion",
        "nuevo",
        "nuevos",
        "usado",
        "usados",
        "deteriorado",
        "deteriorados",
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
        w in norm for w in ("bueno", "malo", "regular", "reparacion", "deterior", "estadistic", "padron", "general")
    ):
        return None
    return s


def extraer_destinatario_instruccion(instruccion_usuario: str) -> tuple[str | None, str | None]:
    """Extrae destinatario y cargo implícitos si el usuario los indicó en lenguaje natural."""
    texto = " ".join(instruccion_usuario.strip().split())
    if not texto:
        return None, None

    patron_con_nombre = re.search(
        r"(?:dirigid[oa]\s+a(?:l)?|destinatari[oa]\s*:?\s*|para\s+(?:el|la)?)\s+"
        r"((?:Ing\.?|Lic\.?|Dr\.?|Dra\.?|M\.?Sc\.?|Arq\.?|Abg\.?|Sr\.?|Sra\.?)\s+"
        r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})"
        r"(?:\s*(?:,|\(|-)\s*([A-Za-zÁÉÍÓÚáéíóúÑñ0-9\s\.\/-]{4,70})(?:\)|$|[.;]))?",
        texto,
    )
    if patron_con_nombre:
        nombre = patron_con_nombre.group(1).strip(" ,.;:-")
        cargo = patron_con_nombre.group(2).strip(" ,.;:-()") if patron_con_nombre.group(2) else None
        return nombre, cargo

    norm = normalizar_texto(texto)
    cargos_directos = (
        ("rector", ("Sr. Rector de la UAGRM", "Rectorado Universitario — UAGRM")),
        ("vicerrector", ("Sr. Vicerrector de la UAGRM", "Vicerrectorado — UAGRM")),
        ("director administrativo", ("Director Administrativo y Financiero (DUAF)", "DUAF — UAGRM")),
        ("daf", ("Director Administrativo y Financiero (DUAF)", "DUAF — UAGRM")),
        ("duaf", ("Director Administrativo y Financiero (DUAF)", "DUAF — UAGRM")),
        ("jefe de activos", ("Jefatura del Departamento de Activos Fijos", "Departamento de Activos Fijos — DUAF")),
        ("decan", ("Sr. Decano de Facultad", "Decanato Facultativo — UAGRM")),
        ("auditoria", ("Jefatura de Auditoría Interna", "Dirección de Auditoría Interna — UAGRM")),
    )
    if any(cue in norm for cue in ("dirigid", "destinatari", "para el", "para la", "al sr", "a la ")):
        for clave, (nombre_c, cargo_c) in cargos_directos:
            if clave in norm:
                return nombre_c, cargo_c
    return None, None


def detectar_exportacion_carta(instruccion_usuario: str) -> list[str]:
    """Detecta si el usuario pidió explícitamente exportar o descargar la carta en Word o PDF."""
    norm = normalizar_texto(instruccion_usuario)
    if any(neg in norm for neg in ("no export", "sin export", "no descarg", "no imprim", "aun no")):
        return []
    verbo_accion = any(
        v in norm
        for v in ("export", "descarg", "genera el ", "generar en", "en formato", "imprim", "sacar en", "listo para")
    )
    formatos: list[str] = []
    if re.search(r"\b(?:word|docx)\b", norm) and (verbo_accion or "en word" in norm or "a word" in norm):
        formatos.append("word")
    if re.search(r"\bpdf\b", norm) and (verbo_accion or "en pdf" in norm or "a pdf" in norm):
        formatos.append("pdf")
    return formatos


def detectar_edicion_parcial_carta(instruccion_usuario: str) -> dict[str, Any]:
    """Identifica si una corrección en sesión temporal apunta a un párrafo o sección específica."""
    norm = normalizar_texto(instruccion_usuario)
    ordinales = {
        "primer": 1,
        "1": 1,
        "primero": 1,
        "segundo": 2,
        "2": 2,
        "tercer": 3,
        "3": 3,
        "tercero": 3,
        "cuarto": 4,
        "4": 4,
        "quinto": 5,
        "5": 5,
        "ultimo": -1,
    }
    indice_parrafo: int | None = None
    m_p = re.search(
        r"parrafo\s*(?:n[°º]?\s*)?(1|2|3|4|5|primer[oa]?|segund[oa]?|tercer[oa]?|cuart[oa]?|quint[oa]?|ultim[oa]?)"
        r"|(primer[oa]?|segund[oa]?|tercer[oa]?|cuart[oa]?|quint[oa]?|ultim[oa]?)\s+parrafo",
        norm,
    )
    if m_p:
        token = (m_p.group(1) or m_p.group(2) or "").rstrip("oa")
        for clave, idx in ordinales.items():
            if token == clave or token.startswith(clave.rstrip("oa")):
                indice_parrafo = idx
                break

    solo_destinatario = any(p in norm for p in ("cambia el destinatario", "cambiar destinatario", "dirigida a ", "dirigido a "))
    solo_referencia = any(p in norm for p in ("cambia la referencia", "cambiar referencia", "ajusta el asunto", "cambia el titulo"))
    return {
        "indice_parrafo_1based": indice_parrafo,
        "toca_destinatario": solo_destinatario,
        "toca_referencia": solo_referencia,
    }


def detectar_formatos_exportacion_reporte(consulta_usuario: str) -> list[str]:
    """Detecta si el usuario pidió exportar el reporte inmediatamente en PDF, Excel y/o Word."""
    norm = normalizar_texto(consulta_usuario)
    if any(
        neg in norm
        for neg in ("no export", "sin export", "no descarg", "todavia no", "aun no", "primero muestrame")
    ):
        return []
    if any(
        p in norm
        for p in ("todos los formatos", "los tres formatos", "los 3 formatos", "en pdf, excel y word", "pdf excel y word")
    ):
        return ["pdf", "excel", "word"]

    quiere_exportar = any(
        v in norm
        for v in (
            "export",
            "descarg",
            "genera en",
            "generar en",
            "sacar en",
            "imprim",
            "formato pdf",
            "formato word",
            "formato excel",
            "en pdf",
            "en excel",
            "en word",
        )
    )
    if not quiere_exportar:
        return []

    formatos: list[str] = []
    if re.search(r"\bpdf\b", norm) or "imprimir" in norm:
        formatos.append("pdf")
    if re.search(r"\b(?:excel|xlsx)\b", norm):
        formatos.append("excel")
    if re.search(r"\b(?:word|docx)\b", norm):
        formatos.append("word")
    return formatos


def detectar_incluir_graficas(consulta_usuario: str) -> bool:
    """Devuelve True solo si el usuario solicita explícitamente gráficas o gráficos en el reporte."""
    norm = normalizar_texto(consulta_usuario)
    if any(neg in norm for neg in ("sin grafic", "no incluir grafic", "no agregues grafic")):
        return False
    return bool(re.search(r"\b(?:grafic[oa]s?|diagramas?|grafico de barras|grafico circular)\b", norm))


def extraer_unidad_solicitante(instruccion_usuario: str) -> str | None:
    """Extrae la facultad, carrera o unidad universitaria solicitante desde lenguaje natural."""
    corregida = corregir_texto_institucional(instruccion_usuario)
    facultad_oficial = resolver_facultad_oficial_uagrm(corregida)
    if facultad_oficial:
        return facultad_oficial

    texto = " ".join(corregida.strip().split())
    if not texto:
        return None
    m = re.search(
        r"(?:de\s+la|de\s+el|del|para\s+la|para\s+el|en\s+la|en\s+el)\s+"
        r"((?:Facultad|Carrera|Direcci[oó]n|Unidad|Departamento|Centro|Instituto|Decanato|Vicerrectorado|Rectorado|Biblioteca)"
        r"(?:\s+(?:de|del|de\s+la|de\s+las|de\s+los|y|e))?(?:\s+[A-ZÁÉÍÓÚÑa-záéíóúñ\.]{2,24}){1,6})",
        texto,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    crudo = re.split(
        r"\b(?:solicito|solicitamos|requiero|requerimos|necesito|necesitamos|pido|pedimos|para\s+nuestr|con\s+el\s+fin|por\s+lo|,\s*solicito)\b",
        m.group(1),
        flags=re.IGNORECASE,
    )[0].strip(" .,;:-")
    if len(crudo) < 6:
        return None
    palabras_min = {"de", "del", "la", "las", "los", "y", "e"}
    partes = crudo.split()
    normalizadas = [
        p.lower() if idx > 0 and p.lower() in palabras_min else p[0].upper() + p[1:]
        for idx, p in enumerate(partes)
    ]
    return " ".join(normalizadas)


def clasificar_intencion_redaccion(instruccion_usuario: str) -> dict[str, Any]:
    """Clasifica el tipo de trámite UAGRM (baja, revalúo, corrección, etc.) desde texto libre."""
    q = normalizar_texto(instruccion_usuario)
    es_correccion_explicita = any(
        p in q
        for p in (
            "datos invalidos",
            "dato invalido",
            "tiene error",
            "tienen error",
            "error en la solicitud",
            "datos errados",
            "corregir los datos",
            "corregir datos",
            "rectificar datos",
        )
    ) and any(p in q for p in ("corregir", "correccion", "datos"))

    if re.search(
        r"\b(?:compras?|comprar|adquisicion(?:es)?|adquirir|dotacion(?:es)?|dotar|"
        r"equipamiento(?:\s+nuevo)?|nuevos?\s+activos?|nuevos?\s+equipos?|provision\s+de)\b",
        q,
    ) or "nuevo activo" in q:
        tipo = "SOLICITUD_NUEVO_ACTIVO"
    elif any(p in q for p in ("descargo", "incidente", "extravio", "robo", "hurto", "siniestro", "sustraccion", "perdida")):
        tipo = "DESCARGO_INCIDENTE"
    elif es_correccion_explicita:
        tipo = "SOLICITUD_CORRECCION_DATOS"
    elif any(
        p in q
        for p in (
            "dar de baja",
            "solicitud de baja",
            "baja definitiva",
            "baja patrimonial",
            "desincorpor",
            "inservible",
            "inutilizad",
            "obsolet",
            "quemad",
        )
    ) or re.search(r"\bbaja\b", q):
        tipo = "SOLICITUD_BAJA"
    elif any(p in q for p in ("revalu", "revaloriz", "valor residual", "vida util", "depreciado a 1")):
        tipo = "SOLICITUD_REVALUO"
    elif any(p in q for p in ("reasigna", "transfer", "traslad", "custodio", "cambio de responsable")):
        tipo = "SOLICITUD_REASIGNACION"
    elif any(p in q for p in ("inventario", "verificacion fisica", "conciliacion", "levantamiento fisico")):
        tipo = "SOLICITUD_INVENTARIO"
    elif any(p in q for p in ("informe tecnico", "dictamen tecnico", "peritaje", "evaluacion tecnica")):
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
    search: str | None = None,
    estado: str | None = None,
    condicion: str | None = None,
    grupo_contable: str | None = None,
    valor_min: float | None = None,
    valor_max: float | None = None,
    gestion: int | None = None,
) -> dict[str, Any]:
    """Obtiene conteos exactos y sumatoria total en Bs. del padrón PostgreSQL UAGRM filtrados por criterios."""
    cliente, seg = _obtener_cliente_y_seguridad()
    return await cliente.post_internal(
        "estadisticas-padron",
        {
            "unidad": _normalizar_unidad_stats(unidad),
            "search": _normalizar_param_opcional(search),
            "estado": _normalizar_param_opcional(estado),
            "condicion": _normalizar_param_opcional(condicion),
            "grupoContable": _normalizar_param_opcional(grupo_contable),
            "valorMin": valor_min,
            "valorMax": valor_max,
            "gestion": gestion,
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
    """Plantilla estructurada de análisis semántico para redactar cartas formales extensas UAGRM."""
    return (
        "Analiza la solicitud administrativa universitaria UAGRM (Bolivia) y genera los bloques "
        "semánticos de alto nivel formal para redactar una carta oficial seria, extensa y precisa.\n\n"
        "ANTECEDENTES OFICIALES:\n"
        f"{recurso_mcp_json}\n\n"
        "SOLICITUD DEL USUARIO:\n"
        f"{instruccion_usuario}\n\n"
        "REGLAS ESTRICTAS:\n"
        "1. Corrige ortografía y errores de dictado (ej. 'famultad' -> 'Facultad', 'conturia' -> 'Contaduría').\n"
        "2. Usa EXCLUSIVAMENTE la unidad o facultad mencionada por el usuario; JAMÁS menciones ni inventes otra facultad distinta.\n"
        "3. PROHIBIDO inventar precios o montos en Bs., códigos patrimoniales ni estados físicos que no estén en los antecedentes.\n"
        "4. PROHIBIDO usar palabras técnicas internas ('MCP', 'JSON', 'true', 'false', 'recurso activo') ni encabezados ('Señor...').\n"
        "5. Redacta cada campo en prosa institucional boliviana formal, específica al pedido real del usuario:\n"
        '   - "asunto_especifico": título oficial en MAYÚSCULAS resumiendo el trámite y bien solicitado.\n'
        '   - "unidad_solicitante": nombre correcto y oficial de la facultad o unidad solicitante.\n'
        '   - "objeto_concreto": frase sustantiva formal que describe exactamente qué se solicita (ej. "la adquisición e incorporación de cuatro (4) televisores destinados al equipamiento audiovisual de las aulas y laboratorios").\n'
        '   - "justificacion_tecnica": 2 a 3 oraciones sólidas y coherentes explicando la necesidad académica, técnica u operativa concreta de este caso específico.\n'
        '   - "impacto_institucional": 1 a 2 oraciones formales sobre el beneficio esperado para docencia, investigación, control patrimonial o continuidad operativa.\n'
        '   - "detalle_complementario": 1 oración formal con precisiones específicas mencionadas por el usuario (fallas, destino, marca, plazos), o "" si no aplica.\n\n'
        "Devuelve EXCLUSIVAMENTE un objeto JSON válido con esas 6 claves exactas."
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
        "Regla: usa exactamente los estados y condiciones del padrón PostgreSQL UAGRM "
        "(estados: BUENO, REGULAR, MALO, EN_REPARACION, BAJA; condiciones: NUEVO, USADO, DETERIORADO) "
        "y no inventes valores que no estén en la consulta o en el resumen.\n"
        "Devuelve SOLO JSON con estas claves:\n"
        '{"filtros": {"search": str|null, "estado": "BUENO"|"REGULAR"|"MALO"|"EN_REPARACION"|"BAJA"|null, '
        '"condicion": "NUEVO"|"USADO"|"DETERIORADO"|null, "grupo_contable": str|null, '
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
