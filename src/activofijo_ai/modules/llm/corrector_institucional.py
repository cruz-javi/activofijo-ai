"""Corrector léxico-fonético sistémico y catálogo oficial de unidades UAGRM."""

from __future__ import annotations

import difflib
import re
import unicodedata


def _sin_tildes(texto: str) -> str:
    nfd = unicodedata.normalize("NFD", texto)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn").lower()


_LEXICO_INSTITUCIONAL: dict[str, str] = {
    _sin_tildes(w): w
    for w in (
        "facultad",
        "facultades",
        "contaduría",
        "contables",
        "auditoría",
        "computación",
        "telecomunicaciones",
        "informática",
        "ingeniería",
        "veterinaria",
        "veterinarias",
        "tecnología",
        "exactas",
        "economía",
        "económicas",
        "empresariales",
        "jurídicas",
        "políticas",
        "humanidades",
        "agronomía",
        "agrícolas",
        "farmacéuticas",
        "bioquímica",
        "politécnica",
        "arquitectura",
        "medicina",
        "administrativa",
        "administrativo",
        "financiera",
        "financiero",
        "universitaria",
        "universitario",
        "universidad",
        "biblioteca",
        "rectorado",
        "vicerrectorado",
        "decanato",
        "postgrado",
        "departamento",
        "dependencia",
        "dirección",
        "carrera",
        "laboratorio",
        "laboratorios",
        "auditorio",
        "televisor",
        "televisores",
        "proyector",
        "proyectores",
        "computadora",
        "computadoras",
        "portátil",
        "portátiles",
        "impresora",
        "impresoras",
        "fotocopiadora",
        "microscopio",
        "servidor",
        "servidores",
        "escritorio",
        "escritorios",
        "archivador",
        "mobiliario",
        "equipamiento",
        "acondicionado",
        "inventario",
        "inventariación",
        "patrimonial",
        "patrimonio",
        "adquisición",
        "dotación",
        "incorporación",
        "revalorización",
        "revalúo",
        "reasignación",
        "transferencia",
        "desincorporación",
        "custodia",
        "custodio",
        "verificación",
        "conciliación",
        "levantamiento",
        "mantenimiento",
        "reparación",
        "deteriorado",
        "deteriorados",
        "inservible",
        "inservibles",
        "obsoleto",
        "obsoletos",
        "obsolescencia",
        "presupuesto",
        "presupuestaria",
        "bolivianos",
        "representante",
        "representación",
        "solicitud",
        "solicitante",
        "institucional",
        "especificaciones",
        "cotizaciones",
        "evaluación",
        "dictamen",
        "informe",
        "reporte",
        "trazabilidad",
        "vehículos",
        "herramientas",
        "maquinaria",
        "semovientes",
        "edificaciones",
        "inmuebles",
        "terrenos",
        "corrección",
        "actualización",
        "descargo",
        "incidente",
    )
}

_PALABRAS_COMUNES_PROTEGIDAS: frozenset[str] = frozenset(
    {
        "solicito",
        "solicitamos",
        "solicitar",
        "requiero",
        "requerimos",
        "necesito",
        "necesitamos",
        "queremos",
        "nuestra",
        "nuestras",
        "nuestro",
        "nuestros",
        "factura",
        "facturas",
        "contrato",
        "contratos",
        "contar",
        "continua",
        "continuo",
        "cultura",
        "cintura",
        "compras",
        "comprar",
        "cuatro",
        "cientos",
        "siempre",
        "tambien",
        "entonces",
        "durante",
        "mediante",
        "presente",
        "anterior",
        "posterior",
        "superior",
        "inferior",
        "general",
        "especial",
        "oficial",
        "tecnico",
        "tecnica",
        "tecnicos",
        "tecnicas",
        "fisico",
        "fisica",
        "fisicos",
        "fisicas",
        "sistema",
        "sistemas",
        "control",
        "gestion",
        "encuentra",
        "encuentran",
        "conocido",
        "conocida",
        "ubicado",
        "ubicados",
        "asignado",
        "asignados",
        "registro",
        "registros",
        "activos",
        "equipos",
        "muebles",
        "enseres",
        "reportes",
        "informes",
        "formato",
        "formatos",
        "exportar",
        "descargar",
        "imprimir",
        "graficas",
        "graficos",
        "unidades",
        "oficinas",
        "carreras",
        "ciencias",
        "publica",
        "publico",
        "sociales",
        "humana",
        "central",
        "valido",
        "validos",
        "invalido",
        "invalidos",
        "corregir",
        "cambiar",
        "agregar",
        "eliminar",
        "modificar",
        "parrafo",
        "parrafos",
        "dirigida",
        "dirigido",
        "urgente",
        "urgencia",
        "docencia",
        "docentes",
        "estudiantes",
        "estudiantil",
        "academico",
        "academica",
        "academicos",
        "academicas",
        "operativo",
        "operativa",
        "operativos",
        "operativas",
    }
)

_FACULTADES_OFICIALES_UAGRM: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("facultad de ciencias contables", "ciencias contables"),
        "Facultad de Ciencias Contables",
    ),
    (
        ("facultad de contaduria", "contaduria publica", "contaduria", "auditoria financiera"),
        "Facultad de Contaduría Pública",
    ),
    (
        (
            "facultad de ciencias de la computacion",
            "facultad de cs de la computacion",
            "facultad de computacion",
            "facultad de ingenieria informatica",
            "facultad de ingenieria en sistemas",
            "facultad de ingenieria de sistemas",
            "ciencias de la computacion",
            "ingenieria informatica",
            "ingenieria en sistemas",
            "ingenieria de sistemas",
            "ficct",
            "la fic",
        ),
        "Facultad de Ciencias de la Computación y Telecomunicaciones (FICCT)",
    ),
    (
        (
            "facultad de ciencias exactas",
            "ciencias exactas",
            "facultad de tecnologia",
            "fcet",
        ),
        "Facultad de Ciencias Exactas y Tecnología",
    ),
    (
        (
            "facultad de ciencias veterinarias",
            "ciencias veterinarias",
            "veterinarias",
            "veterinaria",
            "fcv",
        ),
        "Facultad de Ciencias Veterinarias",
    ),
    (
        (
            "facultad de ciencias de la salud",
            "ciencias de la salud",
            "salud humana",
            "facultad de medicina",
            "medicina",
        ),
        "Facultad de Ciencias de la Salud Humana",
    ),
    (
        (
            "facultad de ciencias economicas",
            "ciencias economicas",
            "economia y empresariales",
            "facultad de economia",
        ),
        "Facultad de Ciencias Económicas y Empresariales",
    ),
    (
        (
            "facultad de ciencias juridicas",
            "ciencias juridicas",
            "facultad de derecho",
        ),
        "Facultad de Ciencias Jurídicas, Políticas, Sociales y Relaciones Internacionales",
    ),
    (
        ("facultad de humanidades", "humanidades"),
        "Facultad de Humanidades",
    ),
    (
        ("facultad de ciencias agricolas", "ciencias agricolas", "agronomia"),
        "Facultad de Ciencias Agrícolas",
    ),
    (
        ("facultad de ciencias del habitat", "ciencias del habitat", "arquitectura"),
        "Facultad de Ciencias del Hábitat, Diseño y Arte",
    ),
    (
        ("facultad de ciencias farmaceuticas", "bioquimica y farmacia", "farmacia"),
        "Facultad de Ciencias Farmacéuticas y Bioquímicas",
    ),
    (
        ("facultad politecnica", "politecnica"),
        "Facultad Politécnica",
    ),
    (
        ("facultad integral del norte", "finor", "montero"),
        "Facultad Integral del Norte (FINOR - Montero)",
    ),
    (
        ("facultad integral del chaco", "fich", "camiri"),
        "Facultad Integral del Chaco (FICH - Camiri)",
    ),
    (
        ("facultad integral de los valles", "valles crucenos", "fivc"),
        "Facultad Integral de los Valles Cruceños (FIVC)",
    ),
    (
        ("direccion de tecnologias de la informacion", "dtic", "cpd"),
        "Dirección de Tecnologías de la Información y Comunicación (DTIC / CPD)",
    ),
    (
        ("departamento de activo fijo", "activo fijo y control patrimonial"),
        "Departamento de Activo Fijo y Control Patrimonial",
    ),
    (
        ("centro experimental el prado", "el prado"),
        "Centro Experimental El Prado (F.C.V.)",
    ),
    (
        ("centro experimental yabare", "yabare"),
        "Centro Experimental Yabaré (F.C.V.)",
    ),
    (
        ("biblioteca central", "biblioteca universitaria"),
        "Biblioteca Central Universitaria",
    ),
    (
        ("direccion administrativa y financiera", "duaf", "daf"),
        "Dirección Universitaria de Administración y Finanzas (DUAF)",
    ),
    (
        ("rectorado", "vicerrectorado"),
        "Rectorado Universitario",
    ),
)

_CLAVES_LEXICO = tuple(_LEXICO_INSTITUCIONAL.keys())


def _preservar_capitalizacion(original: str, reemplazo: str) -> str:
    if original.isupper():
        return reemplazo.upper()
    if original[0].isupper():
        return reemplazo[0].upper() + reemplazo[1:]
    return reemplazo


def corregir_texto_institucional(texto: str) -> str:
    """
    Corrige sistemáticamente distorsiones fonéticas de Whisper y errores de tipeo frecuentes
    sobre el léxico institucional UAGRM sin alterar códigos de activo, marcas ni cifras.
    """
    if not texto or not texto.strip():
        return texto

    salida = re.sub(
        r"\b(como)\s+representan\b",
        r"\1 representante",
        texto,
        flags=re.IGNORECASE,
    )
    salida = re.sub(
        r"\bnecesit[uú][mn]\b",
        "Necesito un",
        salida,
        flags=re.IGNORECASE,
    )
    salida = re.sub(
        r"\brepote(s)?\b",
        r"reporte\1",
        salida,
        flags=re.IGNORECASE,
    )

    def _corregir_token(match: re.Match[str]) -> str:
        original = match.group(0)
        if len(original) < 6 or any(ch.isdigit() for ch in original):
            return original
        norm = _sin_tildes(original)
        if norm in _LEXICO_INSTITUCIONAL or norm in _PALABRAS_COMUNES_PROTEGIDAS:
            return original

        mejor_clave: str | None = None
        mejor_score = 0.0
        for clave in _CLAVES_LEXICO:
            if clave[0] != norm[0]:
                continue
            if abs(len(clave) - len(norm)) > 2:
                continue
            score = difflib.SequenceMatcher(None, norm, clave).ratio()
            if score > mejor_score:
                mejor_score = score
                mejor_clave = clave

        if mejor_clave and mejor_score >= 0.80:
            return _preservar_capitalizacion(original, _LEXICO_INSTITUCIONAL[mejor_clave])
        return original

    salida = re.sub(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{6,}", _corregir_token, salida)
    return salida


def resolver_facultad_oficial_uagrm(texto: str) -> str | None:
    """Identifica el nombre oficial completo de la facultad o unidad UAGRM citada en el texto."""
    corregido = corregir_texto_institucional(texto)
    norm = _sin_tildes(corregido)
    for alias_tuple, nombre_oficial in _FACULTADES_OFICIALES_UAGRM:
        for alias in alias_tuple:
            if re.search(rf"\b{re.escape(alias)}\b", norm):
                return nombre_oficial
    return None


def listar_facultades_mencionadas(texto: str) -> set[str]:
    """Devuelve el conjunto de facultades oficiales mencionadas explícitamente en un texto."""
    corregido = corregir_texto_institucional(texto)
    norm = _sin_tildes(corregido)
    encontradas: set[str] = set()
    for alias_tuple, nombre_oficial in _FACULTADES_OFICIALES_UAGRM:
        if any(re.search(rf"\b{re.escape(alias)}\b", norm) for alias in alias_tuple):
            encontradas.add(nombre_oficial)
    return encontradas
