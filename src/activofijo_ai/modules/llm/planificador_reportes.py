"""Planificador experto multi-fuente y selector dinámico de columnas para Reportes IA UAGRM."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from activofijo_ai.modules.llm.corrector_institucional import (
    corregir_texto_institucional,
    resolver_facultad_oficial_uagrm,
)
from activofijo_ai.modules.llm.mcp_server import (
    _CATALOGO_BIENES_SEARCH,
    _PALABRAS_NO_SEARCH,
    CONDICIONES_PADRON_VALIDAS,
    ESTADOS_PADRON_VALIDOS,
    normalizar_texto,
)

FUENTES_REPORTE_VALIDAS: frozenset[str] = frozenset(
    {
        "PADRON",
        "ALTAS",
        "BAJAS",
        "ASIGNACIONES",
        "TRANSFERENCIAS",
        "REVALUOS",
        "LEVANTAMIENTOS",
    }
)

COLUMNAS_POR_FUENTE: dict[str, tuple[str, ...]] = {
    "PADRON": (
        "codigo",
        "descripcion",
        "grupoContable",
        "ubicacion",
        "custodio",
        "estado",
        "condicion",
        "marca",
        "modelo",
        "nroSerie",
        "fechaAdquisicion",
        "gestion",
        "valorBs",
        "depAcumuladaBs",
        "valorNetoBs",
        "placa",
    ),
    "ALTAS": (
        "codigo",
        "descripcion",
        "fechaIngreso",
        "nroFactura",
        "proveedor",
        "grupoContable",
        "ubicacion",
        "custodio",
        "estado",
        "valorBs",
    ),
    "BAJAS": (
        "nroBaja",
        "codigo",
        "descripcion",
        "fechaBaja",
        "documento",
        "motivo",
        "ubicacion",
        "grupoContable",
        "autorizador",
        "valorBs",
        "valorFinalBs",
    ),
    "ASIGNACIONES": (
        "nroActa",
        "fechaAsignacion",
        "tipoAsignacion",
        "codigo",
        "descripcion",
        "ubicacion",
        "custodio",
        "documentoRespaldo",
        "estado",
        "valorBs",
    ),
    "TRANSFERENCIAS": (
        "nroTransferencia",
        "fechaTransferencia",
        "codigo",
        "descripcion",
        "oficinaOrigen",
        "oficinaDestino",
        "documentoRespaldo",
        "motivo",
        "estado",
        "valorBs",
    ),
    "REVALUOS": (
        "nroRevaluo",
        "fechaRevaluo",
        "codigo",
        "descripcion",
        "ubicacion",
        "disposicionRespaldo",
        "motivo",
        "tecnico",
        "valorAnteriorBs",
        "nuevoValorBs",
        "nuevaVidaUtilMeses",
    ),
    "LEVANTAMIENTOS": (
        "nroPlanilla",
        "fechaLevantamiento",
        "ubicacion",
        "codigo",
        "descripcion",
        "tipoHallazgo",
        "estadoRevision",
        "inspector",
        "observacion",
    ),
}

_ALIAS_COLUMNAS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("nro de acta", "numero de acta", "acta de asignacion", "nro acta"), ("nroActa",)),
    (("nro de baja", "acta de baja", "numero de baja", "nro baja"), ("nroBaja",)),
    (("nro de transferencia", "acta de transferencia", "numero de traslado", "nro transferencia"), ("nroTransferencia",)),
    (("nro de revaluo", "numero de revaluo", "nro revaluo"), ("nroRevaluo",)),
    (("nro de planilla", "nro planilla", "planilla"), ("nroPlanilla",)),
    (("codigo", "codigos", "codigo de activo", "codigo patrimonial"), ("codigo",)),
    (("descripcion", "nombre del bien", "nombre", "detalle del bien"), ("descripcion",)),
    (("grupo contable", "grupo", "categoria contable", "categoria"), ("grupoContable",)),
    (("oficina origen", "unidad origen", "origen", "remitente"), ("oficinaOrigen",)),
    (("oficina destino", "unidad destino", "destino"), ("oficinaDestino",)),
    (("ubicacion", "unidad", "dependencia", "facultad", "oficina"), ("ubicacion",)),
    (("custodio", "responsable", "encargado", "funcionario"), ("custodio",)),
    (("autorizador", "autorizado por"), ("autorizador",)),
    (("perito", "tecnico evaluador", "tecnico"), ("tecnico",)),
    (("inspector",), ("inspector",)),
    (("estado de revision", "revision"), ("estadoRevision",)),
    (("hallazgo", "tipo de hallazgo"), ("tipoHallazgo",)),
    (("estado", "estado fisico", "estado operativo"), ("estado",)),
    (("condicion", "condicion fisica"), ("condicion",)),
    (("marca",), ("marca",)),
    (("modelo",), ("modelo",)),
    (("numero de serie", "nro de serie", "nro serie", "serie"), ("nroSerie",)),
    (("placa", "placa vehicular"), ("placa",)),
    (("factura", "nro factura", "numero de factura"), ("nroFactura",)),
    (("proveedor", "empresa proveedora"), ("proveedor",)),
    (("documento de respaldo", "documento", "resolucion", "respaldo"), ("documento", "documentoRespaldo", "disposicionRespaldo")),
    (("informe pericial", "disposicion"), ("disposicionRespaldo",)),
    (("motivo", "causal", "causa"), ("motivo",)),
    (("observacion", "observaciones", "notas de campo"), ("observacion",)),
    (("tipo de asignacion", "tipo asignacion"), ("tipoAsignacion",)),
    (("fecha de adquisicion", "fecha adquisicion", "fecha de compra", "fecha de alta"), ("fechaAdquisicion", "fechaIngreso")),
    (("fecha de ingreso", "fecha ingreso"), ("fechaIngreso", "fechaAdquisicion")),
    (("fecha de baja", "fecha baja"), ("fechaBaja",)),
    (("fecha de asignacion", "fecha de reasignacion", "fecha asignacion", "fecha reasignacion"), ("fechaAsignacion",)),
    (("fecha de transferencia", "fecha de traslado", "fecha transferencia"), ("fechaTransferencia",)),
    (("fecha de revaluo", "fecha revaluo"), ("fechaRevaluo",)),
    (("fecha de levantamiento", "fecha de inspeccion", "fecha levantamiento"), ("fechaLevantamiento",)),
    (("fecha",), ("fechaAdquisicion", "fechaIngreso", "fechaBaja", "fechaAsignacion", "fechaTransferencia", "fechaRevaluo", "fechaLevantamiento")),
    (("gestion", "anio", "ano"), ("gestion",)),
    (("depreciacion acumulada", "depreciacion"), ("depAcumuladaBs",)),
    (("valor neto", "valor en libros", "neto"), ("valorNetoBs",)),
    (("valor final", "valor residual"), ("valorFinalBs",)),
    (("valor anterior",), ("valorAnteriorBs",)),
    (("nuevo valor", "valor revaluado"), ("nuevoValorBs",)),
    (("vida util", "nueva vida util", "meses"), ("nuevaVidaUtilMeses",)),
    (("valor", "monto", "costo", "precio", "bs"), ("valorBs", "nuevoValorBs", "valorFinalBs")),
)

_MESES_MAP: dict[str, int] = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def detectar_fuente_reporte(consulta: str) -> str:
    norm = normalizar_texto(consulta)
    if any(k in norm for k in ("levantamiento", "levantamientos", "inspeccion fisica", "inspecciones", "planilla de campo", "hallazgo", "hallazgos", "faltantes", "sobrantes")):
        return "LEVANTAMIENTOS"
    if any(k in norm for k in ("transferencia", "transferencias", "traslado", "traslados", "movimiento entre unidades")):
        return "TRANSFERENCIAS"
    if any(k in norm for k in ("reasignacion", "reasignaciones", "asignacion", "asignaciones", "acta de entrega", "actas de custodia", "historial de custodia")):
        return "ASIGNACIONES"
    if any(k in norm for k in ("revaluo", "revaluos", "revalorizacion", "revalorizaciones", "peritaje", "revaluado", "revaluados")):
        return "REVALUOS"
    if any(k in norm for k in ("actas de baja", "reporte de bajas", "bajas realizadas", "bajas registradas", "bajas efectuadas", "fecha de baja", "fechas de baja", "bienes dados de baja", "desincorporacion", "desincorporaciones", "bajas de ", "bajas en ", "bajas del ")):
        return "BAJAS"
    if any(k in norm for k in ("ingresos", "facturas", "factura", "proveedor", "proveedores", "altas realizadas", "altas registradas", "reporte de altas", "fecha de ingreso", "compras realizadas")):
        return "ALTAS"
    return "PADRON"


def _ultimo_dia_mes(anio: int, mes: int) -> int:
    if mes == 2:
        return 29 if (anio % 4 == 0 and (anio % 100 != 0 or anio % 400 == 0)) else 28
    return 30 if mes in (4, 6, 9, 11) else 31


def detectar_fechas_reporte(consulta: str) -> tuple[str | None, str | None]:
    """Extrae rangos de fecha (YYYY-MM-DD) desde lenguaje natural o fechas explícitas."""
    norm = normalizar_texto(consulta)

    # 1. Fechas ISO explícitas YYYY-MM-DD
    isos = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", consulta)
    if len(isos) >= 2:
        return min(isos[0], isos[1]), max(isos[0], isos[1])
    if len(isos) == 1:
        if any(k in norm for k in ("hasta", "antes del", "antes de")):
            return None, isos[0]
        return isos[0], None

    # 2. Fechas DD/MM/YYYY
    dmys = re.findall(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", consulta)
    if dmys:
        conv = [f"{y}-{int(m):02d}-{int(d):02d}" for d, m, y in dmys]
        if len(conv) >= 2:
            return min(conv[0], conv[1]), max(conv[0], conv[1])
        if any(k in norm for k in ("hasta", "antes del", "antes de")):
            return None, conv[0]
        return conv[0], None

    # 3. Rango de meses en un año: "de enero a junio de 2025" / "enero a marzo 2026"
    meses_rx = "|".join(_MESES_MAP.keys())
    m_mes_rango = re.search(
        rf"\b({meses_rx})\s+(?:a|al|hasta)\s+({meses_rx})(?:\s+(?:de|del)\s+(20\d{{2}})|\s+(20\d{{2}}))?",
        norm,
    )
    if m_mes_rango:
        m1 = _MESES_MAP[m_mes_rango.group(1)]
        m2 = _MESES_MAP[m_mes_rango.group(2)]
        anio = int(m_mes_rango.group(3) or m_mes_rango.group(4) or datetime.now().year)
        return f"{anio}-{m1:02d}-01", f"{anio}-{m2:02d}-{_ultimo_dia_mes(anio, m2):02d}"

    # 4. Rango de años: "entre 2023 y 2025", "del 2022 al 2024", "2024 y 2025", "2020-25", "año 2020-25"
    m_anios = re.search(
        r"\b(?:entre|desde|del|de|en\s+el\s+a[nñ]o|en\s+los\s+a[nñ]os|a[nñ]os?)?\s*(199\d|20\d{2})\s*(?:y|a|al|hasta|-|/)\s*(199\d|20\d{2}|\d{2})\b",
        norm,
    )
    if m_anios:
        y1_raw = int(m_anios.group(1))
        g2_str = m_anios.group(2)
        y2_raw = (y1_raw // 100) * 100 + int(g2_str) if len(g2_str) == 2 else int(g2_str)
        if 1990 <= y2_raw <= 2099:
            y1, y2 = sorted((y1_raw, y2_raw))
            return f"{y1}-01-01", f"{y2}-12-31"

    # 5. Desde un mes y año específico: "desde marzo 2025"
    m_desde_mes = re.search(rf"\b(?:desde|a partir de)\s+({meses_rx})\s+(?:de\s+|del\s+)?(20\d{{2}})\b", norm)
    if m_desde_mes:
        m1 = _MESES_MAP[m_desde_mes.group(1)]
        anio = int(m_desde_mes.group(2))
        return f"{anio}-{m1:02d}-01", None

    # 6. Desde un año: "desde 2023" / "a partir de 2024"
    m_desde_anio = re.search(r"\b(?:desde|a partir de|del anio|desde el)\s+(?:el\s+)?(20\d{2})\b", norm)
    if m_desde_anio:
        return f"{int(m_desde_anio.group(1))}-01-01", None

    # 7. Hasta un año: "hasta 2022" / "antes de 2021"
    m_hasta_anio = re.search(r"\b(?:hasta|antes de|antes del)\s+(?:el\s+)?(20\d{2})\b", norm)
    if m_hasta_anio:
        return None, f"{int(m_hasta_anio.group(1))}-12-31"

    return None, None


def detectar_columnas_solicitadas(consulta: str, fuente: str) -> list[str] | None:
    """Detecta si el usuario pidió explícitamente un conjunto específico de columnas."""
    norm = normalizar_texto(consulta)
    pide_columnas = any(
        k in norm
        for k in (
            "columna",
            "columnas",
            "campo",
            "campos",
            "que incluya",
            "que muestre",
            "incluyendo",
            "mostrando",
            "solo con",
            "unicamente con",
            "con los datos",
            "con codigo",
            "solo codigo",
        )
    )
    if not pide_columnas:
        return None

    # Recortar desde el marcador de columnas si existe, para evitar falsos positivos en la parte de filtro
    m_sep = re.search(
        r"\b(?:con\s+las\s+columnas|columnas|campos|que\s+incluya\w*|que\s+muestre\w*|incluyendo|mostrando|solo\s+con|unicamente\s+con|solo\s+codigo)\b(.*)$",
        norm,
    )
    tramo = m_sep.group(0) if m_sep else norm

    validas = set(COLUMNAS_POR_FUENTE.get(fuente, COLUMNAS_POR_FUENTE["PADRON"]))
    posiciones: list[tuple[int, str]] = []
    ya_elegidas: set[str] = set()
    tramo_restante = tramo

    for aliases, candidatas in _ALIAS_COLUMNAS:
        mejor_match: re.Match[str] | None = None
        for al in aliases:
            m = re.search(rf"\b{re.escape(al)}\b", tramo_restante)
            if m and (mejor_match is None or m.start() < mejor_match.start()):
                mejor_match = m
        if mejor_match is None:
            continue
        for key in candidatas:
            if key in validas and key not in ya_elegidas:
                ya_elegidas.add(key)
                posiciones.append((mejor_match.start(), key))
                tramo_restante = (
                    tramo_restante[: mejor_match.start()]
                    + (" " * (mejor_match.end() - mejor_match.start()))
                    + tramo_restante[mejor_match.end() :]
                )
                break

    if len(posiciones) < 2:
        return None
    posiciones.sort(key=lambda pair: pair[0])
    return [key for _, key in posiciones]


def detectar_orden_y_limite(consulta: str) -> tuple[str | None, str | None, int | None]:
    norm = normalizar_texto(consulta)
    limite: int | None = None
    m_top = re.search(r"\b(?:top|los|las|primeros|primeras|solo|unicamente)\s+(\d{1,3})\b", norm)
    if m_top:
        n = int(m_top.group(1))
        if 1 <= n <= 500:
            limite = n

    ordenar_por: str | None = None
    orden_dir: str | None = None
    if any(k in norm for k in ("mas caros", "mayor valor", "mayor costo", "mas costosos", "ordenado por valor", "mayor monto")):
        ordenar_por, orden_dir = "valor", "DESC"
    elif any(k in norm for k in ("mas baratos", "menor valor", "menor costo", "mas economicos")):
        ordenar_por, orden_dir = "valor", "ASC"
    elif any(k in norm for k in ("mas recientes", "ultimos", "ultimas", "ordenado por fecha")):
        ordenar_por, orden_dir = "fecha", "DESC"
    elif any(k in norm for k in ("mas antiguos", "mas viejos", "primeros adquiridos")):
        ordenar_por, orden_dir = "fecha", "ASC"
    return ordenar_por, orden_dir, limite


def validar_termino_search_reporte(
    candidato: str | None,
    consulta_corregida: str,
    ubicacion: str | None = None,
    marca: str | None = None,
) -> str | None:
    """Garantiza que 'search' solo contenga un código patrimonial o bien específico real, jamás verbos, errores de dictado ni facultades."""
    if not candidato or not str(candidato).strip():
        return None
    limpio = " ".join(str(candidato).strip().split())
    m_cod = re.search(r"\b([a-z]\d{6,10}|\d{4}-\d{2}-\d{3,6})\b", limpio, flags=re.IGNORECASE)
    if m_cod:
        return m_cod.group(1).upper()

    norm_cand = normalizar_texto(limpio)
    if not norm_cand or len(norm_cand) < 3:
        return None

    for bien in _CATALOGO_BIENES_SEARCH:
        if re.fullmatch(rf"{re.escape(bien)}(?:s|es)?", norm_cand):
            return bien.upper()

    norm_q = normalizar_texto(consulta_corregida)
    if any(p in norm_q for p in ("todos los activos", "todos los bienes", "todo el padron")):
        return None

    tokens = re.findall(r"[a-z0-9]{3,}", norm_cand)
    if not tokens or len(tokens) > 3:
        return None

    norm_ub = normalizar_texto(ubicacion or "")
    norm_marca = normalizar_texto(marca or "")
    tokens_validos: list[str] = []
    for tk in tokens:
        base = tk[:-1] if len(tk) >= 5 and tk.endswith("s") and not tk.endswith("ss") else tk
        if (
            tk.isdigit()
            or tk in _PALABRAS_NO_SEARCH
            or base in _PALABRAS_NO_SEARCH
            or (norm_ub and (tk in norm_ub or base in norm_ub))
            or (norm_marca and tk == norm_marca)
        ):
            continue
        tokens_validos.append(base.upper())

    if not tokens_validos or len(tokens_validos) != len(tokens):
        return None

    texto_val = " ".join(tokens_validos)
    # El término debe existir realmente en la consulta corregida
    if normalizar_texto(texto_val) not in norm_q and norm_cand not in norm_q:
        return None
    return texto_val


def construir_prompt_planificador_reporte(
    consulta_corregida: str,
    filtros_previos: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Prompt estructurado de alta precisión para planificación SQL multi-fuente en CPU local y Groq."""
    sistema = (
        "Planificador SQL UAGRM. Devuelve SOLO un JSON compacto omitiendo claves nulas.\n"
        "fuente: PADRON (activos del padrón, estados, dados de alta en años) | ALTAS (solo facturas/proveedores/ingresos) | "
        "BAJAS | ASIGNACIONES | TRANSFERENCIAS | REVALUOS | LEVANTAMIENTOS.\n"
        "estados: BUENO|REGULAR|MALO|EN_REPARACION|BAJA. condicion: NUEVO|USADO|DETERIORADO.\n"
        "Reglas estrictas:\n"
        "1. ubicacion: sigla o facultad oficial ('FICCT' para Computación/Ingeniería Informática/Sistemas, 'Ciencias Contables', 'Veterinarias', 'Exactas y Tecnología', 'Salud Humana').\n"
        "2. search: SOLO si piden un código ('U04000012') o un objeto específico ('TELEVISOR','PROYECTOR','CAMIONETA'). Si piden 'todos los activos' o 'bienes', OMITE search.\n"
        "3. Fechas: 'año 2020-25' o 'entre 2020 y 2025' -> fecha_desde='2020-01-01', fecha_hasta='2025-12-31' (y OMITE gestion).\n"
        "4. columnas: lista exacta de claves camelCase SOLO si el usuario pide columnas específicas.\n"
        "Columnas válidas:\n"
        "PADRON: codigo,descripcion,grupoContable,ubicacion,custodio,estado,condicion,marca,modelo,nroSerie,fechaAdquisicion,gestion,valorBs,depAcumuladaBs,valorNetoBs,placa\n"
        "ALTAS: codigo,descripcion,fechaIngreso,nroFactura,proveedor,grupoContable,ubicacion,custodio,estado,valorBs\n"
        "BAJAS: nroBaja,codigo,descripcion,fechaBaja,documento,motivo,ubicacion,grupoContable,autorizador,valorBs,valorFinalBs\n"
        "ASIGNACIONES: nroActa,fechaAsignacion,tipoAsignacion,codigo,descripcion,ubicacion,custodio,documentoRespaldo,estado,valorBs\n"
        "TRANSFERENCIAS: nroTransferencia,fechaTransferencia,codigo,descripcion,oficinaOrigen,oficinaDestino,documentoRespaldo,motivo,estado,valorBs\n"
        "REVALUOS: nroRevaluo,fechaRevaluo,codigo,descripcion,ubicacion,disposicionRespaldo,motivo,tecnico,valorAnteriorBs,nuevoValorBs,nuevaVidaUtilMeses\n"
        "LEVANTAMIENTOS: nroPlanilla,fechaLevantamiento,ubicacion,codigo,descripcion,tipoHallazgo,estadoRevision,inspector,observacion"
    )
    ctx_prev = f" | Prev: {filtros_previos}" if filtros_previos else ""
    return [
        {"role": "system", "content": sistema},
        {"role": "user", "content": f"{consulta_corregida}{ctx_prev}"},
    ]


def normalizar_y_validar_plan(
    plan_llm: dict[str, Any] | None,
    filtros_det: dict[str, Any],
    consulta_corregida: str,
) -> dict[str, Any]:
    """Fusiona y blinda el plan JSON del LLM con reglas estrictas de integridad UAGRM."""
    llm = plan_llm if isinstance(plan_llm, dict) else {}
    norm_q = normalizar_texto(consulta_corregida)

    fuente_det = detectar_fuente_reporte(consulta_corregida)
    fuente_llm = str(llm.get("fuente") or "").strip().upper()
    if fuente_det != "PADRON":
        fuente = fuente_det
    elif fuente_llm == "ALTAS" and not any(
        k in norm_q for k in ("factura", "proveedor", "ingreso", "reporte de altas", "altas realizadas", "altas registradas")
    ):
        fuente = "PADRON"
    elif fuente_llm in FUENTES_REPORTE_VALIDAS:
        fuente = fuente_llm
    else:
        fuente = "PADRON"

    # Unidad estricta: priorizar coincidencia explícita del catálogo oficial UAGRM sobre texto inventado
    unidad_det = filtros_det.get("ubicacion")
    unidad_llm = str(llm.get("ubicacion") or "").strip() or None
    if unidad_det:
        ubicacion = unidad_det
    elif unidad_llm:
        fac_norm = resolver_facultad_oficial_uagrm(unidad_llm)
        ubicacion = unidad_llm if len(unidad_llm) <= 60 else (fac_norm or unidad_llm[:60])
    else:
        ubicacion = None

    estado_llm = str(llm.get("estado") or "").strip().upper()
    estado_det = filtros_det.get("estado")
    estados_extra_fuente = {"VIGENTE", "TRANSFERIDO", "HISTORICO", "ENCONTRADO", "CON_DIFERENCIA", "FALTANTE", "SOBRANTE", "PENDIENTE", "CONCILIADO"}
    estado_valido_llm = estado_llm if (estado_llm in ESTADOS_PADRON_VALIDOS or estado_llm in estados_extra_fuente) else None
    estado = estado_det or estado_valido_llm

    cond_llm = str(llm.get("condicion") or "").strip().upper()
    condicion = filtros_det.get("condicion") or (cond_llm if cond_llm in CONDICIONES_PADRON_VALIDAS else None)

    pide_todos_activos = any(p in norm_q for p in ("todos los activos", "todos los bienes", "todo el padron"))
    grupo_det = filtros_det.get("grupo_contable")
    grupo_llm = str(llm.get("grupo_contable") or "").strip() or None
    grupo_contable = grupo_det or (None if pide_todos_activos else grupo_llm)

    marca = str(llm.get("marca") or "").strip() or None
    custodio = str(llm.get("custodio") or "").strip() or None

    # Blindaje estricto en search (jamás "NECESITUM", "INCLUYA", "COLUMNAS", ni facultades)
    search_cand = validar_termino_search_reporte(
        filtros_det.get("search") or (str(llm.get("search") or "").strip() or None),
        consulta_corregida=consulta_corregida,
        ubicacion=ubicacion,
        marca=marca,
    )

    f_desde_det, f_hasta_det = detectar_fechas_reporte(consulta_corregida)
    f_desde_llm = str(llm.get("fecha_desde") or "").strip()
    f_hasta_llm = str(llm.get("fecha_hasta") or "").strip()
    if re.fullmatch(r"(?:199\d|20\d{2})", f_desde_llm):
        f_desde_llm = f"{f_desde_llm}-01-01"
    if re.fullmatch(r"(?:199\d|20\d{2})", f_hasta_llm):
        f_hasta_llm = f"{f_hasta_llm}-12-31"
    fecha_desde = f_desde_det or (f_desde_llm if re.fullmatch(r"(?:199\d|20\d{2})-\d{2}-\d{2}", f_desde_llm) else None)
    fecha_hasta = f_hasta_det or (f_hasta_llm if re.fullmatch(r"(?:199\d|20\d{2})-\d{2}-\d{2}", f_hasta_llm) else None)

    gestion = filtros_det.get("gestion")
    if gestion is None and isinstance(llm.get("gestion"), int) and 1990 <= llm["gestion"] <= 2100:
        gestion = llm["gestion"]
    elif gestion is None and isinstance(llm.get("gestion"), str):
        f_g1, f_g2 = detectar_fechas_reporte(str(llm.get("gestion")))
        if f_g1 or f_g2:
            fecha_desde = fecha_desde or f_g1
            fecha_hasta = fecha_hasta or f_g2
        elif str(llm["gestion"]).strip().isdigit():
            g_int = int(str(llm["gestion"]).strip())
            if 1990 <= g_int <= 2100:
                gestion = g_int
    # Si hay filtro de rango de fecha explícito (desde/hasta), no restringir adicionalmente a una sola gestión
    if fecha_desde or fecha_hasta:
        gestion = None

    valor_min = filtros_det.get("valor_min")
    if valor_min is None and isinstance(llm.get("valor_min"), (int, float)):
        valor_min = float(llm["valor_min"])
    valor_max = filtros_det.get("valor_max")
    if valor_max is None and isinstance(llm.get("valor_max"), (int, float)):
        valor_max = float(llm["valor_max"])

    ord_det, dir_det, lim_det = detectar_orden_y_limite(consulta_corregida)
    ordenar_por = str(llm.get("ordenar_por") or "").strip().lower() or ord_det
    if ordenar_por not in {"valor", "fecha", "codigo", "descripcion"}:
        ordenar_por = ord_det
    orden_dir = str(llm.get("orden_dir") or "").strip().upper() or dir_det
    if orden_dir not in {"ASC", "DESC"}:
        orden_dir = dir_det

    limite = lim_det
    if limite is None and isinstance(llm.get("limite"), int) and 1 <= llm["limite"] <= 500:
        limite = llm["limite"]

    # Columnas dinámicas: validar claves exactas contra la fuente elegida
    validas = set(COLUMNAS_POR_FUENTE.get(fuente, COLUMNAS_POR_FUENTE["PADRON"]))
    cols_llm_raw = llm.get("columnas")
    cols_llm: list[str] = []
    if isinstance(cols_llm_raw, list):
        for c in cols_llm_raw:
            cs = str(c).strip()
            if cs in validas and cs not in cols_llm:
                cols_llm.append(cs)
    cols_det = detectar_columnas_solicitadas(consulta_corregida, fuente)
    # Fusionar columnas explícitas del usuario (cols_det) y las elegidas por el LLM sin perder ninguna solicitada
    if cols_det and cols_llm:
        columnas: list[str] | None = list(cols_det)
        for c in cols_llm:
            if c not in columnas:
                columnas.append(c)
    elif len(cols_llm) >= 2:
        columnas = cols_llm
    else:
        columnas = cols_det or (cols_llm if cols_llm else None)

    titulo_llm = str(llm.get("titulo_reporte") or "").strip()
    titulo_reporte = corregir_texto_institucional(titulo_llm)[:120] if len(titulo_llm) >= 8 else None

    return {
        "fuente": fuente,
        "ubicacion": ubicacion,
        "estado": estado,
        "condicion": condicion,
        "grupo_contable": grupo_contable,
        "marca": marca,
        "custodio": custodio,
        "search": search_cand,
        "valor_min": valor_min,
        "valor_max": valor_max,
        "gestion": gestion,
        "fecha_desde": fecha_desde,
        "fecha_hasta": fecha_hasta,
        "ordenar_por": ordenar_por,
        "orden_dir": orden_dir,
        "limite": limite,
        "columnas": columnas,
        "titulo_reporte": titulo_reporte,
    }
