from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from activofijo_ai.core.config import Settings
from activofijo_ai.modules.llm.mcp_server import normalizar_texto
from activofijo_ai.modules.llm.provider import (
    LlmErrorExternoError,
    LlmNoConfiguradoError,
    OpenAICompatibleLlmClient,
    extraer_json_objeto,
)

EstadoFisicoInspeccion = Literal["BUENO", "REGULAR", "MALO"]
AccionInspeccionVoz = Literal["completar", "confirmar", "cancelar"]

_PATRONES_MALO = (
    r"\b(?:en\s+)?mal\s+estado\b",
    r"\b(?:estado\s+)?mal[oa]\b",
    r"\bmala\s+condici[oó]n\b",
    r"\bp[eé]sim[oa]\b",
    r"\binservible\b",
    r"\bno\s+(?:funciona|enciende|prende|sirve|arranca|carga)\b",
    r"\b(?:quemad[oa]|quebrad[oa]|rot[oa]|da[ñn]ad[oa]|averiad[oa]|inoperable|obsolet[oa])\b",
    r"\bpara\s+baja\b",
)

_PATRONES_REGULAR = (
    r"\b(?:estado\s+)?regular\b",
    r"\bm[aá]s\s+o\s+menos\b",
    r"\b(?:con\s+)?desgaste\b",
    r"\b(?:desgastad[oa]|rayad[oa]|raspad[oa]|descolorid[oa]|oxidado|deterioro\s+leve|golpead[oa])\b",
    r"\bmedio\s+(?:mal[oa]|da[ñn]ad[oa])\b",
)

_PATRONES_BUENO = (
    r"\b(?:en\s+)?buen\s+est[aá]?\b",
    r"\b(?:en\s+)?buen\s+estado\b",
    r"\b(?:estado\s+)?buen[oa]\b",
    r"\b(?:en\s+)?buena\s+condici[oó]n\b",
    r"\b(?:operativ[oa]|perfect[oa]|excelente|[oó]ptim[oa]|normal)\b",
    r"\btodo\s+(?:bien|ok|correcto)\b",
    r"\bsin\s+novedad\b",
)

_DANIO_ESTRUC_MALO = (
    r"\ble\s+falta\s+(?:una\s+)?(?:pata|pieza|tecla|tapa|pantalla|caj[oó]n|rueda)\b",
    r"\bsin\s+(?:una\s+)?(?:pata|pantalla|cable|bater[ií]a|soporte)\b",
    r"\b(?:no\s+da|no\s+responde|fuent[ae]\s+da[ñn]ada|placa\s+quemada)\b",
)

_PATRONES_CONFIRMACION = (
    r"\benv[ií]a(?:r|lo|la)?\b",
    r"\bacept(?:ar|o|a)\b",
    r"\bconfirm(?:ar|o|a|alo)?\b",
    r"\bguard(?:ar|a|alo)?\b",
    r"\bagreg(?:ar|a|alo)?\b",
    r"\bregistr(?:ar|a|alo)?\b",
    r"\bmand(?:ar|a|alo)\b",
    r"\bs[ií]\b",
    r"\blisto\b",
    r"\bok(?:ay|ey)?\b",
    r"\bdale\b",
    r"\bde\s+acuerdo\b",
    r"\badelante\b",
    r"\b(?:as[ií]\s+)?est[aá]\s+bien\b",
    r"\bnada\s+m[aá]s\b",
    r"\beso\s+(?:es|ser[ií]a)\s+todo\b",
)

_PATRONES_CANCELAR = (
    r"\bcancel(?:ar|a|o)\b",
    r"\bdescart(?:ar|a|o)\b",
    r"\banul(?:ar|a|o)\b",
    r"\bno\s+enviar\b",
    r"\bcerrar\b",
)

_PREFIJOS_ESTADO_RE = re.compile(
    r"^(?:"
    r"(?:eh+|este+|bueno+|mira+|a\s+ver|el\s+activo|el\s+equipo|la\s+mesa|la\s+silla|esto)\s+|"
    r"(?:est[aá]\s+)?(?:en\s+)?(?:muy\s+)?(?:buen\s+estado|buen\s+est[aá]?|bueno|buena|regular|mal\s+estado|malo|mala|perfecto|operativo|todo\s+bien)"
    r"(?:\s*(?:,|y|pero|aunque|con|porque|ya\s+que)\s*)?"
    r")+",
    re.IGNORECASE,
)

_SUFIJOS_CIERRE_RE = re.compile(
    r"(?:^|\s*[,.;]?\s*)(?:"
    r"nada\s+m[aá]s|eso\s+(?:es|ser[ií]a)\s+todo|sin\s+observaci[oó]n(?:es)?|"
    r"ninguna\s+observaci[oó]n|env[ií]a(?:r|lo|la)?|acept(?:ar|o|a)|confirm(?:ar|o|a|alo)?|"
    r"guard(?:ar|a|alo)?|agreg(?:ar|a|alo)?|registr(?:ar|a|alo)?|listo|gracias|ok(?:ay|ey)?"
    r")\.?\s*$",
    re.IGNORECASE,
)

_CONECTORES_INICIALES_RE = re.compile(
    r"^(?:observaci[oó]n\s*:?\s*|nota\s*:?\s*|pon(?:le|er)?\s+que\s+|corregir\s*:?\s*|que\s+|y\s+|pero\s+|con\s+(?=le\s+)|,\s*)+",
    re.IGNORECASE,
)

_TOKENS_SOLO_CONFIRMACION = {
    "enviar",
    "envia",
    "envialo",
    "aceptar",
    "acepto",
    "acepta",
    "confirmar",
    "confirmo",
    "confirma",
    "confirmalo",
    "guardar",
    "guarda",
    "guardalo",
    "agregar",
    "agrega",
    "registrar",
    "registra",
    "mandar",
    "manda",
    "mandalo",
    "si",
    "listo",
    "ok",
    "okay",
    "okey",
    "dale",
    "perfecto",
    "adelante",
    "de acuerdo",
    "asi esta bien",
    "esta bien",
    "nada mas",
    "eso es todo",
    "eso seria todo",
    "gracias",
}


@dataclass(frozen=True)
class ResultadoInterpretacionInspeccion:
    estado_fisico: EstadoFisicoInspeccion | None
    observacion: str
    accion: AccionInspeccionVoz
    confirma_directo: bool
    motor_interpretacion: Literal["reglas_rapidas", "llm_local"]


def _coincide_alguno(texto_norm: str, patrones: tuple[str, ...]) -> bool:
    return any(re.search(pat, texto_norm, flags=re.IGNORECASE) for pat in patrones)


_FRASES_CONFIRMACION_COMPUESTAS_RE = re.compile(
    r"\b(?:de\s+acuerdo|as[ií]\s+est[aá]\s+bien|est[aá]\s+bien|nada\s+m[aá]s|"
    r"eso\s+(?:es|ser[ií]a)\s+todo|por\s+favor)\b",
    re.IGNORECASE,
)


def _es_solo_confirmacion(norm: str) -> bool:
    if norm in _TOKENS_SOLO_CONFIRMACION:
        return True
    sin_frases = _FRASES_CONFIRMACION_COMPUESTAS_RE.sub(" ", norm)
    tokens = [t for t in re.split(r"[\s,.;:!?]+", sin_frases) if t]
    if not tokens:
        return True
    restantes = [t for t in tokens if t not in {"y", "e"}]
    return bool(restantes) and all(t in _TOKENS_SOLO_CONFIRMACION for t in restantes)


def _limpiar_observacion(texto: str) -> str:
    limpio = re.sub(r"\s+", " ", texto).strip()
    for _ in range(4):
        antes = limpio
        sin_sufijo = _SUFIJOS_CIERRE_RE.sub("", limpio).strip(" ,.;:-")
        if sin_sufijo != limpio:
            limpio = re.sub(r"(?:^|\s+)[ye]$", "", sin_sufijo, flags=re.IGNORECASE).strip(" ,.;:-")
        nuevo = _PREFIJOS_ESTADO_RE.sub("", limpio).strip(" ,.;:-")
        nuevo = _CONECTORES_INICIALES_RE.sub("", nuevo).strip(" ,.;:-")
        limpio = nuevo
        if limpio == antes:
            break
    if not limpio:
        return ""
    norm = normalizar_texto(limpio)
    if norm in {
        "buen estado",
        "buen esta",
        "bueno",
        "regular",
        "mal estado",
        "malo",
        "sin observaciones",
        "sin novedad",
        "ninguna",
        "todo bien",
    } or _es_solo_confirmacion(norm):
        return ""
    return f"{limpio[0].upper()}{limpio[1:]}"


def interpretar_inspeccion_rapida(texto: str) -> ResultadoInterpretacionInspeccion:
    """
    Extrae en < 1 ms estado físico, observación limpia o intención de confirmación/cancelación
    desde dictado de voz en campo.
    """
    bruto = re.sub(r"\s+", " ", (texto or "")).strip()
    if not bruto:
        return ResultadoInterpretacionInspeccion(
            estado_fisico=None,
            observacion="",
            accion="completar",
            confirma_directo=False,
            motor_interpretacion="reglas_rapidas",
        )

    norm = normalizar_texto(bruto)
    palabras_norm = [p for p in re.split(r"[\s,.;:!?]+", norm) if p]

    if _coincide_alguno(norm, _PATRONES_CANCELAR) and len(palabras_norm) <= 4:
        return ResultadoInterpretacionInspeccion(
            estado_fisico=None,
            observacion="",
            accion="cancelar",
            confirma_directo=False,
            motor_interpretacion="reglas_rapidas",
        )

    estado: EstadoFisicoInspeccion | None = None
    if _coincide_alguno(norm, _PATRONES_MALO):
        estado = "MALO"
    elif _coincide_alguno(norm, _PATRONES_REGULAR):
        estado = "REGULAR"
    elif _coincide_alguno(norm, _PATRONES_BUENO):
        estado = "BUENO"
    elif _coincide_alguno(norm, _DANIO_ESTRUC_MALO):
        estado = "MALO"

    observacion = _limpiar_observacion(bruto)
    tiene_aceptacion = _coincide_alguno(norm, _PATRONES_CONFIRMACION) or (
        norm in _TOKENS_SOLO_CONFIRMACION
    )

    # Confirmar solo cuando la frase es intención pura de aceptación (sin dictar estado ni observación nueva)
    es_confirmacion_pura = (
        tiene_aceptacion
        and observacion == ""
        and estado is None
        and len(palabras_norm) <= 6
    )

    return ResultadoInterpretacionInspeccion(
        estado_fisico=estado,
        observacion=observacion,
        accion="confirmar" if es_confirmacion_pura else "completar",
        confirma_directo=es_confirmacion_pura,
        motor_interpretacion="reglas_rapidas",
    )


async def interpretar_inspeccion_con_respaldo_ia(
    texto: str,
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ResultadoInterpretacionInspeccion:
    """
    Usa extracción determinista ultrarrápida y solo consulta al LLM local (tope 2 s)
    cuando una frase ambigua no resuelve intención ni estado de forma directa.
    """
    rapido = interpretar_inspeccion_rapida(texto)
    palabras = [p for p in (texto or "").split() if p.strip()]
    necesita_llm = (
        rapido.accion == "completar"
        and rapido.estado_fisico is None
        and len(palabras) >= 2
        and (
            # Frases cortas tipo confirmación coloquial ("mándalo ya", "está listo jefe")
            len(palabras) <= 4
            and not rapido.observacion
            or len(palabras) >= 5
        )
        and (settings.env != "test" or transport is not None)
    )
    if not necesita_llm:
        return rapido

    llm = OpenAICompatibleLlmClient(settings, transport=transport)
    try:
        res = await llm.generar_chat_completado(
            mensajes=[
                {
                    "role": "system",
                    "content": (
                        "Intérprete rápido de voz en inspección UAGRM. Determina:\n"
                        "- accion: 'confirmar' si acepta/envía/confirma sin dictar datos nuevos, "
                        "'cancelar' si cancela, o 'completar' si describe estado u observación.\n"
                        "- estado_fisico: 'BUENO', 'REGULAR', 'MALO' o null si no se menciona.\n"
                        "- observacion: detalle limpio en 1 frase corta (vacío si es confirmación).\n"
                        'Responde SOLO JSON: {"accion": str, "estado_fisico": str|null, "observacion": str}'
                    ),
                },
                {"role": "user", "content": texto[:300]},
            ],
            timeout_override=2.0,
            max_tokens_override=90,
            response_format={"type": "json_object"},
        )
        parsed: dict[str, Any] | None = extraer_json_objeto(res.contenido)
        if parsed:
            acc_raw = str(parsed.get("accion") or "").lower().strip()
            accion_llm: AccionInspeccionVoz = (
                acc_raw if acc_raw in ("completar", "confirmar", "cancelar") else rapido.accion
            )  # type: ignore[assignment]
            est_raw = str(parsed.get("estado_fisico") or "").upper().strip()
            estado_llm: EstadoFisicoInspeccion | None = (
                est_raw if est_raw in ("BUENO", "REGULAR", "MALO") else rapido.estado_fisico
            )
            obs_llm = str(parsed.get("observacion") or "").strip()
            if accion_llm == "confirmar":
                obs_llm = ""
            obs_final = obs_llm if accion_llm == "completar" and obs_llm else ("" if accion_llm != "completar" else rapido.observacion)
            return ResultadoInterpretacionInspeccion(
                estado_fisico=estado_llm,
                observacion=obs_final,
                accion=accion_llm,
                confirma_directo=accion_llm == "confirmar",
                motor_interpretacion="llm_local",
            )
    except (LlmNoConfiguradoError, LlmErrorExternoError, ValueError, httpx.HTTPError, json.JSONDecodeError):
        pass

    return rapido
