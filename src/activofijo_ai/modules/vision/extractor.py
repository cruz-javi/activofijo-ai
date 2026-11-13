import re
from dataclasses import dataclass
from typing import Any

PALABRAS_INSTITUCIONALES_RUIDO = frozenset(
    {
        "UAGRM",
        "ACTIVO",
        "ACTIVOS",
        "FIJO",
        "FIJOS",
        "PROPIEDAD",
        "ESTADO",
        "BOLIVIA",
        "PATRIMONIO",
        "INVENTARIO",
        "CODIGO",
        "SERIE",
        "MODELO",
        "MARCA",
        "DAF",
    }
)

LARGO_MINIMO = 4
LARGO_MAXIMO = 24
MINIMO_DIGITOS = 2


@dataclass(frozen=True)
class ResultadoExtraccionCodigo:
    codigo_principal: str | None
    candidatos: list[str]
    texto_crudo: str
    confianza: float
    coincide_prefijo: bool = False
    puntaje_ranking: float = 0.0


def _limpiar_codigo(texto: str) -> str:
    # Conserva guiones internos útiles y elimina ruido de bordes o signos sueltos.
    limpio = re.sub(r"[^A-Z0-9-]", "", texto.upper())
    return limpio.strip("-")


def _solo_alfanumerico(texto: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", texto.upper())


def _contar_digitos(texto: str) -> int:
    return sum(1 for ch in texto if ch.isdigit())


def extraer_codigos_de_bloques(
    bloques_ocr: list[Any] | None,
    prefijos_conocidos: list[str] | None = None,
) -> ResultadoExtraccionCodigo:
    """Filtra ruido institucional y ordena los mejores códigos patrimoniales detectados."""
    if not bloques_ocr:
        return ResultadoExtraccionCodigo(
            codigo_principal=None,
            candidatos=[],
            texto_crudo="",
            confianza=0.0,
        )

    # Orden visual aproximado de arriba hacia abajo y de izquierda a derecha.
    ordenados = sorted(
        bloques_ocr,
        key=lambda item: (
            round(float(item[0][0][1]) / 18.0) if item[0] else 0,
            float(item[0][0][0]) if item[0] else 0.0,
        ),
    )

    textos_linea: list[str] = []
    tokens_con_score: list[tuple[str, float]] = []

    for entrada in ordenados:
        if len(entrada) < 3:
            continue
        texto = str(entrada[1] or "").strip()
        if not texto:
            continue
        score = float(entrada[2] or 0.0)
        textos_linea.append(texto)

        for parte in re.split(r"\s+", texto):
            if parte:
                tokens_con_score.append((parte, score))

    prefijos_norm = tuple(_solo_alfanumerico(p) for p in (prefijos_conocidos or []) if p.strip())
    # clave -> (visible, puntaje_ranking, confianza_calibrada, coincide_prefijo)
    candidatos_puntaje: dict[str, tuple[str, float, float, bool]] = {}

    def registrar(raw: str, base_score: float, bono: float = 0.0) -> None:
        visible = _limpiar_codigo(raw)
        clave = _solo_alfanumerico(visible)
        if not clave or clave in PALABRAS_INSTITUCIONALES_RUIDO:
            return
        if len(clave) < LARGO_MINIMO or len(clave) > LARGO_MAXIMO:
            return
        digitos = _contar_digitos(clave)
        if digitos < MINIMO_DIGITOS:
            return

        # Premia códigos compatibles con estructura patrimonial (prefijos conocidos + densidad de dígitos + largo completo).
        score_clamped = min(1.0, max(0.0, base_score))
        proporcion_digitos = digitos / len(clave)
        coincide_pref = bool(prefijos_norm and any(clave.startswith(pref) for pref in prefijos_norm))
        bono_largo = 0.14 if 7 <= len(clave) <= 14 else (0.05 if len(clave) >= 5 else 0.0)
        bono_prefijo = 0.35 if coincide_pref else 0.0
        puntaje_rank = score_clamped * 0.72 + proporcion_digitos * 0.14 + bono_largo + bono_prefijo + bono
        confianza_calibrada = min(1.0, score_clamped * 0.85 + proporcion_digitos * 0.15)

        previo = candidatos_puntaje.get(clave)
        if previo is None or puntaje_rank > previo[1]:
            candidatos_puntaje[clave] = (visible, puntaje_rank, confianza_calibrada, coincide_pref)

    for idx, (token, score) in enumerate(tokens_con_score):
        registrar(token, score)
        if idx + 1 < len(tokens_con_score):
            sig_token, sig_score = tokens_con_score[idx + 1]
            # Une pares separados por espacio típico en trazos manuales (ej. "U01 000045").
            registrar(f"{token}{sig_token}", min(score, sig_score), bono=0.02)

    ranking = sorted(
        candidatos_puntaje.values(),
        key=lambda item: (-item[1], -_contar_digitos(item[0]), -len(item[0])),
    )

    if not ranking:
        return ResultadoExtraccionCodigo(
            codigo_principal=None,
            candidatos=[],
            texto_crudo=" | ".join(textos_linea)[:240],
            confianza=0.0,
            coincide_prefijo=False,
            puntaje_ranking=0.0,
        )

    codigo_principal, mejor_rank, mejor_conf, mejor_prefijo = ranking[0]
    confianza = round(min(1.0, max(0.0, mejor_conf)), 4)
    candidatos = [item[0] for item in ranking[:5]]

    return ResultadoExtraccionCodigo(
        codigo_principal=codigo_principal,
        candidatos=candidatos,
        texto_crudo=" | ".join(textos_linea)[:240],
        confianza=confianza,
        coincide_prefijo=mejor_prefijo,
        puntaje_ranking=round(mejor_rank, 4),
    )
