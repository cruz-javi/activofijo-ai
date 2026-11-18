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
from activofijo_ai.skills.registry import construir_instruccion_sistema

EstadoFisicoInspeccion = Literal["BUENO", "REGULAR", "MALO"]
AccionInspeccionVoz = Literal["completar", "confirmar", "cancelar", "ignorar"]
EtapaModalVoz = Literal["dictado_inicial", "esperando_confirmacion"]

MAX_PALABRAS_OBSERVACION = 12

_PATRONES_MALO = (
    r"\b(?:en\s+)?mal\s+estado\b",
    r"\b(?:estado\s+)?mal[oa]\b",
    r"\bmala\s+condici[oó]n\b",
    r"\bp[eé]sim[oa]\b",
    r"\binservible\b",
    r"\bno\s+(?:funciona|enciende|prende|sirve|arranca|carga|da\s+video|da\s+se[nñ]al)\b",
    r"\b(?:quemad[oa]|quebrad[oa]|rot[oa]|da[nñ]ad[oa]|averiad[oa]|inoperable|obsolet[oa]|inutilizable)\b",
    r"\bpara\s+baja\b",
)

_PATRONES_REGULAR = (
    r"\b(?:estado\s+)?regular\b",
    r"\bm[aá]s\s+o\s+menos\b",
    r"\b(?:con\s+)?desgaste\b",
    r"\b(?:desgastad[oa]|rayad[oa]|raspad[oa]|descolorid[oa]|oxidado|deterioro\s+leve|golpead[oa]|floj[oa]|suelt[oa])\b",
    r"\bmedio\s+(?:mal[oa]|da[nñ]ad[oa])\b",
)

_PATRONES_BUENO = (
    r"\b(?:en\s+)?buen\s+estado\b",
    r"\b(?:estado\s+)?buen[oa]\b",
    r"\b(?:en\s+)?buena\s+condici[oó]n\b",
    r"\b(?:operativ[oa]|perfect[oa]|excelente|[oó]ptim[oa]|normal)\b",
    r"\btodo\s+(?:bien|ok|correcto)\b",
    r"\bsin\s+novedad(?:es)?\b",
)

_DANIO_ESTRUC_MALO = (
    r"\ble\s+falta\s+(?:una\s+)?(?:pata|pieza|tecla|tapa|pantalla|caj[oó]n|rueda|asiento|respaldar)\b",
    r"\bfalta\s+(?:una\s+)?(?:pata|pieza|tecla|tapa|pantalla|caj[oó]n|rueda)\b",
    r"\bsin\s+(?:una\s+)?(?:pata|pantalla|cable|bater[ií]a|soporte|fuente|cargador)\b",
    r"\b(?:no\s+da|no\s+responde|fuente\s+da[nñ]ada|placa\s+quemada|se\s+sali[oó]\s+una\s+pata)\b",
)

_PATRONES_CANCELAR = (
    r"\bcancel(?:ar|a|o|alo)\b",
    r"\bdescart(?:ar|a|o|alo)\b",
    r"\banul(?:ar|a|o|alo)\b",
    r"\bno\s+(?:enviar|guardar|agregar|registrar)\b",
    r"\bmejor\s+no\b",
    r"\bno\s+es\s+este\b",
    r"\b(?:cerrar|cierra(?:lo|la)?|cierralo|cierrala)\b",
    r"\bsalir\b",
)

_PATRONES_CONFIRMACION_EXPLICITA = (
    r"\benv[ií](?:ar|a|o|alo|ala|ame)(?:lo|la)?\b",
    r"\bmand(?:ar|a|o|alo|ala)(?:lo|la)?\b",
    r"\bacept(?:ar|a|o|alo|ala)\b",
    r"\bconfirm(?:ar|a|o|alo|ala)\b",
    r"\bguard(?:ar|a|o|alo|ala)\b",
    r"\bagreg(?:ar|a|o|alo|ala)\b",
    r"\bregistr(?:ar|a|o|alo|ala)\b",
    r"\bsub(?:ir|e|a|elo|irlo)(?:lo|la)?\b",
    r"\bproced(?:er|e)\b",
)

_PATRONES_ACUERDO_COLOQUIAL = (
    r"\blisto\b",
    r"\bconforme\b",
    r"\bde\s+acuerdo\b",
    r"\badelante\b",
    r"\bok(?:ay|ey)?\b",
    r"\bdale\b",
    r"\bya\s+qued[oó]\b",
    r"\bas[ií]\s+d[eé]jalo\b",
    r"\bd[eé]jalo\s+as[ií]\b",
    r"\bas[ií]\s+est[aá]\s+bien\b",
    r"\best[aá]\s+bien\s+as[ií]\b",
    r"\btodo\s+correcto\b",
    r"\best[aá]\s+correcto\b",
    r"\bnada\s+m[aá]s\b",
    r"\beso\s+(?:es|ser[ií]a)\s+todo\b",
    r"\bno\s+hay\s+m[aá]s\b",
    r"\bsin\s+cambios\b",
    r"\btal\s+cual\b",
)

# Charla social o ajena a inspección patrimonial que jamás debe guardarse en el bien.
_CHARLA_IRRELEVANTE_COMPLETA_RE = re.compile(
    r"^(?:"
    r"(?:hola+|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|buen\s+d[ií]a|qu[eé]\s+tal|saludos)"
    r"(?:[\s,.;!?]+(?:ingenier[oa]|licenciad[oa]|jefe|amigo|compa[nñ]ero|hermano|se[nñ]or(?:a)?|c[oó]mo\s+est[aá]s?|c[oó]mo\s+te\s+va|c[oó]mo\s+anda[ns]?|todo\s+bien|qu[eé]\s+novedad))*|"
    r"c[oó]mo\s+est[aá]s?(?:\s+todo\s+bien)?|"
    r"al[oó]+|bueno\s+al[oó]+|qu[eé]\s+hora\s+es|"
    r"(?:un\s+(?:ratito|segundo|momento)|esp[eé]r(?:ame|e)|permiso|disculp(?:a|e)|perd[oó]n)(?:[\s,.;]+.*)?|"
    r"p[aá]s(?:ame|eme)\s+.*|muchas\s+gracias|gracias|chau|hasta\s+luego|nos\s+vemos|"
    r"eh+|ah+|mm+|hm+"
    r")[\s,.;!?]*$",
    re.IGNORECASE,
)

_SALUDO_INICIAL_RE = re.compile(
    r"^(?:"
    r"hola+|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|buen\s+d[ií]a|"
    r"c[oó]mo\s+est[aá]s?|qu[eé]\s+tal|disculp(?:a|e)|permiso|al[oó]+|oye"
    r")(?:[\s,.;]+(?:ingenier[oa]|licenciad[oa]|amigo|hermano|jefe|[a-záéíóúñ]+))?[\s,.;:-]*",
    re.IGNORECASE,
)

# Anclas positivas del dominio físico-patrimonial: condiciones, defectos, partes o ubicación.
_ANCLAS_PATRIMONIALES_RE = re.compile(
    r"\b(?:"
    r"buen\s+estado|mal\s+estado|regular|buen[oa]|mal[oa]|operativ[oa]|inservible|obsolet[oa]|"
    r"rot[oa]s?|rayad[oa]s?|rayadura|raspad[oa]s?|astillad[oa]s?|oxidad[oa]s?|floj[oa]s?|suelt[oa]s?|"
    r"doblad[oa]s?|manchad[oa]s?|quemad[oa]s?|quebrad[oa]s?|desgast(?:e|ad[oa]s?)|abollad[oa]s?|"
    r"rajad[oa]s?|suci[oa]s?|descolorid[oa]s?|pintad[oa]s?|pintura|color|barniz|incomplet[oa]s?|"
    r"da[nñ](?:o|ad[oa]s?)|averiad[oa]s?|fall(?:a|ando)|lent[oa]|ruido|zumbido|recalient[ae]|"
    r"parpade[ae]|enciend[ae]|prend[ae]|funcion[ae]|carg[ae]|imprim[ae]|arranc[ae]|sirv[ae]|"
    r"trabad[oa]|trancad[oa]|bloquead[oa]|despegad[oa]|hundid[oa]|torcid[oa]|h[uú]med[oa]|"
    r"mojad[oa]|fisura|grieta|cortad[oa]|pelad[oa]|sulfatad[oa]|inflad[oa]|agotad[oa]|"
    r"golpe(?:s|ad[oa]s?)?|falt(?:a|an|ante)|sin\s+(?:pata|cable|tapa|pantalla|etiqueta|c[oó]digo|placa|llave|rueda|fuente|cargador|bater[ií]a)|"
    r"cambi(?:o|aron|ad[oa])|modificad[oa]|reemplazad[oa]|reparaci[oó]n|mantenimiento|baja|"
    r"pata(?:s)?|rueda(?:s)?|espaldar|respaldo|asiento|caj[oó]n|puerta|vidrio|tabla|borde|esquina|"
    r"superficie|madera|metal|pl[aá]stico|etiqueta|placa|c[oó]digo|serie|marca|modelo|pantalla|"
    r"monitor|teclado|tecla(?:s)?|mouse|rat[oó]n|cable(?:s)?|cargador|fuente|bater[ií]a|disco|"
    r"memoria|bot[oó]n|bisagra(?:s)?|tornillo(?:s|ito)?|tuerca|soporte|brazo|foco|lente|l[aá]mpara|"
    r"filtro|bandeja|t[oó]ner|tinta|carcasa|tapa|enchufe|motor|compresor|aspa(?:s)?|ventilador|"
    r"cerradura|chapa|tapiz|cuero|tela|forro|base|estructura|marco|cristal|espejo|tubo|manguera|"
    r"v[aá]lvula|perilla|interruptor|conector|puerto|usb|hdmi|antena|parlante|bocina|c[aá]mara|"
    r"mesa|silla|escritorio|pupitre|estante|armario|mueble|computadora|laptop|cpu|proyector|impresora|"
    r"televisor|aire|ups|estabilizador|veh[ií]culo|camioneta|"
    r"custodi[oa]|ubicaci[oó]n|oficina|aula|laboratorio|dep[oó]sito|almac[eé]n|pr[eé]stamo|"
    r"prestad[oa]|trasladad[oa]|sobrante|duplicad[oa]|inventario|cat[aá]logo|planilla|"
    r"baila|tambale[ae]|coje[ae]|chispa|humo|fuga|gote[ae]|novedad(?:es)?"
    r")\b",
    re.IGNORECASE,
)

# Indicadores léxicos de conversación personal, doméstica o de pasillo ajena al activo.
_CHARLA_EXTERNA_RE = re.compile(
    r"\b(?:"
    r"almorz(?:ar|amos)|almuerzo|comer|desayun(?:ar|o)|cenar|cena|hambre|sed|"
    r"hace\s+fr[ií]o|hace\s+calor|entra\s+(?:viento|fr[ií]o)|lloviendo|lluvia|clima|"
    r"qu[eé]\s+hora|hora\s+es|vamos\s+a\s+ir|me\s+voy|te\s+vas|ya\s+vengo|ven\s+aqu[ií]|"
    r"p[aá]s(?:ame|eme)|pr[eé]st(?:ame|eme)|tr[aá](?:eme|igame)|ll[eé]vame|ll[aá]mame|"
    r"whatsapp|mi\s+celular|tu\s+celular|mi\s+tel[eé]fono|mi\s+mochila|tu\s+mochila|"
    r"llave\s+del\s+auto|mi\s+auto|mi\s+moto|al\s+ba[nñ]o|caf[eé]|refresco|"
    r"examen|tarea|reuni[oó]n|sueldo|vacaci[oó]n|mam[aá]|pap[aá]|hij[oa]|novi[oa]|espos[oa]|"
    r"doctor|hospital|permiso|disculp(?:a|e)|perd[oó]n|esp[eé]r(?:ame|e)|un\s+ratito|un\s+segundo|"
    r"un\s+momento|al[oó]+|hola+|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|"
    r"c[oó]mo\s+est[aá]s?|qu[eé]\s+tal|chau|hasta\s+luego|nos\s+vemos|"
    r"ci[eé]rr(?:ame|a)\s+la\s+(?:puerta|ventana)|abre\s+la\s+(?:puerta|ventana)"
    r")\b",
    re.IGNORECASE,
)

_PREFIJOS_ESTADO_RE = re.compile(
    r"^(?:"
    r"(?:eh+|este+|pues|mira+|a\s+ver|o\s+sea|digamos|f[ií]jate(?:\s+que)?|creo\s+que|parece\s+que|"
    r"me\s+parece\s+que|b[aá]sicamente|la\s+verdad|el\s+activo|este\s+bien|el\s+equipo|el\s+mueble)\s+|"
    r"(?:est[aá]\s+)?(?:en\s+)?(?:muy\s+)?(?:buen\s+estado|buen[oa]|regular|mal\s+estado|mal[oa]|"
    r"perfect[oa]|operativ[oa]|todo\s+bien|sin\s+novedad(?:es)?)"
    r"(?:\s*(?:,|y|pero|aunque|con|porque|ya\s+que|debido\s+a\s+que)\s*)?"
    r")+",
    re.IGNORECASE,
)

_SUFIJOS_CIERRE_RE = re.compile(
    r"(?:^|\s*[,.;]?\s*)(?:"
    r"nada\s+m[aá]s|eso\s+(?:es|ser[ií]a)\s+todo|no\s+hay\s+m[aá]s|sin\s+observaci[oó]n(?:es)?|"
    r"ninguna\s+observaci[oó]n|sin\s+cambios|tal\s+cual|"
    r"env[ií](?:ar|a|o|alo|ala|ame)(?:lo|la)?|mand(?:ar|a|o|alo|ala)(?:lo|la)?|"
    r"acept(?:ar|a|o|alo|ala)|confirm(?:ar|a|o|alo|ala)|guard(?:ar|a|o|alo|ala)|"
    r"agreg(?:ar|a|o|alo|ala)|registr(?:ar|a|o|alo|ala)|sub(?:ir|e|a|elo|irlo)(?:lo|la)?|"
    r"as[ií]\s+d[eé]jalo|d[eé]jalo\s+as[ií]|as[ií]\s+est[aá]\s+bien|est[aá]\s+bien\s+as[ií]|"
    r"todo\s+correcto|est[aá]\s+correcto|ya\s+qued[oó]|"
    r"listo|conforme|de\s+acuerdo|adelante|dale|gracias|por\s+favor|nom[aá]s|de\s+una\s+vez|ok(?:ay|ey)?|"
    r"hermano|ingenier[oa]|jefe"
    r")\.?\s*$",
    re.IGNORECASE,
)

_CONECTORES_INICIALES_RE = re.compile(
    r"^(?:observaci[oó]n\s*:?\s*|nota\s*:?\s*|pon(?:le|er)?\s+que\s+|corregir\s*:?\s*|"
    r"que\s+|y\s+|pero\s+|con\s+(?=le\s+)|,\s*)+",
    re.IGNORECASE,
)

_MULETILLAS_INTERNAS_RE = re.compile(
    r"\b(?:eh+|este+|pues|o\s+sea|digamos|f[ií]jate\s+que|creo\s+que|parece\s+que|"
    r"b[aá]sicamente|la\s+verdad|por\s+favor|nom[aá]s|hermano)\b",
    re.IGNORECASE,
)

_TOKENS_SOLO_CONFIRMACION = {
    "enviar",
    "envia",
    "envialo",
    "enviala",
    "enviarlo",
    "mandar",
    "manda",
    "mandalo",
    "mandala",
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
    "subir",
    "sube",
    "subelo",
    "proceder",
    "procede",
    "si",
    "listo",
    "ok",
    "okay",
    "okey",
    "dale",
    "perfecto",
    "conforme",
    "adelante",
    "correcto",
    "ya",
    "asi",
    "bien",
    "esta",
    "todo",
    "nomas",
    "favor",
    "por",
    "de",
    "una",
    "vez",
    "hermano",
    "jefe",
    "gracias",
}

_FRASES_CONFIRMACION_COMPUESTAS_RE = re.compile(
    r"\b(?:de\s+acuerdo|as[ií]\s+est[aá]\s+bien|est[aá]\s+bien(?:\s+as[ií])?|as[ií]\s+d[eé]jalo|"
    r"d[eé]jalo\s+as[ií]|todo\s+correcto|est[aá]\s+correcto|ya\s+est[aá]|ya\s+qued[oó]|"
    r"nada\s+m[aá]s|eso\s+(?:es|ser[ií]a)\s+todo|no\s+hay\s+m[aá]s|sin\s+cambios|tal\s+cual|"
    r"este\s+bien|por\s+favor|de\s+una\s+vez)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContextoInspeccionVoz:
    etapa_modal: EtapaModalVoz = "dictado_inicial"
    activo_codigo: str | None = None
    activo_descripcion: str | None = None
    estado_actual: EstadoFisicoInspeccion | None = None
    observacion_actual: str | None = None


@dataclass(frozen=True)
class ResultadoInterpretacionInspeccion:
    estado_fisico: EstadoFisicoInspeccion | None
    observacion: str
    accion: AccionInspeccionVoz
    confirma_directo: bool
    motor_interpretacion: Literal["reglas_rapidas", "llm_local"]


def _coincide_alguno(texto_norm: str, patrones: tuple[str, ...]) -> bool:
    return any(re.search(pat, texto_norm, flags=re.IGNORECASE) for pat in patrones)


def reparar_fonetica_asr(texto: str, activo_descripcion: str | None = None) -> str:
    """
    Herramienta 1 (ASR Predictivo): Corrige errores fonéticos comunes en campo
    apoyándose en el tipo de bien inspeccionado cuando está disponible.
    """
    reparado = re.sub(r"\s+", " ", (texto or "")).strip()
    if not reparado:
        return ""

    reemplazos_generales: tuple[tuple[str, str], ...] = (
        (r"\b(?:en\s+)?(?:buen|ben|guen)\s+est(?:a|ao)\b", "en buen estado"),
        (r"\b(?:en\s+)?mal\s+est(?:a|ao)\b", "en mal estado"),
        (r"\bpa\s+baja\b", "para baja"),
        (r"\b(?:en\s+biar|embiar|inviar)(lo|la)?\b", r"enviar\1"),
        (r"\b(?:haceptar|aseptar|ha\s+aceptar)\b", "aceptar"),
        (r"\brallad([oa]s?)\b", r"rayad\1"),
        (r"\bralladura(s?)\b", r"rayadura\1"),
        (r"\bdescolorad([oa])\b", r"descolorid\1"),
        (r"\bsi\s+novedad\b", "sin novedad"),
    )
    for patron, sustituto in reemplazos_generales:
        reparado = re.sub(patron, sustituto, reparado, flags=re.IGNORECASE)

    desc_norm = normalizar_texto(activo_descripcion or "")
    es_mobiliario = any(
        k in desc_norm for k in ("mesa", "silla", "escritorio", "pupitre", "estante", "banco", "mueble", "armario")
    )
    es_equipo = any(
        k in desc_norm
        for k in ("computadora", "laptop", "cpu", "monitor", "proyector", "impresora", "televisor", "ups", "aire")
    )

    if es_mobiliario or not desc_norm:
        reparado = re.sub(r"\b(falta\s+(?:una\s+)?|sin\s+)pasta\b", r"\1pata", reparado, flags=re.IGNORECASE)
    if es_equipo or not desc_norm:
        reparado = re.sub(r"\bno\s+(?:aprende|emprende)\b", "no enciende", reparado, flags=re.IGNORECASE)

    return reparado


def _tiene_anclaje_activo(texto_norm: str, activo_descripcion: str | None = None) -> bool:
    if _ANCLAS_PATRIMONIALES_RE.search(texto_norm):
        return True
    if activo_descripcion:
        tokens_activo = [
            t for t in re.split(r"[\s,.;:/\-]+", normalizar_texto(activo_descripcion)) if len(t) >= 4
        ]
        if any(re.search(rf"\b{re.escape(t)}\b", texto_norm) for t in tokens_activo):
            return True
    return False


def filtrar_saludo_y_ruido(
    texto: str,
    activo_descripcion: str | None = None,
) -> tuple[str, bool]:
    """
    Herramienta 2 (Anclaje Semántico Patrimonial y Filtro por Cláusulas):
    Descarta saludos, conversaciones personales o frases de terceros sin relación física
    con el activo; si una oración mezcla charla social con un dato técnico, conserva solo la cláusula técnica.
    """
    limpio = re.sub(r"\s+", " ", (texto or "")).strip().strip("¡¿")
    if not limpio:
        return "", True

    norm = normalizar_texto(limpio).strip("¡¿?!., ")
    if _CHARLA_IRRELEVANTE_COMPLETA_RE.match(norm):
        return "", True

    tiene_comando = (
        _coincide_alguno(norm, _PATRONES_CANCELAR)
        or _coincide_alguno(norm, _PATRONES_CONFIRMACION_EXPLICITA)
        or _coincide_alguno(norm, _PATRONES_ACUERDO_COLOQUIAL)
        or norm in {"si", "sip", "sep", "aja", "correcto", "perfecto", "exacto"}
    )

    # Si hay múltiples cláusulas separadas por coma/punto y coma, filtra las cláusulas puramente sociales.
    clausulas_brutas = [c.strip(" ,.;:-¡¿?!") for c in re.split(r"[,;]+|\.\s+", limpio) if c.strip(" ,.;:-¡¿?!")]
    if len(clausulas_brutas) > 1:
        clausulas_utiles: list[str] = []
        for cl in clausulas_brutas:
            cl_norm = normalizar_texto(cl).strip("¡¿?!., ")
            if not cl_norm or _CHARLA_IRRELEVANTE_COMPLETA_RE.match(cl_norm):
                continue
            es_charla_cl = bool(_CHARLA_EXTERNA_RE.search(cl_norm))
            tiene_ancla_cl = _tiene_anclaje_activo(cl_norm, activo_descripcion)
            tiene_cmd_cl = (
                _coincide_alguno(cl_norm, _PATRONES_CANCELAR)
                or _coincide_alguno(cl_norm, _PATRONES_CONFIRMACION_EXPLICITA)
                or _coincide_alguno(cl_norm, _PATRONES_ACUERDO_COLOQUIAL)
            )
            if es_charla_cl and not tiene_ancla_cl and not tiene_cmd_cl:
                continue
            clausulas_utiles.append(cl)
        if not clausulas_utiles:
            return "", True
        limpio = ", ".join(clausulas_utiles)

    # Desprende saludos iniciales residuales ("buenos días, la silla..." -> "la silla...").
    sin_saludo = limpio
    for _ in range(2):
        nuevo = _SALUDO_INICIAL_RE.sub("", sin_saludo).strip(" ,.;:-¡¿?!")
        if nuevo == sin_saludo:
            break
        sin_saludo = nuevo

    if not sin_saludo:
        return "", True

    norm_restante = normalizar_texto(sin_saludo).strip("¡¿?!., ")
    if _CHARLA_IRRELEVANTE_COMPLETA_RE.match(norm_restante):
        return "", True

    # Si contiene marcadores de charla doméstica/pasillo y CERO anclaje patrimonial ni comando -> ignorar.
    tiene_ancla = _tiene_anclaje_activo(norm_restante, activo_descripcion)
    if _CHARLA_EXTERNA_RE.search(norm_restante) and not tiene_ancla and not tiene_comando:
        return "", True

    return sin_saludo, False


def _es_solo_confirmacion(norm: str) -> bool:
    if norm in _TOKENS_SOLO_CONFIRMACION and norm not in {"bueno", "bien", "correcto"}:
        return True
    sin_frases = _FRASES_CONFIRMACION_COMPUESTAS_RE.sub(" ", norm)
    tokens = [t for t in re.split(r"[\s,.;:!?]+", sin_frases) if t]
    if not tokens:
        return True
    restantes = [t for t in tokens if t not in {"y", "e"}]
    return bool(restantes) and all(t in _TOKENS_SOLO_CONFIRMACION for t in restantes)


def sintetizar_observacion_campo(texto: str) -> str:
    """
    Herramienta 3 (Resumidor Técnico): Limpia muletillas, prefijos de estado y
    comandos finales devolviendo una observación técnica breve y clara (<= 12 palabras).
    """
    limpio = re.sub(r"\s+", " ", texto).strip()
    limpio = _MULETILLAS_INTERNAS_RE.sub(" ", limpio)
    limpio = re.sub(r"\s+", " ", limpio).strip(" ,.;:-")

    for _ in range(5):
        antes = limpio
        sin_sufijo = _SUFIJOS_CIERRE_RE.sub("", limpio).strip(" ,.;:-")
        if sin_sufijo != limpio:
            limpio = re.sub(r"(?:^|\s+)[ye]$", "", sin_sufijo, flags=re.IGNORECASE).strip(" ,.;:-")
        nuevo = _PREFIJOS_ESTADO_RE.sub("", limpio).strip(" ,.;:-")
        nuevo = _CONECTORES_INICIALES_RE.sub("", nuevo).strip(" ,.;:-")
        limpio = re.sub(r"\s+", " ", nuevo).strip()
        if limpio == antes:
            break

    if not limpio:
        return ""

    norm = normalizar_texto(limpio)
    if norm in {
        "buen estado",
        "bueno",
        "buena",
        "regular",
        "mal estado",
        "malo",
        "mala",
        "sin observaciones",
        "sin novedad",
        "sin novedades",
        "ninguna",
        "todo bien",
    } or _es_solo_confirmacion(norm):
        return ""

    palabras = [p for p in limpio.split(" ") if p]
    if len(palabras) > MAX_PALABRAS_OBSERVACION:
        limpio = " ".join(palabras[:MAX_PALABRAS_OBSERVACION]).rstrip(" ,.;:-")

    return f"{limpio[0].upper()}{limpio[1:]}"


def _detectar_estado_fisico(norm: str) -> EstadoFisicoInspeccion | None:
    if _coincide_alguno(norm, _PATRONES_MALO):
        return "MALO"
    if _coincide_alguno(norm, _PATRONES_REGULAR):
        return "REGULAR"
    if _coincide_alguno(norm, _PATRONES_BUENO):
        return "BUENO"
    if _coincide_alguno(norm, _DANIO_ESTRUC_MALO):
        return "MALO"
    return None


def interpretar_inspeccion_rapida(
    texto: str,
    contexto: ContextoInspeccionVoz | None = None,
) -> ResultadoInterpretacionInspeccion:
    """
    Pipeline determinista de habilidades y herramientas móviles (< 2 ms):
    reparación fonética -> filtro de charla/ruido -> intención natural multi-palabra -> resumen corto.
    """
    ctx = contexto or ContextoInspeccionVoz()
    reparado = reparar_fonetica_asr(texto, ctx.activo_descripcion)
    sin_saludo, es_irrelevante = filtrar_saludo_y_ruido(reparado, ctx.activo_descripcion)
    if es_irrelevante or not sin_saludo:
        return ResultadoInterpretacionInspeccion(
            estado_fisico=None,
            observacion="",
            accion="ignorar",
            confirma_directo=False,
            motor_interpretacion="reglas_rapidas",
        )

    norm = normalizar_texto(sin_saludo)
    palabras_norm = [p for p in re.split(r"[\s,.;:!?]+", norm) if p]

    tiene_cancelar = _coincide_alguno(norm, _PATRONES_CANCELAR)
    if tiene_cancelar and len(palabras_norm) <= 10:
        sin_cancelar = norm
        for pat in _PATRONES_CANCELAR:
            sin_cancelar = re.sub(pat, " ", sin_cancelar, flags=re.IGNORECASE)
        if not sintetizar_observacion_campo(sin_cancelar):
            return ResultadoInterpretacionInspeccion(
                estado_fisico=None,
                observacion="",
                accion="cancelar",
                confirma_directo=False,
                motor_interpretacion="reglas_rapidas",
            )

    tiene_verbo_envio = _coincide_alguno(norm, _PATRONES_CONFIRMACION_EXPLICITA)
    tiene_acuerdo = _coincide_alguno(norm, _PATRONES_ACUERDO_COLOQUIAL)
    observacion = sintetizar_observacion_campo(sin_saludo)

    # En espera de confirmación, expresiones de acuerdo ("sí está bien, mándalo", "ok", "sí")
    # confirman los datos visibles sin exigir limitarse a 1-3 palabras ni sobreescribir estado previo.
    pide_cambio_estado_explicito = bool(
        re.search(r"\b(?:cambi(?:a|ar|alo)|pon(?:le|er)?|es\s+(?:regular|mal[oa]|buen[oa])|mal\s+estado|buen\s+estado|regular)\b", norm)
    )
    if ctx.etapa_modal == "esperando_confirmacion" and observacion == "":
        solo_afirmacion_etapa2 = (
            norm in {"si", "sip", "sep", "aja", "correcto", "perfecto", "exacto", "eso"}
            or tiene_verbo_envio
            or tiene_acuerdo
        )
        if solo_afirmacion_etapa2 and not pide_cambio_estado_explicito and len(palabras_norm) <= 18:
            return ResultadoInterpretacionInspeccion(
                estado_fisico=None,
                observacion="",
                accion="confirmar",
                confirma_directo=True,
                motor_interpretacion="reglas_rapidas",
            )

    estado = _detectar_estado_fisico(norm)

    # En dictado inicial, si solo dice "sí" sin verbo explícito ni datos, evita envíos accidentales.
    if ctx.etapa_modal == "dictado_inicial" and norm in {"si", "sip", "hola"}:
        return ResultadoInterpretacionInspeccion(
            estado_fisico=None,
            observacion="",
            accion="ignorar",
            confirma_directo=False,
            motor_interpretacion="reglas_rapidas",
        )

    # Confirmación pura multi-palabra sin dictado de nuevos hallazgos.
    mencion_estado_fisico = bool(
        re.search(
            r"\b(?:en\s+buen\s+estado|buen\s+estado|mal\s+estado|regular|buen[oa]|mal[oa]|operativ[oa]|normal|sin\s+novedad(?:es)?)\b",
            norm,
        )
    )
    es_confirmacion_pura = (
        observacion == ""
        and len(palabras_norm) <= 18
        and (
            (tiene_verbo_envio and not mencion_estado_fisico)
            or (tiene_acuerdo and estado is None)
        )
    )

    if es_confirmacion_pura:
        return ResultadoInterpretacionInspeccion(
            estado_fisico=None,
            observacion="",
            accion="confirmar",
            confirma_directo=True,
            motor_interpretacion="reglas_rapidas",
        )

    # Si no se detectó estado físico ni comando y la frase carece por completo de anclaje patrimonial,
    # se descarta como conversación ajena captada por el micrófono.
    if estado is None and not _tiene_anclaje_activo(norm, ctx.activo_descripcion):
        return ResultadoInterpretacionInspeccion(
            estado_fisico=None,
            observacion="",
            accion="ignorar",
            confirma_directo=False,
            motor_interpretacion="reglas_rapidas",
        )

    # Un token suelto incomprensible sin estado ni intención útil se ignora.
    if estado is None and len(palabras_norm) <= 1 and len(observacion) <= 3:
        return ResultadoInterpretacionInspeccion(
            estado_fisico=None,
            observacion="",
            accion="ignorar",
            confirma_directo=False,
            motor_interpretacion="reglas_rapidas",
        )

    return ResultadoInterpretacionInspeccion(
        estado_fisico=estado,
        observacion=observacion,
        accion="completar",
        confirma_directo=False,
        motor_interpretacion="reglas_rapidas",
    )


def _necesita_razonamiento_llm(
    texto: str,
    rapido: ResultadoInterpretacionInspeccion,
    contexto: ContextoInspeccionVoz,
) -> bool:
    palabras = [p for p in (texto or "").split() if p.strip()]
    if rapido.accion in ("confirmar", "cancelar"):
        return False
    # Si ya quedó ignorado por frase corta social claramente detectada, no gastar LLM.
    if rapido.accion == "ignorar":
        return len(palabras) >= 6
    # Frases ambiguas sin estado detectado y con suficiente contexto lingüístico o texto largo a resumir.
    if rapido.estado_fisico is None and len(palabras) >= 3:
        return True
    if len(palabras) >= 13 and len(rapido.observacion.split()) >= 10:
        return True
    return bool(
        contexto.etapa_modal == "esperando_confirmacion"
        and len(palabras) >= 3
        and rapido.estado_fisico is None
    )


async def interpretar_inspeccion_con_respaldo_ia(
    texto: str,
    settings: Settings,
    contexto: ContextoInspeccionVoz | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ResultadoInterpretacionInspeccion:
    """
    Ejecuta primero el motor de reglas/herramientas móviles (< 2 ms) y consulta el LLM local
    inyectando únicamente reglas generales + skill móvil `inspeccion_voz` en casos ambiguos o complejos.
    """
    ctx = contexto or ContextoInspeccionVoz()
    rapido = interpretar_inspeccion_rapida(texto, ctx)

    usar_llm = _necesita_razonamiento_llm(texto, rapido, ctx) and (
        settings.env != "test" or transport is not None
    )
    if not usar_llm:
        return rapido

    instruccion_movil = construir_instruccion_sistema("movil", ("inspeccion_voz",))
    resumen_ctx = (
        f"etapa={ctx.etapa_modal}|"
        f"activo={ctx.activo_descripcion or 'ACTIVO'} ({ctx.activo_codigo or 'S/C'})|"
        f"estado_actual={ctx.estado_actual or 'BUENO'}|"
        f"dicho={texto[:240]}"
    )

    llm = OpenAICompatibleLlmClient(settings, transport=transport)
    try:
        res = await llm.generar_chat_completado(
            mensajes=[
                {"role": "system", "content": instruccion_movil},
                {"role": "user", "content": resumen_ctx},
            ],
            timeout_override=6.5,
            max_tokens_override=40,
            temperature_override=0.0,
            response_format={"type": "json_object"},
        )
        parsed: dict[str, Any] | None = extraer_json_objeto(res.contenido)
        if parsed:
            acc_raw = str(parsed.get("a") or parsed.get("accion") or "").lower().strip()
            accion_llm: AccionInspeccionVoz = (
                acc_raw
                if acc_raw in ("completar", "confirmar", "cancelar", "ignorar")
                else rapido.accion
            )  # type: ignore[assignment]
            est_raw = str(parsed.get("e") or parsed.get("estado_fisico") or "").upper().strip()
            obs_bruta = str(parsed.get("o") or parsed.get("observacion") or "").strip()
            obs_limpia = sintetizar_observacion_campo(obs_bruta) if obs_bruta else ""
            estado_llm: EstadoFisicoInspeccion | None = (
                est_raw
                if est_raw in ("BUENO", "REGULAR", "MALO")
                else (_detectar_estado_fisico(normalizar_texto(obs_limpia)) or rapido.estado_fisico)
            )

            # Regla dura de seguridad: en dictado inicial con nuevos datos no se auto-envía de golpe.
            if (
                accion_llm == "confirmar"
                and ctx.etapa_modal == "dictado_inicial"
                and (estado_llm is not None or obs_limpia != "")
            ):
                accion_llm = "completar"

            if accion_llm in ("confirmar", "cancelar", "ignorar"):
                estado_llm = None
                obs_final = ""
            else:
                obs_final = obs_limpia or rapido.observacion

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
