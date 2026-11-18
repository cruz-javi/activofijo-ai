from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

SuperficieIa = Literal["general", "movil", "web"]

_RAIZ_SKILLS = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SkillEspecificacion:
    identificador: str
    superficie: SuperficieIa
    titulo: str
    contenido: str
    ruta: Path


_CATALOGO_DEFINIDO: tuple[tuple[str, SuperficieIa, str, str], ...] = (
    (
        "reglas_generales",
        "general",
        "Reglas Generales UAGRM",
        "general/reglas_generales.md",
    ),
    (
        "inspeccion_voz",
        "movil",
        "Interpretación de Voz en Inspección Móvil",
        "movil/inspeccion_voz.md",
    ),
)


@lru_cache(maxsize=1)
def cargar_catalogo_skills() -> dict[str, SkillEspecificacion]:
    """Lee una sola vez y cachea en RAM los archivos Markdown de reglas y skills."""
    catalogo: dict[str, SkillEspecificacion] = {}
    for ident, superficie, titulo, ruta_rel in _CATALOGO_DEFINIDO:
        ruta_abs = _RAIZ_SKILLS / ruta_rel
        contenido = ruta_abs.read_text(encoding="utf-8").strip()
        catalogo[ident] = SkillEspecificacion(
            identificador=ident,
            superficie=superficie,
            titulo=titulo,
            contenido=contenido,
            ruta=ruta_abs,
        )
    return catalogo


def listar_skills(superficie: SuperficieIa | None = None) -> list[SkillEspecificacion]:
    """Lista skills disponibles filtrados opcionalmente por superficie (`general`, `movil`, `web`)."""
    todos = list(cargar_catalogo_skills().values())
    if superficie is None:
        return todos
    return [s for s in todos if s.superficie == superficie]


def obtener_skill(identificador: str) -> SkillEspecificacion:
    catalogo = cargar_catalogo_skills()
    if identificador not in catalogo:
        raise KeyError(f"Skill IA desconocido: {identificador}")
    return catalogo[identificador]


def construir_instruccion_sistema(
    superficie: Literal["movil", "web"],
    skills_especificos: tuple[str, ...],
) -> str:
    """
    Ensambla un prompt de sistema estricto y compacto con las reglas generales
    y únicamente los skills habilitados para la superficie solicitada.
    Evita contaminar móvil con contexto web (cartas/reportes).
    """
    catalogo = cargar_catalogo_skills()
    bloques: list[str] = [catalogo["reglas_generales"].contenido]

    for ident in skills_especificos:
        spec = catalogo.get(ident)
        if spec is None:
            raise ValueError(f"El skill '{ident}' no está registrado.")
        if spec.superficie not in ("general", superficie):
            raise ValueError(
                f"El skill '{ident}' ({spec.superficie}) no puede cargarse en superficie '{superficie}'."
            )
        bloques.append(spec.contenido)

    return "\n\n".join(bloques)
