import pytest

from activofijo_ai.skills.registry import (
    cargar_catalogo_skills,
    construir_instruccion_sistema,
    listar_skills,
)


def test_registro_skills_movil_aislado_y_compacto() -> None:
    catalogo = cargar_catalogo_skills()
    assert "reglas_generales" in catalogo
    assert "inspeccion_voz" in catalogo
    assert "vision_etiquetas" in catalogo
    assert "conciliacion_campo" in catalogo

    movil = listar_skills("movil")
    ids_movil = {s.identificador for s in movil}
    assert ids_movil == {"inspeccion_voz", "vision_etiquetas", "conciliacion_campo"}

    prompt = construir_instruccion_sistema("movil", ("inspeccion_voz",))
    assert "Filtro de voz para inspección física" in prompt
    assert 'Responde SOLO JSON: {"a":' in prompt
    assert "cartas" not in prompt.lower()

    with pytest.raises(ValueError):
        construir_instruccion_sistema("web", ("inspeccion_voz",))
