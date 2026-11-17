"""Generador de datasets patrimoniales realistas de la UAGRM usando Faker.

Produce dos archivos en `activofijo-ai/training/datasets/`:
1. `movimientos_patrimoniales_faker.csv`: movimientos individuales (altas, bajas,
   reasignaciones y revalúos) de bienes bajo grupos contables D.S. 24051 / R.R. 567/2018.
2. `series_mensuales_uagrm.csv`: serie histórica mensual consolidada (2017-2026) lista
   para entrenar y validar el ensamble Random Forest Regressor.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from faker import Faker


@dataclass(frozen=True)
class GrupoContableUagrm:
    cod_grupo: int
    des_grupo: str
    vida_util_meses: int
    costo_min_bs: float
    costo_max_bs: float
    prefijo_codigo: str
    ejemplos_bienes: tuple[str, ...]


GRUPOS_UAGRM: tuple[GrupoContableUagrm, ...] = (
    GrupoContableUagrm(
        cod_grupo=2,
        des_grupo="Muebles y Enseres",
        vida_util_meses=120,
        costo_min_bs=650.0,
        costo_max_bs=7500.0,
        prefijo_codigo="U02",
        ejemplos_bienes=(
            "Escritorio ejecutivo metálico",
            "Juego de pupitres universitarios",
            "Estante archivero de acero",
            "Sillón ergonómico administrativo",
            "Mesa de sesiones institucional",
        ),
    ),
    GrupoContableUagrm(
        cod_grupo=3,
        des_grupo="Maquinaria y Equipos",
        vida_util_meses=96,
        costo_min_bs=3500.0,
        costo_max_bs=68000.0,
        prefijo_codigo="U03",
        ejemplos_bienes=(
            "Espectrofotómetro de laboratorio",
            "Bomba centrífuga industrial",
            "Aire acondicionado Split 24000 BTU",
            "Generador eléctrico de respaldo",
            "Centrífuga analítica UTALAB",
        ),
    ),
    GrupoContableUagrm(
        cod_grupo=4,
        des_grupo="Equipos de Computación",
        vida_util_meses=48,
        costo_min_bs=2800.0,
        costo_max_bs=24000.0,
        prefijo_codigo="U04",
        ejemplos_bienes=(
            "Computadora Desktop Core i7",
            "Laptop institucional Latitude",
            "Proyector multimedia aula",
            "Servidor Rack de facultad",
            "Impresora multifuncional red",
        ),
    ),
    GrupoContableUagrm(
        cod_grupo=5,
        des_grupo="Vehículos Automotores",
        vida_util_meses=60,
        costo_min_bs=85000.0,
        costo_max_bs=340000.0,
        prefijo_codigo="U05",
        ejemplos_bienes=(
            "Camioneta doble cabina 4x4",
            "Microbús transporte estudiantil",
            "Vagoneta oficial de comisión",
        ),
    ),
    GrupoContableUagrm(
        cod_grupo=7,
        des_grupo="Semovientes Reproductores",
        vida_util_meses=96,
        costo_min_bs=6000.0,
        costo_max_bs=24000.0,
        prefijo_codigo="U07",
        ejemplos_bienes=(
            "Toro reproductor Nelore Pedigree",
            "Vaquilla lechera Holstein F.C.V.",
            "Semental equino Centro El Prado",
        ),
    ),
    GrupoContableUagrm(
        cod_grupo=9,
        des_grupo="Herramientas y Equipo Técnico",
        vida_util_meses=48,
        costo_min_bs=700.0,
        costo_max_bs=12500.0,
        prefijo_codigo="U09",
        ejemplos_bienes=(
            "Estación total topográfica",
            "Osciloscopio digital de taller",
            "Taladro percutor industrial DMMU",
            "Kit calibración eléctrica",
        ),
    ),
)

UNIDADES_UAGRM: tuple[str, ...] = (
    "Facultad de Ing. en Ciencias de la Computación (FICCT)",
    "Facultad de Ciencias Exactas y Tecnología",
    "Facultad de Ciencias Económicas y Empresariales",
    "Facultad de Ciencias Veterinarias (F.C.V.)",
    "Centro Experimental El Prado (F.C.V.)",
    "Centro Experimental Yabaré (F.C.V.)",
    "Dirección Administrativa y Financiera (DAF)",
    "Departamento de Activo Fijo Central",
    "Facultad de Medicina Humana",
    "Dirección de Tecnologías de la Información (DTI)",
)

CAUSALES_BAJA: tuple[str, ...] = (
    "DETERIORO_TOTAL (R.R. 449/2013 Art. 10.1)",
    "OBSOLESCENCIA_TECNOLOGICA (R.R. 449/2013 Art. 10.4)",
    "DESMANTELAMIENTO_PARTES (R.R. 449/2013 Art. 10.2)",
    "FINES_DIDACTICOS (R.R. 449/2013 Art. 10.5)",
    "DESCARTE_ZOOTECNICO (R.R. 598/2014 Art. 6.I)",
)


def generar_datasets_faker(
    output_dir: Path | None = None,
    seed: int = 2026,
    anios: tuple[int, ...] = tuple(range(2017, 2027)),
) -> tuple[Path, Path]:
    """Genera los datasets sintéticos realistas en CSV y devuelve sus rutas."""
    base_dir = output_dir or (Path(__file__).resolve().parent / "datasets")
    base_dir.mkdir(parents=True, exist_ok=True)

    ruta_movs = base_dir / "movimientos_patrimoniales_faker.csv"
    ruta_series = base_dir / "series_mensuales_uagrm.csv"

    fake = Faker("es_ES")
    Faker.seed(seed)
    rng = random.Random(seed)

    movimientos_rows: list[list[object]] = []
    series_rows: list[list[object]] = []

    mov_seq = 1
    baja_prev_1 = 8.0
    baja_prev_2 = 7.0
    alta_prev_1 = 16.0

    for anio_idx, anio in enumerate(anios):
        # Crecimiento institucional moderado con convergencia presupuestaria (realistic stationary regime)
        factor_gestion = 1.0 + 0.012 * anio_idx

        for mes in range(1, 13):
            # Picos reales universitarios: programación POA (mar-abr), inventarios (ago-oct), cierre (nov-dic)
            angulo = 2.0 * math.pi * (mes / 12.0)
            estacional_altas = 1.0 + 0.25 * math.sin(angulo - 0.5) + (0.18 if mes in (3, 4, 10, 11) else 0.0)
            estacional_bajas = 1.0 + 0.30 * math.sin(angulo + 0.4) + (0.22 if mes in (3, 9, 11, 12) else 0.0)

            ratio_desgaste = min(
                0.62,
                0.24 + 0.014 * anio_idx + 0.06 * math.sin(angulo) + rng.uniform(-0.01, 0.01),
            )

            # Dinámica autorregresiva acotada + desgaste del padrón para que Random Forest generalice entre gestiones
            bajas_cant = max(
                3,
                round(
                    (2.4 + 0.32 * baja_prev_1 + 0.14 * baja_prev_2 + 16.0 * ratio_desgaste)
                    * estacional_bajas
                    + rng.gauss(0.0, 0.7)
                ),
            )

            altas_cant = max(
                6,
                round(
                    (6.5 + 0.38 * alta_prev_1 + 0.55 * bajas_cant)
                    * estacional_altas
                    * factor_gestion
                    + rng.gauss(0.0, 1.1)
                ),
            )
            reasignaciones_cant = max(
                6,
                round((9.0 + 0.72 * altas_cant + 0.4 * bajas_cant) + rng.gauss(0.0, 1.4)),
            )
            revaluos_cant = max(
                2,
                round(
                    (1.5 + 11.0 * ratio_desgaste + (2.0 if mes in (5, 6, 11) else 0.0))
                    + rng.gauss(0.0, 0.6)
                ),
            )

            altas_monto_bs = 0.0
            bajas_monto_bs = 0.0
            reposicion_bs = 0.0
            revaluos_monto_bs = 0.0

            def _agregar_mov(
                tipo: str,
                grupo: GrupoContableUagrm,
                monto: float,
                antiguedad: int,
                cond: str,
                causal: str,
                anio_mov: int = anio,
                mes_mov: int = mes,
            ) -> None:
                nonlocal mov_seq
                dia = rng.randint(1, 28)
                fecha_str = f"{anio_mov}-{mes_mov:02d}-{dia:02d}"
                cod_activo = f"{grupo.prefijo_codigo}{anio_mov % 100:02d}{mov_seq:04d}"
                unidad = rng.choice(UNIDADES_UAGRM)
                responsable = fake.name()
                doc = f"{tipo[:3]}-{anio_mov}-{rng.randint(100, 999)}/{fake.bothify(text='??##').upper()}"
                movimientos_rows.append(
                    [
                        mov_seq,
                        fecha_str,
                        anio_mov,
                        mes_mov,
                        cod_activo,
                        grupo.cod_grupo,
                        grupo.des_grupo,
                        rng.choice(grupo.ejemplos_bienes),
                        unidad,
                        responsable,
                        doc,
                        tipo,
                        cond,
                        grupo.vida_util_meses,
                        antiguedad,
                        round(monto, 2),
                        causal,
                    ]
                )
                mov_seq += 1

            # Grupos mayoritarios con varianza realista (sin saltos aleatorios extremos en el agregado mensual)
            for idx_a in range(altas_cant):
                # Distribución representativa: computación, muebles, maquinaria y herramientas; vehículos solo planificados trimestralmente
                if idx_a == 0 and mes in (4, 9):
                    grp = GRUPOS_UAGRM[3]  # Vehículo planificado semestralmente
                else:
                    grp = rng.choices(
                        (GRUPOS_UAGRM[0], GRUPOS_UAGRM[1], GRUPOS_UAGRM[2], GRUPOS_UAGRM[4], GRUPOS_UAGRM[5]),
                        weights=(28, 16, 40, 5, 11),
                        k=1,
                    )[0]
                monto = rng.uniform(grp.costo_min_bs * 0.9, grp.costo_min_bs * 1.85) * (1.0 + 0.015 * anio_idx)
                altas_monto_bs += monto
                _agregar_mov("ALTA", grp, monto, 0, "NUEVO", "INCORPORACION_POA")

            for _ in range(bajas_cant):
                grp = rng.choices(
                    (GRUPOS_UAGRM[0], GRUPOS_UAGRM[1], GRUPOS_UAGRM[2], GRUPOS_UAGRM[4], GRUPOS_UAGRM[5]),
                    weights=(25, 18, 44, 4, 9),
                    k=1,
                )[0]
                valor_original = rng.uniform(grp.costo_min_bs * 1.1, grp.costo_min_bs * 2.1)
                antiguedad = rng.randint(grp.vida_util_meses, grp.vida_util_meses + 48)
                valor_baja = round(max(1.0, valor_original * rng.uniform(0.03, 0.12)), 2)
                costo_reposicion = valor_original * (1.03 + 0.015 * anio_idx + 0.15 * ratio_desgaste)
                bajas_monto_bs += valor_baja
                reposicion_bs += costo_reposicion
                _agregar_mov(
                    "BAJA",
                    grp,
                    costo_reposicion,
                    antiguedad,
                    rng.choice(("MALO", "REGULAR")),
                    rng.choice(CAUSALES_BAJA),
                )

            for _ in range(reasignaciones_cant):
                grp = rng.choices(GRUPOS_UAGRM, weights=(30, 15, 42, 2, 3, 8), k=1)[0]
                monto_ref = rng.uniform(grp.costo_min_bs, grp.costo_min_bs * 1.8)
                antig = rng.randint(6, grp.vida_util_meses)
                _agregar_mov(
                    "REASIGNACION",
                    grp,
                    monto_ref,
                    antig,
                    rng.choice(("BUENO", "REGULAR")),
                    "ACTA_CUSTODIA_D.S.0181",
                )

            for _ in range(revaluos_cant):
                grp = rng.choices(
                    (GRUPOS_UAGRM[0], GRUPOS_UAGRM[1], GRUPOS_UAGRM[2], GRUPOS_UAGRM[5]),
                    weights=(28, 26, 34, 12),
                    k=1,
                )[0]
                incremento = rng.uniform(grp.costo_min_bs * 0.7, grp.costo_min_bs * 1.4) * (1.0 + 0.25 * ratio_desgaste)
                revaluos_monto_bs += incremento
                antig = rng.randint(grp.vida_util_meses, grp.vida_util_meses + 36)
                _agregar_mov(
                    "REVALUO",
                    grp,
                    incremento,
                    antig,
                    rng.choice(("BUENO", "REGULAR")),
                    "PERITAJE_MRVT_RR567_2018",
                )

            series_rows.append(
                [
                    anio,
                    mes,
                    altas_cant,
                    round(altas_monto_bs, 2),
                    bajas_cant,
                    round(bajas_monto_bs, 2),
                    round(reposicion_bs, 2),
                    reasignaciones_cant,
                    revaluos_cant,
                    round(revaluos_monto_bs, 2),
                    round(ratio_desgaste, 4),
                ]
            )

            baja_prev_2 = baja_prev_1
            baja_prev_1 = float(bajas_cant)
            alta_prev_1 = float(altas_cant)

    with ruta_movs.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "id_movimiento",
                "fecha",
                "gestion",
                "mes",
                "cod_activo",
                "cod_grupo",
                "des_grupo",
                "descripcion",
                "unidad_uagrm",
                "responsable_faker",
                "documento_respaldo",
                "tipo_movimiento",
                "condicion_fisica",
                "vida_util_meses",
                "antiguedad_meses",
                "monto_bs",
                "causal_normativa",
            ]
        )
        writer.writerows(movimientos_rows)

    with ruta_series.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "gestion",
                "mes",
                "altas_cant",
                "altas_monto_bs",
                "bajas_cant",
                "bajas_monto_bs",
                "reposicion_bs",
                "reasignaciones_cant",
                "revaluos_cant",
                "revaluos_monto_bs",
                "ratio_desgaste",
            ]
        )
        writer.writerows(series_rows)

    return ruta_movs, ruta_series


if __name__ == "__main__":
    movs_path, series_path = generar_datasets_faker()
    print(f"Datasets Faker generados:\n- {movs_path}\n- {series_path}")
