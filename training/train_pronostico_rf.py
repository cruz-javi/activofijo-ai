"""Entrenamiento offline del ensamble Random Forest Regressor (`pronostico_rf_v1.joblib`).

Genera (si es necesario) los datasets sintéticos con Faker, extrae ventanas temporales
con lags y estacionalidad cíclica, evalúa métricas OOB y hold-out temporal (2025-2026)
y exporta el artefacto ligero comprimido a `activofijo-ai/models/pronostico_rf_v1.joblib`.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from generar_dataset_faker import generar_datasets_faker

# Permitir ejecución directa sin instalar el paquete en modo editable
RAIZ_AI = Path(__file__).resolve().parents[1]
SRC_AI = RAIZ_AI / "src"
if str(SRC_AI) not in sys.path:
    sys.path.insert(0, str(SRC_AI))

from activofijo_ai.modules.ml.random_forest_engine import (
    FEATURE_NAMES,
    RandomForestRegressorNumpy,
    construir_fila_feature,
    guardar_artefacto_rf,
)

OBJETIVOS: tuple[str, ...] = (
    "bajas_cant",
    "reposicion_bs",
    "altas_cant",
    "altas_monto_bs",
    "reasignaciones_cant",
    "revaluos_cant",
    "revaluos_monto_bs",
)


def cargar_series_csv(ruta_csv: Path) -> list[dict[str, float]]:
    filas: list[dict[str, float]] = []
    with ruta_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            filas.append(
                {
                    "gestion": int(row["gestion"]),
                    "mes": int(row["mes"]),
                    "altas_cant": float(row["altas_cant"]),
                    "altas_monto_bs": float(row["altas_monto_bs"]),
                    "bajas_cant": float(row["bajas_cant"]),
                    "bajas_monto_bs": float(row["bajas_monto_bs"]),
                    "reposicion_bs": float(row["reposicion_bs"]),
                    "reasignaciones_cant": float(row["reasignaciones_cant"]),
                    "revaluos_cant": float(row["revaluos_cant"]),
                    "revaluos_monto_bs": float(row["revaluos_monto_bs"]),
                    "ratio_desgaste": float(row["ratio_desgaste"]),
                }
            )
    return filas


def entrenar_y_exportar(
    forzar_regenerar_dataset: bool = True,
) -> dict[str, Any]:
    datasets_dir = RAIZ_AI / "training" / "datasets"
    ruta_series = datasets_dir / "series_mensuales_uagrm.csv"
    ruta_movs = datasets_dir / "movimientos_patrimoniales_faker.csv"

    if forzar_regenerar_dataset or not ruta_series.is_file() or not ruta_movs.is_file():
        ruta_movs, ruta_series = generar_datasets_faker(datasets_dir)

    filas = cargar_series_csv(ruta_series)
    modelos_dict: dict[str, Any] = {}
    medias_entrenamiento: dict[str, float] = {}
    metricas_validacion: dict[str, dict[str, float]] = {}

    for idx_obj, clave in enumerate(OBJETIVOS):
        serie = [float(f[clave]) for f in filas]
        X_list = []
        y_list = []
        gestiones = []

        for i in range(3, len(filas)):
            X_list.append(
                construir_fila_feature(
                    int(filas[i]["gestion"]),
                    int(filas[i]["mes"]),
                    serie[:i],
                    float(filas[i]["ratio_desgaste"]),
                )
            )
            y_list.append(serie[i])
            gestiones.append(int(filas[i]["gestion"]))

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)
        gest_arr = np.array(gestiones, dtype=int)

        # Validación temporal hold-out: entrenar 2017-2024 y evaluar 2025-2026
        mask_train = gest_arr <= 2024
        mask_test = gest_arr >= 2025

        rf_eval = RandomForestRegressorNumpy(
            n_estimators=100, max_depth=6, min_samples_split=4, min_samples_leaf=2, random_state=42 + idx_obj
        ).fit(X[mask_train], y[mask_train])
        preds_test = rf_eval.predict(X[mask_test])
        err_test = y[mask_test] - preds_test
        holdout_rmse = float(np.sqrt(np.mean(err_test * err_test)))
        ss_res = float(np.sum(err_test * err_test))
        ss_tot = float(np.sum((y[mask_test] - np.mean(y[mask_test])) ** 2))
        holdout_r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

        # Entrenamiento final sobre todo el histórico 2017-2026 para producción
        rf_full = RandomForestRegressorNumpy(
            n_estimators=100, max_depth=6, min_samples_split=4, min_samples_leaf=2, random_state=100 + idx_obj
        ).fit(X, y)

        modelos_dict[clave] = rf_full.to_dict()
        medias_entrenamiento[clave] = round(float(np.mean(y)), 4)
        metricas_validacion[clave] = {
            "oob_r2": round(rf_full.oob_r2, 4),
            "oob_rmse": round(rf_full.oob_rmse, 2),
            "oob_mae": round(rf_full.oob_mae, 2),
            "holdout_2025_2026_r2": round(holdout_r2, 4),
            "holdout_2025_2026_rmse": round(holdout_rmse, 2),
        }

    with ruta_movs.open("r", encoding="utf-8") as fh:
        total_movimientos = max(0, sum(1 for _ in fh) - 1)

    artefacto = {
        "version": "rf-patrimonio-v1.0",
        "algoritmo": "Random Forest Regressor (100 CART Bootstrap)",
        "entrenado_en": datetime.now(UTC).isoformat(),
        "n_estimators": 100,
        "feature_names": list(FEATURE_NAMES),
        "dataset_info": {
            "periodos_mensuales": len(filas),
            "movimientos_faker": total_movimientos,
            "gestiones": "2017-2026",
        },
        "medias_entrenamiento": medias_entrenamiento,
        "metricas_validacion": metricas_validacion,
        "modelos": modelos_dict,
    }

    ruta_modelo = RAIZ_AI / "models" / "pronostico_rf_v1.joblib"
    guardar_artefacto_rf(ruta_modelo, artefacto)

    resumen = {
        "modelo": str(ruta_modelo),
        "tamano_bytes": ruta_modelo.stat().st_size,
        "dataset_info": artefacto["dataset_info"],
        "metricas_validacion": metricas_validacion,
    }
    ruta_resumen = datasets_dir / "resumen_entrenamiento_rf.json"
    ruta_resumen.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumen


if __name__ == "__main__":
    resultado = entrenar_y_exportar(forzar_regenerar_dataset=True)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
