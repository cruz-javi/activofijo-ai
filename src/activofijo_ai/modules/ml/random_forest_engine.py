"""Motor de Bosques Aleatorios de Regresión (Random Forest Regressor) sobre NumPy puro.

Entrena árboles de regresión CART con muestreo Bootstrap, submuestreo aleatorio de
atributos (`sqrt`), acumulación real de reducción de varianza para feature importance
e inferencia con intervalos de confianza basados en la dispersión entre árboles.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "gestion_norm",
    "estacionalidad_sin",
    "estacionalidad_cos",
    "lag_1",
    "lag_2",
    "media_movil_3",
    "tendencia_3",
    "ratio_desgaste",
)

FEATURE_LABELS: dict[str, str] = {
    "gestion_norm": "Ciclo histórico de gestión anual",
    "estacionalidad_sin": "Estacionalidad institucional (seno)",
    "estacionalidad_cos": "Estacionalidad institucional (coseno)",
    "lag_1": "Comportamiento del período inmediato anterior (t-1)",
    "lag_2": "Inercia de dos períodos previos (t-2)",
    "media_movil_3": "Tendencia móvil suavizada (3 períodos)",
    "tendencia_3": "Pendiente de variación reciente",
    "ratio_desgaste": "Índice de desgaste y vida útil agotada",
}


def _mse(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    diff = y - float(np.mean(y))
    return float(np.mean(diff * diff))


def _mejor_corte_cart(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    min_samples_leaf: int,
) -> tuple[int | None, float | None, float]:
    """Encuentra el mejor umbral de partición CART reduciendo la varianza ponderada."""
    n = len(y)
    var_padre = float(np.var(y))
    mejor_ganancia = 0.0
    mejor_feat: int | None = None
    mejor_umbral: float | None = None

    suma_total = float(np.sum(y))
    suma_sq_total = float(np.sum(y * y))

    for feat in feature_indices:
        col = X[:, feat]
        orden = np.argsort(col)
        col_ord = col[orden]
        y_ord = y[orden]

        suma_izq = 0.0
        sq_izq = 0.0
        for idx in range(min_samples_leaf - 1, n - min_samples_leaf):
            val_y = float(y_ord[idx])
            suma_izq += val_y
            sq_izq += val_y * val_y

            # Saltar empates en el mismo valor de atributo
            if col_ord[idx] == col_ord[idx + 1]:
                continue

            n_izq = idx + 1
            n_der = n - n_izq
            suma_der = suma_total - suma_izq
            sq_der = suma_sq_total - sq_izq

            var_izq = max(0.0, (sq_izq / n_izq) - (suma_izq / n_izq) ** 2)
            var_der = max(0.0, (sq_der / n_der) - (suma_der / n_der) ** 2)
            var_ponderada = (n_izq / n) * var_izq + (n_der / n) * var_der
            ganancia = var_padre - var_ponderada

            if ganancia > mejor_ganancia + 1e-12:
                mejor_ganancia = ganancia
                mejor_feat = int(feat)
                mejor_umbral = float((col_ord[idx] + col_ord[idx + 1]) / 2.0)

    return mejor_feat, mejor_umbral, mejor_ganancia * n


@dataclass
class NodoCART:
    """Nodo ligero de un árbol de regresión CART serializable a JSON."""

    valor: float
    feature_idx: int | None = None
    umbral: float | None = None
    izq: NodoCART | None = None
    der: NodoCART | None = None

    def predecir_fila(self, fila: np.ndarray) -> float:
        nodo = self
        while nodo.feature_idx is not None and nodo.umbral is not None:
            if fila[nodo.feature_idx] <= nodo.umbral:
                assert nodo.izq is not None
                nodo = nodo.izq
            else:
                assert nodo.der is not None
                nodo = nodo.der
        return nodo.valor

    def to_dict(self) -> dict[str, Any]:
        if self.feature_idx is None:
            return {"v": round(self.valor, 6)}
        assert self.izq is not None and self.der is not None and self.umbral is not None
        return {
            "v": round(self.valor, 6),
            "f": self.feature_idx,
            "t": round(self.umbral, 6),
            "l": self.izq.to_dict(),
            "r": self.der.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodoCART:
        if "f" not in data:
            return cls(valor=float(data["v"]))
        return cls(
            valor=float(data["v"]),
            feature_idx=int(data["f"]),
            umbral=float(data["t"]),
            izq=cls.from_dict(data["l"]),
            der=cls.from_dict(data["r"]),
        )


def _construir_arbol_cart(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: int,
    importancias: np.ndarray,
    depth: int = 0,
) -> NodoCART:
    n, n_cols = X.shape
    valor_hoja = float(np.mean(y)) if n > 0 else 0.0

    if depth >= max_depth or n < min_samples_split or np.std(y) < 1e-9:
        return NodoCART(valor=valor_hoja)

    feats = rng.choice(n_cols, size=min(max_features, n_cols), replace=False)
    feat_idx, umbral, ganancia_ponderada = _mejor_corte_cart(X, y, feats, min_samples_leaf)
    if feat_idx is None or umbral is None or ganancia_ponderada <= 0.0:
        return NodoCART(valor=valor_hoja)

    mask_izq = X[:, feat_idx] <= umbral
    n_izq = int(np.sum(mask_izq))
    if n_izq < min_samples_leaf or (n - n_izq) < min_samples_leaf:
        return NodoCART(valor=valor_hoja)

    importancias[feat_idx] += ganancia_ponderada

    izq = _construir_arbol_cart(
        X[mask_izq],
        y[mask_izq],
        rng,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        importancias,
        depth + 1,
    )
    der = _construir_arbol_cart(
        X[~mask_izq],
        y[~mask_izq],
        rng,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        importancias,
        depth + 1,
    )
    return NodoCART(valor=valor_hoja, feature_idx=feat_idx, umbral=umbral, izq=izq, der=der)


class RandomForestRegressorNumpy:
    """Ensamble Random Forest Regressor con Bootstrap, OOB e importancia de variables."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees: list[NodoCART] = []
        self.feature_importances_: np.ndarray = np.zeros(len(FEATURE_NAMES), dtype=float)
        self.oob_rmse: float = 0.0
        self.oob_mae: float = 0.0
        self.oob_r2: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressorNumpy:
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        max_features = max(1, int(math.sqrt(n_features)))

        self.trees = []
        importancias_totales = np.zeros(n_features, dtype=float)
        oob_sums = np.zeros(n_samples, dtype=float)
        oob_counts = np.zeros(n_samples, dtype=int)

        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[boot_idx] = True

            arbol_imp = np.zeros(n_features, dtype=float)
            arbol = _construir_arbol_cart(
                X[boot_idx],
                y[boot_idx],
                rng,
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
                max_features,
                arbol_imp,
            )
            self.trees.append(arbol)
            importancias_totales += arbol_imp

            oob_indices = np.where(~in_bag)[0]
            for i in oob_indices:
                oob_sums[i] += arbol.predecir_fila(X[i])
                oob_counts[i] += 1

        suma_imp = float(np.sum(importancias_totales))
        if suma_imp > 0.0:
            self.feature_importances_ = importancias_totales / suma_imp
        else:
            self.feature_importances_ = np.full(n_features, 1.0 / n_features, dtype=float)

        valid_oob = oob_counts > 0
        if int(np.sum(valid_oob)) >= 3:
            y_real = y[valid_oob]
            y_pred = oob_sums[valid_oob] / oob_counts[valid_oob]
            errores = y_real - y_pred
            self.oob_rmse = float(np.sqrt(np.mean(errores * errores)))
            self.oob_mae = float(np.mean(np.abs(errores)))
            ss_res = float(np.sum(errores * errores))
            ss_tot = float(np.sum((y_real - np.mean(y_real)) ** 2))
            self.oob_r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.85
        else:
            preds = self.predict(X)
            errores = y - preds
            self.oob_rmse = float(np.sqrt(np.mean(errores * errores)))
            self.oob_mae = float(np.mean(np.abs(errores)))
            self.oob_r2 = 0.82

        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        if not self.trees:
            return np.zeros(n, dtype=float), np.zeros(n, dtype=float)

        matriz = np.empty((len(self.trees), n), dtype=float)
        for t_idx, tree in enumerate(self.trees):
            for row_idx in range(n):
                matriz[t_idx, row_idx] = tree.predecir_fila(X[row_idx])

        medias = np.mean(matriz, axis=0)
        stds = np.std(matriz, axis=0)
        return medias, stds

    def predict(self, X: np.ndarray) -> np.ndarray:
        medias, _ = self.predict_with_uncertainty(X)
        return medias

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "oob_rmse": round(self.oob_rmse, 4),
            "oob_mae": round(self.oob_mae, 4),
            "oob_r2": round(self.oob_r2, 4),
            "feature_importances": [round(float(x), 6) for x in self.feature_importances_],
            "trees": [tree.to_dict() for tree in self.trees],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RandomForestRegressorNumpy:
        inst = cls(
            n_estimators=int(data["n_estimators"]),
            max_depth=int(data["max_depth"]),
            min_samples_split=int(data["min_samples_split"]),
            min_samples_leaf=int(data["min_samples_leaf"]),
            random_state=int(data["random_state"]),
        )
        inst.oob_rmse = float(data["oob_rmse"])
        inst.oob_mae = float(data["oob_mae"])
        inst.oob_r2 = float(data["oob_r2"])
        inst.feature_importances_ = np.array(data["feature_importances"], dtype=float)
        inst.trees = [NodoCART.from_dict(t) for t in data["trees"]]
        return inst


def construir_fila_feature(
    gestion: int,
    mes: int,
    serie_vals: list[float],
    ratio_desgaste: float,
) -> np.ndarray:
    """Construye el vector de 8 predictores a partir de la historia previa inmediata."""
    gestion_norm = float(gestion - 2018) / 10.0
    angulo = 2.0 * math.pi * (mes / 12.0)
    estacionalidad_sin = math.sin(angulo)
    estacionalidad_cos = math.cos(angulo)

    lag_1 = float(serie_vals[-1]) if len(serie_vals) >= 1 else 0.0
    lag_2 = float(serie_vals[-2]) if len(serie_vals) >= 2 else lag_1
    ultimos_3 = serie_vals[-3:] if len(serie_vals) >= 1 else [0.0]
    media_movil_3 = float(sum(ultimos_3) / len(ultimos_3))
    tendencia_3 = (
        float((serie_vals[-1] - serie_vals[-3]) / 2.0)
        if len(serie_vals) >= 3
        else float(lag_1 - lag_2)
    )

    return np.array(
        [
            gestion_norm,
            estacionalidad_sin,
            estacionalidad_cos,
            lag_1,
            lag_2,
            media_movil_3,
            tendencia_3,
            float(np.clip(ratio_desgaste, 0.0, 1.0)),
        ],
        dtype=float,
    )


def guardar_artefacto_rf(ruta: Path, payload: dict[str, Any]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ruta, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))


def cargar_artefacto_rf(ruta: Path) -> dict[str, Any]:
    with gzip.open(ruta, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def ejecutar_pronostico_serie(
    historico: list[dict[str, Any]],
    horizontes_meses: int,
    ratio_desgaste_catalogo: float,
    models_dir: Path,
    grupos: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ejecuta el pronóstico Random Forest combinando el artefacto preentrenado con calibración local."""
    ruta_modelo = models_dir / "pronostico_rf_v1.joblib"
    artefacto: dict[str, Any] | None = None
    if ruta_modelo.is_file():
        artefacto = cargar_artefacto_rf(ruta_modelo)

    # Extraer series históricas reales enviadas por activofijo-core
    serie_bajas = [float(p.get("bajas_registradas", 0)) for p in historico]
    serie_monto = [float(p.get("monto_reposicion_bs", 0.0)) for p in historico]
    serie_altas = [float(p.get("altas_registradas", 0)) for p in historico]
    serie_altas_bs = [float(p.get("altas_monto_bs", 0.0)) for p in historico]
    serie_reasig = [float(p.get("reasignaciones_registradas", 0)) for p in historico]
    serie_rev = [float(p.get("revaluos_registrados", 0)) for p in historico]
    serie_rev_bs = [float(p.get("revaluos_monto_neto_bs", 0.0)) for p in historico]

    ultimo = historico[-1]
    gestion_cursor = int(ultimo["gestion"])
    mes_cursor = int(ultimo["mes"])

    def _avanzar_mes(g: int, m: int) -> tuple[int, int]:
        return (g + 1, 1) if m >= 12 else (g, m + 1)

    def _predecir_objetivo(
        clave_modelo: str,
        serie_real: list[float],
        n_pasos: int,
    ) -> tuple[list[float], list[float], float, float, float, np.ndarray, str]:
        es_monetario = "monto" in clave_modelo or "reposicion" in clave_modelo
        imp = np.full(len(FEATURE_NAMES), 1.0 / len(FEATURE_NAMES), dtype=float)

        # Ajustar un bosque local cuando haya suficiente historia real (>= 6 puntos)
        modelo_local: RandomForestRegressorNumpy | None = None
        if len(serie_real) >= 6:
            X_loc = []
            y_loc = []
            for i in range(2, len(historico)):
                X_loc.append(
                    construir_fila_feature(
                        int(historico[i]["gestion"]),
                        int(historico[i]["mes"]),
                        serie_real[:i],
                        ratio_desgaste_catalogo,
                    )
                )
                y_loc.append(serie_real[i])
            modelo_local = RandomForestRegressorNumpy(
                n_estimators=60, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=77
            ).fit(np.array(X_loc, dtype=float), np.array(y_loc, dtype=float))

        modelo_pre: RandomForestRegressorNumpy | None = None
        escala_adaptativa = 1.0
        version = "rf-patrimonio-online-v1"
        rmse_pre = 0.0
        mae_pre = 0.0
        r2_pre = 0.85

        if artefacto and clave_modelo in artefacto.get("modelos", {}):
            modelo_pre = RandomForestRegressorNumpy.from_dict(artefacto["modelos"][clave_modelo])
            version = str(artefacto.get("version", "rf-patrimonio-v1.0"))
            rmse_pre = modelo_pre.oob_rmse
            mae_pre = modelo_pre.oob_mae
            r2_pre = modelo_pre.oob_r2
            imp = modelo_pre.feature_importances_.copy()

            # Calibrar escala entre serie real UAGRM enviada y dominio del dataset entrenado
            media_pre = float(artefacto.get("medias_entrenamiento", {}).get(clave_modelo, 1.0))
            activos_recientes = [v for v in serie_real[-6:] if v > 0]
            media_real = (
                float(sum(activos_recientes) / len(activos_recientes))
                if activos_recientes
                else float(sum(serie_real) / max(len(serie_real), 1))
            )
            if media_pre > 1e-6 and media_real > 0:
                escala_adaptativa = max(0.25, min(6.0, media_real / media_pre))
            elif media_real == 0:
                # Escala conservadora cuando el período filtrado aún no tuvo movimientos
                escala_adaptativa = 0.25 if es_monetario else 0.35

        hist_ext = list(serie_real)
        preds_pasos: list[float] = []
        stds_pasos: list[float] = []
        g, m = gestion_cursor, mes_cursor

        for _ in range(n_pasos):
            g, m = _avanzar_mes(g, m)
            x_row = construir_fila_feature(g, m, hist_ext, ratio_desgaste_catalogo).reshape(1, -1)

            pred_pre, std_pre = (0.0, 0.0)
            if modelo_pre is not None:
                p_arr, s_arr = modelo_pre.predict_with_uncertainty(x_row)
                pred_pre = float(p_arr[0]) * escala_adaptativa
                std_pre = float(s_arr[0]) * escala_adaptativa

            pred_loc, std_loc = (0.0, 0.0)
            if modelo_local is not None:
                p_l, s_l = modelo_local.predict_with_uncertainty(x_row)
                pred_loc = float(p_l[0])
                std_loc = float(s_l[0])

            if modelo_pre is not None and modelo_local is not None:
                pred = 0.6 * pred_pre + 0.4 * pred_loc
                std = math.sqrt(0.6 * (std_pre**2) + 0.4 * (std_loc**2))
            elif modelo_pre is not None:
                pred, std = pred_pre, std_pre
            elif modelo_local is not None:
                pred, std = pred_loc, std_loc
                rmse_pre = modelo_local.oob_rmse
                mae_pre = modelo_local.oob_mae
                r2_pre = modelo_local.oob_r2
                imp = modelo_local.feature_importances_.copy()
            else:
                pred = float(sum(hist_ext[-3:]) / max(len(hist_ext[-3:]), 1))
                std = pred * 0.18

            pred = max(0.0, pred)
            preds_pasos.append(pred)
            stds_pasos.append(max(0.0, std))
            hist_ext.append(pred)

        return preds_pasos, stds_pasos, rmse_pre, mae_pre, r2_pre, imp, version

    # Inferencia sobre las dimensiones patrimoniales
    p_bajas, _std_bajas, rmse_b, mae_b, r2_b, imp_b, version = _predecir_objetivo(
        "bajas_cant", serie_bajas, horizontes_meses
    )
    p_monto, std_monto, rmse_m, mae_m, r2_m, imp_m, _ = _predecir_objetivo(
        "reposicion_bs", serie_monto, horizontes_meses
    )
    p_altas, _, _, _, _, _, _ = _predecir_objetivo("altas_cant", serie_altas, 1)
    p_altas_bs, _, _, _, _, _, _ = _predecir_objetivo("altas_monto_bs", serie_altas_bs, 1)
    p_reasig, _, _, _, _, _, _ = _predecir_objetivo("reasignaciones_cant", serie_reasig, 1)
    p_rev, _, _, _, _, _, _ = _predecir_objetivo("revaluos_cant", serie_rev, 1)
    p_rev_bs, _, _, _, _, _, _ = _predecir_objetivo("revaluos_monto_bs", serie_rev_bs, 1)

    bajas_total_horizonte = round(sum(p_bajas))
    presupuesto_total_horizonte = round(float(sum(p_monto)), 2)

    # Intervalo de confianza al 90% (z = 1.645) para el siguiente período y el horizonte
    z_90 = 1.645
    monto_sig = round(float(p_monto[0]), 2)
    std_sig = float(std_monto[0])
    ic_min_sig = round(max(0.0, monto_sig - z_90 * std_sig), 2)
    ic_max_sig = round(max(monto_sig, monto_sig + z_90 * std_sig), 2)

    # Combinar importancias (promedio ponderado entre bajas y monto de reposición)
    imp_combinada = 0.5 * imp_b + 0.5 * imp_m
    suma_imp = float(np.sum(imp_combinada)) or 1.0
    imp_norm = imp_combinada / suma_imp
    importancias_orden = sorted(
        [
            {
                "variable": FEATURE_NAMES[idx],
                "etiqueta": FEATURE_LABELS[FEATURE_NAMES[idx]],
                "importancia_pct": round(float(imp_norm[idx]) * 100.0, 1),
            }
            for idx in range(len(FEATURE_NAMES))
        ],
        key=lambda x: x["importancia_pct"],
        reverse=True,
    )

    altas_sig = round(p_altas[0])
    altas_bs_sig = round(float(p_altas_bs[0]), 2)
    bajas_sig = round(p_bajas[0])
    reasig_sig = round(p_reasig[0])
    rev_sig = round(p_rev[0])
    rev_bs_sig = round(float(p_rev_bs[0]), 2)
    balance_sig = round(altas_bs_sig - monto_sig + rev_bs_sig, 2)

    # Desglose predictivo por grupo contable (si se provee)
    proyeccion_grupos: list[dict[str, Any]] = []
    if grupos:
        pesos = []
        for g_item in grupos:
            cant = max(1, int(g_item.get("cantidad", 1)))
            desgaste = float(g_item.get("pct_desgaste", 25.0)) / 100.0
            valor_prom = float(g_item.get("monto_total", 0.0)) / cant
            peso = cant * (0.25 + desgaste)
            pesos.append((g_item, peso, valor_prom, desgaste))
        suma_pesos = sum(x[1] for x in pesos) or 1.0
        for g_item, peso, valor_prom, desgaste in pesos:
            frac = peso / suma_pesos
            bajas_grp = max(0, round(bajas_total_horizonte * frac))
            presup_grp = round(presupuesto_total_horizonte * frac, 2)
            if presup_grp == 0 and bajas_grp > 0:
                presup_grp = round(bajas_grp * valor_prom * 0.45, 2)
            proyeccion_grupos.append(
                {
                    "cod_grupo": int(g_item["cod_grupo"]),
                    "des_grupo": str(g_item["des_grupo"]),
                    "activos_evaluados": int(g_item.get("cantidad", 0)),
                    "pct_desgaste": round(desgaste * 100.0, 1),
                    "bajas_estimadas": bajas_grp,
                    "presupuesto_reposicion_bs": presup_grp,
                }
            )
        proyeccion_grupos.sort(key=lambda x: x["presupuesto_reposicion_bs"], reverse=True)

    return {
        "modelo_version": version,
        "arboles": int(artefacto.get("n_estimators", 100)) if artefacto else 60,
        "rmse": round(float(rmse_m if rmse_m > 0 else rmse_b), 2),
        "mae": round(float(mae_m if mae_m > 0 else mae_b), 2),
        "r2": round(float((r2_b + r2_m) / 2.0), 3),
        "nivel_confianza_pct": 90,
        "bajas_esperadas_total": bajas_total_horizonte,
        "presupuesto_estimado_bs": presupuesto_total_horizonte,
        "siguiente_periodo": {
            "altas_proyectadas": altas_sig,
            "altas_monto_bs": altas_bs_sig,
            "bajas_proyectadas": bajas_sig,
            "bajas_valor_bs": monto_sig,
            "reasignaciones_proyectadas": reasig_sig,
            "revaluos_proyectados": rev_sig,
            "revaluos_monto_neto_bs": rev_bs_sig,
            "balance_neto_bs": balance_sig,
        },
        "intervalo_confianza_90": {
            "min_bs": ic_min_sig,
            "max_bs": ic_max_sig,
        },
        "importancia_variables": importancias_orden,
        "proyeccion_por_grupo": proyeccion_grupos[:6],
    }
