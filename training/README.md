# Entrenamiento Offline (`training/`)

Esta carpeta alojará los scripts reproducibles de entrenamiento y calibración fuera de línea (no se ejecutan en el contenedor de producción EC2):

1. `train_priorizador_cart_lgbm.py` *(pendiente Sprint 3)*: entrenamiento de Árboles de Decisión (CART) y LightGBM para priorización normativa (`HU2-07`, `HU2-14`, `HU3-20`), exportando artefactos `.joblib` y reglas de corte explicables a `../models/`.
2. `train_pronostico_rf.py` *(pendiente Sprint 3)*: entrenamiento de *Random Forest Regressor* sobre series temporales de bajas y reposición patrimonial con validación temporal ($RMSE$ y $R^2$).
3. `export_crnn_onnx.py` *(pendiente Sprint 3)*: exportación a ONNX Runtime ligero para extracción óptica documental en Web (`HU2-14`).

> Para instalar dependencias científicas de entrenamiento localmente:
> `uv pip install -e ".[dev,ml]"`

