# Entrenamiento Offline (`training/`)

Esta carpeta aloja los scripts reproducibles de generación de datasets sintéticos con **Faker** y entrenamiento fuera de línea para `activofijo-ai`:

1. `generar_dataset_faker.py`: genera con `Faker("es_ES")` y parámetros contables UAGRM (`D.S. 24051` / `R.R. 449/2013` / `R.R. 567/2018`):
   - `training/datasets/movimientos_patrimoniales_faker.csv` (movimientos individuales de altas, bajas, reasignaciones y revalúos 2017–2026),
   - `training/datasets/series_mensuales_uagrm.csv` (120 períodos mensuales consolidados).
2. `train_pronostico_rf.py`: entrena los ensambles **Random Forest Regressor** (`n_estimators=100`, muestreo Bootstrap, submuestreo `sqrt` e importancia por reducción de varianza), valida métricas OOB y hold-out temporal (`2025-2026`), y exporta el artefacto ligero `models/pronostico_rf_v1.joblib`.
3. `train_priorizador_cart_lgbm.py` / `export_crnn_onnx.py` *(pendientes Sprint 3)*.

> Para ejecutar la generación y reentrenamiento localmente:
> ```bash
> .venv/Scripts/python.exe training/train_pronostico_rf.py
> ```
