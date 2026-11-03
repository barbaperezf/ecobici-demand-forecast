"""
Modelo de demanda: salidas por estación y hora (regresión).

Pasos (los usa notebooks/02_modelo.ipynb):
    1. construir_tabla()   tabla estación x fecha x hora con ceros incluidos
    2. split_por_fechas()  primeros ~80 % de los días para entrenar, resto prueba
    3. baseline_promedio() promedio histórico por estación, hora y tipo de día
    4. entrenar_lineal()   regresión lineal (vista en clase), con variables escaladas
    5. entrenar_mlp()      perceptrón multicapa (visto en clase), con early stopping
    6. buscar_hiperparametros() / entrenar_xgb()  XGBoost global
    7. metricas()          RMSE y MAE

Versión 2 (notebooks/03_modelo_v2.ipynb): agrega calendario (vacaciones SEP y
festivos) con agregar_calendario() y clima por hora con agregar_clima().

Horizonte de 1 día y validación en ventanas (notebooks/04_horizonte_y_ventanas.ipynb):
agregar_rezagos_1d() y ventanas_por_fechas().

Validación mes por mes (notebooks/06_validacion_mensual.ipynb): ventanas_mensuales().

Los ids de estación son el `ce` de los viajes (incluye pares como "271-272").
"""

import os

import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

FECHA_INICIO = "2026-03-01"   # el periodo del proyecto; descarta viajes con fechas raras
FECHA_FIN = "2026-08-31"      # último día del periodo del proyecto (None = sin límite); evita que meses
                              # posteriores que haya en el parquet (sep 2026) cambien los resultados de 02 a 05
FEATURES_CAT = ["estacion", "cp", "alcaldia"]
FEATURES_REZAGO = ["lag_7d", "media_4sem", "nivel_semana"]
FEATURES_NUM = ["hora", "dia_semana", "es_fin_de_semana", "lat", "lon", "capacidad"] + FEATURES_REZAGO
FEATURES = FEATURES_CAT + FEATURES_NUM
TARGET = "salidas"

# Versión 2: calendario escolar SEP 2025-2026 y clima
VACACIONES_SEP = [("2026-03-30", "2026-04-10"), ("2026-07-16", "2026-08-30")]   # Semana Santa y verano
FESTIVOS = ["2026-03-16", "2026-04-02", "2026-04-03", "2026-05-01"]   # Benito Juárez, Jueves y Viernes Santo, 1 de mayo
FEATURES_CALENDARIO = ["vacaciones_sep", "es_festivo"]
FEATURES_CLIMA = ["lluvia_mm", "temperatura"]
FEATURES_V2 = FEATURES + FEATURES_CALENDARIO + FEATURES_CLIMA

# Horizonte de 1 día: además de los rezagos de 7 días o más, lo que pasó ayer
FEATURES_REZAGO_1D = ["lag_1d", "nivel_ayer"]
FEATURES_1D = FEATURES + FEATURES_REZAGO_1D


# ---------------------------------------------------------------- datos

def cargar_validos() -> pd.DataFrame:
    """Viajes de 1 a 45 min dentro del periodo, solo con las columnas que se usan."""
    trips = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "trips_clean.parquet"),
        columns=["estacion_origen", "origen_dt", "hora_del_dia", "duracion_min"],
    )
    ok = (trips["duracion_min"] >= 1) & (trips["duracion_min"] <= 45)
    ok &= trips["origen_dt"] >= FECHA_INICIO
    if FECHA_FIN:
        ok &= trips["origen_dt"] < pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)
    ok &= trips["estacion_origen"].notna()
    return trips.loc[ok].reset_index(drop=True)


def cargar_estaciones() -> pd.DataFrame:
    geo = pd.read_csv(
        os.path.join(PROCESSED_DIR, "stations_geo.csv"),
        dtype={"ce": "string", "cp": "string"},
    )
    geo = geo.rename(columns={"ce": "estacion"})
    return geo[["estacion", "lat", "lon", "capacidad", "cp", "alcaldia"]].drop_duplicates("estacion")


def construir_tabla(validos: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Una fila por estación-fecha-hora, con `salidas` (cero si no hubo viajes).

    Cada estación se considera activa entre su primer y su último viaje en el
    periodo: así no se rellenan con ceros días en que la estación aún no existía.
    """
    v = validos.assign(
        estacion=validos["estacion_origen"].astype("string"),
        fecha=validos["origen_dt"].dt.normalize(),
        hora=validos["hora_del_dia"].astype(int),
    )
    conteo = v.groupby(["estacion", "fecha", "hora"]).size().rename(TARGET).reset_index()

    rango = conteo.groupby("estacion")["fecha"].agg(["min", "max"]).reset_index()
    dias = pd.DataFrame({"fecha": np.sort(conteo["fecha"].unique())})
    grid = rango.merge(dias, how="cross")
    grid = grid[(grid["fecha"] >= grid["min"]) & (grid["fecha"] <= grid["max"])]
    grid = grid[["estacion", "fecha"]].merge(pd.DataFrame({"hora": range(24)}), how="cross")

    tabla = grid.merge(conteo, on=["estacion", "fecha", "hora"], how="left")
    tabla[TARGET] = tabla[TARGET].fillna(0).astype("int32")

    tabla["dia_semana"] = tabla["fecha"].dt.dayofweek.astype("int8")
    tabla["es_fin_de_semana"] = (tabla["dia_semana"] >= 5).astype("int8")
    tabla = tabla.merge(estaciones, on="estacion", how="left")

    for c in FEATURES_CAT:
        tabla[c] = tabla[c].astype(object).where(tabla[c].notna(), "desconocido").astype(str).astype("category")
    tabla["hora"] = tabla["hora"].astype("int8")
    tabla = tabla.sort_values(["fecha", "estacion", "hora"]).reset_index(drop=True)
    return agregar_rezagos(tabla)


def agregar_rezagos(tabla: pd.DataFrame) -> pd.DataFrame:
    """Variables de contexto reciente, todas con al menos 7 días de rezago.

    - lag_7d:       salidas de la misma estación y hora, 7 días antes
    - media_4sem:   promedio de la misma estación, hora y día de la semana en las 4 semanas previas (7, 14, 21, 28 días)
    - nivel_semana: salidas por hora de la estación en la semana d-13 .. d-7 (su nivel de demanda reciente)

    Así el modelo "ve" si el nivel de demanda cambió (verano, vacaciones) sin usar datos que
    no se conocerían al predecir con una semana de anticipación. Se descartan las filas sin
    historia suficiente (primera semana de cada estación).
    """
    base = tabla[["estacion", "fecha", "hora", TARGET]]
    lags = []
    for dias in (7, 14, 21, 28):
        sh = base.assign(fecha=base["fecha"] + pd.Timedelta(days=dias)).rename(columns={TARGET: f"_l{dias}"})
        tabla = tabla.merge(sh, on=["estacion", "fecha", "hora"], how="left")
        lags.append(f"_l{dias}")
    tabla["lag_7d"] = tabla["_l7"]
    tabla["media_4sem"] = tabla[lags].mean(axis=1)
    tabla = tabla.drop(columns=lags)

    diario = base.groupby(["estacion", "fecha"], observed=True)[TARGET].mean().rename("_d").reset_index()
    diario = diario.sort_values(["estacion", "fecha"])
    diario["nivel_semana"] = diario.groupby("estacion", observed=True)["_d"].transform(
        lambda s: s.rolling(7, min_periods=3).mean().shift(7)
    )
    tabla = tabla.merge(diario[["estacion", "fecha", "nivel_semana"]], on=["estacion", "fecha"], how="left")
    tabla = tabla.dropna(subset=FEATURES_REZAGO)
    for c in FEATURES_REZAGO:
        tabla[c] = tabla[c].astype("float32")
    return tabla.reset_index(drop=True)


def agregar_calendario(tabla: pd.DataFrame) -> pd.DataFrame:
    """Marca los días de vacaciones escolares SEP (ciclo 2025-2026) y los festivos oficiales."""
    vac = np.zeros(len(tabla), dtype="int8")
    for ini, fin in VACACIONES_SEP:
        vac |= tabla["fecha"].between(ini, fin).to_numpy().astype("int8")
    tabla["vacaciones_sep"] = vac
    tabla["es_festivo"] = tabla["fecha"].isin(pd.to_datetime(FESTIVOS)).astype("int8")
    return tabla


def descargar_clima(inicio: str, fin: str, lat: float = 19.42, lon: float = -99.16) -> pd.DataFrame:
    """Lluvia (mm) y temperatura (°C) por hora en la zona de Ecobici, del histórico de Open-Meteo.

    Se guarda en data/processed/clima_cdmx_hora.csv y se reutiliza si ya existe.
    Las horas vienen en hora local de la CDMX, igual que los viajes.
    """
    ruta = os.path.join(PROCESSED_DIR, "clima_cdmx_hora.csv")
    if os.path.exists(ruta):
        return pd.read_csv(ruta, parse_dates=["fecha"])
    import requests

    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat, "longitude": lon, "start_date": inicio, "end_date": fin,
            "hourly": "precipitation,temperature_2m", "timezone": "America/Mexico_City",
        },
        timeout=60,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    dt = pd.to_datetime(h["time"])
    clima = pd.DataFrame({
        "fecha": dt.normalize(),
        "hora": dt.hour.astype("int8"),
        "lluvia_mm": h["precipitation"],
        "temperatura": h["temperature_2m"],
    })
    clima.to_csv(ruta, index=False)
    return clima


def agregar_clima(tabla: pd.DataFrame, clima: pd.DataFrame) -> pd.DataFrame:
    """Une el clima observado de esa fecha y hora (el mismo para todas las estaciones).

    Es el clima real, no un pronóstico: en uso real habría que alimentarlo con el pronóstico.
    """
    c = clima.assign(hora=clima["hora"].astype("int8"))
    tabla = tabla.merge(c, on=["fecha", "hora"], how="left")
    for col in FEATURES_CLIMA:
        tabla[col] = tabla[col].astype("float32")
    return tabla


def agregar_rezagos_1d(tabla: pd.DataFrame) -> pd.DataFrame:
    """Rezagos de 1 día, para predecir el día siguiente (por ejemplo, al cierre de cada día).

    - lag_1d:     salidas de la misma estación y hora, el día anterior
    - nivel_ayer: salidas por hora de la estación en todo el día anterior
    Si la estación no tiene registro ayer, quedan vacías (XGBoost las trata como faltantes).
    """
    base = tabla[["estacion", "fecha", "hora", TARGET]]
    sh = base.assign(fecha=base["fecha"] + pd.Timedelta(days=1)).rename(columns={TARGET: "lag_1d"})
    tabla = tabla.merge(sh, on=["estacion", "fecha", "hora"], how="left")

    diario = base.groupby(["estacion", "fecha"], observed=True)[TARGET].mean().rename("nivel_ayer").reset_index()
    diario["fecha"] = diario["fecha"] + pd.Timedelta(days=1)
    tabla = tabla.merge(diario, on=["estacion", "fecha"], how="left")
    for c in FEATURES_REZAGO_1D:
        tabla[c] = tabla[c].astype("float32")
    return tabla


def ventanas_por_fechas(tabla: pd.DataFrame, n: int = 3, dias: int = 35):
    """Validación en ventanas (walk-forward): n periodos de prueba consecutivos de `dias` días al final.

    Cada ventana entrena con todos los días anteriores a ella. La última ventana es la
    partición original (últimos 35 días = ~20 %). Regresa una lista de (train, test, inicio, fin).
    """
    fechas = np.sort(tabla["fecha"].unique())
    res = []
    for k in range(n, 0, -1):
        ini = fechas[len(fechas) - k * dias]
        fin = fechas[len(fechas) - (k - 1) * dias - 1]
        train = tabla[tabla["fecha"] < ini]
        test = tabla[(tabla["fecha"] >= ini) & (tabla["fecha"] <= fin)]
        res.append((train, test, pd.Timestamp(ini), pd.Timestamp(fin)))
    return res


def ventanas_mensuales(tabla: pd.DataFrame, desde: str = "2026-03-01", min_dias_train: int = 42):
    """Validación mes por mes (walk-forward): cada mes calendario completo es una ventana de prueba.

    Cada mes entrena con todos los días anteriores. Solo se usan los meses que empiezan en `desde`
    o después y que tienen al menos `min_dias_train` días de entrenamiento antes de ellos.
    Regresa una lista de (train, test, inicio, fin), igual que ventanas_por_fechas().
    """
    fechas = pd.DatetimeIndex(np.sort(tabla["fecha"].unique()))
    meses = pd.Series(fechas.to_period("M")).unique()
    res = []
    for m in meses:
        ini, fin = m.start_time, m.end_time.normalize()
        if ini < pd.Timestamp(desde) or fin > fechas.max():
            continue   # mes anterior a `desde` o incompleto
        if (fechas < ini).sum() < min_dias_train:
            continue   # muy poca historia para entrenar
        train = tabla[tabla["fecha"] < ini]
        test = tabla[(tabla["fecha"] >= ini) & (tabla["fecha"] <= fin)]
        res.append((train, test, ini, fin))
    return res


def split_por_fechas(tabla: pd.DataFrame, frac_train: float = 0.8):
    """Entrena con los primeros días y prueba con los últimos (sin mezclar el tiempo)."""
    fechas = np.sort(tabla["fecha"].unique())
    corte = fechas[int(len(fechas) * frac_train)]
    train = tabla[tabla["fecha"] < corte]
    test = tabla[tabla["fecha"] >= corte]
    return train, test, pd.Timestamp(corte)


# ---------------------------------------------------------------- baseline

def baseline_promedio(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Promedio histórico de salidas por estación, hora y tipo de día (entre semana / fin de semana).

    Si una combinación no existe en train, se usa el promedio por hora y tipo de día.
    """
    llaves = ["estacion", "hora", "es_fin_de_semana"]
    prom = train.groupby(llaves, observed=True)[TARGET].mean().rename("pred").reset_index()
    prom_global = (
        train.groupby(["hora", "es_fin_de_semana"], observed=True)[TARGET].mean().rename("pred_g").reset_index()
    )
    m = test[llaves].merge(prom, on=llaves, how="left").merge(
        prom_global, on=["hora", "es_fin_de_semana"], how="left"
    )
    return m["pred"].fillna(m["pred_g"]).to_numpy()


def baseline_semana_pasada(test: pd.DataFrame) -> np.ndarray:
    """Predice lo mismo que pasó hace 7 días en esa estación y hora."""
    return test["lag_7d"].to_numpy(dtype=float)


def baseline_media_4sem(test: pd.DataFrame) -> np.ndarray:
    """Predice el promedio de las 4 semanas previas en esa estación, hora y día de la semana."""
    return test["media_4sem"].to_numpy(dtype=float)


# ---------------------------------------------------------------- métricas

def metricas(y_real, y_pred) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_real
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "Sesgo": float(np.mean(err)),   # >0: el modelo predice de más; <0: de menos
    }


def error_por(df: pd.DataFrame, y_pred, por: str) -> pd.DataFrame:
    """RMSE y MAE agrupados por una columna (hora, alcaldia, estacion...)."""
    d = df[[por, TARGET]].copy()
    d["err"] = np.asarray(y_pred, dtype=float) - d[TARGET].to_numpy(dtype=float)
    g = d.groupby(por, observed=True)["err"]
    return pd.DataFrame({
        "RMSE": g.apply(lambda e: float(np.sqrt(np.mean(e ** 2)))),
        "MAE": g.apply(lambda e: float(np.mean(np.abs(e)))),
        "n": g.size(),
    })


# ---------------------------------------------------------------- XGBoost

PARAM_BASE = dict(
    tree_method="hist",
    enable_categorical=True,
    n_estimators=1000,          # tope; early stopping decide cuántos usar
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
)

ESPACIO_BUSQUEDA = {
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "max_depth": [4, 6, 8, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 20],
    "objective": ["reg:squarederror", "count:poisson"],   # las salidas son conteos
}


def split_validacion(train: pd.DataFrame, frac: float = 0.8):
    """Parte train en (fit, validación) por fechas, para elegir hiperparámetros sin mirar test."""
    fit, val, _ = split_por_fechas(train, frac)
    return fit, val


def buscar_hiperparametros(
    train: pd.DataFrame, n_iter: int = 12, semilla: int = 42, features=FEATURES
) -> pd.DataFrame:
    """Random search: prueba n_iter combinaciones y las evalúa en los últimos días de train.

    Regresa una tabla ordenada por RMSE de validación, con el número de árboles
    que eligió el early stopping (`mejor_iteracion`).
    """
    import xgboost as xgb
    from sklearn.model_selection import ParameterSampler

    fit, val = split_validacion(train)
    resultados = []
    for i, params in enumerate(ParameterSampler(ESPACIO_BUSQUEDA, n_iter=n_iter, random_state=semilla)):
        modelo = xgb.XGBRegressor(**{**PARAM_BASE, **params})
        modelo.fit(fit[features], fit[TARGET], eval_set=[(val[features], val[TARGET])], verbose=False)
        pred = np.clip(modelo.predict(val[features]), 0, None)
        m = metricas(val[TARGET], pred)
        resultados.append({**params, "mejor_iteracion": int(modelo.best_iteration) + 1, **m})
        print(f"[{i + 1}/{n_iter}] RMSE val={m['RMSE']:.3f}  MAE val={m['MAE']:.3f}  {params}")
    return pd.DataFrame(resultados).sort_values("RMSE").reset_index(drop=True)


def elegir_arboles(train: pd.DataFrame, params: dict, features=FEATURES) -> int:
    """Número de árboles por early stopping en los últimos días de train, con params fijos."""
    import xgboost as xgb

    fit, val = split_validacion(train)
    modelo = xgb.XGBRegressor(**{**PARAM_BASE, **params})
    modelo.fit(fit[features], fit[TARGET], eval_set=[(val[features], val[TARGET])], verbose=False)
    return int(modelo.best_iteration) + 1


def entrenar_xgb(train: pd.DataFrame, params: dict, n_arboles: int, features=FEATURES):
    """Entrena el modelo final con todo train y el número de árboles ya elegido."""
    import xgboost as xgb

    base = {k: v for k, v in PARAM_BASE.items() if k not in ("n_estimators", "early_stopping_rounds")}
    modelo = xgb.XGBRegressor(**{**base, "n_estimators": n_arboles, **params})
    modelo.fit(train[features], train[TARGET], verbose=False)
    return modelo


# ------------------------------------------- modelos vistos en clase (necesitan escalar)

FEATURES_ONEHOT = ["estacion", "dia_semana", "hora"]
FEATURES_ESCALAR = ["lat", "lon", "capacidad"] + FEATURES_REZAGO
FEATURES_BIN = ["es_fin_de_semana"]


def preprocesador():
    """One-hot para las variables categóricas y estandarización (z-score) para las numéricas.

    La regresión lineal y la red neuronal son sensibles a la escala de las
    variables (ver "Normalization and Scaling" en clase); los árboles no lo son,
    por eso XGBoost usa las variables tal cual.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_ONEHOT),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), FEATURES_ESCALAR),
        ("bin", "passthrough", FEATURES_BIN),
    ])


class ModeloEscalado:
    """Une el preprocesador ya ajustado con un modelo de sklearn; predice desde la tabla original."""

    def __init__(self, prep, modelo):
        self.prep = prep
        self.modelo = modelo

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.prep.transform(df[FEATURES])
        return np.clip(self.modelo.predict(X), 0, None)


def entrenar_lineal(train: pd.DataFrame) -> ModeloEscalado:
    """Regresión lineal (mínimos cuadrados) sobre variables escaladas y one-hot."""
    from sklearn.linear_model import LinearRegression

    prep = preprocesador()
    X = prep.fit_transform(train[FEATURES])
    modelo = LinearRegression().fit(X, train[TARGET].to_numpy(dtype=float))
    return ModeloEscalado(prep, modelo)


def entrenar_mlp(
    train: pd.DataFrame,
    capas=(64, 32),
    lr: float = 1e-3,
    batch: int = 2048,
    epocas: int = 15,
    paciencia: int = 3,
    max_filas: int = 1_500_000,
    semilla: int = 42,
):
    """Perceptrón multicapa (ReLU, Adam, pérdida MSE) con early stopping por validación.

    - Entrena con los primeros ~80 % de los días de `train` y valida con el 20 % final.
    - Si hay más de `max_filas` filas, usa una muestra aleatoria para acotar el tiempo.
    - Cada época es una pasada completa (partial_fit). Se queda con la época de menor
      RMSE de validación y se detiene si no mejora en `paciencia` épocas.

    Regresa (ModeloEscalado, historial) con RMSE de train y de validación por época.
    """
    import copy

    from sklearn.neural_network import MLPRegressor

    fit, val = split_validacion(train)
    if max_filas and len(fit) > max_filas:
        fit = fit.sample(max_filas, random_state=semilla)
    fit = fit.sample(frac=1.0, random_state=semilla)    # mezclar: las filas vienen ordenadas por fecha

    prep = preprocesador()
    Xf = prep.fit_transform(fit[FEATURES])
    yf = fit[TARGET].to_numpy(dtype=float)
    Xv = prep.transform(val[FEATURES])
    yv = val[TARGET].to_numpy(dtype=float)
    idx = np.random.default_rng(semilla).choice(len(yf), size=min(200_000, len(yf)), replace=False)

    mlp = MLPRegressor(
        hidden_layer_sizes=capas, activation="relu", solver="adam",
        learning_rate_init=lr, batch_size=batch, random_state=semilla,
    )
    historial, mejor_rmse, mejor_modelo, sin_mejora = [], np.inf, None, 0
    for ep in range(1, epocas + 1):
        mlp.partial_fit(Xf, yf)
        r_tr = metricas(yf[idx], np.clip(mlp.predict(Xf[idx]), 0, None))["RMSE"]
        r_val = metricas(yv, np.clip(mlp.predict(Xv), 0, None))["RMSE"]
        historial.append({"epoca": ep, "RMSE_train": r_tr, "RMSE_val": r_val})
        print(f"época {ep:2d}  RMSE train={r_tr:.3f}  RMSE val={r_val:.3f}")
        if r_val < mejor_rmse - 1e-4:
            mejor_rmse, mejor_modelo, sin_mejora = r_val, copy.deepcopy(mlp), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                print("early stopping")
                break
    return ModeloEscalado(prep, mejor_modelo), pd.DataFrame(historial)
