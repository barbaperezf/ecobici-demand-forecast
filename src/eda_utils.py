"""
Funciones reusables para el EDA. Cada quien las importa en su notebook:

    import sys; sys.path.append("../src")
    from eda_utils import load_clean_trips, load_stations, plot_outliers_duracion, ...
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_clean_trips() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "trips_clean.parquet")
    return pd.read_parquet(path)


def load_stations() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "stations.csv")
    return pd.read_csv(path, dtype={"station_id": "string"})


def load_stations_ce() -> pd.DataFrame:
    """Catálogo de estaciones con la llave de los VIAJES.

    Los ids de estación en los viajes son el número CE (columna `ce` de
    stations_geo.csv), NO el station_id del feed GBFS. Esta función regresa
    station_id = ce (texto) + nombre, lat, lon, para poder hacer
    `stations.set_index("station_id")` directo con estacion_origen/destino.
    Incluye las estaciones pareadas (ej. "271-272").
    """
    geo = pd.read_csv(os.path.join(PROCESSED_DIR, "stations_geo.csv"), dtype={"ce": "string"})
    out = geo[["ce", "nombre", "lat", "lon", "capacidad"]].rename(columns={"ce": "station_id"})
    return out.drop_duplicates("station_id")


def load_status_snapshot() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "station_status_snapshot.csv")
    return pd.read_csv(path, dtype={"station_id": "string"})


def demanda_por_estacion_hora(trips: pd.DataFrame) -> pd.DataFrame:
    """Agrega el histórico de viajes a nivel estación-fecha-hora: esto es la
    'demanda' que van a predecir (salidas por estación por hora)."""
    g = (
        trips.groupby(["estacion_origen", "fecha", "hora_del_dia"])
        .size()
        .reset_index(name="salidas")
    )
    return g


def plot_outliers_duracion(trips: pd.DataFrame, max_minutos: int = 180):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    trips["duracion_min"].clip(upper=max_minutos).plot.box(ax=axes[0])
    axes[0].set_title("Duración del viaje (min), recortado a %d min" % max_minutos)

    trips["duracion_min"].clip(upper=max_minutos).plot.hist(bins=50, ax=axes[1])
    axes[1].set_title("Distribución de duración del viaje")
    plt.tight_layout()
    return fig


def plot_viajes_por_hora(trips: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    trips.groupby("hora_del_dia").size().plot(kind="bar", ax=ax)
    ax.set_title("Viajes totales por hora del día")
    ax.set_xlabel("Hora")
    ax.set_ylabel("# viajes")
    plt.tight_layout()
    return fig


def plot_top_estaciones(trips: pd.DataFrame, n: int = 10):
    fig, ax = plt.subplots(figsize=(8, 4))
    trips["estacion_origen"].value_counts().head(n).plot(kind="bar", ax=ax)
    ax.set_title(f"Top {n} estaciones por salidas")
    ax.set_xlabel("station_id")
    ax.set_ylabel("# salidas")
    plt.tight_layout()
    return fig
