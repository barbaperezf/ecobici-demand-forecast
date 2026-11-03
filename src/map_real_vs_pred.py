"""
Genera un mapa HTML interactivo con las estaciones de Ecobici, coloreadas
por qué tan lejos está la demanda predicha del estado real observado.

Esto es exactamente lo que necesitan para "presentar comportamiento en
tiempo real vs predicción": un mapa donde cada punto es una estación, con
un popup que muestra bicis reales disponibles vs. las que tu modelo
hubiera esperado.

Uso:
    python src/map_real_vs_pred.py

Por ahora usa una predicción "placeholder" (promedio histórico por hora)
solo para dejar el mapa funcionando de punta a punta. En cuanto tengan el
modelo de verdad, reemplacen `predecir_demanda_placeholder` por la salida
real del modelo (misma forma: station_id -> bicis esperadas).
"""

import datetime as dt
import os

import folium
import pandas as pd

from eda_utils import demanda_por_estacion_hora, load_clean_trips, load_stations, load_status_snapshot

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def predecir_demanda_placeholder(trips: pd.DataFrame, hora: int) -> pd.DataFrame:
    """Placeholder: promedio histórico de salidas por estación a esa hora.
    Reemplazar con la predicción real del modelo cuando exista."""
    demanda = demanda_por_estacion_hora(trips)
    demanda_hora = demanda[demanda["hora_del_dia"] == hora]
    promedio = (
        demanda_hora.groupby("estacion_origen")["salidas"]
        .mean()
        .reset_index(name="salidas_esperadas")
    )
    return promedio


def build_map(hora: int = None) -> folium.Map:
    stations = load_stations()
    status = load_status_snapshot()
    trips = load_clean_trips()

    if hora is None:
        hora = dt.datetime.now().hour

    pred = predecir_demanda_placeholder(trips, hora)
    # La predicción está indexada por id de viaje (`ce`); el estado en vivo y
    # el catálogo GBFS por station_id. Se unen vía stations_geo.csv (ce <-> station_id).
    pred = pred.rename(columns={"estacion_origen": "ce"})
    geo = pd.read_csv(
        os.path.join(PROCESSED_DIR, "stations_geo.csv"),
        dtype={"station_id": "string", "ce": "string"},
    )[["station_id", "ce"]]

    df = (
        stations.merge(status, on="station_id", how="left")
        .merge(geo, on="station_id", how="left")
        .merge(pred, on="ce", how="left")
    )

    center_lat, center_lon = df["lat"].mean(), df["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

    for _, row in df.iterrows():
        real = row.get("bicis_disponibles")
        esperado = row.get("salidas_esperadas")

        if pd.isna(real) or pd.isna(esperado):
            color = "gray"
        else:
            # heurística simple: si hay menos bicis reales que demanda
            # esperada, la estación probablemente se queda vacía -> rojo
            diff = real - esperado
            color = "red" if diff < -2 else ("orange" if diff < 2 else "green")

        popup = folium.Popup(
            f"<b>{row.get('nombre', row['station_id'])}</b><br>"
            f"Bicis disponibles (real, ahora): {real}<br>"
            f"Salidas esperadas hist. a las {hora}:00: "
            f"{esperado:.1f}" if pd.notna(esperado) else "N/D",
            max_width=250,
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=popup,
        ).add_to(m)

    return m


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    mapa = build_map()
    out_path = os.path.join(FIGURES_DIR, "mapa_real_vs_prediccion.html")
    mapa.save(out_path)
    print(f"Mapa guardado en: {out_path} (ábrelo en el navegador)")
