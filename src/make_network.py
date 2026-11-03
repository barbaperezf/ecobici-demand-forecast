import os
import numpy as np
import pandas as pd

from eda_utils import load_stations_ce
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# Mismo esquema de color que make_figures.py, para consistencia visual.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
GRID = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# ---------------------------------------------------------------------------
# Parámetros de filtrado ("filtros típicos" acordados):
#   - EDGE_MIN_TRIPS: sólo se dibujan pares de estaciones con al menos este
#     número de viajes combinados (origen->destino + destino->origen). Esto
#     evita intentar dibujar las ~677*676 combinaciones posibles.
#   - Dentro de los pares que sí pasan el filtro, el grosor/opacidad de cada
#     línea sigue escalando con su volumen, así las conexiones más débiles
#     (pero aún relevantes) se ven tenues en vez de desaparecer de golpe.
#   - TOP_EDGES: además del umbral, nos quedamos con un máximo de conexiones
#     (las de mayor volumen) para no saturar la imagen si el umbral por sí
#     solo deja pasar demasiadas.
#   - IMBALANCE_RATIO: un par se considera "direccional" (naranja) si más del
#     65% de sus viajes van en un solo sentido; si el ir y venir es parecido
#     se considera "balanceado" (azul).
# ---------------------------------------------------------------------------
NODE_TOP_N = 250          # cuántas estaciones se muestran como nodos (de 677)
EDGE_MIN_TRIPS = 150      # piso absoluto: un par necesita al menos estos viajes
EDGE_PERCENTILE = 85      # y además estar en el 15% de pares con más viajes
TOP_EDGES = 300           # tope duro de conexiones dibujadas
IMBALANCE_RATIO = 0.65

NODE_SIZE_MIN = 25
NODE_SIZE_MAX = 1200


def load_data():
    trips = pd.read_parquet(os.path.join(PROCESSED_DIR, "trips_clean.parquet"))
    # OJO: los viajes usan el id `ce`, no el station_id de GBFS (ver eda_utils.load_stations_ce)
    stations = load_stations_ce()
    validos = trips[(trips["duracion_min"] >= 1) & (trips["duracion_min"] <= 45)].copy()
    return validos, stations


def build_pair_flows(validos: pd.DataFrame) -> pd.DataFrame:
    """Regresa un DataFrame con una fila por par de estaciones (sin importar
    el orden), con el conteo de viajes en cada sentido y el sentido dominante.
    """
    viajes = validos.dropna(subset=["estacion_origen", "estacion_destino"])
    viajes = viajes[viajes["estacion_origen"] != viajes["estacion_destino"]]

    conteo_dirigido = (
        viajes.groupby(["estacion_origen", "estacion_destino"])
        .size()
        .reset_index(name="viajes")
    )

    # Clave de par sin orden (A-B es igual a B-A) para agregar ambos sentidos.
    a = conteo_dirigido["estacion_origen"].astype(str)
    b = conteo_dirigido["estacion_destino"].astype(str)
    conteo_dirigido["par_id"] = np.where(a < b, a + "|" + b, b + "|" + a)

    pares = []
    for par_id, grupo in conteo_dirigido.groupby("par_id"):
        est_a, est_b = par_id.split("|")
        fila_ab = grupo[(grupo["estacion_origen"].astype(str) == est_a) & (grupo["estacion_destino"].astype(str) == est_b)]
        fila_ba = grupo[(grupo["estacion_origen"].astype(str) == est_b) & (grupo["estacion_destino"].astype(str) == est_a)]
        viajes_ab = int(fila_ab["viajes"].sum())
        viajes_ba = int(fila_ba["viajes"].sum())
        total = viajes_ab + viajes_ba
        if total == 0:
            continue
        if viajes_ab >= viajes_ba:
            origen, destino, mayor, menor = est_a, est_b, viajes_ab, viajes_ba
        else:
            origen, destino, mayor, menor = est_b, est_a, viajes_ba, viajes_ab
        pares.append({
            "origen": origen,
            "destino": destino,
            "viajes_total": total,
            "viajes_dominante": mayor,
            "viajes_opuesto": menor,
            "proporcion_dominante": mayor / total,
        })

    return pd.DataFrame(pares)


def filtrar_conexiones(pares: pd.DataFrame) -> pd.DataFrame:
    if pares.empty:
        return pares
    umbral_percentil = pares["viajes_total"].quantile(EDGE_PERCENTILE / 100)
    umbral = max(EDGE_MIN_TRIPS, umbral_percentil)
    filtrado = pares[pares["viajes_total"] >= umbral].copy()
    filtrado = filtrado.sort_values("viajes_total", ascending=False).head(TOP_EDGES)
    filtrado["direccional"] = filtrado["proporcion_dominante"] >= IMBALANCE_RATIO
    return filtrado


def volumen_por_estacion(validos: pd.DataFrame) -> pd.Series:
    salidas = validos["estacion_origen"].value_counts()
    llegadas = validos["estacion_destino"].value_counts()
    volumen = salidas.add(llegadas, fill_value=0)
    volumen.index = volumen.index.astype(str)
    return volumen


def escalar_tamano_nodo(volumenes: pd.Series) -> pd.Series:
    # Escala raíz cuadrada: el área del punto es proporcional al volumen,
    # que es la forma correcta de mapear una magnitud a un tamaño visual.
    raiz = np.sqrt(volumenes.clip(lower=0))
    if raiz.max() == raiz.min():
        return pd.Series(NODE_SIZE_MIN, index=volumenes.index)
    normalizado = (raiz - raiz.min()) / (raiz.max() - raiz.min())
    return NODE_SIZE_MIN + normalizado * (NODE_SIZE_MAX - NODE_SIZE_MIN)


def fig_red_estaciones(validos: pd.DataFrame, stations: pd.DataFrame):
    volumenes = volumen_por_estacion(validos)

    # Paso 1: elegir qué estaciones se muestran como nodos. En vez de
    # limitarnos a las que sobreviven el filtro de conexiones (lo que dejaría
    # sólo un puñado de hubs), tomamos las NODE_TOP_N estaciones con más
    # volumen del sistema. Así la mayoría de las estaciones relevantes
    # aparecen -aunque muchas sin conexiones resaltadas-, sólo que las de bajo
    # volumen se ven pequeñas y tenues en vez de abrumar el dibujo.
    estaciones_top = volumenes.sort_values(ascending=False).head(NODE_TOP_N).index

    coords = stations.set_index("station_id")[["lat", "lon"]]
    coords = coords[coords.index.isin(estaciones_top)]
    faltantes = set(estaciones_top) - set(coords.index)
    if faltantes:
        print(f"  aviso: {len(faltantes)} estación(es) sin coordenadas en stations_geo.csv, se omiten del grafo")

    # Paso 2: de todos los pares posibles, sólo calculamos/filtramos flujo
    # entre estaciones que ya quedaron seleccionadas como nodos.
    validos_top = validos[
        validos["estacion_origen"].isin(coords.index) & validos["estacion_destino"].isin(coords.index)
    ]
    pares = build_pair_flows(validos_top)
    conexiones = filtrar_conexiones(pares)

    if conexiones.empty:
        raise ValueError(
            "Ninguna conexión superó los filtros de EDGE_MIN_TRIPS/EDGE_PERCENTILE. "
            "Baja esos umbrales en make_network.py."
        )

    tamanos = escalar_tamano_nodo(volumenes.reindex(coords.index).fillna(0))

    fig, ax = plt.subplots(figsize=(16, 14))

    # --- Conexiones (dibujadas primero, detrás de los nodos) ---
    max_viajes = conexiones["viajes_total"].max()
    min_viajes = conexiones["viajes_total"].min()

    for _, fila in conexiones.iterrows():
        x0, y0 = coords.loc[fila["origen"], "lon"], coords.loc[fila["origen"], "lat"]
        x1, y1 = coords.loc[fila["destino"], "lon"], coords.loc[fila["destino"], "lat"]

        # Peso normalizado 0-1 dentro de las conexiones que sí se dibujan,
        # para variar grosor y opacidad (de-enfatizar las conexiones más
        # débiles sin ocultarlas por completo).
        if max_viajes == min_viajes:
            peso = 1.0
        else:
            peso = (fila["viajes_total"] - min_viajes) / (max_viajes - min_viajes)

        color = ORANGE if fila["direccional"] else BLUE
        ancho = 0.6 + peso * 2.6
        alpha = 0.35 + peso * 0.55

        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>",
            mutation_scale=10 + peso * 12,
            shrinkA=6, shrinkB=6,
            connectionstyle="arc3,rad=0.08",
            color=color,
            linewidth=ancho,
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(arrow)

    # --- Nodos ---
    for est_id, fila_coord in coords.iterrows():
        size = tamanos.get(est_id, NODE_SIZE_MIN)
        # De-enfatizar estaciones de bajo volumen: más chicas y más claras.
        alpha_nodo = 0.45 + 0.55 * (size - NODE_SIZE_MIN) / max(1, (NODE_SIZE_MAX - NODE_SIZE_MIN))
        ax.scatter(
            fila_coord["lon"], fila_coord["lat"],
            s=size, color=INK_2, alpha=alpha_nodo,
            edgecolors=SURFACE, linewidths=0.6, zorder=3,
        )

    lon_margin = (coords["lon"].max() - coords["lon"].min()) * 0.06
    lat_margin = (coords["lat"].max() - coords["lat"].min()) * 0.06
    ax.set_xlim(coords["lon"].min() - lon_margin, coords["lon"].max() + lon_margin)
    # Margen extra abajo para dejar espacio limpio a las leyendas.
    ax.set_ylim(coords["lat"].min() - lat_margin * 4.5, coords["lat"].max() + lat_margin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    n_estaciones = len(coords)
    n_totales = len(volumenes)
    pct_direccional = 100 * conexiones["direccional"].mean()
    ax.set_title(
        f"El tráfico se concentra en pocas rutas: {len(conexiones)} conexiones entre las "
        f"{n_estaciones} estaciones de mayor volumen (de {n_totales} en el sistema)\n"
        f"Tamaño del nodo = volumen total de viajes  ·  naranja = flujo direccional, {pct_direccional:.0f}% de las conexiones mostradas  ·  azul = flujo balanceado (ida y vuelta)",
        fontsize=16, fontweight="bold", loc="left", wrap=True, pad=16,
    )

    # --- Leyendas (con fondo sólido para no perderse contra la red) ---
    leyenda_color = [
        mpatches.Patch(color=ORANGE, label="Flujo direccional (>65% en un sentido)"),
        mpatches.Patch(color=BLUE, label="Flujo balanceado (ida y vuelta similar)"),
    ]
    leg1 = ax.legend(
        handles=leyenda_color, loc="lower left", fontsize=12,
        frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95,
        borderpad=1.0,
    )
    ax.add_artist(leg1)

    handles_tam = []
    for tam, etiqueta in [(NODE_SIZE_MIN, "Volumen bajo"), ((NODE_SIZE_MIN + NODE_SIZE_MAX) / 2, "Volumen medio"), (NODE_SIZE_MAX, "Volumen alto")]:
        handles_tam.append(ax.scatter([], [], s=tam, color=INK_2, alpha=0.7, edgecolors=SURFACE, linewidths=0.6, label=etiqueta))
    ax.legend(
        handles=handles_tam, loc="lower right", fontsize=12, labelspacing=1.8, borderpad=1.2,
        title="Estación (tamaño = viajes totales)", title_fontsize=12,
        frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95,
    )

    return fig


def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    validos, stations = load_data()
    print("Calculando flujos entre estaciones...")
    fig = fig_red_estaciones(validos, stations)
    out_path = os.path.join(FIGURES_DIR, "07_red_estaciones.png")
    _savefig(fig, out_path)
    print(f"Listo. Grafo guardado en: {out_path}")