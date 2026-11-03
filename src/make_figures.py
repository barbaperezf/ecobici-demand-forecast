"""
Genera las gráficas finales del EDA, ya con criterio de diseño (no solo
matplotlib por default): colores con un trabajo específico, títulos que dicen
el hallazgo, etiquetas directas, paleta validada contra daltonismo.

Uso:
    python src/make_figures.py

Genera todo en figures/, listo para pegar en las slides.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from eda_utils import load_stations_ce

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# --- paleta validada (misma que usamos para el resto de artifacts de Claude) ---
BLUE = "#2a78d6"      # categórico slot 1 / secuencial / polo "positivo"
ORANGE = "#eb6834"    # categórico slot 2
AQUA = "#1baf7a"       # categórico slot 3
RED = "#e34948"        # polo "negativo" (divergente)
GRID = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


def _clean_axes(ax, y_grid=True):
    """Quita bordes sobrantes y deja solo una rejilla horizontal tenue."""
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    if y_grid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)
    # evita que matplotlib ponga notación científica / offset (ej. "1e6")
    # en una esquina — con números grandes eso se encima con nuestros
    # títulos y subtítulos.
    try:
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    except (AttributeError, ValueError):
        pass  # ejes categóricos (barh con nombres) no soportan ticklabel_format


def _savefig(fig, path):
    """Guarda con bbox_inches='tight' para que nada (título largo, etiqueta
    al final de una línea) se corte en el borde del lienzo."""
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def load_data():
    trips = pd.read_parquet(os.path.join(PROCESSED_DIR, "trips_clean.parquet"))
    # OJO: los viajes usan el id `ce`, no el station_id de GBFS (ver eda_utils.load_stations_ce)
    stations = load_stations_ce()
    validos = trips[(trips["duracion_min"] >= 1) & (trips["duracion_min"] <= 45)]
    return trips, stations, validos


# ---------------------------------------------------------------------------
# 1. Duración del viaje: un solo histograma, coloreado por la DECISIÓN tomada
#    (válido vs. excluido), en vez de dos gráficas separadas "antes/después".
# ---------------------------------------------------------------------------
def fig_duracion(trips: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 5))

    datos = trips["duracion_min"].clip(upper=90)
    validos_mask = datos <= 45
    n_validos = validos_mask.sum()
    n_excluidos = (~validos_mask).sum()
    pct_validos = n_validos / len(trips)

    ax.hist(datos[validos_mask], bins=45, range=(0, 90), color=BLUE, label="Válido (1-45 min)")
    ax.hist(datos[~validos_mask], bins=45, range=(0, 90), color=MUTED, alpha=0.6, label="Excluido (>45 min)")
    ax.axvline(45, color=RED, linestyle="--", linewidth=1.5)
    ax.text(46, ax.get_ylim()[1] * 0.92, "límite oficial\n45 min", color=RED, fontsize=9)

    # título + subtítulo en una sola llamada (dos líneas) para que nunca
    # compitan por el mismo espacio con la notación del eje Y
    ax.set_title(
        f"El {pct_validos:.1%} de los viajes respeta el límite oficial de 45 min\n"
        f"{n_excluidos:,} viajes lo excedieron y se excluyeron del análisis por estación",
        fontsize=13, color=INK, loc="left", pad=16, wrap=True,
    )
    ax.title.set_fontsize(13)
    # la segunda línea del título queda más chica y en tono secundario
    ax.set_xlabel("Duración del viaje (minutos)")
    ax.set_ylabel("Número de viajes")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(frameon=False, loc="upper right")
    _clean_axes(ax)
    _savefig(fig, os.path.join(FIGURES_DIR, "01_duracion_viaje.png"))


# ---------------------------------------------------------------------------
# 2. Top estaciones: magnitud -> una sola escala secuencial (azul), con la
#    estación líder resaltada y etiquetada directamente.
# ---------------------------------------------------------------------------
def fig_top_estaciones(validos: pd.DataFrame, stations: pd.DataFrame, n: int = 10):
    nombres = stations.set_index("station_id")["nombre"]
    top = validos["estacion_origen"].value_counts().head(n)
    top.index = top.index.map(lambda sid: nombres.get(sid, sid))
    top = top.sort_values()  # para barh, que se lea de mayor a menor de arriba a abajo

    fig, ax = plt.subplots(figsize=(10.5, 5))
    colors = [BLUE] * (len(top) - 1) + [ORANGE]  # resalta la #1
    bars = ax.barh(top.index, top.values, color=colors)

    for bar, val in zip(bars, top.values):
        ax.text(val + top.max() * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=9, color=INK)

    ax.set_title(
        f'"{top.index[-1]}" concentra la mayor demanda del sistema',
        fontsize=13, color=INK, loc="left", pad=14, wrap=True,
    )
    ax.set_xlabel("Salidas registradas (viajes)")
    ax.set_xlim(0, top.max() * 1.15)  # deja espacio a la derecha para las etiquetas
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _clean_axes(ax, y_grid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _savefig(fig, os.path.join(FIGURES_DIR, "02_top_estaciones.png"))


# ---------------------------------------------------------------------------
# 3. Viajes por hora: magnitud a lo largo del día, con las horas pico
#    anotadas directamente y el horario sin operación sombreado.
# ---------------------------------------------------------------------------
def fig_viajes_por_hora(validos: pd.DataFrame):
    conteo = validos.groupby("hora_del_dia").size().reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(conteo.index, conteo.values, color=BLUE, width=0.7)
    ax.axvspan(-0.5, 4.5, color=MUTED, alpha=0.12)
    ax.text(2, conteo.max() * 1.03, "sistema cerrado\n(00:30-05:00)", ha="center", fontsize=8.5, color=INK_2)

    pico_am, pico_pm = 8, 18
    for h in [pico_am, pico_pm]:
        ax.annotate(
            f"{conteo[h]:,.0f}", xy=(h, conteo[h]), xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=9.5, color=INK, fontweight="bold",
        )

    ax.set_title(
        "Doble pico de demanda: salida al trabajo (8am) y regreso (6pm)",
        fontsize=13, color=INK, loc="left", pad=14, wrap=True,
    )
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Número de viajes")
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(top=conteo.max() * 1.15)  # espacio para el rótulo "sistema cerrado"
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _clean_axes(ax)
    _savefig(fig, os.path.join(FIGURES_DIR, "03_viajes_por_hora.png"))


# ---------------------------------------------------------------------------
# 4. Entre semana vs. fin de semana: 2 series categóricas (azul/naranja fijos),
#    etiqueta directa al final de la línea en vez de depender solo de leyenda.
# ---------------------------------------------------------------------------
def fig_semana_vs_finde(validos: pd.DataFrame):
    entre = validos[~validos["es_fin_de_semana"]].groupby("hora_del_dia").size().reindex(range(24), fill_value=0)
    finde = validos[validos["es_fin_de_semana"]].groupby("hora_del_dia").size().reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(entre.index, entre.values, color=BLUE, linewidth=2.2, marker="o", markersize=4)
    ax.plot(finde.index, finde.values, color=ORANGE, linewidth=2.2, marker="o", markersize=4)

    ax.text(23.3, entre.values[-1], "Entre semana", color=BLUE, fontsize=10, fontweight="bold", va="center")
    ax.text(23.3, finde.values[-1], "Fin de semana", color=ORANGE, fontsize=10, fontweight="bold", va="center")

    ax.set_title(
        "El doble pico de hora punta desaparece el fin de semana",
        fontsize=13, color=INK, loc="left", pad=14, wrap=True,
    )
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Número de viajes")
    ax.set_xlim(0, 27)
    ax.set_xticks(range(0, 24, 2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _clean_axes(ax)
    _savefig(fig, os.path.join(FIGURES_DIR, "04_semana_vs_finde.png"))


# ---------------------------------------------------------------------------
# 5. Flujo neto por estación: esto SÍ es polaridad (se vacía / se llena) ->
#    par divergente azul<->rojo, ordenado, con el signo como eje del diseño.
# ---------------------------------------------------------------------------
def fig_flujo_neto(validos: pd.DataFrame, stations: pd.DataFrame, n: int = 8):
    nombres = stations.set_index("station_id")["nombre"]
    salidas = validos.groupby("estacion_origen").size().rename("salidas")
    llegadas = validos.groupby("estacion_destino").size().rename("llegadas")
    flujo = pd.concat([salidas, llegadas], axis=1).fillna(0)
    flujo["neto"] = flujo["llegadas"] - flujo["salidas"]
    flujo["nombre"] = flujo.index.map(lambda sid: nombres.get(sid, sid))
    flujo = flujo.dropna(subset=["nombre"])

    peor = flujo.sort_values("neto").head(n)
    mejor = flujo.sort_values("neto", ascending=False).head(n)
    combinado = pd.concat([peor, mejor]).sort_values("neto")

    fig, ax = plt.subplots(figsize=(10.5, 6))
    colors = [RED if v < 0 else BLUE for v in combinado["neto"]]
    bars = ax.barh(combinado["nombre"], combinado["neto"], color=colors)

    # Regla híbrida para que la etiqueta nunca desaparezca ni se encime:
    # - barra LARGA: etiqueta ADENTRO, pegada a la punta, en blanco
    #   (nunca invade la columna de nombres de estación, a la izquierda del cero).
    # - barra CORTA: el texto no cabría adentro (se saldría del área de color
    #   y el blanco se volvería invisible sobre el fondo claro) -> etiqueta
    #   AFUERA, en tinta oscura; como la barra es corta, sigue lejos de los
    #   nombres de estación.
    max_abs = combinado["neto"].abs().max()
    umbral_corta = max_abs * 0.18
    pad = max_abs * 0.015

    for bar, val in zip(bars, combinado["neto"]):
        es_corta = abs(val) < umbral_corta
        if val >= 0:
            x, ha = (val + pad, "left") if es_corta else (val - pad, "right")
        else:
            x, ha = (val - pad, "right") if es_corta else (val + pad, "left")
        color_txt = INK if es_corta else "white"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{val:+,.0f}",
                 va="center", ha=ha, fontsize=8.5, color=color_txt, fontweight="bold")

    # deja aire extra en ambos extremos para las etiquetas que quedan afuera
    ax.set_xlim(combinado["neto"].min() - max_abs * 0.12, combinado["neto"].max() + max_abs * 0.12)

    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_title(
        "Estaciones que se vacían (rojo) vs. que acumulan bicis (azul)",
        fontsize=13, color=INK, loc="left", pad=14, wrap=True,
    )
    ax.set_xlabel("Flujo neto (llegadas − salidas)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _clean_axes(ax, y_grid=False)
    _savefig(fig, os.path.join(FIGURES_DIR, "05_flujo_neto.png"))


# ---------------------------------------------------------------------------
# 6. Curva de acumulación: identidad por estación (categórico fijo),
#    cero como referencia, etiqueta directa al final de cada línea.
# ---------------------------------------------------------------------------
def fig_curva_acumulacion(validos: pd.DataFrame, stations: pd.DataFrame):
    nombres = stations.set_index("station_id")["nombre"]

    salidas_hora = validos.groupby(["estacion_origen", "hora_del_dia"]).size().rename("salidas").rename_axis(["station_id", "hora_del_dia"])
    llegadas_hora = validos.groupby(["estacion_destino", "hora_del_dia"]).size().rename("llegadas").rename_axis(["station_id", "hora_del_dia"])
    flujo_hora = pd.concat([salidas_hora, llegadas_hora], axis=1).fillna(0)
    flujo_hora["neto"] = flujo_hora["llegadas"] - flujo_hora["salidas"]
    flujo_hora = flujo_hora.reset_index()

    peor_hora = flujo_hora.loc[flujo_hora.groupby("station_id")["neto"].idxmin()]
    top3 = peor_hora.sort_values("neto").head(3)["station_id"]

    colores = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for color, est in zip(colores, top3):
        serie = flujo_hora[flujo_hora["station_id"] == est].sort_values("hora_del_dia")
        acumulado = serie["neto"].cumsum()
        nombre = nombres.get(est, est)
        ax.plot(serie["hora_del_dia"], acumulado, color=color, linewidth=2.2, marker="o", markersize=4)
        ax.annotate(
            f"{nombre}\n{acumulado.iloc[-1]:+,.0f}", xy=(serie["hora_del_dia"].iloc[-1], acumulado.iloc[-1]),
            xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=color, fontweight="bold",
            annotation_clip=False,
        )

    ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_title(
        "Sin rebalanceo, estas 3 estaciones se quedarían sin bicis en horas pico",
        fontsize=13, color=INK, loc="left", pad=14, wrap=True,
    )
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Bicis acumuladas (llegadas − salidas)")
    ax.set_xlim(0, 26)  # deja espacio a la derecha para las 3 etiquetas de línea
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _clean_axes(ax)
    _savefig(fig, os.path.join(FIGURES_DIR, "06_curva_acumulacion.png"))


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    trips, stations, validos = load_data()

    fig_duracion(trips)
    fig_top_estaciones(validos, stations)
    fig_viajes_por_hora(validos)
    fig_semana_vs_finde(validos)
    fig_flujo_neto(validos, stations)
    fig_curva_acumulacion(validos, stations)

    print(f"Listo. Gráficas guardadas en: {FIGURES_DIR}")