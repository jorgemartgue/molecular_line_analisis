#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS


# ============================================================
# CONFIGURACIÓN
# ============================================================

REGION = "E_NORTH"
MOLECULA = "CH3CCH_v0"

RUTA_MAPAS = (
    Path.home()
    / "TFM"
    / "maps"
    / "diagrot"
    / REGION
    / MOLECULA
)

GUARDAR_FIGURA = True


# ============================================================
# ARCHIVOS
# ============================================================

ARCHIVOS = {
    "T": RUTA_MAPAS / f"{MOLECULA}_Tex.fits",
    "N": RUTA_MAPAS / f"{MOLECULA}_Ncol.fits",
    "dT": RUTA_MAPAS / f"{MOLECULA}_Delta_Tex.fits",
    "dN": RUTA_MAPAS / f"{MOLECULA}_Delta_Ncol.fits",
}


# ============================================================
# FUNCIONES
# ============================================================

def cargar_mapa(path):
    """
    Carga un mapa FITS y su WCS celestial.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo:\n{path}"
        )

    with fits.open(path) as hdul:
        data = np.asarray(
            hdul[0].data,
            dtype=float,
        )

        header = hdul[0].header.copy()

    data = np.squeeze(data)

    if data.ndim != 2:
        raise ValueError(
            f"{path.name} no es un mapa 2D: {data.shape}"
        )

    return data, WCS(header).celestial


def limites_robustos(data, pmin=2, pmax=98):
    """
    Calcula límites robustos para la escala de color.
    """

    valores = data[np.isfinite(data)]

    if valores.size == 0:
        return 0, 1

    vmin, vmax = np.nanpercentile(
        valores,
        [pmin, pmax],
    )

    if vmin == vmax:
        margen = max(
            abs(vmin) * 0.1,
            1,
        )

        vmin -= margen
        vmax += margen

    return vmin, vmax


def representar_mapa(
        fig,
        posicion,
        data,
        wcs,
        titulo,
        colorbar_label,
        logaritmico=False,
        cmap="viridis"):

    data_plot = data.copy()

    if logaritmico:
        data_plot[data_plot <= 0] = np.nan
        data_plot = np.log10(data_plot)

    vmin, vmax = limites_robustos(
        data_plot,
    )

    ax = fig.add_subplot(
        2,
        2,
        posicion,
        projection=wcs,
    )

    imagen = ax.imshow(
        data_plot,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_title(titulo)

    ax.coords[0].set_axislabel("RA")
    ax.coords[1].set_axislabel("Dec")

    ax.coords[0].set_major_formatter(
        "hh:mm:ss.s"
    )

    ax.coords[1].set_major_formatter(
        "dd:mm:ss.s"
    )

    ax.coords[0].display_minor_ticks(True)
    ax.coords[1].display_minor_ticks(True)

    colorbar = fig.colorbar(
        imagen,
        ax=ax,
        pad=0.02,
        fraction=0.046,
    )

    colorbar.set_label(
        colorbar_label
    )


# ============================================================
# CARGAR MAPAS
# ============================================================

mapas = {}
wcs_referencia = None

for tipo, path in ARCHIVOS.items():

    data, wcs = cargar_mapa(path)

    mapas[tipo] = data

    if wcs_referencia is None:
        wcs_referencia = wcs


# ============================================================
# REPRESENTACIÓN
# ============================================================

fig = plt.figure(
    figsize=(13, 11),
)

fig.suptitle(
    f"Diagrama rotacional — {REGION} — {MOLECULA}",
    fontsize=15,
)

representar_mapa(
    fig=fig,
    posicion=1,
    data=mapas["T"],
    wcs=wcs_referencia,
    titulo=r"$T_\mathrm{ex}$",
    colorbar_label="K",
    cmap="inferno",
)

representar_mapa(
    fig=fig,
    posicion=2,
    data=mapas["N"],
    wcs=wcs_referencia,
    titulo=r"$N_\mathrm{col}$",
    colorbar_label=r"$\log_{10}(N/\mathrm{cm}^{-2})$",
    logaritmico=True,
    cmap="viridis",
)

representar_mapa(
    fig=fig,
    posicion=3,
    data=mapas["dT"],
    wcs=wcs_referencia,
    titulo=r"$\Delta T_\mathrm{ex}$",
    colorbar_label="K",
    cmap="magma",
)

representar_mapa(
    fig=fig,
    posicion=4,
    data=mapas["dN"],
    wcs=wcs_referencia,
    titulo=r"$\Delta N_\mathrm{col}$",
    colorbar_label=r"$\log_{10}(\Delta N/\mathrm{cm}^{-2})$",
    logaritmico=True,
    cmap="plasma",
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96],
)


# ============================================================
# GUARDAR Y MOSTRAR
# ============================================================

if GUARDAR_FIGURA:

    path_salida = (
        RUTA_MAPAS
        / f"{MOLECULA}_mapas_diagrot.pdf"
    )

    fig.savefig(
        path_salida,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Figura guardada en: {path_salida}")


plt.show()