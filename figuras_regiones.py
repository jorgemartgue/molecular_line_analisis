#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 11:27:33 2026

@author: jorge
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import (
    ImageNormalize,
    LinearStretch,
    LogStretch,
    SqrtStretch,
    ZScaleInterval,
)
from astropy.visualization.wcsaxes import add_beam
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales
from regions import Regions


# ============================================================
# 1. Rutas
# ============================================================

RUTA_FITS = "/home/jorge/TFM/W51_te_continuum_best.fits"
RUTA_REGIONES = "/home/jorge/TFM/regiones/regiones_TFM.reg"


# ============================================================
# 2. Leer mapa de continuo
# ============================================================

with fits.open(RUTA_FITS, memmap=True) as hdul:

    header = hdul[0].header

    # El continuo ya es una imagen 2D
    imagen = hdul[0].data

    # WCS celeste
    wcs = WCS(header).celestial


print("======================================")
print("Mapa de continuo")
print("======================================")
print(f"Shape: {imagen.shape}")
print(f"Unidad: {header.get('BUNIT', '')}")

if all(k in header for k in ["BMAJ", "BMIN", "BPA"]):

    print(
        f"Beam: "
        f"{header['BMAJ'] * 3600:.3f}\" x "
        f"{header['BMIN'] * 3600:.3f}\" "
        f"(PA={header['BPA']:.1f} deg)"
    )
# ============================================================
# 3. Normalización de imagen
# ============================================================


interval = ZScaleInterval(
    contrast=0.25
)


def añadir_barra_escala(
    ax,
    wcs,
    longitud_pc,
    distancia_pc=5400,
    color="white",
):
    """
    Añade una barra de escala física en la esquina inferior derecha.
    """

    # Longitud física -> tamaño angular
    longitud_arcsec = (
        longitud_pc / distancia_pc
        * 206265
    )

    # Escala del píxel en arcsec/pixel
    escalas = proj_plane_pixel_scales(wcs) * 3600
    escala_x = escalas[0]

    longitud_pix = longitud_arcsec / escala_x

    # Límites actuales del zoom
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ancho = abs(xmax - xmin)
    alto = abs(ymax - ymin)

    # Posición: esquina inferior derecha
    x2 = xmax - 0.07 * ancho
    x1 = x2 - longitud_pix

    y = ymin + 0.08 * alto

    # Barra
    ax.plot(
        [x1, x2],
        [y, y],
        color=color,
        linewidth=3,
        solid_capstyle="butt",
    )

    # Texto
    ax.text(
        (x1 + x2) / 2,
        y + 0.025 * alto,
        f"{longitud_pc:g} pc",
        color=color,
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="bottom",
        path_effects=[
            pe.Stroke(
                linewidth=2,
                foreground="black"
            ),
            pe.Normal()
        ]
    )


def valores_zoom(
    imagen,
    wcs,
    ra_deg,
    dec_deg,
    ancho_arcsec
):

    centro = SkyCoord(
        ra_deg * u.deg,
        dec_deg * u.deg,
        frame="fk5"
    )

    xcen, ycen = wcs.world_to_pixel(centro)

    escalas = proj_plane_pixel_scales(wcs) * 3600

    escala_x = escalas[0]
    escala_y = escalas[1]

    ancho_pix = ancho_arcsec / escala_x
    alto_pix = ancho_arcsec / escala_y

    x0 = int(xcen - ancho_pix / 2)
    x1 = int(xcen + ancho_pix / 2)

    y0 = int(ycen - alto_pix / 2)
    y1 = int(ycen + alto_pix / 2)

    recorte = imagen[y0:y1, x0:x1]

    return recorte[np.isfinite(recorte)]

# ============================================================
# Normalización específica del zoom central
# ============================================================

valores_centro = valores_zoom(
    imagen,
    wcs,
    ra_deg=290.9162,
    dec_deg=14.51815,
    ancho_arcsec=8
)

interval = ZScaleInterval(contrast=0.25)

vmin_centro, vmax_centro = interval.get_limits(valores_centro)

norm_centro = ImageNormalize(
    vmin=vmin_centro,
    vmax=vmax_centro,
    stretch=LinearStretch()
)

print(
    f"Centro: vmin={vmin_centro:.4f}, "
    f"vmax={vmax_centro:.4f} Jy/beam"
)


# ============================================================
# Normalización específica de MM14
# ============================================================

valores_mm14 = valores_zoom(
    imagen,
    wcs,
    ra_deg=290.91077479,
    dec_deg=14.51159009,
    ancho_arcsec=5
)

vmin_mm14, vmax_mm14 = interval.get_limits(valores_mm14)

norm_mm14 = ImageNormalize(
    vmin=vmin_mm14,
    vmax=vmax_mm14,
    stretch=LinearStretch()
)

print(
    f"MM14: vmin={vmin_mm14:.4f}, "
    f"vmax={vmax_mm14:.4f} Jy/beam"
)

def hacer_zoom(ax, wcs, ra_deg, dec_deg, ancho_arcsec):

    centro = SkyCoord(
        ra_deg * u.deg,
        dec_deg * u.deg,
        frame="fk5"
    )

    xcen, ycen = wcs.world_to_pixel(centro)

    escalas = proj_plane_pixel_scales(wcs) * 3600

    escala_x = escalas[0]
    escala_y = escalas[1]

    ancho_pix = ancho_arcsec / escala_x
    alto_pix = ancho_arcsec / escala_y

    ax.set_xlim(
        xcen - ancho_pix / 2,
        xcen + ancho_pix / 2
    )

    ax.set_ylim(
        ycen - alto_pix / 2,
        ycen + alto_pix / 2
    )

# ============================================================
# 4. Crear figura
# ============================================================

fig = plt.figure(
    figsize=(8, 7),
    layout="constrained"
)

ax = fig.add_subplot(
    111,
    projection=wcs
)

im = ax.imshow(
    imagen,
    origin="lower",
    cmap="inferno",
    norm=norm_centro
)


# ============================================================
# 5. Regiones DS9
# ============================================================

regiones = Regions.read(
    RUTA_REGIONES,
    format="ds9"
)

FUENTES_CENTRO = [
    "d2",
    "ALMAmm31",
    "ALMAmm24",
    "ALMAmm35",
    "north",
]

OFFSETS_CENTRO = {
    "north":      (-20, 18),
    "ALMAmm35":   (-10, 18),
    "ALMAmm24":   (8, 15),
    "ALMAmm31":   (8, -28),
    "d2":         (8, -28),
}


FUENTES_MM14 = [
    "ALMAmm14",
]

OFFSETS_MM14 = {
    "ALMAmm14": (12, 10),
}

def dibujar_regiones(
    ax,
    wcs,
    regiones,
    fuentes=None,
    offsets=None
):

    if offsets is None:
        offsets = {}

    for region in regiones:

        texto = region.meta.get("text", "")

        if not texto:
            continue

        # Nombre de la fuente = texto antes de la primera coma
        nombre = texto.split(",")[0].strip()

        # Si hemos dado una lista de fuentes,
        # solo dibujamos esas
        if fuentes is not None and nombre not in fuentes:
            continue

        # Convertir región celeste a píxeles
        region_pix = region.to_pixel(wcs)

        # Dibujar elipse
        artist = region_pix.as_artist(
            edgecolor="cyan",
            facecolor="none",
            linewidth=1.8
        )

        ax.add_artist(artist)

        # ----------------------------------------------------
        # Preparar etiqueta
        # ----------------------------------------------------

        partes = texto.split(",")

        if len(partes) > 1:

            velocidad = partes[1].strip()

            velocidad = velocidad.replace(
                "km/s",
                r"km s$^{-1}$"
            )

            texto_plot = (
                f"{nombre}\n"
                f"{velocidad}"
            )

        else:
            texto_plot = nombre

        # Desplazamiento de la etiqueta
        dx, dy = offsets.get(nombre, (8, 8))

        etiqueta = ax.annotate(
            texto_plot,

            xy=(
                region_pix.center.x,
                region_pix.center.y
            ),

            xytext=(dx, dy),
            textcoords="offset points",

            color="cyan",
            fontsize=8.5,
            fontweight="bold",

            ha="left",
            va="bottom",
        )

        # Contorno negro para mejorar contraste
        etiqueta.set_path_effects([
            pe.Stroke(
                linewidth=2.0,
                foreground="black"
            ),
            pe.Normal()
        ])

dibujar_regiones(
    ax,
    wcs,
    regiones,
    fuentes=FUENTES_CENTRO,
    offsets=OFFSETS_CENTRO
)

hacer_zoom(
    ax,
    wcs,
    ra_deg=290.9162,
    dec_deg=14.51815,
    ancho_arcsec=8
)
añadir_barra_escala(
    ax,
    wcs,
    longitud_pc=0.05,
    distancia_pc=5400
)
add_beam(
    ax,
    header=header,
    corner="bottom left",
    frame=False,
    facecolor="white",
    edgecolor="black",
    linewidth=1.0
)

# ============================================================
# 6. Coordenadas
# ============================================================

ax.coords[0].set_axislabel(
    "Right Ascension (J2000)",
    fontsize=12
)

ax.coords[1].set_axislabel(
    "Declination (J2000)",
    fontsize=12
)

ax.coords[0].set_major_formatter("hh:mm:ss.s")
ax.coords[1].set_major_formatter("dd:mm:ss")

ax.coords[0].set_ticklabel(size=10)
ax.coords[1].set_ticklabel(size=10)


# ============================================================
# 7. Colorbar
# ============================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
    fraction=0.046
)

cbar.set_label(
    "Intensity [Jy beam$^{-1}$]",
    fontsize=11
)


# ============================================================
# 8. Título
# ============================================================

ax.set_title(
    "W51 IRS2 — central region — continuum",
    fontsize=14,
    pad=12
)

# ============================================================
# 9. Guardar
# ============================================================

plt.savefig(
    "W51_IRS2_central.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "W51_IRS2_central.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. Figura MM14
# ============================================================

fig = plt.figure(
    figsize=(8, 7),
    layout="constrained"
)
ax = fig.add_subplot(
    111,
    projection=wcs
)

im = ax.imshow(
    imagen,
    origin="lower",
    cmap="inferno",
    norm=norm_mm14
)


dibujar_regiones(
    ax,
    wcs,
    regiones,
    fuentes=FUENTES_MM14,
    offsets=OFFSETS_MM14
)


# Zoom sobre MM14
hacer_zoom(
    ax,
    wcs,
    ra_deg=290.91077479,
    dec_deg=14.51159009,
    ancho_arcsec=5
)

añadir_barra_escala(
    ax,
    wcs,
    longitud_pc=0.05,
    distancia_pc=5400
)
add_beam(
    ax,
    header=header,
    corner="bottom left",
    frame=False,
    facecolor="white",
    edgecolor="black",
    linewidth=1.0
)

FUENTES_CENTRO = [
    "d2",
    "ALMAmm31",
    "ALMAmm24",
    "ALMAmm35",
    "north",
]

OFFSETS_CENTRO = {
    "north":      (-20, 18),
    "ALMAmm35":   (-10, 18),
    "ALMAmm24":   (8, 15),
    "ALMAmm31":   (8, -28),
    "d2":         (8, -28),
}


FUENTES_MM14 = [
    "ALMAmm14",
]

OFFSETS_MM14 = {
    "ALMAmm14": (12, 10),
}

# Coordenadas
ax.coords[0].set_axislabel(
    "Right Ascension (J2000)",
    fontsize=12
)

ax.coords[1].set_axislabel(
    "Declination (J2000)",
    fontsize=12
)

ax.coords[0].set_major_formatter("hh:mm:ss.s")
ax.coords[1].set_major_formatter("dd:mm:ss")

ax.coords[0].set_ticklabel(size=10)
ax.coords[1].set_ticklabel(size=10)


# Colorbar
cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
    fraction=0.046
)

cbar.set_label(
    "Intensity [Jy beam$^{-1}$]",
    fontsize=11
)


# Título
ax.set_title(
    "W51 IRS2 — MM14 — continuum",
    fontsize=14,
    pad=12
)

plt.savefig(
    "W51_IRS2_MM14.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "W51_IRS2_MM14.pdf",
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 11. W51-E
# ============================================================

FUENTES_E = [
    "e2e",
    "e2w",
    "e8mm",
]

OFFSETS_E = {
    "e2e": (10, 18),
    "e2w": (10, -28),
    "e8mm": (10, 12),
}

CENTRO_E_RA = 290.9330
CENTRO_E_DEC = 14.5088

ANCHO_E = 14.0
# ============================================================
# 13. Regiones de W51-E
# ============================================================

FUENTES_E = [
    "e2e",
    "e2w",
    "e8mm",
]

OFFSETS_E = {
    "e2e": (10, 18),
    "e2w": (10, -28),
    "e8mm": (10, 12),
}


CENTRO_E_RA = 290.9330
CENTRO_E_DEC = 14.5088

ANCHO_E = 13


# ============================================================
# Normalización W51-E
# ============================================================

valores_E = valores_zoom(
    imagen,
    wcs,
    ra_deg=CENTRO_E_RA,
    dec_deg=CENTRO_E_DEC,
    ancho_arcsec=ANCHO_E
)

# ============================================================
# Normalización W51-E
# ============================================================

vmin_E = np.nanpercentile(valores_E, 1)
vmax_E = np.nanmax(valores_E)

norm_E = ImageNormalize(
    vmin=vmin_E,
    vmax=vmax_E,
    stretch=LinearStretch()
)

print(
    f"W51-E: vmin={vmin_E:.5f}, "
    f"vmax={vmax_E:.5f} Jy/beam"
)

# ============================================================
# Figura W51-E
# ============================================================

fig = plt.figure(figsize=(8, 7))

ax = fig.add_subplot(
    111,
    projection=wcs
)

im = ax.imshow(
    imagen,
    origin="lower",
    cmap="inferno",
    norm=norm_E,
    interpolation="none"
)


dibujar_regiones(
    ax,
    wcs,
    regiones,
    fuentes=FUENTES_E,
    offsets=OFFSETS_E
)

hacer_zoom(
    ax,
    wcs,
    ra_deg=CENTRO_E_RA,
    dec_deg=CENTRO_E_DEC,
    ancho_arcsec=ANCHO_E
)
add_beam(
    ax,
    header=header,
    corner="bottom left",
    frame=False,
    facecolor="white",
    edgecolor="black",
    linewidth=1.0
)
añadir_barra_escala(
    ax,
    wcs,
    longitud_pc=0.05,
    distancia_pc=5400
)

# Coordenadas
ax.coords[0].set_axislabel(
    "Right Ascension (J2000)",
    fontsize=12
)

ax.coords[1].set_axislabel(
    "Declination (J2000)",
    fontsize=12
)

ax.coords[0].set_major_formatter(
    "hh:mm:ss.ss"
)

ax.coords[1].set_major_formatter(
    "dd:mm:ss.s"
)

ax.coords[0].set_ticklabel(
    size=10
)

ax.coords[1].set_ticklabel(
    size=10
)


# Colorbar
cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
    fraction=0.046
)

cbar.set_label(
    "Intensity [Jy beam$^{-1}$]",
    fontsize=11
)


# Título
ax.set_title(
    "W51-E — continuum",
    fontsize=14,
    pad=12
)

# Guardar
plt.savefig(
    "W51_E.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "W51_E.pdf",
    bbox_inches="tight"
)

plt.show()

######################
# MAPAS
######################

RUTA_MAP_MM31_D2 = (
    "/home/jorge/TFM/regiones/mm31_d2.reg"
)

RUTA_MAP_NORTH = (
    "/home/jorge/TFM/regiones/regionNORTH_mm35_mm24.reg"
)

RUTA_MAP_MM14 = (
    "/home/jorge/TFM/regiones/regionMM14map.reg"
)

RUTA_MAP_E = (
    "/home/jorge/TFM/regiones/regionEnorth.reg"
)

# ============================================================
# 12. Regiones utilizadas para los mapas
# ============================================================

RUTA_MAP_MM31_D2 = "/home/jorge/TFM/regiones/mm31_d2.reg"

RUTA_MAP_NORTH = (
    "/home/jorge/TFM/regiones/regionNORTH_mm35_mm24.reg"
)

RUTA_MAP_MM14 = (
    "/home/jorge/TFM/regiones/regionMM14map.reg"
)

RUTA_MAP_E = (
    "/home/jorge/TFM/regiones/regionEnorth.reg"
)


reg_mm31_d2 = Regions.read(
    RUTA_MAP_MM31_D2,
    format="ds9"
)

reg_north = Regions.read(
    RUTA_MAP_NORTH,
    format="ds9"
)

reg_mm14 = Regions.read(
    RUTA_MAP_MM14,
    format="ds9"
)

reg_E = Regions.read(
    RUTA_MAP_E,
    format="ds9"
)

def dibujar_cajas_mapa(
    ax,
    wcs,
    regiones,
    etiqueta=None,
    color="cyan",
    offset=(8, 8)
):

    for region in regiones:

        region_pix = region.to_pixel(wcs)

        artist = region_pix.as_artist(
            edgecolor=color,
            facecolor="none",
            linewidth=2.0
        )

        ax.add_artist(artist)

        if etiqueta is not None:

            dx, dy = offset

            texto = ax.annotate(
                etiqueta,
                xy=(
                    region_pix.center.x,
                    region_pix.center.y
                ),
                xytext=(dx, dy),
                textcoords="offset points",
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="bottom"
            )

            texto.set_path_effects([
                pe.Stroke(
                    linewidth=2.0,
                    foreground="black"
                ),
                pe.Normal()
            ])

# ============================================================
# 13. Regiones de mapas — IRS2 central
# ============================================================

valores_map_centro = valores_zoom(
    imagen,
    wcs,
    ra_deg=290.9162,
    dec_deg=14.51815,
    ancho_arcsec=8
)

vmin_map_centro, vmax_map_centro = interval.get_limits(
    valores_map_centro
)

norm_map_centro = ImageNormalize(
    vmin=vmin_map_centro,
    vmax=vmax_map_centro,
    stretch=LinearStretch()
)


fig = plt.figure(
    figsize=(8, 7)
)

ax = fig.add_subplot(
    111,
    projection=wcs
)

im = ax.imshow(
    imagen,
    origin="lower",
    cmap="inferno",
    norm=norm_map_centro,
    interpolation="none"
)


dibujar_cajas_mapa(
    ax,
    wcs,
    reg_mm31_d2,
    etiqueta="MM31 + d2",
    offset=(8, -22)
)

dibujar_cajas_mapa(
    ax,
    wcs,
    reg_north,
    etiqueta="NORTH + MM35 + MM24",
    offset=(8, 10)
)


hacer_zoom(
    ax,
    wcs,
    ra_deg=290.9162,
    dec_deg=14.51815,
    ancho_arcsec=8
)


añadir_barra_escala(
    ax,
    wcs,
    longitud_pc=0.05,
    distancia_pc=5400
)


add_beam(
    ax,
    header=header,
    corner="bottom left",
    frame=False,
    facecolor="white",
    edgecolor="black",
    linewidth=1.0
)


ax.coords[0].set_axislabel(
    "Right Ascension (J2000)",
    fontsize=12
)

ax.coords[1].set_axislabel(
    "Declination (J2000)",
    fontsize=12
)

ax.coords[0].set_major_formatter(
    "hh:mm:ss.s"
)

ax.coords[1].set_major_formatter(
    "dd:mm:ss"
)

ax.coords[0].set_ticklabel(size=10)
ax.coords[1].set_ticklabel(size=10)


cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
    fraction=0.046
)

cbar.set_label(
    "Intensity [Jy beam$^{-1}$]",
    fontsize=11
)


ax.set_title(
    "W51 IRS2 — map regions",
    fontsize=14,
    pad=12
)


plt.savefig(
    "W51_IRS2_map_regions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "W51_IRS2_map_regions.pdf",
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 14. Región de mapa — MM14
# ============================================================

valores_map_mm14 = valores_zoom(
    imagen,
    wcs,
    ra_deg=290.91077479,
    dec_deg=14.51159009,
    ancho_arcsec=5
)

vmin_map_mm14, vmax_map_mm14 = interval.get_limits(
    valores_map_mm14
)

norm_map_mm14 = ImageNormalize(
    vmin=vmin_map_mm14,
    vmax=vmax_map_mm14,
    stretch=LinearStretch()
)


fig = plt.figure(
    figsize=(8, 7)
)

ax = fig.add_subplot(
    111,
    projection=wcs
)

im = ax.imshow(
    imagen,
    origin="lower",
    cmap="inferno",
    norm=norm_map_mm14,
    interpolation="none"
)


dibujar_cajas_mapa(
    ax,
    wcs,
    reg_mm14,
    etiqueta="MM14",
    offset=(8, 8)
)

hacer_zoom(
    ax,
    wcs,
    ra_deg=290.91077479,
    dec_deg=14.51159009,
    ancho_arcsec=5
)


añadir_barra_escala(
    ax,
    wcs,
    longitud_pc=0.05,
    distancia_pc=5400
)


add_beam(
    ax,
    header=header,
    corner="bottom left",
    frame=False,
    facecolor="white",
    edgecolor="black",
    linewidth=1.0
)


ax.coords[0].set_axislabel(
    "Right Ascension (J2000)",
    fontsize=12
)

ax.coords[1].set_axislabel(
    "Declination (J2000)",
    fontsize=12
)

ax.coords[0].set_major_formatter(
    "hh:mm:ss.s"
)

ax.coords[1].set_major_formatter(
    "dd:mm:ss"
)

ax.coords[0].set_ticklabel(size=10)
ax.coords[1].set_ticklabel(size=10)


cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
    fraction=0.046
)

cbar.set_label(
    "Intensity [Jy beam$^{-1}$]",
    fontsize=11
)


ax.set_title(
    "W51 IRS2 — MM14 map region",
    fontsize=14,
    pad=12
)


plt.savefig(
    "W51_IRS2_MM14_map_region.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "W51_IRS2_MM14_map_region.pdf",
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 15. Región de mapa — W51-E
# ============================================================

valores_map_E = valores_zoom(
    imagen,
    wcs,
    ra_deg=CENTRO_E_RA,
    dec_deg=CENTRO_E_DEC,
    ancho_arcsec=ANCHO_E
)

vmin_map_E = np.nanpercentile(
    valores_map_E,
    1
)

vmax_map_E = np.nanmax(
    valores_map_E
)

norm_map_E = ImageNormalize(
    vmin=vmin_map_E,
    vmax=vmax_map_E,
    stretch=LinearStretch()
)


fig = plt.figure(
    figsize=(8, 7)
)

ax = fig.add_subplot(
    111,
    projection=wcs
)

im = ax.imshow(
    imagen,
    origin="lower",
    cmap="inferno",
    norm=norm_map_E,
    interpolation="none"
)


dibujar_cajas_mapa(
    ax,
    wcs,
    reg_E,
    etiqueta="e2e + e2w",
    offset=(8, 8)
)

hacer_zoom(
    ax,
    wcs,
    ra_deg=CENTRO_E_RA,
    dec_deg=CENTRO_E_DEC,
    ancho_arcsec=ANCHO_E
)


añadir_barra_escala(
    ax,
    wcs,
    longitud_pc=0.05,
    distancia_pc=5400
)


add_beam(
    ax,
    header=header,
    corner="bottom left",
    frame=False,
    facecolor="white",
    edgecolor="black",
    linewidth=1.0
)


ax.coords[0].set_axislabel(
    "Right Ascension (J2000)",
    fontsize=12
)

ax.coords[1].set_axislabel(
    "Declination (J2000)",
    fontsize=12
)

ax.coords[0].set_major_formatter(
    "hh:mm:ss.ss"
)

ax.coords[1].set_major_formatter(
    "dd:mm:ss.s"
)

ax.coords[0].set_ticklabel(size=10)
ax.coords[1].set_ticklabel(size=10)


cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
    fraction=0.046
)

cbar.set_label(
    "Intensity [Jy beam$^{-1}$]",
    fontsize=11
)


ax.set_title(
    "W51-E — map region",
    fontsize=14,
    pad=12
)


plt.savefig(
    "W51_E_map_region.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "W51_E_map_region.pdf",
    bbox_inches="tight"
)

plt.show()