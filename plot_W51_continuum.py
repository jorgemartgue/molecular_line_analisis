#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figuras de continuo de W51 para el TFM.

Genera:
1. W51 IRS2 — región central (fuentes compactas)
2. W51 IRS2 — MM14 (fuente compacta)
3. W51-E (fuentes compactas)
4. W51 IRS2 — regiones usadas para los mapas
5. W51 IRS2 — MM14, región usada para los mapas
6. W51-E — región usada para los mapas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import (
    ImageNormalize,
    LinearStretch,
    LogStretch,
    SqrtStretch,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization.wcsaxes import add_beam
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales
from regions import Regions


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

RUTA_BEAM_ANALISIS = (
    "/home/jorge/TFM/reprojected/"
    "W51-IRS2_B6_spw0_12M_spw0.JvM.image.pbcor-reprojected-cube.fits"
)
RUTA_FITS = "/home/jorge/TFM/W51_te_continuum_best.fits"
RUTA_REGIONES = "/home/jorge/TFM/regiones/regiones_TFM.reg"

# Regiones usadas para generar los mapas
RUTA_MAP_MM31_D2 = "/home/jorge/TFM/regiones/mm31_d2.reg"
RUTA_MAP_NORTH = "/home/jorge/TFM/regiones/regionNORTH_mm35_mm24.reg"
RUTA_MAP_MM14 = "/home/jorge/TFM/regiones/regionMM14map.reg"
RUTA_MAP_E_NORTH = "/home/jorge/TFM/regiones/regionEnorth.reg"
RUTA_MAP_E_SOUTH = "/home/jorge/TFM/regiones/regionEsouth.reg"
# Distancia adoptada a W51
DISTANCIA_PC = 5400

# Centros y campos de visión
CENTRO_IRS2_RA = 290.9162
CENTRO_IRS2_DEC = 14.51815
ANCHO_IRS2 = 8.0

CENTRO_MM14_RA = 290.91077479
CENTRO_MM14_DEC = 14.51159009
ANCHO_MM14 = 5.0

# W51-E North
CENTRO_E_NORTH_RA = 290.9330
CENTRO_E_NORTH_DEC = 14.50985
ANCHO_E_NORTH = 9.0
ALTO_E_NORTH = 8.0

# W51-E South
CENTRO_E_SOUTH_RA = 290.9330
CENTRO_E_SOUTH_DEC = 14.50730
ANCHO_E_SOUTH = 9.0
ALTO_E_SOUTH = 8.0

# Zoom rectangular en E para quitar zona vacía lateral
ANCHO_E = 9.0
ALTO_E = 14

# Barra de escala
LONGITUD_ESCALA_PC = 0.05


# ============================================================
# 2. LEER MAPA DE CONTINUO
# ============================================================

with fits.open(RUTA_FITS, memmap=True) as hdul:
    header = hdul[0].header.copy()
    imagen = np.squeeze(hdul[0].data)

if imagen.ndim != 2:
    raise ValueError(
        f"El FITS de continuo debería ser 2D tras squeeze(), "
        f"pero tiene shape {imagen.shape}."
    )

wcs = WCS(header).celestial

print("======================================")
print("Mapa de continuo W51")
print("======================================")
print(f"Shape: {imagen.shape}")
print(f"Unidad: {header.get('BUNIT', '')}")

if all(k in header for k in ["BMAJ", "BMIN", "BPA"]):
    print(
        f"Beam continuo: "
        f"{header['BMAJ'] * 3600:.3f}\" x "
        f"{header['BMIN'] * 3600:.3f}\" "
        f"(PA={header['BPA']:.1f} deg)"
    )
else:
    print("ADVERTENCIA: el FITS de continuo no contiene BMAJ/BMIN/BPA.")

with fits.open(RUTA_BEAM_ANALISIS, memmap=True) as hdul:

    header_analisis = hdul[0].header


print(
    "Beam continuo: "
    f"{header['BMAJ'] * 3600:.3f}\" x "
    f"{header['BMIN'] * 3600:.3f}\" "
    f"(PA={header['BPA']:.1f} deg)"
)

print(
    "Beam análisis: "
    f"{header_analisis['BMAJ'] * 3600:.3f}\" x "
    f"{header_analisis['BMIN'] * 3600:.3f}\" "
    f"(PA={header_analisis['BPA']:.1f} deg)"
)

# ============================================================
# 3. FUNCIONES AUXILIARES
# ============================================================

def añadir_barra_escala(
    ax,
    wcs,
    longitud_pc,
    distancia_pc=DISTANCIA_PC,
    color="white",
):
    """Añade una barra de escala física en la esquina inferior derecha."""

    longitud_arcsec = longitud_pc / distancia_pc * 206265

    escalas = proj_plane_pixel_scales(wcs) * 3600
    escala_x = escalas[0]

    longitud_pix = longitud_arcsec / escala_x

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ancho = abs(xmax - xmin)
    alto = abs(ymax - ymin)

    x2 = xmax - 0.07 * ancho
    x1 = x2 - longitud_pix
    y = ymin + 0.08 * alto

    ax.plot(
        [x1, x2],
        [y, y],
        color=color,
        linewidth=3,
        solid_capstyle="butt",
    )

    ax.text(
        (x1 + x2) / 2,
        y + 0.025 * alto,
        f"{longitud_pc:g} pc",
        color=color,
        fontsize=FS_SCALE,
        fontweight="bold",
        ha="center",
        va="bottom",
        path_effects=[
            pe.Stroke(linewidth=2, foreground="black"),
            pe.Normal(),
        ],
    )


def valores_zoom(
    imagen,
    wcs,
    ra_deg,
    dec_deg,
    ancho_arcsec,
    alto_arcsec=None,
):
    """Devuelve los valores finitos dentro del zoom indicado."""

    if alto_arcsec is None:
        alto_arcsec = ancho_arcsec

    centro = SkyCoord(
        ra_deg * u.deg,
        dec_deg * u.deg,
        frame="fk5",
    )

    xcen, ycen = wcs.world_to_pixel(centro)

    escalas = proj_plane_pixel_scales(wcs) * 3600
    escala_x = escalas[0]
    escala_y = escalas[1]

    ancho_pix = ancho_arcsec / escala_x
    alto_pix = alto_arcsec / escala_y

    x0 = max(0, int(np.floor(xcen - ancho_pix / 2)))
    x1 = min(imagen.shape[1], int(np.ceil(xcen + ancho_pix / 2)))

    y0 = max(0, int(np.floor(ycen - alto_pix / 2)))
    y1 = min(imagen.shape[0], int(np.ceil(ycen + alto_pix / 2)))

    recorte = imagen[y0:y1, x0:x1]
    valores = recorte[np.isfinite(recorte)]

    if valores.size == 0:
        raise ValueError("El zoom seleccionado no contiene valores finitos.")

    return valores


def hacer_zoom(
    ax,
    wcs,
    ra_deg,
    dec_deg,
    ancho_arcsec,
    alto_arcsec=None,
):
    """Aplica un zoom cuadrado o rectangular a un eje WCSAxes."""

    if alto_arcsec is None:
        alto_arcsec = ancho_arcsec

    centro = SkyCoord(
        ra_deg * u.deg,
        dec_deg * u.deg,
        frame="fk5",
    )

    xcen, ycen = wcs.world_to_pixel(centro)

    escalas = proj_plane_pixel_scales(wcs) * 3600
    escala_x = escalas[0]
    escala_y = escalas[1]

    ancho_pix = ancho_arcsec / escala_x
    alto_pix = alto_arcsec / escala_y

    ax.set_xlim(
        xcen - ancho_pix / 2,
        xcen + ancho_pix / 2,
    )

    ax.set_ylim(
        ycen - alto_pix / 2,
        ycen + alto_pix / 2,
    )


def dibujar_regiones(
    ax,
    wcs,
    regiones,
    fuentes=None,
    offsets=None,
    colores=None,
):
    """Dibuja las elipses DS9 y sus etiquetas/velocidades."""

    if offsets is None:
        offsets = {}

    if colores is None:
        colores = {}
    
    for region in regiones:
        texto = region.meta.get("text", "")

        if not texto:
            continue

        nombre = texto.split(",")[0].strip()
        color = colores.get(nombre, "cyan")

        if fuentes is not None and nombre not in fuentes:
            continue

        region_pix = region.to_pixel(wcs)

        artist = region_pix.as_artist(
            edgecolor=color,
            facecolor="none",
            linewidth=1.8,
        )
        ax.add_artist(artist)

        partes = texto.split(",")

        if len(partes) > 1:
            velocidad = partes[1].strip()
            velocidad = velocidad.replace(
                "km/s",
                r"km s$^{-1}$",
            )
            texto_plot = f"{nombre}\n{velocidad}"
        else:
            texto_plot = nombre

        dx, dy = offsets.get(nombre, (8, 8))

        etiqueta = ax.annotate(
    texto_plot,
    xy=(
        region_pix.center.x,
        region_pix.center.y,
    ),
    xytext=(dx, dy),
    textcoords="offset points",
    color = color,
    fontsize=FS_SOURCE,
    fontweight="bold",
    ha="left",
    va="bottom",
)

        etiqueta.set_path_effects([
            pe.Stroke(linewidth=2.0, foreground="black"),
            pe.Normal(),
        ])


def dibujar_cajas_mapa(
    ax,
    wcs,
    regiones,
    etiqueta=None,
    color="cyan",
    offset=(8, 8),
):
    """Dibuja las cajas DS9 utilizadas para generar los mapas."""

    for region in regiones:
        region_pix = region.to_pixel(wcs)

        artist = region_pix.as_artist(
            edgecolor=color,
            facecolor="none",
            linewidth=2.0,
        )
        ax.add_artist(artist)

        if etiqueta is not None:
            dx, dy = offset

            texto = ax.annotate(
    etiqueta,
    xy=(
        region_pix.center.x,
        region_pix.center.y,
    ),
    xytext=(dx, dy),
    textcoords="offset points",
    color=color,
    fontsize=FS_MAP_REGION,
    fontweight="bold",
    ha="left",
    va="bottom",
)

            texto.set_path_effects([
                pe.Stroke(linewidth=2.0, foreground="black"),
                pe.Normal(),
            ])


def configurar_ejes(ax, formato_ra="hh:mm:ss.s", formato_dec="dd:mm:ss"):
    """Formato común de coordenadas."""

    ax.coords[0].set_axislabel(
    "Right Ascension (J2000)",
    fontsize=FS_AXIS,
    )

    ax.coords[1].set_axislabel(
    "Declination (J2000)",
    fontsize=FS_AXIS,
    )

    ax.coords[0].set_ticklabel(size=FS_TICKS)
    ax.coords[1].set_ticklabel(size=FS_TICKS)

    ax.coords[0].set_major_formatter(formato_ra)
    ax.coords[1].set_major_formatter(formato_dec)




def añadir_beams(ax):
    """
    Añade los dos beams:
    - continuo
    - cubos utilizados para el análisis
    """

    añadir_doble_beam(
        ax,
        wcs,
        header_continuo=header,
        header_analisis=header_analisis
    )


def crear_figura_base(norm, figsize=(6.2, 5.5)):
    """Crea figura, WCSAxes e imagen con formato común."""

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(
        111,
        projection=wcs,
    )

    im = ax.imshow(
        imagen,
        origin="lower",
        cmap="inferno",
        norm=norm,
        interpolation="none",
    )

    return fig, ax, im

def añadir_titulo_interno(ax, titulo):
    """Añade el título dentro del panel, en la esquina superior izquierda."""

    texto = ax.text(
    0.03,
    0.97,
    titulo,
    transform=ax.transAxes,
    color="white",
    fontsize=FS_TITLE,
    fontweight="bold",
    ha="left",
    va="top",
    zorder=20,
)

    texto.set_path_effects([
        pe.Stroke(
            linewidth=2.5,
            foreground="black",
        ),
        pe.Normal(),
    ])

def añadir_colorbar(fig, ax, im):
    """Añade una colorbar con la misma altura que el panel principal."""

    divider = make_axes_locatable(ax)

    cax = divider.append_axes(
        "right",
        size="4%",
        pad=0.12,
        axes_class=plt.Axes,
    )

    cbar = fig.colorbar(
        im,
        cax=cax,
    )

    cbar.set_label(
    "Intensity [Jy beam$^{-1}$]",
    fontsize=FS_COLORBAR,
)

    cbar.ax.tick_params(
    labelsize=FS_COLORBAR_TICKS,
)

    return cbar


def guardar_figura(nombre):
    """Guarda la figura actual en PNG y PDF."""

    plt.savefig(
        f"{nombre}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        f"{nombre}.pdf",
        bbox_inches="tight",
    )

    plt.show()

def añadir_doble_beam(
    ax,
    wcs,
    header_continuo,
    header_analisis
):
    """
    Dibuja el beam del continuo y el beam de los cubos
    utilizados para el análisis en la esquina inferior izquierda.
    """

    # ========================================================
    # Beam continuo
    # ========================================================

    bmaj_cont = header_continuo["BMAJ"] * 3600
    bmin_cont = header_continuo["BMIN"] * 3600
    bpa_cont = header_continuo["BPA"]

    # ========================================================
    # Beam análisis
    # ========================================================

    bmaj_ana = header_analisis["BMAJ"] * 3600
    bmin_ana = header_analisis["BMIN"] * 3600
    bpa_ana = header_analisis["BPA"]

    # ========================================================
    # Escala angular del píxel
    # ========================================================

    escalas = proj_plane_pixel_scales(wcs) * 3600

    escala_x = escalas[0]
    escala_y = escalas[1]

    # Dimensiones en píxeles
    width_cont = bmin_cont / escala_x
    height_cont = bmaj_cont / escala_y

    width_ana = bmin_ana / escala_x
    height_ana = bmaj_ana / escala_y

    # ========================================================
    # Límites actuales de la figura
    # ========================================================

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ancho = abs(xmax - xmin)
    alto = abs(ymax - ymin)

    # ========================================================
    # Posiciones
    # ========================================================

    # Altura común de ambos beams
    y_beam = ymin + 0.080 * alto

    # Separación horizontal
    x_cont = xmin + 0.080 * ancho
    x_ana = xmin + 0.230 * ancho

    # ========================================================
    # Beam continuo
    # ========================================================

    beam_cont = Ellipse(
        (x_cont, y_beam),
        width=width_cont,
        height=height_cont,
        angle=bpa_cont,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=10
    )

    ax.add_patch(beam_cont)

    # ========================================================
    # Beam análisis
    # ========================================================

    beam_ana = Ellipse(
        (x_ana, y_beam),
        width=width_ana,
        height=height_ana,
        angle=bpa_ana,
        facecolor="none",
        edgecolor="cyan",
        linewidth=1.8,
        zorder=10
    )

    ax.add_patch(beam_ana)

    # ========================================================
    # Etiquetas
    # ========================================================

    y_texto = ymin + 0.025 * alto

    texto_cont = ax.text(
        x_cont,
        y_texto,
        "Continuum",
        color="white",
        fontsize=FS_BEAM,
        ha="center",
        va="bottom",
        zorder=11
    )

    texto_cont.set_path_effects([
        pe.Stroke(
            linewidth=2.0,
            foreground="black"
        ),
        pe.Normal()
    ])

    texto_ana = ax.text(
        x_ana,
        y_texto,
        "Analysis",
        color="cyan",
        fontsize=8.5,
        ha="center",
        va="bottom",
        zorder=11
    )

    texto_ana.set_path_effects([
        pe.Stroke(
            linewidth=2.0,
            foreground="black"
        ),
        pe.Normal()
    ])
    
    
# ============================================================
# 4. LEER REGIONES
# ============================================================

regiones = Regions.read(
    RUTA_REGIONES,
    format="ds9",
)

reg_mm31_d2 = Regions.read(
    RUTA_MAP_MM31_D2,
    format="ds9",
)

reg_north = Regions.read(
    RUTA_MAP_NORTH,
    format="ds9",
)

reg_mm14 = Regions.read(
    RUTA_MAP_MM14,
    format="ds9",
)

reg_E_north = Regions.read(
    RUTA_MAP_E_NORTH,
    format="ds9",
)

reg_E_south = Regions.read(
    RUTA_MAP_E_SOUTH,
    format="ds9",
)
# ============================================================
# 6. CONFIGURACIÓN DE FUENTES
# ============================================================

FUENTES_CENTRO = [
    "d2",
    "ALMAmm31",
    "ALMAmm24",
    "ALMAmm35",
    "north",
]

OFFSETS_CENTRO = {
    "north": (-40, 25),
    "ALMAmm35": (18, 10),
    "ALMAmm24": (-65, -42),
    "ALMAmm31": (-10, -45),
    "d2": (-20, -45),
}

FUENTES_MM14 = [
    "ALMAmm14",
]

OFFSETS_MM14 = {
    "ALMAmm14": (12, 10),
}

FUENTES_E_NORTH = [
    "e2e",
    "e2w",
]

OFFSETS_E_NORTH = {
    "e2e": (10, 18),
    "e2w": (10, -28),
}

FUENTES_E_SOUTH = [
    "e8mm",
]

OFFSETS_E_SOUTH = {
    "e8mm": (10, 12),
}

# ============================================================
# TAMAÑOS DE TEXTO
# ============================================================

FS_TITLE = 16
FS_AXIS = 18
FS_TICKS = 15
FS_SOURCE = 11
FS_MAP_REGION = 10
FS_COLORBAR = 16
FS_COLORBAR_TICKS = 14
FS_SCALE = 13
FS_BEAM = 11

# ============================================================
# COLORES DE LAS REGIONES COMPACTAS
# ============================================================

COLORES_COMPACTAS = {
    # IRS2 central
    "north": "red",
    "ALMAmm35": "lime",
    "ALMAmm24": "yellow",
    "ALMAmm31": "magenta",
    "d2": "springgreen",

    # MM14
    "ALMAmm14": "violet",

    # W51-E North
    "e2e": "magenta",
    "e2w": "lime",

    # W51-E South
    "e8mm": "magenta",
}

COLORES_CENTRO = {
    nombre: COLORES_COMPACTAS[nombre]
    for nombre in FUENTES_CENTRO
}

COLORES_MM14 = {
    nombre: COLORES_COMPACTAS[nombre]
    for nombre in FUENTES_MM14
}

COLORES_E_NORTH = {
    nombre: COLORES_COMPACTAS[nombre]
    for nombre in FUENTES_E_NORTH
}

COLORES_E_SOUTH = {
    nombre: COLORES_COMPACTAS[nombre]
    for nombre in FUENTES_E_SOUTH
}

# ============================================================
# 7. NORMALIZACIONES
# ============================================================

# ------------------------------------------------------------
# IRS2 central
# Miriam: mínimo ~1 sigma = 0.003 Jy/beam,
# máximo 0.3 Jy/beam, log scale con alpha/a = 15.
# ------------------------------------------------------------

vmin_centro = 3e-3
vmax_centro = 0.3

norm_centro = ImageNormalize(
    vmin=vmin_centro,
    vmax=vmax_centro,
    stretch=LogStretch(a=15),
    clip=True,
)

print(
    f"IRS2 central: vmin={vmin_centro:.4f}, "
    f"vmax={vmax_centro:.4f} Jy/beam, LogStretch(a=15)"
)


# ------------------------------------------------------------
# MM14
# Miriam: mínimo 0.003 Jy/beam, máximo 0.35 Jy/beam,
# square-root.
# ------------------------------------------------------------

vmin_mm14 = 3e-3
vmax_mm14 = 0.1

norm_mm14 = ImageNormalize(
    vmin=vmin_mm14,
    vmax=vmax_mm14,
    stretch=SqrtStretch(),
    clip=True,
)

print(
    f"MM14: vmin={vmin_mm14:.4f}, "
    f"vmax={vmax_mm14:.4f} Jy/beam, SqrtStretch"
)


# ------------------------------------------------------------
# W51-E
# Miriam la ve bien: mantenemos la normalización que estaba
# funcionando, pero calculada sobre el nuevo zoom rectangular.
# ------------------------------------------------------------

valores_E = valores_zoom(
    imagen,
    wcs,
    ra_deg=290.9330,
    dec_deg=14.5088,
    ancho_arcsec=ANCHO_E,
    alto_arcsec=14.0,
)

vmin_E = 0.0
vmax_E = np.nanmax(valores_E)

norm_E = ImageNormalize(
    vmin=vmin_E,
    vmax=vmax_E,
    stretch=LinearStretch(),
    clip=True,
)

print(
    f"W51-E: vmin={vmin_E:.5f}, "
    f"vmax={vmax_E:.5f} Jy/beam"
)


# ============================================================
# 8. FIGURA 1 — W51 IRS2 CENTRAL
# ============================================================

fig, ax, im = crear_figura_base(norm_centro)

# Regiones compactas
dibujar_regiones(
    ax,
    wcs,
    regiones,
    fuentes=FUENTES_CENTRO,
    offsets=OFFSETS_CENTRO,
    colores=COLORES_CENTRO,
)

# Regiones utilizadas para los mapas
dibujar_cajas_mapa(
    ax,
    wcs,
    reg_mm31_d2,
    etiqueta="MM31 + d2",
    offset=(0, 55),
)

dibujar_cajas_mapa(
    ax,
    wcs,
    reg_north,
    etiqueta="NORTH + MM35 + MM24",
    offset=(-75, 55),
)

hacer_zoom(
    ax,
    wcs,
    ra_deg=CENTRO_IRS2_RA,
    dec_deg=CENTRO_IRS2_DEC,
    ancho_arcsec=ANCHO_IRS2,
)

añadir_barra_escala(ax, wcs, LONGITUD_ESCALA_PC)
añadir_beams(ax)
configurar_ejes(ax)
añadir_colorbar(fig, ax, im)

añadir_titulo_interno(
    ax,
    "a) W51 IRS2 — central region",
)

guardar_figura("W51_IRS2_central")


# ============================================================
# 9. FIGURA 2 — W51 IRS2 MM14
# ============================================================

fig, ax, im = crear_figura_base(norm_mm14)

dibujar_regiones(
    ax,
    wcs,
    regiones,
    fuentes=FUENTES_MM14,
    offsets=OFFSETS_MM14,
    colores= COLORES_MM14
)

dibujar_cajas_mapa(
    ax,
    wcs,
    reg_mm14,
    etiqueta="MM14",
    offset=(80, 95),
)

hacer_zoom(
    ax,
    wcs,
    ra_deg=CENTRO_MM14_RA,
    dec_deg=CENTRO_MM14_DEC,
    ancho_arcsec=ANCHO_MM14,
)

añadir_barra_escala(ax, wcs, LONGITUD_ESCALA_PC)
añadir_beams(ax)
configurar_ejes(ax)
añadir_colorbar(fig, ax, im)

añadir_titulo_interno(
    ax,
    "b) W51 IRS2 — MM14",
)

guardar_figura("W51_IRS2_MM14")


# ============================================================
# 10. FIGURA 3 — W51-E NORTH
# ============================================================

fig, ax, im = crear_figura_base(norm_E)

dibujar_regiones(
    ax,
    wcs,
    regiones,
    fuentes=FUENTES_E_NORTH,
    offsets=OFFSETS_E_NORTH,
    colores= COLORES_E_NORTH
)

dibujar_cajas_mapa(
    ax,
    wcs,
    reg_E_north,
    etiqueta="E North",
    offset=(60, 60),
)

hacer_zoom(
    ax,
    wcs,
    ra_deg=CENTRO_E_NORTH_RA,
    dec_deg=CENTRO_E_NORTH_DEC,
    ancho_arcsec=ANCHO_E_NORTH,
    alto_arcsec=ALTO_E_NORTH,
)

añadir_barra_escala(ax, wcs, LONGITUD_ESCALA_PC)
añadir_beams(ax)

configurar_ejes(
    ax,
    formato_ra="hh:mm:ss.ss",
    formato_dec="dd:mm:ss.s",
)

añadir_colorbar(fig, ax, im)

añadir_titulo_interno(
    ax,
    "c) W51-E North",
)

guardar_figura("W51_E_north")


# ============================================================
# 11. FIGURA 4 — W51-E SOUTH
# ============================================================

fig, ax, im = crear_figura_base(norm_E)

dibujar_regiones(
    ax,
    wcs,
    regiones,
    fuentes=FUENTES_E_SOUTH,
    offsets=OFFSETS_E_SOUTH,
    colores= COLORES_E_SOUTH    
)

dibujar_cajas_mapa(
    ax,
    wcs,
    reg_E_south,
    etiqueta="E South",
    offset=(-50, -40),
)

hacer_zoom(
    ax,
    wcs,
    ra_deg=CENTRO_E_SOUTH_RA,
    dec_deg=CENTRO_E_SOUTH_DEC,
    ancho_arcsec=ANCHO_E_SOUTH,
    alto_arcsec=ALTO_E_SOUTH,
)

añadir_barra_escala(ax, wcs, LONGITUD_ESCALA_PC)
añadir_beams(ax)

configurar_ejes(
    ax,
    formato_ra="hh:mm:ss.ss",
    formato_dec="dd:mm:ss.s",
)

añadir_colorbar(fig, ax, im)

añadir_titulo_interno(
    ax,
    "d) W51-E South",
)

guardar_figura("W51_E_south")

