#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.patheffects as pe
from regions import Regions
from matplotlib.patches import Ellipse
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u

# ============================================================
# CONFIGURACIÓN
# ============================================================

REGION = "E_NORTH"
MOLECULA = "CH3CN"
# MOLECULA_LABEL = r"$\mathrm{CH_3OH}\ v_t=0$"
# MOLECULA_LABEL = r"$\mathrm{anti-C_2H_5OH}$"
# MOLECULA_LABEL = r"$\mathrm{CH_3OCHO}\ v_t=1$"
# MOLECULA_LABEL = r"$\mathrm{C_2H_5CN}$"
MOLECULA_LABEL = r"$\mathrm{CH_3CN}$"

REGIONES_DIR = (
    Path.home()
    / "TFM"
    / "regiones"
)

RUTA_REGION_NORTH = (
    REGIONES_DIR
    / "regionNORTH.reg"
)

RUTA_REGION_MM35 = (
    REGIONES_DIR
    / "regionMM35.reg"
)

RUTA_REGION_MM24 = (
    REGIONES_DIR
    / "regionMM24.reg"
)

BEAM_FITS = (
    Path.home()
    / "TFM"
    / "reprojected"
    / "W51-IRS2_B6_spw0_12M_spw0.JvM.image.pbcor-reprojected-cube.fits"
)


# ============================================================
# ESTILO DE FIGURAS
# ============================================================

FS_AXIS = 20
FS_TICKS = 16
FS_REGION = 18
FS_METHOD = 17
FS_PANEL = 19
FS_COLORBAR = 18
FS_COLORBAR_TICKS = 15
FS_MOLECULE = 25

# ------------------------------------------------------------
# Directorio base de mapas
# ------------------------------------------------------------

MAPS_DIR = (
    Path.home()
    / "TFM"
    / "maps_21Agosto"
    / "maps"
)


# ------------------------------------------------------------
# Mapas chi2
# ------------------------------------------------------------

CHI2_DIR = (
    MAPS_DIR
    / "chi2"
    / REGION
    / MOLECULA
)

# ============================================================
# REGIONES COMPACTAS
# ============================================================

REGIONES_COMPACTAS = {
    "mm31_d2": [
        ("d2", REGIONES_DIR / "regionMF2.reg"),
        ("MM31", REGIONES_DIR / "regionMM31.reg"),
    ],

    "E_NORTH": [
        ("NORTH", REGIONES_DIR / "regionNORTH.reg"),
        ("MM35", REGIONES_DIR / "regionMM35.reg"),
        ("MM24", REGIONES_DIR / "regionMM24.reg"),
    ],
}


# ------------------------------------------------------------
# Mapas de diagrama rotacional
# ------------------------------------------------------------

DIAGROT_DIR = (
    MAPS_DIR
    / "diagrot"
    / REGION
    / MOLECULA
)

REGIONES_MAPA = []

if REGION not in REGIONES_COMPACTAS:
    raise ValueError(
        f"No hay regiones compactas definidas para {REGION}"
    )

for nombre, ruta_region in REGIONES_COMPACTAS[REGION]:

    if not ruta_region.exists():
        raise FileNotFoundError(
            f"No existe la región: {ruta_region}"
        )

    regs = Regions.read(
        ruta_region,
        format="ds9"
    )

    for region in regs:
        REGIONES_MAPA.append(
            (nombre, region)
        )
    
print("[plot_maps] Directorio chi2:")
print(CHI2_DIR)

print("[plot_maps] Directorio diagrot:")
print(DIAGROT_DIR)

if not BEAM_FITS.exists():
    raise FileNotFoundError(
        f"No existe el cubo usado para el beam:\n{BEAM_FITS}"
    )

with fits.open(BEAM_FITS) as hdul:
    HEADER_BEAM = hdul[0].header.copy()

print("\n[beam]")
print("FITS :", BEAM_FITS.name)
print("BMAJ :", HEADER_BEAM.get("BMAJ"))
print("BMIN :", HEADER_BEAM.get("BMIN"))
print("BPA  :", HEADER_BEAM.get("BPA"))

# ============================================================
# FUNCIONES
# ============================================================

def cargar_fits(path):
    """
    Carga un FITS 2D y devuelve data, header.
    """

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    with fits.open(path) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header

    # Por si el FITS tiene dimensiones extra de longitud 1
    data = np.squeeze(data)

    if data.ndim != 2:
        raise ValueError(
            f"El FITS {path.name} no es 2D después de aplicar squeeze(). "
            f"Shape encontrada: {data.shape}"
        )

    return data, header


def buscar_fits_por_nombre(base_dir, patron):
    """
    Busca un FITS usando un patrón parcial.
    """

    archivos = sorted(base_dir.glob(patron))

    if len(archivos) == 0:
        raise FileNotFoundError(
            f"No he encontrado ningún archivo con patrón "
            f"{patron} en {base_dir}"
        )

    if len(archivos) > 1:
        print(f"[aviso] Hay varios archivos para {patron}. Uso el primero:")

        for archivo in archivos:
            print(f"    {archivo.name}")

    return archivos[0]


def buscar_mapa_diagrot(base_dir, parametro):
    """
    Busca los mapas del diagrama rotacional.

    parametro:
        'Tex'
        'Ncol'

    Primero prueba nombres habituales y, si no existen,
    busca automáticamente dentro de la carpeta.
    """

    if parametro == "Tex":

        candidatos = [
            f"{MOLECULA}_Tex.fits",
            f"{MOLECULA}_T_ex.fits",
            "Tex.fits",
            "T_ex.fits",
        ]

        palabras = ["tex"]

    elif parametro == "Ncol":

        candidatos = [
            f"{MOLECULA}_Ncol.fits",
            f"{MOLECULA}_N_col.fits",
            "Ncol.fits",
            "N_col.fits",
        ]

        palabras = ["ncol"]

    else:
        raise ValueError(
            "parametro debe ser 'Tex' o 'Ncol'"
        )

    # --------------------------------------------------------
    # 1. Probar nombres concretos
    # --------------------------------------------------------

    for nombre in candidatos:

        path = base_dir / nombre

        if path.exists():
            return path

    # --------------------------------------------------------
    # 2. Búsqueda automática
    # --------------------------------------------------------

    archivos = sorted(base_dir.glob("*.fits"))

    encontrados = []

    for archivo in archivos:

        nombre = archivo.name.lower()

        # Evitamos mapas de incertidumbre
        if (
            "delta" in nombre
            or "error" in nombre
            or "sigma" in nombre
            or "uncert" in nombre
        ):
            continue

        if any(palabra in nombre for palabra in palabras):
            encontrados.append(archivo)

    if len(encontrados) == 0:

        print("\n[error] FITS disponibles en:")
        print(base_dir)

        for archivo in archivos:
            print("   ", archivo.name)

        raise FileNotFoundError(
            f"No he encontrado el mapa {parametro} "
            f"del diagrama rotacional."
        )

    if len(encontrados) > 1:

        print(
            f"[aviso] He encontrado varios candidatos "
            f"para {parametro}:"
        )

        for archivo in encontrados:
            print("   ", archivo.name)

        print(
            f"[aviso] Uso: {encontrados[0].name}"
        )

    return encontrados[0]

def dibujar_beam(
    ax,
    wcs_mapa,
    header_beam,
    color="white",
    edgecolor="black",
    x_frac=0.12,
    y_frac=0.12,
):
    """
    Dibuja el synthesized beam usando BMAJ, BMIN y BPA
    del cubo ALMA original.
    """

    bmaj = header_beam.get("BMAJ")
    bmin = header_beam.get("BMIN")
    bpa = header_beam.get("BPA")

    if (
        bmaj is None
        or bmin is None
        or bpa is None
    ):
        print(
            "[beam] El cubo original no contiene "
            "BMAJ/BMIN/BPA."
        )
        return

    # Escala angular del mapa representado [deg/pixel]
    pixel_scales = proj_plane_pixel_scales(
        wcs_mapa
    )

    pixscale_x = abs(pixel_scales[0])
    pixscale_y = abs(pixel_scales[1])

    # BMAJ y BMIN están en grados
    beam_width = bmin / pixscale_x
    beam_height = bmaj / pixscale_y

    # Tamaño del mapa
    ny, nx = ax.images[0].get_array().shape

    # Posición: esquina inferior izquierda
    x = x_frac * nx
    y = y_frac * ny

    print(
        f"[beam] "
        f"{bmaj * 3600:.3f}\" × "
        f"{bmin * 3600:.3f}\"  "
        f"PA={bpa:.1f} deg"
    )

    beam = Ellipse(
        (x, y),
        width=beam_width,
        height=beam_height,
        angle=90 + bpa,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=1.5,
        zorder=30,
    )

    ax.add_patch(beam)

def dibujar_regiones_mapa(
    ax,
    wcs,
    regiones,
    color="cyan"
):
    """
    Dibuja las regiones DS9 de d2 y MM31
    y añade el nombre de cada una.
    """

    for nombre, region in regiones:

        region_pix = region.to_pixel(wcs)

        # Dibujar región
        artist = region_pix.as_artist(
            edgecolor=color,
            facecolor="none",
            linewidth=2.5,
            zorder=10
        )

        ax.add_artist(artist)

        # Etiqueta
        etiqueta = ax.annotate(
            nombre,
            xy=(
                region_pix.center.x,
                region_pix.center.y
            ),
            xytext=(7, 7),
            textcoords="offset points",
            color=color,
            fontsize=FS_REGION,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=11
        )

        etiqueta.set_path_effects([
            pe.Stroke(
                linewidth=2.0,
                foreground="black"
            ),
            pe.Normal()
        ])
        
def plot_mapa(
    data,
    header,
    titulo,
    label_cbar,
    cmap="inferno",
    log=False,
    percent_min=5,
    percent_max=95,
    save_path=None,
):
    """
    Representa un mapa 2D usando las coordenadas celestes
    contenidas en el header FITS.
    """

    data_plot = np.array(data, dtype=float)

    if log:
        data_plot = np.where(
            data_plot > 0,
            np.log10(data_plot),
            np.nan,
        )

    if np.all(~np.isfinite(data_plot)):
        print(
            f"[aviso] El mapa {titulo} "
            f"no tiene valores finitos."
        )
        return

    vmin = np.nanpercentile(
        data_plot,
        percent_min,
    )

    vmax = np.nanpercentile(
        data_plot,
        percent_max,
    )

    # ========================================================
    # WCS
    # ========================================================

    wcs = WCS(header).celestial

    fig = plt.figure(
        figsize=(6, 5)
    )

    ax = fig.add_subplot(
        111,
        projection=wcs,
    )
    im = ax.imshow(
        data_plot,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    dibujar_regiones_mapa(
    ax,
    wcs,
    REGIONES_MAPA
)


    # ========================================================
    # Coordenadas
    # ========================================================

    ra = ax.coords[0]
    dec = ax.coords[1]

    ra.set_axislabel(
    "R.A. (J2000)",
    fontsize=FS_AXIS
)

    dec.set_axislabel(
    "Dec. (J2000)",
    fontsize=FS_AXIS
)

    ra.set_ticklabel(
    size=FS_TICKS
)

    dec.set_ticklabel(
    size=FS_TICKS
)

    ra.set_major_formatter(
        "hh:mm:ss.s"
    )

    dec.set_major_formatter(
        "dd:mm:ss"
    )

    ra.set_ticks_position("bt")
    dec.set_ticks_position("lr")
    
    ra.set_ticklabel_position("b")
    dec.set_ticklabel_position("l")
    
    # ========================================================
    # Colorbar
    # ========================================================

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.025,
        fraction=0.035,
        shrink=0.80,
        aspect=30,
    )

    cbar.set_label(
        label_cbar,
        fontsize=14
    )

    cbar.ax.tick_params(
        labelsize=12
    )

    # ========================================================
    # Título
    # ========================================================
    
    ax.set_title(
        titulo,
        fontsize=15,
        pad=12
    )

    plt.tight_layout()

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"[plot] Guardado: {save_path}"
        )

    plt.show()


def comparar_mapas(
    data1,
    data2,
    header1,
    header2,
    titulo1,
    titulo2,
    label_cbar,
    panel_label,
    cmap="inferno",
    log=False,
    percent_min=5,
    percent_max=95,
    save_path=None,
):
    """
    Representa dos mapas utilizando coordenadas celestes
    y exactamente la misma escala de color.

    Panel izquierdo:
        Rotational diagram

    Panel derecho:
        chi2 fit
    """

    # ========================================================
    # Preparar datos
    # ========================================================

    data1_plot = np.array(
        data1,
        dtype=float,
    )

    data2_plot = np.array(
        data2,
        dtype=float,
    )

    if log:

        data1_plot = np.where(
            data1_plot > 0,
            np.log10(data1_plot),
            np.nan,
        )

        data2_plot = np.where(
            data2_plot > 0,
            np.log10(data2_plot),
            np.nan,
        )

    # ========================================================
    # Valores válidos
    # ========================================================

    valores1 = data1_plot[
        np.isfinite(data1_plot)
    ]

    valores2 = data2_plot[
        np.isfinite(data2_plot)
    ]

    if (
        len(valores1) == 0
        or len(valores2) == 0
    ):

        print(
            "[aviso] Uno de los mapas no tiene "
            "valores válidos para comparar."
        )

        return

    valores = np.concatenate(
        [
            valores1,
            valores2,
        ]
    )

    # ========================================================
    # Escala de color común
    # ========================================================

    vmin = np.nanpercentile(
        valores,
        percent_min,
    )

    vmax = np.nanpercentile(
        valores,
        percent_max,
    )

    # ========================================================
    # WCS
    # ========================================================

    wcs1 = WCS(header1).celestial
    wcs2 = WCS(header2).celestial

    # ========================================================
    # Figura
    # ========================================================

    fig = plt.figure(
        figsize=(15, 6)
    )

    ax1 = fig.add_subplot(
        121,
        projection=wcs1,
    )

    ax2 = fig.add_subplot(
        122,
        projection=wcs2,
    )

    # ========================================================
    # PANEL IZQUIERDO — ROTATIONAL DIAGRAM
    # ========================================================

    im = ax1.imshow(
        data1_plot,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Regiones compactas
    dibujar_regiones_mapa(
        ax1,
        wcs1,
        REGIONES_MAPA
    )

    # Beam
    dibujar_beam(
        ax1,
        wcs1,
        HEADER_BEAM
    )

    # --------------------------------------------------------
    # Nombre del método
    # --------------------------------------------------------

    ax1.text(
        0.96,
        0.05,
        titulo1,
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=FS_METHOD,
        color="white",
        fontweight="bold",
        zorder=20,
        path_effects=[
            pe.Stroke(
                linewidth=2.5,
                foreground="black"
            ),
            pe.Normal()
        ]
    )

    # --------------------------------------------------------
    # Identificador a) / b)
    # --------------------------------------------------------

    ax1.text(
        0.03,
        0.96,
        panel_label,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=FS_PANEL,
        fontweight="bold",
        color="white",
        zorder=20,
        path_effects=[
            pe.Stroke(
                linewidth=2.5,
                foreground="black"
            ),
            pe.Normal()
        ]
    )

    # --------------------------------------------------------
    # Coordenadas panel izquierdo
    # --------------------------------------------------------

    ra1 = ax1.coords[0]
    dec1 = ax1.coords[1]

    ra1.set_axislabel(
        "R.A. (J2000)",
        fontsize=FS_AXIS
    )

    dec1.set_axislabel(
        "Dec. (J2000)",
        fontsize=FS_AXIS
    )

    # Más ticks principales
    ra1.set_ticks(
        spacing=0.75 * u.arcsec
    )

    dec1.set_ticks(
        spacing=0.5 * u.arcsec
    )

    # Formato
    ra1.set_major_formatter(
        "hh:mm:ss.ss"
    )

    dec1.set_major_formatter(
        "dd:mm:ss.s"
    )

    # Tamaño de números
    ra1.set_ticklabel(
        size=FS_TICKS
    )

    dec1.set_ticklabel(
        size=FS_TICKS
    )

    # Ticks en los bordes
    ra1.set_ticks_position("bt")
    dec1.set_ticks_position("lr")

    ra1.set_ticklabel_position("b")
    dec1.set_ticklabel_position("l")

    # Ticks menores
    ra1.display_minor_ticks(True)
    dec1.display_minor_ticks(True)

    ra1.set_minor_frequency(2)
    dec1.set_minor_frequency(2)

    # ========================================================
    # PANEL DERECHO — CHI2 FIT
    # ========================================================

    ax2.imshow(
        data2_plot,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Regiones compactas
    dibujar_regiones_mapa(
        ax2,
        wcs2,
        REGIONES_MAPA
    )

    # --------------------------------------------------------
    # Nombre de la molécula
    # --------------------------------------------------------

    ax2.text(
        0.96,
        0.96,
        MOLECULA_LABEL,
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=FS_MOLECULE,
        fontweight="bold",
        color="white",
        zorder=20,
        path_effects=[
            pe.Stroke(
                linewidth=2.5,
                foreground="black"
                ),
            pe.Normal()
            ]
        )

    # --------------------------------------------------------
    # Nombre del método
    # --------------------------------------------------------

    ax2.text(
        0.96,
        0.05,
        titulo2,
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=FS_METHOD,
        color="white",
        fontweight="bold",
        zorder=20,
        path_effects=[
            pe.Stroke(
                linewidth=2.5,
                foreground="black"
            ),
            pe.Normal()
        ]
    )

    # --------------------------------------------------------
    # Coordenadas panel derecho
    # --------------------------------------------------------

    ra2 = ax2.coords[0]
    dec2 = ax2.coords[1]

    ra2.set_axislabel(
        "R.A. (J2000)",
        fontsize=FS_AXIS
    )

    # No repetimos Dec
    dec2.set_axislabel("")

    # Mismo espaciado que el panel izquierdo
    ra2.set_ticks(
        spacing=0.75 * u.arcsec
    )

    dec2.set_ticks(
        spacing=0.5 * u.arcsec
    )

    # Formato
    ra2.set_major_formatter(
        "hh:mm:ss.ss"
    )

    dec2.set_major_formatter(
        "dd:mm:ss.s"
    )

    # Tamaño de números
    ra2.set_ticklabel(
        size=FS_TICKS
    )

    dec2.set_ticklabel(
        size=FS_TICKS
    )

    # Ticks en los bordes
    ra2.set_ticks_position("bt")
    dec2.set_ticks_position("lr")

    ra2.set_ticklabel_position("b")

    # No mostramos números de Dec
    # en el segundo panel
    dec2.set_ticklabel_visible(False)

    # Ticks menores
    ra2.display_minor_ticks(True)
    dec2.display_minor_ticks(True)

    ra2.set_minor_frequency(2)
    dec2.set_minor_frequency(2)

    # ========================================================
    # Espaciado de la figura
    # ========================================================

    plt.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.15,
        top=0.95,
        wspace=0.28,
    )

    # ========================================================
    # Colorbar común
    # ========================================================

    # Cogemos exactamente la posición del segundo mapa
    pos = ax2.get_position()

    cax = fig.add_axes([
        pos.x1 + 0.018,
        pos.y0,
        0.015,
        pos.height,
    ])

    cbar = fig.colorbar(
        im,
        cax=cax,
    )

    cbar.set_label(
        label_cbar,
        fontsize=FS_COLORBAR,
        labelpad=12,
    )

    cbar.ax.tick_params(
        labelsize=FS_COLORBAR_TICKS
    )

    # ========================================================
    # Guardar
    # ========================================================

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"[plot] Guardado: {save_path}"
        )

    # ========================================================
    # Mostrar
    # ========================================================

    plt.show()

# ============================================================
# CARGAR MAPAS CHI2
# ============================================================

path_T = (
    CHI2_DIR
    / f"{MOLECULA}_Tex_chi2.fits"
)

path_N = (
    CHI2_DIR
    / f"{MOLECULA}_Ncol_chi2.fits"
)

path_dT = (
    CHI2_DIR
    / f"{MOLECULA}_deltaTex_chi2.fits"
)

path_dN = (
    CHI2_DIR
    / f"{MOLECULA}_deltaNcol_chi2.fits"
)

path_chi2 = (
    CHI2_DIR
    / f"{MOLECULA}_chi2_min.fits"
)


print("\n[plot_maps] Archivos chi2 usados:")

print("  T     :", path_T.name)
print("  N     :", path_N.name)
print("  dT    :", path_dT.name)
print("  dN    :", path_dN.name)
print("  chi2  :", path_chi2.name)


T_fit_map, header = cargar_fits(
    path_T
)

N_fit_map, _ = cargar_fits(
    path_N
)

deltaT_map, _ = cargar_fits(
    path_dT
)

deltaN_map, _ = cargar_fits(
    path_dN
)

chi2_map, _ = cargar_fits(
    path_chi2
)

# ============================================================
# CARGAR MAPAS DEL DIAGRAMA ROTACIONAL
# ============================================================

path_T_diagrot = buscar_mapa_diagrot(
    DIAGROT_DIR,
    "Tex",
)

path_N_diagrot = buscar_mapa_diagrot(
    DIAGROT_DIR,
    "Ncol",
)


print(
    "\n[plot_maps] Archivos del "
    "diagrama rotacional usados:"
)

print(
    "  T     :",
    path_T_diagrot.name,
)

print(
    "  N     :",
    path_N_diagrot.name,
)


T_diagrot_map, header_diagrot = cargar_fits(
    path_T_diagrot
)

N_diagrot_map, _ = cargar_fits(
    path_N_diagrot
)


# ============================================================
# COMPROBAR DIMENSIONES
# ============================================================

print("\n[shapes]")

print(
    "  chi2 T    :",
    T_fit_map.shape,
)

print(
    "  diagrot T :",
    T_diagrot_map.shape,
)

print(
    "  chi2 N    :",
    N_fit_map.shape,
)

print(
    "  diagrot N :",
    N_diagrot_map.shape,
)


if T_fit_map.shape != T_diagrot_map.shape:

    print(
        "[aviso] Los mapas de temperatura "
        "chi2 y diagrot no tienen la misma forma."
    )


if N_fit_map.shape != N_diagrot_map.shape:

    print(
        "[aviso] Los mapas de columna "
        "chi2 y diagrot no tienen la misma forma."
    )


# ============================================================
# MÁSCARA DE CALIDAD CHI2
# ============================================================

frac_deltaT = (
    deltaT_map
    / T_fit_map
)

frac_deltaN = (
    deltaN_map
    / N_fit_map
)


frac_deltaT = np.where(
    np.isfinite(frac_deltaT),
    frac_deltaT,
    np.nan,
)

frac_deltaN = np.where(
    np.isfinite(frac_deltaN),
    frac_deltaN,
    np.nan,
)


quality_mask = (

    np.isfinite(T_fit_map)

    & np.isfinite(N_fit_map)

    & np.isfinite(deltaT_map)

    & np.isfinite(deltaN_map)

    & np.isfinite(chi2_map)

    & (T_fit_map > 20)

    & (T_fit_map < 600)

    & (N_fit_map > 0)

    & (frac_deltaT < 1.0)

    & (frac_deltaN < 2.0)

    & (chi2_map < 250)

)


print(
    "\n[quality] píxeles totales:",
    np.isfinite(T_fit_map).sum(),
)

print(
    "[quality] píxeles aceptados:",
    quality_mask.sum(),
)


T_masked = np.where(
    quality_mask,
    T_fit_map,
    np.nan,
)

N_masked = np.where(
    quality_mask,
    N_fit_map,
    np.nan,
)

chi2_masked = np.where(
    quality_mask,
    chi2_map,
    np.nan,
)

logN_masked = np.where(
    N_masked > 0,
    np.log10(N_masked),
    np.nan,
)


# ============================================================
# DERIVADOS CHI2
# ============================================================

logN_map = np.where(
    N_fit_map > 0,
    np.log10(N_fit_map),
    np.nan,
)


# ============================================================
# DERIVADOS DIAGRAMA ROTACIONAL
# ============================================================

logN_diagrot_map = np.where(
    N_diagrot_map > 0,
    np.log10(N_diagrot_map),
    np.nan,
)


# ============================================================
# CARPETA DE SALIDA
# ============================================================

OUT_DIR = CHI2_DIR / "plots"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PLOTS CHI2
# ============================================================

plot_mapa(
    T_fit_map,
    header,
    titulo=f"{MOLECULA} - T_fit - {REGION}",
    label_cbar=r"$T_{\rm ex}$ [K]",
    cmap="inferno",
    save_path=OUT_DIR / f"{MOLECULA}_{REGION}_T_fit.pdf",
)

plot_mapa(
    N_fit_map,
    header,
    titulo=f"{MOLECULA} - log10(N_fit) - {REGION}",
    label_cbar=r"$\log_{10}(N_{\rm col}\,[{\rm cm}^{-2}])$",
    cmap="viridis",
    log=True,
    save_path=OUT_DIR / f"{MOLECULA}_{REGION}_logN_fit.pdf",
)

plot_mapa(
    deltaT_map,
    header,
    titulo=(
        f"{MOLECULA} - delta T - {REGION}"
    ),
    label_cbar=r"$\Delta T$ [K]",
    cmap="magma",
    save_path=(
        OUT_DIR
        / f"{MOLECULA}_{REGION}_deltaT.pdf"
    ),
)

plot_mapa(
    deltaN_map,
    header,
    titulo=(
        f"{MOLECULA} - log10(delta N) - {REGION}"
    ),
    label_cbar=(
        r"log10($\Delta N$ [cm$^{-2}$])"
    ),
    cmap="magma",
    log=True,
    save_path=(
        OUT_DIR
        / f"{MOLECULA}_{REGION}_log_deltaN.pdf"
    ),
)


plot_mapa(
    chi2_map,
    header,
    titulo=(
        f"{MOLECULA} - chi2 mínimo - {REGION}"
    ),
    label_cbar=r"$\chi^2_{\rm min}$",
    cmap="cividis",
    save_path=(
        OUT_DIR
        / f"{MOLECULA}_{REGION}_chi2_min.pdf"
    ),
)


plot_mapa(
    frac_deltaT,
    header,
    titulo=(
        f"{MOLECULA} - deltaT / T_fit - {REGION}"
    ),
    label_cbar=r"$\Delta T/T_{\rm fit}$",
    cmap="plasma",
    save_path=(
        OUT_DIR
        / f"{MOLECULA}_{REGION}_frac_deltaT.pdf"
    ),
)


plot_mapa(
    frac_deltaN,
    header,
    titulo=(
        f"{MOLECULA} - deltaN / N_fit - {REGION}"
    ),
    label_cbar=r"$\Delta N/N_{\rm fit}$",
    cmap="plasma",
    save_path=(
        OUT_DIR
        / f"{MOLECULA}_{REGION}_frac_deltaN.pdf"
    ),
)


# ============================================================
# PLOTS DIAGRAMA ROTACIONAL
# ============================================================

plot_mapa(
    T_diagrot_map,
    header_diagrot,
    titulo=f"{MOLECULA} - T_ex rotational diagram - {REGION}",
    label_cbar=r"$T_{\rm ex}$ [K]",
    cmap="inferno",
    save_path=OUT_DIR / f"{MOLECULA}_{REGION}_Tex_diagrot.pdf",
)


plot_mapa(
    N_diagrot_map,
    header_diagrot,
    titulo=f"{MOLECULA} - N_col rotational diagram - {REGION}",
    label_cbar=r"$\log_{10}(N_{\rm col}\,[{\rm cm}^{-2}])$",
    cmap="viridis",
    log=True,
    save_path=OUT_DIR / f"{MOLECULA}_{REGION}_logN_diagrot.pdf",
)

# ============================================================
# COMPARACIÓN CHI2 vs DIAGRAMA ROTACIONAL
# ============================================================

if T_fit_map.shape == T_diagrot_map.shape:

    comparar_mapas(
    T_diagrot_map,
    T_fit_map,
    header_diagrot,
    header,
    titulo1="Rotational diagram",
    titulo2=r"$\chi^2$ fit",
    panel_label=r"a) $T_{\rm ex}$ [K]",
    label_cbar=r"$T_{\rm ex}$ [K]",
    cmap="inferno",
    save_path=(
        OUT_DIR
        / f"{MOLECULA}_{REGION}_comparison_Tex.pdf"
    ),
)

else:

    print(
        "[aviso] No hago comparación de Tex "
        "porque las dimensiones son diferentes."
    )


if N_fit_map.shape == N_diagrot_map.shape:

    comparar_mapas(
    N_diagrot_map,
    N_fit_map,
    header_diagrot,
    header,
    titulo1="Rotational diagram",
    titulo2=r"$\chi^2$ fit",
    panel_label=(
        r"b) $N_{\rm col}$ [cm$^{-2}$]"
    ),
    label_cbar=(
        r"$\log_{10}(N_{\rm col}\,[{\rm cm}^{-2}])$"
    ),
    cmap="viridis",
    log=True,
    save_path=(
        OUT_DIR
        / f"{MOLECULA}_{REGION}_comparison_Ncol.pdf"
    ),
)

else:

    print(
        "[aviso] No hago comparación de Ncol "
        "porque las dimensiones son diferentes."
    )


print("\n[plot_maps] Terminado.")