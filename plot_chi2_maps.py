#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.patheffects as pe
from regions import Regions

# ============================================================
# CONFIGURACIÓN
# ============================================================

REGION = "NORTH_MAP"            # cambia esto si hace falta
MOLECULA = "C2H5OH_g"      # cambia esto por la molécula ajustada
REGIONES_DIR = Path.home() / "TFM" / "regiones"

RUTA_REGION_NORTH = REGIONES_DIR / "regionNORTH.reg"
RUTA_REGION_MM35 = REGIONES_DIR / "regionMM35.reg"
RUTA_REGION_MM24 = REGIONES_DIR / "regionMM24.reg"
# ------------------------------------------------------------
# Rutas
# ------------------------------------------------------------

CHI2_DIR = (
    Path.home()
    / "TFM"
    / "tables"
    / "chi2_maps"
    / REGION
    / MOLECULA
)

DIAGROT_DIR = (
    Path.home()
    / "TFM"
    / "maps"
    / "diagrot"
    / REGION
    / MOLECULA
)

REGIONES_MAPA = []

for nombre, ruta_region in [
    ("NORTH", RUTA_REGION_NORTH),
    ("MM35", RUTA_REGION_MM35),
    ("MM24", RUTA_REGION_MM24)
]:

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
            linewidth=2.0,
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
            fontsize=9,
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
        figsize=(7, 6)
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
        "Right Ascension (J2000)",
        fontsize=16
    )

    dec.set_axislabel(
        "Declination (J2000)",
        fontsize=16
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

    ra.set_ticklabel(
        size=13
    )

    dec.set_ticklabel(
        size=13
    )

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
    cmap="inferno",
    log=False,
    percent_min=5,
    percent_max=95,
    save_path=None,
):
    """
    Representa dos mapas utilizando coordenadas celestes
    y exactamente la misma escala de color.
    """

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
    # Escala común
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
    # Primer mapa
    # ========================================================

    im = ax1.imshow(
        data1_plot,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    dibujar_regiones_mapa(
        ax1,
        wcs1,
        REGIONES_MAPA
    )

    ax1.set_title(
        titulo1,
        fontsize=14,
        pad=10
    )

    ra1 = ax1.coords[0]
    dec1 = ax1.coords[1]

    ra1.set_axislabel(
        "Right Ascension (J2000)",
        fontsize=15
    )

    dec1.set_axislabel(
        "Declination (J2000)",
        fontsize=15
    )

    ra1.set_major_formatter(
        "hh:mm:ss.s"
    )

    dec1.set_major_formatter(
        "dd:mm:ss"
    )

    ra1.set_ticklabel(
        size=12
    )

    dec1.set_ticklabel(
        size=12
    )

    # ========================================================
    # Segundo mapa
    # ========================================================

    ax2.imshow(
        data2_plot,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    dibujar_regiones_mapa(
        ax2,
        wcs2,
        REGIONES_MAPA
    )

    ax2.set_title(
        titulo2,
        fontsize=14,
        pad=10
    )

    ra2 = ax2.coords[0]
    dec2 = ax2.coords[1]

    ra2.set_axislabel(
        "Right Ascension (J2000)",
        fontsize=15
    )

    # --------------------------------------------------------
    # IMPORTANTE:
    # quitamos la etiqueta de Dec en el segundo panel
    # para no duplicar información y evitar solapamiento.
    # --------------------------------------------------------

    dec2.set_axislabel("")

    ra2.set_major_formatter(
        "hh:mm:ss.s"
    )

    dec2.set_major_formatter(
        "dd:mm:ss"
    )

    ra2.set_ticklabel(
        size=12
    )

    dec2.set_ticklabel(
        size=12
    )

    # Opcional: ocultamos las etiquetas numéricas de Dec
    # del segundo panel porque ambos mapas cubren la misma región.
    dec2.set_ticklabel_visible(False)

    # ========================================================
    # Colorbar común
    # ========================================================

    # Reservamos un eje independiente para la colorbar
    cax = fig.add_axes([
        0.92,   # posición horizontal
        0.18,   # posición vertical
        0.015,  # ancho
        0.64    # alto
    ])

    cbar = fig.colorbar(
        im,
        cax=cax
    )

    cbar.set_label(
        label_cbar,
        fontsize=14,
        labelpad=10
    )

    cbar.ax.tick_params(
        labelsize=11
    )

    # ========================================================
    # Espaciado
    # ========================================================

    plt.subplots_adjust(
        left=0.08,
        right=0.90,
        bottom=0.15,
        top=0.90,
        wspace=0.28
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
    save_path=OUT_DIR / f"{MOLECULA}_{REGION}_T_fit.png",
)

plot_mapa(
    N_fit_map,
    header,
    titulo=f"{MOLECULA} - log10(N_fit) - {REGION}",
    label_cbar=r"$\log_{10}(N_{\rm col}\,[{\rm cm}^{-2}])$",
    cmap="viridis",
    log=True,
    save_path=OUT_DIR / f"{MOLECULA}_{REGION}_logN_fit.png",
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
        / f"{MOLECULA}_{REGION}_deltaT.png"
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
        / f"{MOLECULA}_{REGION}_log_deltaN.png"
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
        / f"{MOLECULA}_{REGION}_chi2_min.png"
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
        / f"{MOLECULA}_{REGION}_frac_deltaT.png"
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
        / f"{MOLECULA}_{REGION}_frac_deltaN.png"
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
    save_path=OUT_DIR / f"{MOLECULA}_{REGION}_Tex_diagrot.png",
)


plot_mapa(
    N_diagrot_map,
    header_diagrot,
    titulo=f"{MOLECULA} - N_col rotational diagram - {REGION}",
    label_cbar=r"$\log_{10}(N_{\rm col}\,[{\rm cm}^{-2}])$",
    cmap="viridis",
    log=True,
    save_path=OUT_DIR / f"{MOLECULA}_{REGION}_logN_diagrot.png",
)

# ============================================================
# COMPARACIÓN CHI2 vs DIAGRAMA ROTACIONAL
# ============================================================

if T_fit_map.shape == T_diagrot_map.shape:

    comparar_mapas(
    T_fit_map,
    T_diagrot_map,
    header,
    header_diagrot,
    titulo1=r"$\chi^2$ fit",
    titulo2="Rotational diagram",
    label_cbar=r"$T_{\rm ex}$ [K]",
    cmap="inferno",
    save_path=(
        OUT_DIR
        / f"{MOLECULA}_{REGION}_comparison_Tex.png"
    ),
)

else:

    print(
        "[aviso] No hago comparación de Tex "
        "porque las dimensiones son diferentes."
    )


if N_fit_map.shape == N_diagrot_map.shape:

    comparar_mapas(
    N_fit_map,
    N_diagrot_map,
    header,
    header_diagrot,
    titulo1=r"$\chi^2$ fit",
    titulo2="Rotational diagram",
    label_cbar=(
        r"$\log_{10}(N_{\rm col}\,[{\rm cm}^{-2}])$"
    ),
    cmap="viridis",
    log=True,
    save_path=(
        OUT_DIR
        / f"{MOLECULA}_{REGION}_comparison_Ncol.png"
    ),
)

else:

    print(
        "[aviso] No hago comparación de Ncol "
        "porque las dimensiones son diferentes."
    )


print("\n[plot_maps] Terminado.")