#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:05:59 2026

@author: jorge
"""

'''
Esto es para hacer el analisis de los mapas, crear como varía la T_ex y la 
N_col según el radio
'''

from pathlib import Path
from astropy.wcs import WCS
from regions import Regions, SkyRegion
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.wcs.utils import proj_plane_pixel_scales

ruta_fits_chi2 = Path("/home/jorge/TFM/tables/chi2_maps")

ruta_regiones = Path("/home/jorge/TFM/regiones")

REGION = "mm31_d2"
MOLECULA = "C2H5CN"

# Carpeta concreta de la región y molécula
ruta_mapas = (ruta_fits_chi2 / REGION / MOLECULA)

# Nombres generados por save_chi2_maps_fits()
rutas_fits = {
    "T_fit_map": (
        ruta_mapas
        / f"{MOLECULA}_Tex_chi2.fits"
    ),
    "N_fit_map": (
        ruta_mapas
        / f"{MOLECULA}_Ncol_chi2.fits"
    ),
    "deltaT_map": (
        ruta_mapas
        / f"{MOLECULA}_deltaTex_chi2.fits"
    ),
    "deltaN_map": (
        ruta_mapas
        / f"{MOLECULA}_deltaNcol_chi2.fits"
    ),
}

# Comprobar que existen
for nombre, ruta in rutas_fits.items():
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encuentra {nombre}: {ruta}"
        )

# Cargar los datos
mapas_chi2 = {}

for nombre, ruta in rutas_fits.items():

    with fits.open(ruta) as hdul:
        mapas_chi2[nombre] = (
            hdul[0].data.astype(float).copy()
        )

# El header puede tomarse del mapa de temperatura
with fits.open(rutas_fits["T_fit_map"]) as hdul:
    header_chi2 = hdul[0].header.copy()

print(
    f"Mapas χ² cargados para {MOLECULA} "
    f"en {REGION}:"
)

for nombre, mapa in mapas_chi2.items():
    print(
        f"    {nombre}: "
        f"shape={mapa.shape}, "
        f"píxeles finitos={np.count_nonzero(np.isfinite(mapa))}"
    )

#Rutas de las regiones compactas

regiones_compactas = {
    "NORTH_MAP":["regionMM24.reg", "regionMM35.reg", "regionNORTH.reg"],
    "mm31_d2": ["regionMM31.reg", "regionMF2.reg"],
    "MM14_MAP": ["regionMM14.reg"], 
    "E_NORTH": ["regione2e.reg", "regione2w.reg"] 
    }

rutas_regiones_compactas = [] 
for regionesc in regiones_compactas[REGION]: 
    rutas_regiones_compactas.append(ruta_regiones / regionesc) 

# ============================================================
# RECORTAR LOS MAPAS EN LAS REGIONES COMPACTAS
# ============================================================

ruta_salida_recortes = (
    ruta_fits_chi2
    / REGION
    / MOLECULA
    / "regiones_compactas"
)

ruta_salida_recortes.mkdir(
    parents=True,
    exist_ok=True,
)

wcs_chi2 = WCS(header_chi2).celestial
shape_mapas = mapas_chi2["T_fit_map"].shape


def construir_mascara_region(
        ruta_region,
        wcs,
        shape):
    """
    Lee una región DS9 y construye una máscara booleana 2D.

    Si el archivo contiene varias figuras, se utiliza la unión
    de todas ellas.
    """

    regiones = Regions.read(
        ruta_region,
        format="ds9",
    )

    mascara_total = np.zeros(
        shape,
        dtype=bool,
    )

    for region in regiones:

        # Ignorar regiones marcadas como excluidas en DS9.
        if region.meta.get("include", 1) == 0:
            continue

        # Convertir coordenadas celestes a coordenadas de píxel.
        if isinstance(region, SkyRegion):
            region_pixel = region.to_pixel(wcs)
        else:
            region_pixel = region

        mascara_region = region_pixel.to_mask(
            mode="center",
        )

        imagen_mascara = mascara_region.to_image(
            shape,
        )

        if imagen_mascara is not None:
            mascara_total |= imagen_mascara > 0

    if not np.any(mascara_total):
        raise ValueError(
            f"La región {ruta_region.name} no contiene "
            "ningún píxel del mapa."
        )

    return mascara_total


def recortar_mapa_con_mascara(
        mapa,
        mascara):
    """
    Recorta un mapa al rectángulo mínimo que contiene la región.

    Los píxeles del rectángulo que quedan fuera de la región
    se establecen como NaN.
    """

    coordenadas_y, coordenadas_x = np.where(
        mascara
    )

    y_min = coordenadas_y.min()
    y_max = coordenadas_y.max() + 1

    x_min = coordenadas_x.min()
    x_max = coordenadas_x.max() + 1

    corte_y = slice(y_min, y_max)
    corte_x = slice(x_min, x_max)

    mapa_recortado = mapa[
        corte_y,
        corte_x,
    ].copy()

    mascara_recortada = mascara[
        corte_y,
        corte_x,
    ]

    mapa_recortado[
        ~mascara_recortada
    ] = np.nan

    return (
        mapa_recortado,
        mascara_recortada,
        corte_y,
        corte_x,
    )


recortes_regiones = {}

for ruta_region in rutas_regiones_compactas:

    nombre_region_compacta = ruta_region.stem

    # Eliminar el prefijo "region" del nombre.
    if nombre_region_compacta.startswith("region"):
        nombre_region_compacta = (
            nombre_region_compacta[len("region"):]
        )

    print(
        f"\n[recorte] Procesando "
        f"{nombre_region_compacta}"
    )

    if not ruta_region.exists():
        raise FileNotFoundError(
            f"No se encuentra la región: {ruta_region}"
        )

    mascara = construir_mascara_region(
        ruta_region=ruta_region,
        wcs=wcs_chi2,
        shape=shape_mapas,
    )

    recortes_regiones[
        nombre_region_compacta
    ] = {}

    # Todos los mapas tienen la misma forma, por lo que los
    # límites del recorte serán iguales.
    (
        _,
        mascara_recortada,
        corte_y,
        corte_x,
    ) = recortar_mapa_con_mascara(
        mapas_chi2["T_fit_map"],
        mascara,
    )

    # Actualizar el WCS para el nuevo recorte.
    wcs_recortado = wcs_chi2.slice(
        (
            corte_y,
            corte_x,
        )
    )

    header_recortado_base = (
        wcs_recortado.to_header()
    )

    ruta_salida_region = (
        ruta_salida_recortes
        / nombre_region_compacta
    )

    ruta_salida_region.mkdir(
        parents=True,
        exist_ok=True,
    )

    for nombre_mapa, mapa in mapas_chi2.items():

        mapa_recortado = mapa[
            corte_y,
            corte_x,
        ].copy()

        mapa_recortado[
            ~mascara_recortada
        ] = np.nan

        recortes_regiones[
            nombre_region_compacta
        ][nombre_mapa] = mapa_recortado

        header_salida = (
            header_recortado_base.copy()
        )

        header_salida["MOLEC"] = MOLECULA
        header_salida["REGION"] = (
            nombre_region_compacta
        )
        header_salida["MAPTYPE"] = nombre_mapa

        if nombre_mapa in {
            "T_fit_map",
            "deltaT_map",
        }:
            header_salida["BUNIT"] = "K"

        elif nombre_mapa in {
            "N_fit_map",
            "deltaN_map",
        }:
            header_salida["BUNIT"] = "cm-2"

        ruta_salida = (
            ruta_salida_region
            / f"{MOLECULA}_{nombre_mapa}_"
              f"{nombre_region_compacta}.fits"
        )

        fits.PrimaryHDU(
            data=mapa_recortado,
            header=header_salida,
        ).writeto(
            ruta_salida,
            overwrite=True,
        )

        print(
            f"[recorte] Guardado: {ruta_salida}"
        )

    # Guardar también la máscara.
    ruta_mascara = (
        ruta_salida_region
        / f"mask_{nombre_region_compacta}.fits"
    )

    header_mascara = (
        header_recortado_base.copy()
    )
    header_mascara["REGION"] = (
        nombre_region_compacta
    )

    fits.PrimaryHDU(
        data=mascara_recortada.astype(np.uint8),
        header=header_mascara,
    ).writeto(
        ruta_mascara,
        overwrite=True,
    )

    print(
        f"[recorte] Píxeles incluidos: "
        f"{np.count_nonzero(mascara_recortada)}"
    )
    
# ============================================================
# PERFILES RADIALES DE Tex Y Ncol
# ============================================================

ruta_salida_perfiles = (
    ruta_fits_chi2
    / REGION
    / MOLECULA
    / "perfiles_radiales"
)

ruta_salida_perfiles.mkdir(
    parents=True,
    exist_ok=True,
)

def calcular_perfil_radial(
        T_map,
        N_map,
        deltaT_map,
        deltaN_map,
        mascara,
        header,
        n_anillos=10):
    """
    Divide la región en anillos concéntricos equiespaciados.

    Cada fila de la tabla corresponde a un píxel individual.
    No calcula medianas ni promedios.
    """

    shape = T_map.shape

    mapas = [
        N_map,
        deltaT_map,
        deltaN_map,
        mascara,
    ]

    if any(mapa.shape != shape for mapa in mapas):
        raise ValueError(
            "Todos los mapas y la máscara deben tener "
            "la misma forma."
        )

    if n_anillos < 1:
        raise ValueError(
            "n_anillos debe ser mayor o igual que 1."
        )

    mascara = mascara.astype(bool)

    # --------------------------------------------------------
    # Centro geométrico de la región
    # --------------------------------------------------------

    y_region, x_region = np.where(mascara)

    if y_region.size == 0:
        raise ValueError(
            "La máscara no contiene ningún píxel."
        )

    y_centro = np.mean(y_region)
    x_centro = np.mean(x_region)

    # --------------------------------------------------------
    # Escala angular de los píxeles
    # --------------------------------------------------------

    wcs = WCS(header).celestial

    escalas_pixel = (
        proj_plane_pixel_scales(wcs)
        * 3600.0
    )

    escala_x_arcsec = abs(
        escalas_pixel[0]
    )

    escala_y_arcsec = abs(
        escalas_pixel[1]
    )

    # --------------------------------------------------------
    # Radio de cada píxel
    # --------------------------------------------------------

    yy, xx = np.indices(shape)

    delta_x_arcsec = (
        (xx - x_centro)
        * escala_x_arcsec
    )

    delta_y_arcsec = (
        (yy - y_centro)
        * escala_y_arcsec
    )

    radio_pixel_arcsec = np.sqrt(
        delta_x_arcsec**2
        + delta_y_arcsec**2
    )

    radio_maximo = np.nanmax(
        radio_pixel_arcsec[mascara]
    )

    # Diez anillos con la misma anchura radial.
    limites_radiales = np.linspace(
        0,
        radio_maximo,
        n_anillos + 1,
    )

    resultados = []

    # --------------------------------------------------------
    # Guardar individualmente todos los píxeles
    # --------------------------------------------------------

    for numero_anillo in range(n_anillos):

        radio_interior = (
            limites_radiales[numero_anillo]
        )

        radio_exterior = (
            limites_radiales[numero_anillo + 1]
        )

        radio_central = 0.5 * (
            radio_interior
            + radio_exterior
        )

        if numero_anillo == n_anillos - 1:

            mascara_anillo = (
                mascara
                & (
                    radio_pixel_arcsec
                    >= radio_interior
                )
                & (
                    radio_pixel_arcsec
                    <= radio_exterior
                )
            )

        else:

            mascara_anillo = (
                mascara
                & (
                    radio_pixel_arcsec
                    >= radio_interior
                )
                & (
                    radio_pixel_arcsec
                    < radio_exterior
                )
            )

        pixeles_y, pixeles_x = np.where(
            mascara_anillo
        )

        for y, x in zip(
                pixeles_y,
                pixeles_x):

            T_valor = T_map[y, x]
            N_valor = N_map[y, x]
            deltaT_valor = deltaT_map[y, x]
            deltaN_valor = deltaN_map[y, x]

            # Ignorar píxeles sin ningún resultado válido.
            T_valida = (
                np.isfinite(T_valor)
                and T_valor > 0
            )

            N_valida = (
                np.isfinite(N_valor)
                and N_valor > 0
            )

            if not T_valida and not N_valida:
                continue

            if not T_valida:
                T_valor = np.nan
                deltaT_valor = np.nan

            if not N_valida:
                N_valor = np.nan
                deltaN_valor = np.nan

            if (
                not np.isfinite(deltaT_valor)
                or deltaT_valor < 0
            ):
                deltaT_valor = np.nan

            if (
                not np.isfinite(deltaN_valor)
                or deltaN_valor < 0
            ):
                deltaN_valor = np.nan

            resultados.append(
                (
                    numero_anillo + 1,
                    radio_central,
                    radio_pixel_arcsec[y, x],
                    radio_interior,
                    radio_exterior,
                    x,
                    y,
                    T_valor,
                    deltaT_valor,
                    N_valor,
                    deltaN_valor,
                )
            )

    perfil = Table(
        rows=resultados,
        names=[
            "anillo",
            "radio_anillo_arcsec",
            "radio_pixel_arcsec",
            "radio_interior_arcsec",
            "radio_exterior_arcsec",
            "x_pixel",
            "y_pixel",
            "T_ex",
            "deltaT",
            "N_col",
            "deltaN",
        ],
    )

    for columna in [
        "radio_anillo_arcsec",
        "radio_pixel_arcsec",
        "radio_interior_arcsec",
        "radio_exterior_arcsec",
    ]:
        perfil[columna].unit = "arcsec"

    perfil["T_ex"].unit = "K"
    perfil["deltaT"].unit = "K"
    perfil["N_col"].unit = "cm-2"
    perfil["deltaN"].unit = "cm-2"

    return perfil, (x_centro, y_centro)


def representar_perfil_radial(
        perfil,
        nombre_region,
        molecula,
        ruta_salida,
        representar_errores=True):
    """
    Representa todos los píxeles de cada uno de los 10 anillos.

    Los píxeles pertenecientes a un mismo anillo tienen el mismo
    valor en el eje radial.
    """

    radio = np.asarray(
        perfil["radio_anillo_arcsec"],
        dtype=float,
    )

    T = np.asarray(
        perfil["T_ex"],
        dtype=float,
    )

    deltaT = np.asarray(
        perfil["deltaT"],
        dtype=float,
    )

    N = np.asarray(
        perfil["N_col"],
        dtype=float,
    )

    deltaN = np.asarray(
        perfil["deltaN"],
        dtype=float,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # Tex frente al radio
    # --------------------------------------------------------

    validos_T = (
        np.isfinite(radio)
        & np.isfinite(T)
        & (T > 0)
    )

    if representar_errores:

        errores_T = np.where(
            np.isfinite(deltaT[validos_T])
            & (deltaT[validos_T] >= 0),
            deltaT[validos_T],
            0,
        )

        axes[0].errorbar(
            radio[validos_T],
            T[validos_T],
            yerr=errores_T,
            fmt="o",
            linestyle="none",
            color="tab:red",
            ecolor="lightcoral",
            markersize=4,
            capsize=2,
            alpha=0.65,
        )

    else:

        axes[0].scatter(
            radio[validos_T],
            T[validos_T],
            color="tab:red",
            s=18,
            alpha=0.65,
        )

    axes[0].set_xlabel(
        "Radius (arcsec)"
    )

    axes[0].set_ylabel(
        r"$T_{\mathrm{ex}}$ (K)"
    )

    axes[0].grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # Ncol frente al radio
    # --------------------------------------------------------

    validos_N = (
        np.isfinite(radio)
        & np.isfinite(N)
        & (N > 0)
    )

    if representar_errores:

        errores_N = np.where(
            np.isfinite(deltaN[validos_N])
            & (deltaN[validos_N] >= 0),
            deltaN[validos_N],
            0,
        )

        axes[1].errorbar(
            radio[validos_N],
            N[validos_N],
            yerr=errores_N,
            fmt="o",
            linestyle="none",
            color="tab:blue",
            ecolor="lightblue",
            markersize=4,
            capsize=2,
            alpha=0.65,
        )

    else:

        axes[1].scatter(
            radio[validos_N],
            N[validos_N],
            color="tab:blue",
            s=18,
            alpha=0.65,
        )

    axes[1].set_xlabel(
        "Radius (arcsec)"
    )

    axes[1].set_ylabel(
        r"$N_{\mathrm{col}}$ (cm$^{-2}$)"
    )

    axes[1].set_yscale(
        "log"
    )

    axes[1].grid(
        alpha=0.25
    )

    fig.suptitle(
        f"{molecula} — {nombre_region}"
    )

    ruta_figura = (
        ruta_salida
        / f"{molecula}_{nombre_region}_"
          "perfil_radial.pdf"
    )

    fig.savefig(
        ruta_figura,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(
        f"[perfil_radial] Figura guardada: "
        f"{ruta_figura}"
    )


def representar_perfil_radial_medio(
        perfil,
        nombre_region,
        molecula,
        ruta_salida):
    """
    Calcula y representa la media de Tex y Ncol en cada anillo.

    Las incertidumbres se propagan suponiendo que las
    incertidumbres de los píxeles son independientes.
    """

    anillos = np.asarray(
        perfil["anillo"],
        dtype=int,
    )

    radio = np.asarray(
        perfil["radio_anillo_arcsec"],
        dtype=float,
    )

    T = np.asarray(
        perfil["T_ex"],
        dtype=float,
    )

    deltaT = np.asarray(
        perfil["deltaT"],
        dtype=float,
    )

    N = np.asarray(
        perfil["N_col"],
        dtype=float,
    )

    deltaN = np.asarray(
        perfil["deltaN"],
        dtype=float,
    )

    resultados = []

    for numero_anillo in np.unique(anillos):

        mascara_anillo = (
            anillos == numero_anillo
        )

        radio_anillo = np.nanmean(
            radio[mascara_anillo]
        )

        # ----------------------------------------------------
        # Media de Tex
        # ----------------------------------------------------

        validos_T = (
            mascara_anillo
            & np.isfinite(T)
            & (T > 0)
        )

        if np.any(validos_T):

            T_media = np.mean(
                T[validos_T]
            )

            n_T = np.count_nonzero(
                validos_T
            )

            validos_error_T = (
                validos_T
                & np.isfinite(deltaT)
                & (deltaT >= 0)
            )

            if np.any(validos_error_T):

                error_T_media = (
                    np.sqrt(
                        np.sum(
                            deltaT[
                                validos_error_T
                            ]**2
                        )
                    )
                    / np.count_nonzero(
                        validos_error_T
                    )
                )

            else:
                error_T_media = np.nan

        else:

            T_media = np.nan
            error_T_media = np.nan
            n_T = 0

        # ----------------------------------------------------
        # Media de Ncol
        # ----------------------------------------------------

        validos_N = (
            mascara_anillo
            & np.isfinite(N)
            & (N > 0)
        )

        if np.any(validos_N):

            N_media = np.mean(
                N[validos_N]
            )

            n_N = np.count_nonzero(
                validos_N
            )

            validos_error_N = (
                validos_N
                & np.isfinite(deltaN)
                & (deltaN >= 0)
            )

            if np.any(validos_error_N):

                error_N_media = (
                    np.sqrt(
                        np.sum(
                            deltaN[
                                validos_error_N
                            ]**2
                        )
                    )
                    / np.count_nonzero(
                        validos_error_N
                    )
                )

            else:
                error_N_media = np.nan

        else:

            N_media = np.nan
            error_N_media = np.nan
            n_N = 0

        resultados.append(
            (
                numero_anillo,
                radio_anillo,
                T_media,
                error_T_media,
                N_media,
                error_N_media,
                n_T,
                n_N,
            )
        )

    tabla_medias = Table(
        rows=resultados,
        names=[
            "anillo",
            "radio_arcsec",
            "T_media",
            "error_T_media",
            "N_media",
            "error_N_media",
            "n_pix_T",
            "n_pix_N",
        ],
    )

    tabla_medias["radio_arcsec"].unit = "arcsec"
    tabla_medias["T_media"].unit = "K"
    tabla_medias["error_T_media"].unit = "K"
    tabla_medias["N_media"].unit = "cm-2"
    tabla_medias["error_N_media"].unit = "cm-2"

    # ========================================================
    # REPRESENTACIÓN
    # ========================================================

    radio_medio = np.asarray(
        tabla_medias["radio_arcsec"],
        dtype=float,
    )

    T_media = np.asarray(
        tabla_medias["T_media"],
        dtype=float,
    )

    error_T = np.asarray(
        tabla_medias["error_T_media"],
        dtype=float,
    )

    N_media = np.asarray(
        tabla_medias["N_media"],
        dtype=float,
    )

    error_N = np.asarray(
        tabla_medias["error_N_media"],
        dtype=float,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # Temperatura media
    # --------------------------------------------------------

    validos_T = (
        np.isfinite(radio_medio)
        & np.isfinite(T_media)
        & (T_media > 0)
    )

    errores_T_plot = np.where(
        np.isfinite(error_T[validos_T]),
        error_T[validos_T],
        0,
    )

    axes[0].errorbar(
        radio_medio[validos_T],
        T_media[validos_T],
        yerr=errores_T_plot,
        fmt="o",
        color="tab:red",
        ecolor="lightcoral",
        markersize=6,
        linewidth=1.2,
        capsize=3,
        label="Mean",
    )

    axes[0].set_xlabel(
        "Radius (arcsec)"
    )

    axes[0].set_ylabel(
        r"Mean $T_{\mathrm{ex}}$ (K)"
    )

    axes[0].grid(
        alpha=0.25
    )

    axes[0].legend()

    # --------------------------------------------------------
    # Densidad de columna media
    # --------------------------------------------------------

    validos_N = (
        np.isfinite(radio_medio)
        & np.isfinite(N_media)
        & (N_media > 0)
    )

    errores_N_plot = np.where(
        np.isfinite(error_N[validos_N]),
        error_N[validos_N],
        0,
    )

    # Evitar que las barras entren en valores negativos
    # dentro de un eje logarítmico.
    errores_N_plot = np.minimum(
        errores_N_plot,
        0.99 * N_media[validos_N],
    )

    axes[1].errorbar(
        radio_medio[validos_N],
        N_media[validos_N],
        yerr=errores_N_plot,
        fmt="o",
        color="tab:blue",
        ecolor="lightblue",
        markersize=6,
        linewidth=1.2,
        capsize=3,
        label="Mean",
    )

    axes[1].set_xlabel(
        "Radius (arcsec)"
    )

    axes[1].set_ylabel(
        r"Mean $N_{\mathrm{col}}$ (cm$^{-2}$)"
    )

    axes[1].set_yscale(
        "log"
    )

    axes[1].grid(
        alpha=0.25
    )

    axes[1].legend()

    fig.suptitle(
        f"{molecula} — {nombre_region} — Mean radial profile"
    )

    ruta_figura = (
        ruta_salida
        / f"{molecula}_{nombre_region}_"
          "perfil_radial_media.pdf"
    )

    fig.savefig(
        ruta_figura,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(
        f"[perfil_radial] Figura de medias guardada: "
        f"{ruta_figura}"
    )

    return tabla_medias

# ============================================================
# CALCULAR TODAS LAS REGIONES COMPACTAS
# ============================================================

perfiles_radiales = {}

for ruta_region in rutas_regiones_compactas:

    nombre_region_compacta = ruta_region.stem

    if nombre_region_compacta.startswith("region"):
        nombre_region_compacta = (
            nombre_region_compacta[len("region"):]
        )

    ruta_recorte = (
        ruta_fits_chi2
        / REGION
        / MOLECULA
        / "regiones_compactas"
        / nombre_region_compacta
    )

    rutas_recortadas = {
        "T": (
            ruta_recorte
            / f"{MOLECULA}_T_fit_map_"
              f"{nombre_region_compacta}.fits"
        ),
        "N": (
            ruta_recorte
            / f"{MOLECULA}_N_fit_map_"
              f"{nombre_region_compacta}.fits"
        ),
        "deltaT": (
            ruta_recorte
            / f"{MOLECULA}_deltaT_map_"
              f"{nombre_region_compacta}.fits"
        ),
        "deltaN": (
            ruta_recorte
            / f"{MOLECULA}_deltaN_map_"
              f"{nombre_region_compacta}.fits"
        ),
        "mask": (
            ruta_recorte
            / f"mask_{nombre_region_compacta}.fits"
        ),
    }

    for nombre, ruta in rutas_recortadas.items():
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encuentra {nombre}: {ruta}"
            )

    with fits.open(rutas_recortadas["T"]) as hdul:
        T_recorte = hdul[0].data.astype(float).copy()
        header_recorte = hdul[0].header.copy()

    with fits.open(rutas_recortadas["N"]) as hdul:
        N_recorte = hdul[0].data.astype(float).copy()

    with fits.open(rutas_recortadas["deltaT"]) as hdul:
        deltaT_recorte = hdul[0].data.astype(float).copy()

    with fits.open(rutas_recortadas["deltaN"]) as hdul:
        deltaN_recorte = hdul[0].data.astype(float).copy()

    with fits.open(rutas_recortadas["mask"]) as hdul:
        mascara_recorte = (
            hdul[0].data.astype(bool).copy()
        )

    perfil, centro_pixel = calcular_perfil_radial(
        T_map=T_recorte,
        N_map=N_recorte,
        deltaT_map=deltaT_recorte,
        deltaN_map=deltaN_recorte,
        mascara=mascara_recorte,
        header=header_recorte,
        n_anillos=30,
    )
    perfiles_radiales[
        nombre_region_compacta
    ] = perfil

    ruta_tabla = (
        ruta_salida_perfiles
        / f"{MOLECULA}_{nombre_region_compacta}_"
          "perfil_radial.ecsv"
    )

    perfil.write(
        ruta_tabla,
        format="ascii.ecsv",
        overwrite=True,
    )

    print(
        f"\n[perfil_radial] Región: "
        f"{nombre_region_compacta}"
    )

    print(
        f"[perfil_radial] Centro en el recorte: "
        f"x={centro_pixel[0]:.2f}, "
        f"y={centro_pixel[1]:.2f}"
    )

    print(
        f"[perfil_radial] Número de anillos: "
        f"{len(perfil)}"
    )

    representar_perfil_radial(
        perfil=perfil,
        nombre_region=nombre_region_compacta,
        molecula=MOLECULA,
        ruta_salida=ruta_salida_perfiles,
        representar_errores=True,
    )
    
    tabla_medias = representar_perfil_radial_medio(
        perfil=perfil,
        nombre_region=nombre_region_compacta,
        molecula=MOLECULA,
        ruta_salida=ruta_salida_perfiles,
    )

    ruta_tabla_medias = (
        ruta_salida_perfiles
        / f"{MOLECULA}_{nombre_region_compacta}_"
          "perfil_radial_media.ecsv"
    )

    tabla_medias.write(
        ruta_tabla_medias,
        format="ascii.ecsv",
        overwrite=True,
    )

    print(
        f"[perfil_radial] Tabla de medias guardada: "
        f"{ruta_tabla_medias}"
    )