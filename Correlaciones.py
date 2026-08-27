#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:24:57 2026

@author: jorge
"""

"""
Análisis de correlaciones entre los mapas chi2.

Para una región extensa:
    1. Detecta automáticamente todas las moléculas disponibles.
    2. Carga Tex, Ncol, deltaTex y deltaNcol.
    3. Selecciona los píxeles de cada región compacta.
    4. Representa:
        - Tex vs log10(Ncol) para cada molécula.
        - log10(Nmol1) vs log10(Nmol2) para todas las parejas.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from regions import Regions, SkyRegion
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    TextArea,
    HPacker,
    VPacker,
)
from scipy.stats import pearsonr, spearmanr


# ============================================================
# CONFIGURACIÓN
# ============================================================

ruta_fits_chi2 = Path(
    "/home/jorge/TFM/maps_24Agosto/maps/chi2"
)

ruta_regiones = Path(
    "/home/jorge/TFM/regiones"
)

ruta_salida_base = Path(
    "/home/jorge/TFM/figures/correlaciones"
)




REGION = "GLOBAL"

AJUSTE_CONJUNTO = True


regiones_compactas = {
    "NORTH_MAP": [
        "regionMM24.reg",
        "regionMM35.reg",
        "regionNORTH.reg",
    ],
    "mm31_d2": [
        "regionMM31.reg",
        "regionMF2.reg",
    ],
    "MM14_MAP": [
        "regionMM14.reg",
    ],
    "E_NORTH": [
        "regione2e.reg",
        "regione2w.reg",
    ],
}

# ============================================================
# NOMBRES Y COLORES FIJOS DE LAS REGIONES COMPACTAS
# ============================================================

labels_compactas = {
    "MF2": "d2",
    "MM31": "mm31",
    "MM24": "mm24",
    "MM35": "mm35",
    "MM14": "mm14",
    "NORTH": "north",
    "e2e": "e2e",
    "e2w": "e2w",
}


colores_compactas_fijos = {
    "MF2": "tab:blue",
    "MM31": "tab:orange",
    "MM24": "tab:green",
    "MM35": "tab:red",
    "MM14": "tab:purple",
    "NORTH": "tab:brown",
    "e2e": "tab:pink",
    "e2w": "tab:olive",
}

MODO_GLOBAL = (
    REGION == "GLOBAL"
)

if (
    not MODO_GLOBAL
    and REGION not in regiones_compactas
):
    raise ValueError(
        f"Región no reconocida: {REGION}"
    )


# ============================================================
# COMPROBACIONES INICIALES
# ============================================================

# ============================================================
# BUSCAR MOLÉCULAS DISPONIBLES
# ============================================================

def buscar_moleculas(ruta_region):
    """
    Busca automáticamente las moléculas que contienen los
    cuatro mapas chi2 necesarios.

    Devuelve una lista ordenada con sus nombres.
    """

    moleculas = []

    for carpeta in sorted(ruta_region.iterdir()):

        if not carpeta.is_dir():
            continue

        molecula = carpeta.name

        rutas = {
            "T": (
                carpeta
                / f"{molecula}_Tex_chi2.fits"
            ),
            "N": (
                carpeta
                / f"{molecula}_Ncol_chi2.fits"
            ),
            "deltaT": (
                carpeta
                / f"{molecula}_deltaTex_chi2.fits"
            ),
            "deltaN": (
                carpeta
                / f"{molecula}_deltaNcol_chi2.fits"
            ),
        }

        faltan = [
            nombre
            for nombre, ruta in rutas.items()
            if not ruta.exists()
        ]

        if faltan:
            print(
                f"[aviso] {molecula}: "
                f"faltan {', '.join(faltan)}. "
                "No se utilizará."
            )
            continue

        moleculas.append(molecula)

    return moleculas



# ============================================================
# CARGAR MAPAS CHI2
# ============================================================

def cargar_fits(ruta):
    """
    Carga un FITS y devuelve:
        data   -> array 2D
        header -> cabecera FITS
        wcs    -> WCS celestial
    """

    with fits.open(ruta) as hdul:
        data = np.squeeze(
            hdul[0].data
        ).astype(float)

        header = hdul[0].header.copy()

    wcs = WCS(header).celestial

    if data.ndim != 2:
        raise ValueError(
            f"El mapa {ruta} no es 2D. "
            f"Shape encontrada: {data.shape}"
        )

    return data, header, wcs


def cargar_mapas_molecula(
    ruta_region,
    molecula,
):
    """
    Carga los cuatro mapas chi2 de una molécula.
    """

    carpeta = (
        ruta_region
        / molecula
    )

    rutas = {
        "T": (
            carpeta
            / f"{molecula}_Tex_chi2.fits"
        ),
        "N": (
            carpeta
            / f"{molecula}_Ncol_chi2.fits"
        ),
        "deltaT": (
            carpeta
            / f"{molecula}_deltaTex_chi2.fits"
        ),
        "deltaN": (
            carpeta
            / f"{molecula}_deltaNcol_chi2.fits"
        ),
    }

    T, header, wcs = cargar_fits(
        rutas["T"]
    )

    N, _, _ = cargar_fits(
        rutas["N"]
    )

    deltaT, _, _ = cargar_fits(
        rutas["deltaT"]
    )

    deltaN, _, _ = cargar_fits(
        rutas["deltaN"]
    )

    # Comprobar que los cuatro mapas
    # de la molécula tienen la misma forma
    formas = {
        T.shape,
        N.shape,
        deltaT.shape,
        deltaN.shape,
    }

    if len(formas) != 1:
        raise ValueError(
            f"{molecula}: los mapas chi2 "
            f"no tienen la misma shape."
        )

    return {
        "T": T,
        "N": N,
        "deltaT": deltaT,
        "deltaN": deltaN,
        "header": header,
        "wcs": wcs,
    }

def nombre_region_compacta(
    archivo_region,
):
    """
    Convierte, por ejemplo:

        regionMM31.reg -> MM31
        regione2e.reg  -> e2e
        regionMF2.reg  -> MF2
    """

    nombre = Path(
        archivo_region
    ).stem

    if nombre.lower().startswith("region"):
        nombre = nombre[6:]

    return nombre
def crear_mascara_region(
    ruta_region_ds9,
    wcs,
    shape,
):
    """
    Construye una máscara booleana 2D a partir
    de una región DS9.

    Devuelve True para los píxeles que están
    dentro de la región.
    """

    regiones = Regions.read(
        ruta_region_ds9,
        format="ds9",
    )

    mascara_total = np.zeros(
        shape,
        dtype=bool,
    )

    for region in regiones:

        if isinstance(
            region,
            SkyRegion,
        ):
            region_pix = region.to_pixel(
                wcs
            )
        else:
            region_pix = region

        mask = region_pix.to_mask(
            mode="center"
        )

        if mask is None:
            continue

        imagen_mask = mask.to_image(
            shape
        )

        if imagen_mask is None:
            continue

        mascara_total |= (
            imagen_mask > 0
        )

    return mascara_total

# ============================================================
# CARGAR UNA REGIÓN EXTENSA COMPLETA
# ============================================================

def cargar_region_extensa(
    nombre_region_extensa,
):
    """
    Carga todos los mapas chi2 y las regiones compactas
    asociadas a una región extensa.

    Devuelve:
        - moléculas disponibles
        - mapas de cada molécula
        - máscara conjunta de regiones compactas
        - datos de cada región compacta
    """

    ruta_region_extensa = (
        ruta_fits_chi2
        / nombre_region_extensa
    )

    if not ruta_region_extensa.exists():
        raise FileNotFoundError(
            f"No existe la carpeta:\n"
            f"{ruta_region_extensa}"
        )

    # ========================================================
    # BUSCAR MOLÉCULAS
    # ========================================================

    moleculas_region = buscar_moleculas(
        ruta_region_extensa
    )

    if len(moleculas_region) == 0:
        print(
            f"[aviso] No hay moléculas válidas "
            f"en {nombre_region_extensa}."
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"CARGANDO REGIÓN EXTENSA: "
        f"{nombre_region_extensa}"
    )

    print(
        "=" * 70
    )

    print(
        f"Moléculas encontradas: "
        f"{len(moleculas_region)}"
    )

    # ========================================================
    # CARGAR MAPAS
    # ========================================================

    datos_region = {}

    for molecula in moleculas_region:

        datos_region[
            molecula
        ] = cargar_mapas_molecula(
            ruta_region_extensa,
            molecula,
        )

        print(
            f"    {molecula:20s} "
            f"{datos_region[molecula]['N'].shape}"
        )

    # ========================================================
    # MÁSCARA CONJUNTA DE TODAS LAS REGIONES COMPACTAS
    # ========================================================

    mascaras_totales = {}

    for molecula in moleculas_region:

        d = datos_region[
            molecula
        ]

        mascara_total = np.zeros(
            d["N"].shape,
            dtype=bool,
        )

        for archivo_region in regiones_compactas[
            nombre_region_extensa
        ]:

            ruta_region_ds9 = (
                ruta_regiones
                / archivo_region
            )

            if not ruta_region_ds9.exists():
                raise FileNotFoundError(
                    f"No existe la región DS9:\n"
                    f"{ruta_region_ds9}"
                )

            mascara = crear_mascara_region(
                ruta_region_ds9,
                d["wcs"],
                d["N"].shape,
            )

            mascara_total |= mascara

        mascaras_totales[
            molecula
        ] = mascara_total

    # ========================================================
    # EXTRAER DATOS DE CADA REGIÓN COMPACTA
    # ========================================================

    datos_compactas = {}

    for archivo_region in regiones_compactas[
        nombre_region_extensa
    ]:

        nombre_compacta = (
            nombre_region_compacta(
                archivo_region
            )
        )

        ruta_region_ds9 = (
            ruta_regiones
            / archivo_region
        )

        if not ruta_region_ds9.exists():
            raise FileNotFoundError(
                f"No existe la región DS9:\n"
                f"{ruta_region_ds9}"
            )

        datos_compactas[
            nombre_compacta
        ] = {}

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"REGIÓN COMPACTA: "
            f"{nombre_compacta}"
        )

        print(
            "-" * 70
        )

        for molecula in moleculas_region:

            d = datos_region[
                molecula
            ]

            mascara_region = crear_mascara_region(
                ruta_region_ds9,
                d["wcs"],
                d["N"].shape,
            )

            # ----------------------------------------
            # Máscara de calidad
            # ----------------------------------------

            mascara_valida = (
                mascara_region
                & np.isfinite(d["T"])
                & np.isfinite(d["N"])
                & np.isfinite(d["deltaT"])
                & np.isfinite(d["deltaN"])
                & (d["T"] > 0)
                & (d["N"] > 0)
                & (d["deltaT"] >= 0)
                & (d["deltaN"] >= 0)
            )

            y, x = np.where(
                mascara_valida
            )

            T = d["T"][
                mascara_valida
            ]

            N = d["N"][
                mascara_valida
            ]

            deltaT = d["deltaT"][
                mascara_valida
            ]

            deltaN = d["deltaN"][
                mascara_valida
            ]

            # ----------------------------------------
            # Ncol en escala logarítmica
            # ----------------------------------------

            logN = np.log10(
                N
            )

            delta_logN = (
                deltaN
                / (
                    N
                    * np.log(10.0)
                )
            )

            datos_compactas[
                nombre_compacta
            ][
                molecula
            ] = {
                "T": T,
                "N": N,
                "logN": logN,

                "deltaT": deltaT,
                "deltaN": deltaN,
                "delta_logN": delta_logN,

                "x": x,
                "y": y,

                "mask": mascara_valida,
            }

            print(
                f"{molecula:20s}: "
                f"{len(N):5d} píxeles"
            )

    return {
        "moleculas": moleculas_region,
        "datos": datos_region,
        "mascaras_totales": mascaras_totales,
        "compactas": datos_compactas,
    }

# ============================================================
# CARGA DE DATOS SEGÚN EL MODO
# ============================================================

if MODO_GLOBAL:

    # ========================================================
    # MODO GLOBAL
    # ========================================================

    regiones_seleccionadas = list(
        regiones_compactas.keys()
    )

    datos_globales = {}

    for region_extensa in regiones_seleccionadas:

        datos_globales[
            region_extensa
        ] = cargar_region_extensa(
            region_extensa
        )

    # --------------------------------------------------------
    # Unión de todas las moléculas disponibles
    # --------------------------------------------------------

    moleculas = sorted(
        set().union(
            *[
                set(
                    contexto["moleculas"]
                )
                for contexto
                in datos_globales.values()
            ]
        )
    )

    if len(moleculas) == 0:
        raise RuntimeError(
            "No se han encontrado moléculas "
            "válidas en ninguna región."
        )

    # --------------------------------------------------------
    # Lista de todas las regiones compactas
    # --------------------------------------------------------

    compactas_globales = []

    for (
        region_extensa,
        contexto
    ) in datos_globales.items():

        for nombre_compacta in contexto[
            "compactas"
        ]:

            compactas_globales.append(
                (
                    region_extensa,
                    nombre_compacta,
                )
            )

    # --------------------------------------------------------
    # Colores globales
    # --------------------------------------------------------

    colores_compactas_global = {
    (
        region_extensa,
        nombre_compacta,
    ): colores_compactas_fijos[
        nombre_compacta
    ]

    for (
        region_extensa,
        nombre_compacta,
    )
    in compactas_globales
}

    # --------------------------------------------------------
    # Información
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODO GLOBAL"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRegiones extensas: "
        f"{len(datos_globales)}"
    )

    for region_extensa in datos_globales:

        print(
            f"    - {region_extensa}"
        )

    print(
        f"\nMoléculas totales: "
        f"{len(moleculas)}"
    )

    for molecula in moleculas:

        print(
            f"    - {molecula}"
        )

    print(
        f"\nRegiones compactas: "
        f"{len(compactas_globales)}"
    )

    for (
        region_extensa,
        nombre_compacta
    ) in compactas_globales:

        print(
            f"    - {nombre_compacta} "
            f"({region_extensa})"
        )


else:

    # ========================================================
    # MODO REGIÓN INDIVIDUAL
    # ========================================================

    contexto_region = cargar_region_extensa(
        REGION
    )

    # --------------------------------------------------------
    # Recuperamos los nombres de variables que ya utilizaba
    # el código antiguo
    # --------------------------------------------------------

    moleculas = contexto_region[
        "moleculas"
    ]

    datos = contexto_region[
        "datos"
    ]

    mascaras_compactas_totales = (
        contexto_region[
            "mascaras_totales"
        ]
    )

    datos_regiones = contexto_region[
        "compactas"
    ]

    nombres_compactas = list(
        datos_regiones.keys()
    )

    # --------------------------------------------------------
    # Colores
    # --------------------------------------------------------

    colores_compactas = {
    nombre: colores_compactas_fijos[
        nombre
    ]
    for nombre in nombres_compactas
}


    if len(moleculas) == 0:
        raise RuntimeError(
            f"No se han encontrado moléculas "
            f"válidas para {REGION}."
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"REGIÓN SELECCIONADA: "
        f"{REGION}"
    )

    print(
        "=" * 70
    )

    print(
        f"\nMoléculas: "
        f"{len(moleculas)}"
    )

    for molecula in moleculas:

        print(
            f"    - {molecula}"
        )


# ============================================================
# AJUSTE LINEAL
# ============================================================

def ajustar_lineal(
    x,
    y,
    sx=None,
    sy=None,
):
    """
    Ajuste lineal ordinario por mínimos cuadrados:

        y = m*x + b

    Las incertidumbres sx y sy no intervienen en el ajuste.
    Se utilizan únicamente para estimar la dispersión
    esperada debida a las medidas.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if sx is not None:
        sx = np.asarray(sx, dtype=float)

    if sy is not None:
        sy = np.asarray(sy, dtype=float)

    # ----------------------------------------
    # Píxeles válidos
    # ----------------------------------------

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if sx is not None:
        mask &= np.isfinite(sx)

    if sy is not None:
        mask &= np.isfinite(sy)

    x_fit = x[mask]
    y_fit = y[mask]

    if sx is not None:
        sx_fit = sx[mask]
    else:
        sx_fit = None

    if sy is not None:
        sy_fit = sy[mask]
    else:
        sy_fit = None

    if len(x_fit) < 3:
        return None

    # ----------------------------------------
    # Ajuste lineal y = m*x + b
    # ----------------------------------------

    coeficientes, cov = np.polyfit(
        x_fit,
        y_fit,
        1,
        cov=True,
    )

    pendiente = coeficientes[0]
    intercepto = coeficientes[1]

    error_pendiente = np.sqrt(
        cov[0, 0]
    )

    error_intercepto = np.sqrt(
        cov[1, 1]
    )

    # ----------------------------------------
    # Modelo
    # ----------------------------------------

    y_modelo = (
        pendiente * x_fit
        + intercepto
    )

    residuos = (
        y_fit
        - y_modelo
    )

    # ----------------------------------------
    # R²
    # ----------------------------------------

    ss_res = np.sum(
        residuos**2
    )

    ss_tot = np.sum(
        (
            y_fit
            - np.mean(y_fit)
        )**2
    )

    if ss_tot > 0:
        r2 = (
            1.0
            - ss_res / ss_tot
        )
    else:
        r2 = np.nan

    # ----------------------------------------
    # Dispersión residual
    # ----------------------------------------

    if len(residuos) > 2:
        sigma_res = np.std(
            residuos,
            ddof=2,
        )
    else:
        sigma_res = np.nan

    # ----------------------------------------
    # Dispersión esperada por las medidas
    # ----------------------------------------

    if (
        sx_fit is not None
        and sy_fit is not None
    ):

        sigma_medida_pix = np.sqrt(
            sy_fit**2
            + (
                pendiente
                * sx_fit
            )**2
        )

        sigma_meas = np.sqrt(
            np.mean(
                sigma_medida_pix**2
            )
        )

        sigma_int = np.sqrt(
            max(
                sigma_res**2
                - sigma_meas**2,
                0.0,
            )
        )

    else:

        sigma_meas = np.nan
        sigma_int = np.nan

    # ----------------------------------------
    # Correlaciones
    # ----------------------------------------

    if (
        np.std(x_fit) > 0
        and np.std(y_fit) > 0
    ):

        pearson, _ = pearsonr(
            x_fit,
            y_fit,
        )

        spearman, _ = spearmanr(
            x_fit,
            y_fit,
        )

    else:

        pearson = np.nan
        spearman = np.nan

    return {
        "x": x_fit,
        "y": y_fit,
        "sx": sx_fit,
        "sy": sy_fit,

        "slope": pendiente,
        "slope_err": error_pendiente,

        "intercept": intercepto,
        "intercept_err": error_intercepto,

        "pearson": pearson,
        "spearman": spearman,

        "r2": r2,

        "sigma_res": sigma_res,
        "sigma_meas": sigma_meas,
        "sigma_int": sigma_int,

        "cov_beta": cov,

        "npix": len(x_fit),
    }

# ============================================================
# BANDA DE INCERTIDUMBRE DEL AJUSTE LINEAL
# ============================================================

def banda_lineal(
    x,
    slope,
    intercept,
    cov_beta,
):
    """
    Banda 1-sigma asociada a la incertidumbre
    de la pendiente y del intercepto.
    """

    var_m = cov_beta[0, 0]
    var_b = cov_beta[1, 1]
    cov_mb = cov_beta[0, 1]

    var_y = (
        x**2 * var_m
        + var_b
        + 2.0 * x * cov_mb
    )

    var_y = np.maximum(
        var_y,
        0.0,
    )

    sigma_y = np.sqrt(
        var_y
    )

    y = (
        slope * x
        + intercept
    )

    return y, sigma_y

# ============================================================
# EXTRAER PÍXELES FUERA DE TODAS LAS REGIONES COMPACTAS
# ============================================================

def extraer_pixeles_exteriores_molecula(
    molecula,
):
    """
    Extrae los píxeles válidos que no pertenecen
    a ninguna región compacta.
    """

    d = datos[molecula]

    mascara_exterior = (
        ~mascaras_compactas_totales[molecula]
        & np.isfinite(d["T"])
        & np.isfinite(d["N"])
        & np.isfinite(d["deltaT"])
        & np.isfinite(d["deltaN"])
        & (d["T"] > 0)
        & (d["N"] > 0)
        & (d["deltaT"] >= 0)
        & (d["deltaN"] >= 0)
    )

    T = d["T"][
        mascara_exterior
    ]

    N = d["N"][
        mascara_exterior
    ]

    deltaT = d["deltaT"][
        mascara_exterior
    ]

    deltaN = d["deltaN"][
        mascara_exterior
    ]

    logN = np.log10(
        N
    )

    delta_logN = (
        deltaN
        / (
            N
            * np.log(10.0)
        )
    )

    return {
        "T": T,
        "logN": logN,
        "deltaT": deltaT,
        "delta_logN": delta_logN,
        "npix": len(N),
    }

# ============================================================
# EXTRAER TODOS LOS HOT CORES PARA UNA MOLÉCULA
# ============================================================

def extraer_todos_hotcores_molecula(
    molecula,
):
    """
    Extrae conjuntamente todos los píxeles pertenecientes
    a cualquiera de las regiones compactas.
    """

    d = datos[molecula]

    mascara = (
        mascaras_compactas_totales[molecula]
        & np.isfinite(d["T"])
        & np.isfinite(d["N"])
        & np.isfinite(d["deltaT"])
        & np.isfinite(d["deltaN"])
        & (d["T"] > 0)
        & (d["N"] > 0)
        & (d["deltaT"] >= 0)
        & (d["deltaN"] >= 0)
    )

    T = d["T"][
        mascara
    ]

    N = d["N"][
        mascara
    ]

    deltaT = d["deltaT"][
        mascara
    ]

    deltaN = d["deltaN"][
        mascara
    ]

    return {
        "x": np.log10(N),
        "y": T,
        "sx": (
            deltaN
            / (
                N
                * np.log(10.0)
            )
        ),
        "sy": deltaT,
        "npix": len(N),
    }

# ============================================================
# FIGURA Tex vs log(Ncol) - REGIÓN EXTENSA
# ============================================================

def plot_tex_vs_logn(
    molecula,
    ruta_salida,
):
    """
    Figura única para toda la región extensa.

    - Cada región compacta aparece en un color distinto.
    - Los píxeles exteriores aparecen en gris.
    - Se ajusta cada región compacta por separado.
    - Se realiza además un ajuste conjunto de todos
      los hot cores.
    """

    exterior = (
        extraer_pixeles_exteriores_molecula(
            molecula
        )
    )

    todos_cores = (
        extraer_todos_hotcores_molecula(
            molecula
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 7.5)
    )

    # ========================================================
    # PÍXELES EXTERIORES
    # ========================================================

    if exterior["npix"] > 0:

        ax.errorbar(
            exterior["logN"],
            exterior["T"],
            xerr=exterior["delta_logN"],
            yerr=exterior["deltaT"],
            fmt="o",
            markersize=3.5,
            alpha=0.20,
            color="0.65",
            ecolor="0.75",
            elinewidth=0.6,
            capsize=0,
            label="Outside compact regions",
            zorder=1,
        )

    # ========================================================
    # AJUSTES INDIVIDUALES
    # ========================================================

    resultados_individuales = {}

    lineas_stats = []

    for nombre_region in nombres_compactas:

        label_region = labels_compactas[
        nombre_region
    ]        

        datos_mol = datos_regiones[
            nombre_region
        ][
            molecula
        ]

        x = datos_mol["logN"]
        y = datos_mol["T"]

        sx = datos_mol["delta_logN"]
        sy = datos_mol["deltaT"]

        color = colores_compactas[
            nombre_region
        ]

        # ----------------------------------------
        # Puntos + barras de error
        # ----------------------------------------

        ax.errorbar(
            x,
            y,
            xerr=sx,
            yerr=sy,
            fmt="o",
            markersize=5,
            alpha=0.75,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=0,
            label=label_region,
            zorder=3,
        )

        # ----------------------------------------
        # Ajuste individual
        # ----------------------------------------

        resultado = ajustar_lineal(
            x,
            y,
            sx,
            sy,
        )

        resultados_individuales[
            nombre_region
        ] = resultado

        if resultado is None:
            continue

        xmin = np.min(
            resultado["x"]
        )

        xmax = np.max(
            resultado["x"]
        )

        x_modelo = np.linspace(
            xmin,
            xmax,
            300,
        )

        y_modelo = (
            resultado["slope"]
            * x_modelo
            + resultado["intercept"]
        )

        ax.plot(
            x_modelo,
            y_modelo,
            color=color,
            linewidth=2.2,
            zorder=4,
        )

        lineas_stats.append(
            f"{label_region}: "
            f"N={resultado['npix']}, "
            f"r={resultado['pearson']:.2f}, "
            f"rho={resultado['spearman']:.2f}, "
            f"m={resultado['slope']:.2f}"
            f"+/-{resultado['slope_err']:.2f}, "
            f"R2={resultado['r2']:.2f}"
        )

    # ========================================================
    # AJUSTE CONJUNTO DE TODOS LOS HOT CORES
    # ========================================================

    resultado_total = ajustar_lineal(
        todos_cores["x"],
        todos_cores["y"],
        todos_cores["sx"],
        todos_cores["sy"],
    )

    if (
        resultado_total is not None
        and len(nombres_compactas) > 1
    ):

        xmin = np.min(
            resultado_total["x"]
        )

        xmax = np.max(
            resultado_total["x"]
        )

        x_modelo = np.linspace(
            xmin,
            xmax,
            300,
        )

        y_modelo, sigma_modelo = (
            banda_lineal(
                x_modelo,
                resultado_total["slope"],
                resultado_total["intercept"],
                resultado_total["cov_beta"],
            )
        )

        ax.plot(
            x_modelo,
            y_modelo,
            color="black",
            linestyle="--",
            linewidth=2.8,
            label="All hot cores fit",
            zorder=6,
        )

        ax.fill_between(
            x_modelo,
            y_modelo - sigma_modelo,
            y_modelo + sigma_modelo,
            color="black",
            alpha=0.08,
            zorder=2,
        )

        lineas_stats.append(
            "All cores: "
            f"N={resultado_total['npix']}, "
            f"r={resultado_total['pearson']:.2f}, "
            f"rho={resultado_total['spearman']:.2f}, "
            f"m={resultado_total['slope']:.2f}"
            f"+/-{resultado_total['slope_err']:.2f}, "
            f"R2={resultado_total['r2']:.2f}"
        )

    # ========================================================
    # ESTADÍSTICAS
    # ========================================================

    if lineas_stats:

        texto = "\n".join(
            lineas_stats
        )

        ax.text(
            0.02,
            0.98,
            texto,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox=dict(
                boxstyle="square,pad=0.45",
                facecolor="white",
                alpha=0.88,
            ),
            zorder=10,
        )

    # ========================================================
    # EJES
    # ========================================================

    ax.set_xlabel(
        r"$\log_{10}"
        r"\left("
        r"N_{\rm col}/\mathrm{cm}^{-2}"
        r"\right)$",
        fontsize=16,
    )

    ax.set_ylabel(
        r"$T_{\rm ex}$ (K)",
        fontsize=16,
    )

    ax.set_title(
        f"{REGION}: {molecula}",
        fontsize=17,
    )

    ax.tick_params(
        axis="both",
        labelsize=13,
    )

    ax.legend(
        loc="best",
        fontsize=14,
        markerscale = 1.2
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    # ========================================================
    # GUARDAR
    # ========================================================

    ruta_salida.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_pdf = (
        ruta_salida
        / f"{molecula}_Tex_vs_logN.pdf"
    )

    fig.savefig(
        ruta_pdf,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)

    print(
        f"[guardado] {ruta_pdf}"
    )

    return {
        "individuales": resultados_individuales,
        "todos_cores": resultado_total,
    }

# ============================================================
# FIGURA GLOBAL Tex vs log(Ncol)
# ============================================================

def plot_tex_vs_logn_global(
    molecula,
    ruta_salida,
):
    """
    Representa Tex frente a log10(Ncol) utilizando todas
    las regiones extensas disponibles.

    - Cada región compacta tiene un color fijo.
    - Cada región compacta tiene su propio ajuste lineal.
    - Se representa el intervalo de confianza del 95 %.
    - Los píxeles exteriores aparecen en gris.
    - Se realiza un ajuste conjunto de todas las regiones
      compactas si AJUSTE_CONJUNTO=True.
    - Se utiliza una única leyenda.
    """

    fig, ax = plt.subplots(
        figsize=(11, 8.5)
    )

    resultados_individuales = {}
    filas_leyenda = []

    # Datos para el ajuste conjunto
    x_todos = []
    y_todos = []
    sx_todos = []
    sy_todos = []

    exterior_etiquetado = False

    # ========================================================
    # RECORRER TODAS LAS REGIONES EXTENSAS
    # ========================================================

    for (
        region_extensa,
        contexto
    ) in datos_globales.items():

        # La molécula debe existir en esta región
        if molecula not in contexto["moleculas"]:
            continue

        d = contexto["datos"][molecula]

        mascara_compactas = contexto[
            "mascaras_totales"
        ][molecula]

        # ====================================================
        # PÍXELES EXTERIORES
        # ====================================================

        mascara_exterior = (
            ~mascara_compactas
            & np.isfinite(d["T"])
            & np.isfinite(d["N"])
            & np.isfinite(d["deltaT"])
            & np.isfinite(d["deltaN"])
            & (d["T"] > 0)
            & (d["N"] > 0)
            & (d["deltaT"] >= 0)
            & (d["deltaN"] >= 0)
        )

        T_ext = d["T"][mascara_exterior]
        N_ext = d["N"][mascara_exterior]

        deltaT_ext = d["deltaT"][
            mascara_exterior
        ]

        deltaN_ext = d["deltaN"][
            mascara_exterior
        ]

        if len(N_ext) > 0:

            logN_ext = np.log10(
                N_ext
            )

            delta_logN_ext = (
                deltaN_ext
                / (
                    N_ext
                    * np.log(10.0)
                )
            )

            ax.errorbar(
                logN_ext,
                T_ext,
                xerr=delta_logN_ext,
                yerr=deltaT_ext,
                fmt="o",
                markersize=3.5,
                alpha=0.18,
                color="0.65",
                ecolor="0.75",
                elinewidth=0.6,
                capsize=0,
                zorder=1,
            )

            exterior_etiquetado = True

        # ====================================================
        # REGIONES COMPACTAS INDIVIDUALES
        # ====================================================

        for (
            nombre_compacta,
            datos_compacta
        ) in contexto["compactas"].items():

            if molecula not in datos_compacta:
                continue

            label_compacta = labels_compactas[
                nombre_compacta
            ]

            dm = datos_compacta[
                molecula
            ]

            x = dm["logN"]
            y = dm["T"]

            sx = dm["delta_logN"]
            sy = dm["deltaT"]

            if len(x) == 0:
                continue

            color = colores_compactas_global[
                (
                    region_extensa,
                    nombre_compacta,
                )
            ]

            # ------------------------------------------------
            # Puntos
            # ------------------------------------------------

            ax.errorbar(
                x,
                y,
                xerr=sx,
                yerr=sy,
                fmt="o",
                markersize=5,
                alpha=0.75,
                color=color,
                ecolor=color,
                elinewidth=0.8,
                capsize=0,
                zorder=3,
            )

            # ------------------------------------------------
            # Ajuste individual
            # ------------------------------------------------

            resultado = ajustar_lineal(
                x,
                y,
                sx,
                sy,
            )

            resultados_individuales[
                (
                    region_extensa,
                    nombre_compacta,
                )
            ] = resultado

            if resultado is None:
                continue

            xmin = np.min(
                resultado["x"]
            )

            xmax = np.max(
                resultado["x"]
            )

            x_modelo = np.linspace(
                xmin,
                xmax,
                300,
            )

            y_modelo, sigma_modelo = (
                banda_lineal(
                    x_modelo,
                    resultado["slope"],
                    resultado["intercept"],
                    resultado["cov_beta"],
                )
            )

            # Recta
            ax.plot(
                x_modelo,
                y_modelo,
                color=color,
                linewidth=2.2,
                zorder=5,
            )

            # Intervalo de confianza del 95 %
            ax.fill_between(
                x_modelo,
                y_modelo - 1.96 * sigma_modelo,
                y_modelo + 1.96 * sigma_modelo,
                color=color,
                alpha=0.15,
                linewidth=0,
                zorder=2,
            )

            filas_leyenda.append(
                {
                    "nombre": label_compacta,
                    "color": color,
                    "rho": resultado["spearman"],
                    "slope": resultado["slope"],
                    "slope_err": resultado[
                        "slope_err"
                    ],
                }
            )

        # ====================================================
        # DATOS PARA EL AJUSTE CONJUNTO
        # ====================================================

        mascara_cores = (
            mascara_compactas
            & np.isfinite(d["T"])
            & np.isfinite(d["N"])
            & np.isfinite(d["deltaT"])
            & np.isfinite(d["deltaN"])
            & (d["T"] > 0)
            & (d["N"] > 0)
            & (d["deltaT"] >= 0)
            & (d["deltaN"] >= 0)
        )

        T_core = d["T"][
            mascara_cores
        ]

        N_core = d["N"][
            mascara_cores
        ]

        deltaT_core = d["deltaT"][
            mascara_cores
        ]

        deltaN_core = d["deltaN"][
            mascara_cores
        ]

        if len(N_core) > 0:

            x_todos.append(
                np.log10(
                    N_core
                )
            )

            y_todos.append(
                T_core
            )

            sx_todos.append(
                deltaN_core
                / (
                    N_core
                    * np.log(10.0)
                )
            )

            sy_todos.append(
                deltaT_core
            )

    # ========================================================
    # AJUSTE CONJUNTO GLOBAL
    # ========================================================

    resultado_total = None
    fila_ajuste_global = None

    if (
        AJUSTE_CONJUNTO
        and len(x_todos) > 0
    ):

        x_total = np.concatenate(
            x_todos
        )

        y_total = np.concatenate(
            y_todos
        )

        sx_total = np.concatenate(
            sx_todos
        )

        sy_total = np.concatenate(
            sy_todos
        )

        resultado_total = ajustar_lineal(
            x_total,
            y_total,
            sx_total,
            sy_total,
        )

        if resultado_total is not None:

            xmin = np.min(
                resultado_total["x"]
            )

            xmax = np.max(
                resultado_total["x"]
            )

            x_modelo = np.linspace(
                xmin,
                xmax,
                300,
            )

            y_modelo, sigma_modelo = (
                banda_lineal(
                    x_modelo,
                    resultado_total["slope"],
                    resultado_total["intercept"],
                    resultado_total["cov_beta"],
                )
            )

            ax.plot(
                x_modelo,
                y_modelo,
                color="black",
                linestyle="--",
                linewidth=3.0,
                zorder=7,
            )

            ax.fill_between(
                x_modelo,
                y_modelo - 1.96 * sigma_modelo,
                y_modelo + 1.96 * sigma_modelo,
                color="black",
                alpha=0.10,
                linewidth=0,
                zorder=2,
            )

            fila_ajuste_global = {
                "nombre": (
                    "All compact regions fit"
                ),
                "color": "black",
                "rho": resultado_total[
                    "spearman"
                ],
                "slope": resultado_total[
                    "slope"
                ],
                "slope_err": resultado_total[
                    "slope_err"
                ],
            }

    # ========================================================
    # EJES
    # ========================================================

    ax.set_xlabel(
        r"$\log_{10}"
        r"\left("
        r"N_{\rm col}/\mathrm{cm}^{-2}"
        r"\right)$",
        fontsize=16,
    )

    ax.set_ylabel(
        r"$T_{\rm ex}$ (K)",
        fontsize=16,
    )

    ax.set_title(
        f"GLOBAL: {molecula}",
        fontsize=17,
    )

    ax.tick_params(
        axis="both",
        labelsize=13,
    )

    # ========================================================
    # LEYENDA ÚNICA
    # ========================================================

    filas = []

    # Exterior
    if exterior_etiquetado:

        filas.append(
            TextArea(
                "Outside compact regions",
                textprops={
                    "color": "0.55",
                    "fontsize": 12,
                },
            )
        )

    # Regiones compactas
    for info in filas_leyenda:

        texto_nombre = TextArea(
            info["nombre"],
            textprops={
                "color": info["color"],
                "fontsize": 12,
            },
        )

        texto_ajuste = TextArea(
            (
                rf"   $\rho={info['rho']:.2f}$, "
                rf"$m={info['slope']:.2f}"
                rf"\pm{info['slope_err']:.2f}$"
            ),
            textprops={
                "color": "black",
                "fontsize": 12,
            },
        )

        filas.append(
            HPacker(
                children=[
                    texto_nombre,
                    texto_ajuste,
                ],
                align="baseline",
                pad=0,
                sep=2,
            )
        )

    # Ajuste global
    if fila_ajuste_global is not None:

        texto_nombre = TextArea(
            fila_ajuste_global["nombre"],
            textprops={
                "color": "black",
                "fontsize": 12,
            },
        )

        texto_ajuste = TextArea(
            (
                rf"   $\rho="
                rf"{fila_ajuste_global['rho']:.2f}$, "
                rf"$m="
                rf"{fila_ajuste_global['slope']:.2f}"
                rf"\pm"
                rf"{fila_ajuste_global['slope_err']:.2f}$"
            ),
            textprops={
                "color": "black",
                "fontsize": 12,
            },
        )

        filas.append(
            HPacker(
                children=[
                    texto_nombre,
                    texto_ajuste,
                ],
                align="baseline",
                pad=0,
                sep=2,
            )
        )

    if filas:

        contenido_leyenda = VPacker(
            children=filas,
            align="left",
            pad=0,
            sep=4,
        )

        leyenda = AnchoredOffsetbox(
            loc="upper left",
            child=contenido_leyenda,
            pad=0.5,
            borderpad=0.8,
            frameon=True,
        )

        leyenda.patch.set_facecolor(
            "white"
        )

        leyenda.patch.set_alpha(
            0.92
        )

        ax.add_artist(
            leyenda
        )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    # ========================================================
    # GUARDAR
    # ========================================================

    ruta_salida.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_pdf = (
        ruta_salida
        / f"{molecula}_Tex_vs_logN.pdf"
    )

    fig.savefig(
        ruta_pdf,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(
        fig
    )

    print(
        f"[guardado] {ruta_pdf}"
    )

    return {
        "individuales": resultados_individuales,
        "todos_cores": resultado_total,
    }

# ============================================================
# GENERAR TODAS LAS FIGURAS Tex vs log(Ncol)
# ============================================================

resultados_tex_logn = {}

ruta_salida = (
    ruta_salida_base
    / REGION
    / "Tex_vs_Ncol"
)


for molecula in moleculas:

    print(
        f"\n>>> {molecula}"
    )

    if MODO_GLOBAL:

        resultados_tex_logn[
            molecula
        ] = plot_tex_vs_logn_global(
            molecula,
            ruta_salida,
        )

    else:

        resultados_tex_logn[
            molecula
        ] = plot_tex_vs_logn(
            molecula,
            ruta_salida,
        )

# ============================================================
# COMPROBAR COMPATIBILIDAD WCS
# ============================================================

def wcs_compatibles(
    datos1,
    datos2,
    tolerancia_arcsec=1e-3,
):
    """
    Comprueba que dos mapas tienen la misma shape y que
    sus WCS asignan las mismas coordenadas celestes a
    varios píxeles de referencia.

    tolerancia_arcsec:
        diferencia máxima permitida en coordenadas.
    """

    shape1 = datos1["N"].shape
    shape2 = datos2["N"].shape

    if shape1 != shape2:
        return False

    ny, nx = shape1

    puntos = [
        (0, 0),
        (nx - 1, 0),
        (0, ny - 1),
        (nx - 1, ny - 1),
        (
            (nx - 1) / 2,
            (ny - 1) / 2,
        ),
    ]

    for x, y in puntos:

        coord1 = datos1[
            "wcs"
        ].pixel_to_world(
            x,
            y,
        )

        coord2 = datos2[
            "wcs"
        ].pixel_to_world(
            x,
            y,
        )

        separacion = coord1.separation(
            coord2
        ).arcsec

        if separacion > tolerancia_arcsec:
            return False

    return True

# ============================================================
# EXTRAER PÍXELES DE UNA REGIÓN COMPACTA ENTRE DOS MOLÉCULAS
# ============================================================

def extraer_pareja_molecular(
    nombre_region,
    mol1,
    mol2,
):
    """
    Extrae los píxeles comunes entre dos moléculas
    dentro de una región compacta concreta.
    """

    d1 = datos[mol1]
    d2 = datos[mol2]

    if not wcs_compatibles(
        d1,
        d2,
    ):
        print(
            f"[aviso] {mol1} vs {mol2}: "
            "los mapas no tienen la misma rejilla WCS."
        )
        return None

    mask1 = datos_regiones[
        nombre_region
    ][
        mol1
    ][
        "mask"
    ]

    mask2 = datos_regiones[
        nombre_region
    ][
        mol2
    ][
        "mask"
    ]

    mascara_comun = (
        mask1
        & mask2
    )

    N1 = d1["N"][
        mascara_comun
    ]

    N2 = d2["N"][
        mascara_comun
    ]

    dN1 = d1["deltaN"][
        mascara_comun
    ]

    dN2 = d2["deltaN"][
        mascara_comun
    ]

    validos = (
        np.isfinite(N1)
        & np.isfinite(N2)
        & np.isfinite(dN1)
        & np.isfinite(dN2)
        & (N1 > 0)
        & (N2 > 0)
        & (dN1 > 0)
        & (dN2 > 0)
    )

    N1 = N1[validos]
    N2 = N2[validos]

    dN1 = dN1[validos]
    dN2 = dN2[validos]

    return {
        "x": np.log10(N1),
        "y": np.log10(N2),

        "sx": (
            dN1
            / (
                N1
                * np.log(10.0)
            )
        ),

        "sy": (
            dN2
            / (
                N2
                * np.log(10.0)
            )
        ),

        "npix": len(N1),
    }

# ============================================================
# EXTRAER PÍXELES COMUNES ENTRE DOS MOLÉCULAS
# ============================================================

def extraer_pareja_exterior(
    mol1,
    mol2,
):
    """
    Extrae los píxeles comunes entre dos moléculas
    que están fuera de todas las regiones compactas.
    """

    d1 = datos[mol1]
    d2 = datos[mol2]

    if not wcs_compatibles(
        d1,
        d2,
    ):
        return None

    mascara_exterior = (
        ~mascaras_compactas_totales[mol1]
        & ~mascaras_compactas_totales[mol2]
    )

    mascara_valida = (
        mascara_exterior
        & np.isfinite(d1["N"])
        & np.isfinite(d2["N"])
        & np.isfinite(d1["deltaN"])
        & np.isfinite(d2["deltaN"])
        & (d1["N"] > 0)
        & (d2["N"] > 0)
        & (d1["deltaN"] >= 0)
        & (d2["deltaN"] >= 0)
    )

    N1 = d1["N"][
        mascara_valida
    ]

    N2 = d2["N"][
        mascara_valida
    ]

    dN1 = d1["deltaN"][
        mascara_valida
    ]

    dN2 = d2["deltaN"][
        mascara_valida
    ]

    return {
        "x": np.log10(N1),
        "y": np.log10(N2),

        "sx": (
            dN1
            / (
                N1
                * np.log(10.0)
            )
        ),

        "sy": (
            dN2
            / (
                N2
                * np.log(10.0)
            )
        ),

        "npix": len(N1),
    }

# ============================================================
# Hot cores por molecula 
# ============================================================

def extraer_pareja_todos_hotcores(
    mol1,
    mol2,
):
    """
    Extrae conjuntamente todos los píxeles pertenecientes
    a regiones compactas y válidos para ambas moléculas.
    """

    d1 = datos[mol1]
    d2 = datos[mol2]

    if not wcs_compatibles(
        d1,
        d2,
    ):
        return None

    mascara_cores = (
        mascaras_compactas_totales[mol1]
        & mascaras_compactas_totales[mol2]
    )

    mascara_valida = (
        mascara_cores
        & np.isfinite(d1["N"])
        & np.isfinite(d2["N"])
        & np.isfinite(d1["deltaN"])
        & np.isfinite(d2["deltaN"])
        & (d1["N"] > 0)
        & (d2["N"] > 0)
        & (d1["deltaN"] > 0)
        & (d2["deltaN"] > 0)
    )

    N1 = d1["N"][
        mascara_valida
    ]

    N2 = d2["N"][
        mascara_valida
    ]

    dN1 = d1["deltaN"][
        mascara_valida
    ]

    dN2 = d2["deltaN"][
        mascara_valida
    ]

    return {
        "x": np.log10(N1),
        "y": np.log10(N2),

        "sx": (
            dN1
            / (
                N1
                * np.log(10.0)
            )
        ),

        "sy": (
            dN2
            / (
                N2
                * np.log(10.0)
            )
        ),

        "npix": len(N1),
    }





# ============================================================
# FIGURA log(Nmol1) vs log(Nmol2) - REGIÓN EXTENSA
# ============================================================

def plot_ncol_vs_ncol(
    mol1,
    mol2,
    ruta_salida,
):
    """
    Figura única para toda la región extensa.

    Cada región compacta:
        - color diferente
        - barras de error
        - ajuste lineal independiente

    Además:
        - píxeles exteriores en gris
        - ajuste conjunto de todos los hot cores
    """

    exterior = extraer_pareja_exterior(
        mol1,
        mol2,
    )

    todos_cores = extraer_pareja_todos_hotcores(
        mol1,
        mol2,
    )

    if todos_cores is None:
        return None

    fig, ax = plt.subplots(
        figsize=(9, 7.5)
    )

    # ========================================================
    # EXTERIOR
    # ========================================================

    if (
        exterior is not None
        and exterior["npix"] > 0
    ):

        ax.errorbar(
            exterior["x"],
            exterior["y"],
            xerr=exterior["sx"],
            yerr=exterior["sy"],
            fmt="o",
            markersize=3.5,
            alpha=0.20,
            color="0.65",
            ecolor="0.75",
            elinewidth=0.6,
            capsize=0,
            label="Outside compact regions",
            zorder=1,
        )

    # ========================================================
    # REGIONES INDIVIDUALES
    # ========================================================

    resultados_individuales = {}

    lineas_stats = []

    for nombre_region in nombres_compactas:

        label_region = labels_compactas[
    nombre_region
]
        
        pareja = extraer_pareja_molecular(
            nombre_region,
            mol1,
            mol2,
        )

        if (
            pareja is None
            or pareja["npix"] == 0
        ):
            continue

        color = colores_compactas[
            nombre_region
        ]

        # ----------------------------------------
        # Puntos + errores
        # ----------------------------------------

        ax.errorbar(
            pareja["x"],
            pareja["y"],
            xerr=pareja["sx"],
            yerr=pareja["sy"],
            fmt="o",
            markersize=5,
            alpha=0.75,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=0,
            label=label_region,
            zorder=3,
        )

        # ----------------------------------------
        # Ajuste individual
        # ----------------------------------------

        resultado = ajustar_lineal(
            pareja["x"],
            pareja["y"],
            pareja["sx"],
            pareja["sy"],
        )

        resultados_individuales[
            nombre_region
        ] = resultado

        if resultado is None:
            continue

        xmin = np.min(
            resultado["x"]
        )

        xmax = np.max(
            resultado["x"]
        )

        x_modelo = np.linspace(
            xmin,
            xmax,
            300,
        )

        y_modelo = (
            resultado["slope"]
            * x_modelo
            + resultado["intercept"]
        )

        ax.plot(
            x_modelo,
            y_modelo,
            color=color,
            linewidth=2.2,
            zorder=5,
        )

        lineas_stats.append(
            f"{label_region}: "
            f"N={resultado['npix']}, "
            f"r={resultado['pearson']:.2f}, "
            f"rho={resultado['spearman']:.2f}, "
            f"m={resultado['slope']:.2f}"
            f"+/-{resultado['slope_err']:.2f}, "
            f"R2={resultado['r2']:.2f}"
        )

    # ========================================================
    # AJUSTE DE TODOS LOS HOT CORES
    # ========================================================

    resultado_total = ajustar_lineal(
        todos_cores["x"],
        todos_cores["y"],
        todos_cores["sx"],
        todos_cores["sy"],
    )

    if (
        resultado_total is not None
        and len(nombres_compactas) > 1
    ):

        xmin = np.min(
            resultado_total["x"]
        )

        xmax = np.max(
            resultado_total["x"]
        )

        x_modelo = np.linspace(
            xmin,
            xmax,
            300,
        )

        y_modelo, sigma_modelo = (
            banda_lineal(
                x_modelo,
                resultado_total["slope"],
                resultado_total["intercept"],
                resultado_total["cov_beta"],
            )
        )

        ax.plot(
            x_modelo,
            y_modelo,
            color="black",
            linestyle="--",
            linewidth=2.8,
            label="All hot cores fit",
            zorder=7,
        )

        ax.fill_between(
            x_modelo,
            y_modelo - sigma_modelo,
            y_modelo + sigma_modelo,
            color="black",
            alpha=0.08,
            zorder=2,
        )

        lineas_stats.append(
            "All cores: "
            f"N={resultado_total['npix']}, "
            f"r={resultado_total['pearson']:.2f}, "
            f"rho={resultado_total['spearman']:.2f}, "
            f"m={resultado_total['slope']:.2f}"
            f"+/-{resultado_total['slope_err']:.2f}, "
            f"R2={resultado_total['r2']:.2f}"
        )

    # ========================================================
    # ESTADÍSTICAS
    # ========================================================

    if lineas_stats:

        texto = "\n".join(
            lineas_stats
        )

        ax.text(
            0.02,
            0.98,
            texto,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12.5,
            linespacing=1.25,
            bbox=dict(
                boxstyle="square,pad=0.55",
                facecolor="white",
                alpha=0.90,
                ),
            zorder=10,
            )
    # ========================================================
    # EJES
    # ========================================================

    ax.set_xlabel(
        rf"$\log_{{10}}"
        rf"\left[N({mol1})/"
        rf"\mathrm{{cm}}^{{-2}}\right]$",
        fontsize=15,
    )

    ax.set_ylabel(
        rf"$\log_{{10}}"
        rf"\left[N({mol2})/"
        rf"\mathrm{{cm}}^{{-2}}\right]$",
        fontsize=15,
    )

    ax.set_title(
        f"{REGION}: {mol1} vs {mol2}",
        fontsize=16,
    )

    ax.tick_params(
        axis="both",
        labelsize=13,
    )

    ax.legend(
        loc="best",
        fontsize=14,
        markerscale = 1.2
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    # ========================================================
    # GUARDAR
    # ========================================================

    ruta_salida.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_pdf = (
        ruta_salida
        / f"{mol1}_vs_{mol2}.pdf"
    )

    fig.savefig(
        ruta_pdf,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)

    print(
        f"[guardado] {ruta_pdf}"
    )

    return {
        "individuales": resultados_individuales,
        "todos_cores": resultado_total,
    }
# ============================================================
# FIGURA GLOBAL log(Nmol1) vs log(Nmol2)
# ============================================================

def plot_ncol_vs_ncol_global(
    mol1,
    mol2,
    ruta_salida,
):
    """
    Representa log10(Nmol2) frente a log10(Nmol1)
    utilizando todas las regiones extensas disponibles.

    - Cada región compacta tiene un color fijo.
    - Cada región compacta tiene su propio ajuste lineal.
    - Se representa el intervalo de confianza del 95 %.
    - Los píxeles exteriores aparecen en gris.
    - Se realiza un ajuste conjunto de todas las regiones
      compactas si AJUSTE_CONJUNTO=True.
    - Una región extensa se omite si no contiene ambas
      moléculas.
    - Se utiliza una única leyenda.
    """

    fig, ax = plt.subplots(
        figsize=(11, 8.5)
    )

    resultados_individuales = {}
    filas_leyenda = []

    x_todos = []
    y_todos = []
    sx_todos = []
    sy_todos = []

    exterior_etiquetado = False
    hay_datos = False

    # ========================================================
    # RECORRER TODAS LAS REGIONES EXTENSAS
    # ========================================================

    for (
        region_extensa,
        contexto
    ) in datos_globales.items():

        # Ambas moléculas deben existir
        if (
            mol1 not in contexto["moleculas"]
            or mol2 not in contexto["moleculas"]
        ):
            continue

        d1 = contexto["datos"][mol1]
        d2 = contexto["datos"][mol2]

        # Los mapas deben compartir rejilla
        if not wcs_compatibles(
            d1,
            d2,
        ):

            print(
                f"[aviso] {region_extensa}: "
                f"{mol1} vs {mol2}: "
                "WCS incompatibles. "
                "Se omite esta región."
            )

            continue

        hay_datos = True

        # ====================================================
        # PÍXELES EXTERIORES
        # ====================================================

        mascara_exterior = (
            ~contexto[
                "mascaras_totales"
            ][mol1]
            & ~contexto[
                "mascaras_totales"
            ][mol2]
            & np.isfinite(d1["N"])
            & np.isfinite(d2["N"])
            & np.isfinite(d1["deltaN"])
            & np.isfinite(d2["deltaN"])
            & (d1["N"] > 0)
            & (d2["N"] > 0)
            & (d1["deltaN"] >= 0)
            & (d2["deltaN"] >= 0)
        )

        N1_ext = d1["N"][
            mascara_exterior
        ]

        N2_ext = d2["N"][
            mascara_exterior
        ]

        dN1_ext = d1["deltaN"][
            mascara_exterior
        ]

        dN2_ext = d2["deltaN"][
            mascara_exterior
        ]

        if len(N1_ext) > 0:

            x_ext = np.log10(
                N1_ext
            )

            y_ext = np.log10(
                N2_ext
            )

            sx_ext = (
                dN1_ext
                / (
                    N1_ext
                    * np.log(10.0)
                )
            )

            sy_ext = (
                dN2_ext
                / (
                    N2_ext
                    * np.log(10.0)
                )
            )

            ax.errorbar(
                x_ext,
                y_ext,
                xerr=sx_ext,
                yerr=sy_ext,
                fmt="o",
                markersize=3.5,
                alpha=0.18,
                color="0.65",
                ecolor="0.75",
                elinewidth=0.6,
                capsize=0,
                zorder=1,
            )

            exterior_etiquetado = True

        # ====================================================
        # REGIONES COMPACTAS
        # ====================================================

        for (
            nombre_compacta,
            datos_compacta
        ) in contexto["compactas"].items():

            if (
                mol1 not in datos_compacta
                or mol2 not in datos_compacta
            ):
                continue

            label_compacta = labels_compactas[
                nombre_compacta
            ]

            mask1 = datos_compacta[
                mol1
            ]["mask"]

            mask2 = datos_compacta[
                mol2
            ]["mask"]

            mascara_comun = (
                mask1
                & mask2
            )

            N1 = d1["N"][
                mascara_comun
            ]

            N2 = d2["N"][
                mascara_comun
            ]

            dN1 = d1["deltaN"][
                mascara_comun
            ]

            dN2 = d2["deltaN"][
                mascara_comun
            ]

            validos = (
                np.isfinite(N1)
                & np.isfinite(N2)
                & np.isfinite(dN1)
                & np.isfinite(dN2)
                & (N1 > 0)
                & (N2 > 0)
                & (dN1 >= 0)
                & (dN2 >= 0)
            )

            N1 = N1[
                validos
            ]

            N2 = N2[
                validos
            ]

            dN1 = dN1[
                validos
            ]

            dN2 = dN2[
                validos
            ]

            if len(N1) == 0:
                continue

            x = np.log10(
                N1
            )

            y = np.log10(
                N2
            )

            sx = (
                dN1
                / (
                    N1
                    * np.log(10.0)
                )
            )

            sy = (
                dN2
                / (
                    N2
                    * np.log(10.0)
                )
            )

            color = colores_compactas_global[
                (
                    region_extensa,
                    nombre_compacta,
                )
            ]

            # ------------------------------------------------
            # Puntos
            # ------------------------------------------------

            ax.errorbar(
                x,
                y,
                xerr=sx,
                yerr=sy,
                fmt="o",
                markersize=5,
                alpha=0.75,
                color=color,
                ecolor=color,
                elinewidth=0.8,
                capsize=0,
                zorder=3,
            )

            # ------------------------------------------------
            # Ajuste individual
            # ------------------------------------------------

            resultado = ajustar_lineal(
                x,
                y,
                sx,
                sy,
            )

            resultados_individuales[
                (
                    region_extensa,
                    nombre_compacta,
                )
            ] = resultado

            if resultado is None:
                continue

            xmin = np.min(
                resultado["x"]
            )

            xmax = np.max(
                resultado["x"]
            )

            x_modelo = np.linspace(
                xmin,
                xmax,
                300,
            )

            y_modelo, sigma_modelo = (
                banda_lineal(
                    x_modelo,
                    resultado["slope"],
                    resultado["intercept"],
                    resultado["cov_beta"],
                )
            )

            # Recta
            ax.plot(
                x_modelo,
                y_modelo,
                color=color,
                linewidth=2.2,
                zorder=5,
            )

            # Intervalo de confianza del 95 %
            ax.fill_between(
                x_modelo,
                y_modelo - 1.96 * sigma_modelo,
                y_modelo + 1.96 * sigma_modelo,
                color=color,
                alpha=0.15,
                linewidth=0,
                zorder=2,
            )

            filas_leyenda.append(
                {
                    "nombre": label_compacta,
                    "color": color,
                    "rho": resultado["spearman"],
                    "slope": resultado["slope"],
                    "slope_err": resultado[
                        "slope_err"
                    ],
                }
            )

        # ====================================================
        # DATOS PARA EL AJUSTE CONJUNTO
        # ====================================================

        mascara_cores = (
            contexto[
                "mascaras_totales"
            ][mol1]
            & contexto[
                "mascaras_totales"
            ][mol2]
            & np.isfinite(d1["N"])
            & np.isfinite(d2["N"])
            & np.isfinite(d1["deltaN"])
            & np.isfinite(d2["deltaN"])
            & (d1["N"] > 0)
            & (d2["N"] > 0)
            & (d1["deltaN"] >= 0)
            & (d2["deltaN"] >= 0)
        )

        N1_core = d1["N"][
            mascara_cores
        ]

        N2_core = d2["N"][
            mascara_cores
        ]

        dN1_core = d1["deltaN"][
            mascara_cores
        ]

        dN2_core = d2["deltaN"][
            mascara_cores
        ]

        if len(N1_core) > 0:

            x_todos.append(
                np.log10(
                    N1_core
                )
            )

            y_todos.append(
                np.log10(
                    N2_core
                )
            )

            sx_todos.append(
                dN1_core
                / (
                    N1_core
                    * np.log(10.0)
                )
            )

            sy_todos.append(
                dN2_core
                / (
                    N2_core
                    * np.log(10.0)
                )
            )

    # ========================================================
    # NINGUNA REGIÓN VÁLIDA
    # ========================================================

    if not hay_datos:

        plt.close(
            fig
        )

        return None

    # ========================================================
    # AJUSTE CONJUNTO GLOBAL
    # ========================================================

    resultado_total = None
    fila_ajuste_global = None

    if (
        AJUSTE_CONJUNTO
        and len(x_todos) > 0
    ):

        x_total = np.concatenate(
            x_todos
        )

        y_total = np.concatenate(
            y_todos
        )

        sx_total = np.concatenate(
            sx_todos
        )

        sy_total = np.concatenate(
            sy_todos
        )

        resultado_total = ajustar_lineal(
            x_total,
            y_total,
            sx_total,
            sy_total,
        )

        if resultado_total is not None:

            xmin = np.min(
                resultado_total["x"]
            )

            xmax = np.max(
                resultado_total["x"]
            )

            x_modelo = np.linspace(
                xmin,
                xmax,
                300,
            )

            y_modelo, sigma_modelo = (
                banda_lineal(
                    x_modelo,
                    resultado_total["slope"],
                    resultado_total["intercept"],
                    resultado_total["cov_beta"],
                )
            )

            ax.plot(
                x_modelo,
                y_modelo,
                color="black",
                linestyle="--",
                linewidth=3.0,
                zorder=7,
            )

            ax.fill_between(
                x_modelo,
                y_modelo - 1.96 * sigma_modelo,
                y_modelo + 1.96 * sigma_modelo,
                color="black",
                alpha=0.10,
                linewidth=0,
                zorder=2,
            )

            fila_ajuste_global = {
                "nombre": (
                    "All compact regions fit"
                ),
                "color": "black",
                "rho": resultado_total[
                    "spearman"
                ],
                "slope": resultado_total[
                    "slope"
                ],
                "slope_err": resultado_total[
                    "slope_err"
                ],
            }

    # ========================================================
    # EJES
    # ========================================================

    ax.set_xlabel(
        rf"$\log_{{10}}"
        rf"\left[N({mol1})/"
        rf"\mathrm{{cm}}^{{-2}}\right]$",
        fontsize=15,
    )

    ax.set_ylabel(
        rf"$\log_{{10}}"
        rf"\left[N({mol2})/"
        rf"\mathrm{{cm}}^{{-2}}\right]$",
        fontsize=15,
    )

    ax.set_title(
        f"GLOBAL: {mol1} vs {mol2}",
        fontsize=16,
    )

    ax.tick_params(
        axis="both",
        labelsize=13,
    )

    # ========================================================
    # LEYENDA ÚNICA
    # ========================================================

    filas = []

    # Exterior
    if exterior_etiquetado:

        filas.append(
            TextArea(
                "Outside compact regions",
                textprops={
                    "color": "0.55",
                    "fontsize": 12,
                },
            )
        )

    # Regiones compactas
    for info in filas_leyenda:

        texto_nombre = TextArea(
            info["nombre"],
            textprops={
                "color": info["color"],
                "fontsize": 12,
            },
        )

        texto_ajuste = TextArea(
            (
                rf"   $\rho={info['rho']:.2f}$, "
                rf"$m={info['slope']:.2f}"
                rf"\pm{info['slope_err']:.2f}$"
            ),
            textprops={
                "color": "black",
                "fontsize": 12,
            },
        )

        filas.append(
            HPacker(
                children=[
                    texto_nombre,
                    texto_ajuste,
                ],
                align="baseline",
                pad=0,
                sep=2,
            )
        )

    # Ajuste global
    if fila_ajuste_global is not None:

        texto_nombre = TextArea(
            fila_ajuste_global["nombre"],
            textprops={
                "color": "black",
                "fontsize": 12,
            },
        )

        texto_ajuste = TextArea(
            (
                rf"   $\rho="
                rf"{fila_ajuste_global['rho']:.2f}$, "
                rf"$m="
                rf"{fila_ajuste_global['slope']:.2f}"
                rf"\pm"
                rf"{fila_ajuste_global['slope_err']:.2f}$"
            ),
            textprops={
                "color": "black",
                "fontsize": 12,
            },
        )

        filas.append(
            HPacker(
                children=[
                    texto_nombre,
                    texto_ajuste,
                ],
                align="baseline",
                pad=0,
                sep=2,
            )
        )

    if filas:

        contenido_leyenda = VPacker(
            children=filas,
            align="left",
            pad=0,
            sep=4,
        )

        leyenda = AnchoredOffsetbox(
            loc="upper left",
            child=contenido_leyenda,
            pad=0.5,
            borderpad=0.8,
            frameon=True,
        )

        leyenda.patch.set_facecolor(
            "white"
        )

        leyenda.patch.set_alpha(
            0.92
        )

        ax.add_artist(
            leyenda
        )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    # ========================================================
    # GUARDAR
    # ========================================================

    ruta_salida.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_pdf = (
        ruta_salida
        / f"{mol1}_vs_{mol2}.pdf"
    )

    fig.savefig(
        ruta_pdf,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(
        fig
    )

    print(
        f"[guardado] {ruta_pdf}"
    )

    return {
        "individuales": resultados_individuales,
        "todos_cores": resultado_total,
    }

# ============================================================
# GENERAR TODAS LAS COMPARACIONES Ncol vs Ncol
# ============================================================

resultados_ncol_ncol = {}

ruta_salida = (
    ruta_salida_base
    / REGION
    / "Ncol_vs_Ncol"
)


if MODO_GLOBAL:

    # ========================================================
    # PAREJAS GLOBALES VÁLIDAS
    # ========================================================

    parejas_moleculares = []

    for mol1, mol2 in combinations(
        moleculas,
        2,
    ):

        # Las dos moléculas deben coexistir
        # al menos en una región extensa
        existe_region_comun = any(
            (
                mol1 in contexto["moleculas"]
                and mol2 in contexto["moleculas"]
            )
            for contexto
            in datos_globales.values()
        )

        if existe_region_comun:

            parejas_moleculares.append(
                (
                    mol1,
                    mol2,
                )
            )

    print(
        f"\nParejas moleculares globales: "
        f"{len(parejas_moleculares)}"
    )

    # ========================================================
    # GENERAR FIGURAS GLOBAL
    # ========================================================

    for mol1, mol2 in parejas_moleculares:

        print(
            f"\n>>> GLOBAL: "
            f"{mol1} vs {mol2}"
        )

        resultados_ncol_ncol[
            (
                mol1,
                mol2,
            )
        ] = plot_ncol_vs_ncol_global(
            mol1,
            mol2,
            ruta_salida,
        )


else:

    # ========================================================
    # COMPORTAMIENTO ANTIGUO
    # ========================================================

    for mol1, mol2 in combinations(
        moleculas,
        2,
    ):

        print(
            f"\n>>> {mol1} vs {mol2}"
        )

        resultados_ncol_ncol[
            (
                mol1,
                mol2,
            )
        ] = plot_ncol_vs_ncol(
            mol1,
            mol2,
            ruta_salida,
        )