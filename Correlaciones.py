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


# ÚNICA VARIABLE QUE CAMBIAMOS MANUALMENTE
REGION = "mm31_d2"


# Regiones compactas contenidas en cada región extensa
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
# COMPROBACIONES INICIALES
# ============================================================

if REGION not in regiones_compactas:
    raise ValueError(
        f"La región {REGION!r} no está definida en "
        "'regiones_compactas'."
    )

ruta_region_extensa = (
    ruta_fits_chi2
    / REGION
)

if not ruta_region_extensa.exists():
    raise FileNotFoundError(
        f"No existe la carpeta:\n"
        f"{ruta_region_extensa}"
    )


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


moleculas = buscar_moleculas(
    ruta_region_extensa
)


if len(moleculas) == 0:
    raise RuntimeError(
        f"No se han encontrado moléculas válidas "
        f"para {REGION}."
    )


print("\n" + "=" * 70)
print(f"REGIÓN EXTENSA: {REGION}")
print("=" * 70)

print(
    f"\nMoléculas encontradas: "
    f"{len(moleculas)}"
)

for molecula in moleculas:
    print(
        f"    - {molecula}"
    )
    
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


# ============================================================
# CARGAR TODAS LAS MOLÉCULAS
# ============================================================

datos = {}

print("\nCargando mapas chi2...")

for molecula in moleculas:

    datos[molecula] = cargar_mapas_molecula(
        ruta_region_extensa,
        molecula,
    )

    print(
        f"    {molecula:20s} "
        f"{datos[molecula]['N'].shape}"
    )
# ============================================================
# REGIONES COMPACTAS
# ============================================================

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
# MÁSCARA CONJUNTA DE TODAS LAS REGIONES COMPACTAS
# ============================================================

mascaras_compactas_totales = {}

for molecula in moleculas:

    d = datos[molecula]

    mascara_total = np.zeros(
        d["N"].shape,
        dtype=bool,
    )

    for archivo_region in regiones_compactas[REGION]:

        ruta_region_ds9 = (
            ruta_regiones
            / archivo_region
        )

        mascara = crear_mascara_region(
            ruta_region_ds9,
            d["wcs"],
            d["N"].shape,
        )

        mascara_total |= mascara

    mascaras_compactas_totales[
        molecula
    ] = mascara_total

# ============================================================
# EXTRAER DATOS DE LAS REGIONES COMPACTAS
# ============================================================

datos_regiones = {}


for archivo_region in regiones_compactas[
    REGION
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

    datos_regiones[
        nombre_compacta
    ] = {}

    for molecula in moleculas:

        d = datos[molecula]

        mascara_region = crear_mascara_region(
            ruta_region_ds9,
            d["wcs"],
            d["N"].shape,
        )

        # ----------------------------------------
        # Máscara de calidad básica
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
        # Pasar N a log10
        # ----------------------------------------

        logN = np.log10(
            N
        )

        # Propagación del error:
        #
        # sigma_logN =
        # sigma_N / (N ln(10))
        #
        delta_logN = (
            deltaN
            / (
                N
                * np.log(10.0)
            )
        )

        datos_regiones[
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

    mascara_compactas = (
        mascaras_compactas_totales[
            molecula
        ]
    )

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

    T = d["T"][
        mascara_exterior
    ]

    N = d["N"][
        mascara_exterior
    ]

    return {
        "T": T,
        "logN": np.log10(N),
        "npix": len(N),
    }

# ============================================================
# FIGURA Tex vs log(Ncol)
# ============================================================

def plot_tex_vs_logn(
    nombre_region,
    molecula,
    datos_mol,
    ruta_salida,
):
    """
    Representa Tex frente a log10(Ncol) para todos
    los píxeles válidos de una región compacta.
    """

    x = datos_mol[
        "logN"
    ]

    y = datos_mol[
        "T"
    ]

    sx = datos_mol[
        "delta_logN"
    ]

    sy = datos_mol[
        "deltaT"
    ]

    resultado = ajustar_lineal(
        x,
        y,
        sx,
        sy,
        )
    exterior = extraer_pixeles_exteriores_molecula(molecula)

    if resultado is None:
        print(
            f"[aviso] {nombre_region} - "
            f"{molecula}: no hay suficientes "
            "píxeles para realizar el ajuste."
        )
        return None

    # ----------------------------------------
    # Figura
    # ----------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    # Píxeles fuera de todas las regiones compactas
    ax.scatter(
        exterior["logN"],
        exterior["T"],
        s=22,
        alpha=0.30,
        color="0.65",
        edgecolors="none",
        label="Outside compact regions",
        zorder=1,
        )

    # Píxeles de la región compacta
    ax.scatter(
        resultado["x"],
        resultado["y"],
        s=32,
        alpha=0.75,
        color="tab:blue",
        edgecolors="none",
        label=nombre_region,
        zorder=3,
        )

    # ----------------------------------------
    # Ajuste
    # ----------------------------------------

    xmin = np.min(
        resultado["x"]
    )

    xmax = np.max(
        resultado["x"]
    )

    margen = (
        0.04
        * (
            xmax
            - xmin
        )
    )

    if margen == 0:
        margen = 0.05

    x_modelo = np.linspace(
        xmin - margen,
        xmax + margen,
        300,
    )

    y_modelo, sigma_modelo = banda_lineal(
    x_modelo,
    resultado["slope"],
    resultado["intercept"],
    resultado["cov_beta"],
    )

    ax.plot(
        x_modelo,
        y_modelo,
        linewidth=2.5,
        label="Linear fit",
        zorder=4,
    )

    ax.fill_between(
        x_modelo,
        y_modelo - sigma_modelo,
        y_modelo + sigma_modelo,
        alpha=0.18,
        zorder=2,
    )

    # ----------------------------------------
    # Caja de estadísticas
    # ----------------------------------------

    texto = (
        rf"$N_{{\rm pix}}$ = "
        f"{resultado['npix']}\n\n"

        rf"Pearson = "
        f"{resultado['pearson']:.3f}\n"

        rf"Spearman = "
        f"{resultado['spearman']:.3f}\n\n"

        rf"Slope = "
        f"{resultado['slope']:.3f}"
        rf" $\pm$ "
        f"{resultado['slope_err']:.3f}\n"

        rf"Intercept = "
        f"{resultado['intercept']:.3f}"
        rf" $\pm$ "
        f"{resultado['intercept_err']:.3f}\n\n"

        rf"$\sigma_{{\rm res}}$ = "
        f"{resultado['sigma_res']:.3f} K\n\n"

        rf"$R^2$ = "
        f"{resultado['r2']:.3f}"
    )

    ax.text(
        0.03,
        0.97,
        texto,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(
            boxstyle="square,pad=0.5",
            facecolor="white",
            alpha=0.9,
        ),
    )

    # ----------------------------------------
    # Ejes
    # ----------------------------------------

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
        f"{nombre_region}: {molecula}",
        fontsize=17,
    )

    ax.tick_params(
        axis="both",
        labelsize=13,
    )

    ax.legend(
        loc="lower right",
        fontsize=11,
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    # ----------------------------------------
    # Guardar
    # ----------------------------------------

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

    return resultado

# ============================================================
# GENERAR TODAS LAS FIGURAS Tex vs log(Ncol)
# ============================================================

resultados_tex_logn = {}


print(
    "\n"
    + "=" * 70
)

print(
    "GENERANDO Tex vs log(Ncol)"
)

print(
    "=" * 70
)


for nombre_region, datos_region in datos_regiones.items():

    print(
        f"\n>>> {nombre_region}"
    )

    ruta_salida = (
        ruta_salida_base
        / REGION
        / nombre_region
        / "Tex_vs_Ncol"
    )

    resultados_tex_logn[
        nombre_region
    ] = {}

    for molecula, datos_mol in datos_region.items():

        resultado = plot_tex_vs_logn(
            nombre_region,
            molecula,
            datos_mol,
            ruta_salida,
        )

        resultados_tex_logn[
            nombre_region
        ][
            molecula
        ] = resultado

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
# EXTRAER PÍXELES COMUNES ENTRE DOS MOLÉCULAS
# ============================================================

def extraer_pareja_molecular(
    nombre_region,
    mol1,
    mol2,
):
    """
    Extrae únicamente los píxeles válidos y comunes
    entre dos moléculas dentro de una región compacta.

    Devuelve logN1, logN2 y sus incertidumbres.
    """

    d1 = datos[mol1]
    d2 = datos[mol2]

    # ----------------------------------------
    # Comprobar WCS
    # ----------------------------------------

    if not wcs_compatibles(
        d1,
        d2,
    ):
        print(
            f"[aviso] {mol1} vs {mol2}: "
            "los mapas no tienen la misma "
            "rejilla WCS. Se omite."
        )

        return None

    # Máscaras ya calculadas para esta
    # región compacta
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

    # ----------------------------------------
    # Intersección espacial
    # ----------------------------------------

    mask_comun = (
        mask1
        & mask2
    )

    N1 = d1["N"][
        mask_comun
    ]

    N2 = d2["N"][
        mask_comun
    ]

    dN1 = d1["deltaN"][
        mask_comun
    ]

    dN2 = d2["deltaN"][
        mask_comun
    ]

    # Protección adicional
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

    # ----------------------------------------
    # Logaritmos
    # ----------------------------------------

    logN1 = np.log10(
        N1
    )

    logN2 = np.log10(
        N2
    )

    dlogN1 = (
        dN1
        / (
            N1
            * np.log(10.0)
        )
    )

    dlogN2 = (
        dN2
        / (
            N2
            * np.log(10.0)
        )
    )

    return {
        "x": logN1,
        "y": logN2,
        "sx": dlogN1,
        "sy": dlogN2,
        "npix": len(logN1),
    }

# ============================================================
# EXTRAER PÍXELES EXTERIORES COMUNES ENTRE DOS MOLÉCULAS
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

    # Fuera de todas las regiones compactas
    # para ambas moléculas
    mascara_exterior = (
        ~mascaras_compactas_totales[mol1]
        & ~mascaras_compactas_totales[mol2]
    )

    # Píxeles válidos en ambas moléculas
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

    return {
        "x": np.log10(N1),
        "y": np.log10(N2),
        "npix": len(N1),
    }

# ============================================================
# FIGURA log(Nmol1) vs log(Nmol2)
# ============================================================

def plot_ncol_vs_ncol(
    nombre_region,
    mol1,
    mol2,
    ruta_salida,
):
    """
    Representa log10(Nmol2) frente a log10(Nmol1)
    para los píxeles comunes de una región compacta.
    """

    pareja = extraer_pareja_molecular(
        nombre_region,
        mol1,
        mol2,
    )
    exterior = extraer_pareja_exterior(
        mol1,
        mol2,
        )   

    if pareja is None:
        return None

    if pareja["npix"] < 3:

        print(
            f"[aviso] {nombre_region}: "
            f"{mol1} vs {mol2}: "
            "menos de 3 píxeles comunes."
        )

        return None

    # ----------------------------------------
    # ODR
    # ----------------------------------------

    resultado = ajustar_lineal(
        pareja["x"],
        pareja["y"],
        pareja["sx"],
        pareja["sy"],
        )

    if resultado is None:
        return None

    # ----------------------------------------
    # Figura
    # ----------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    # ----------------------------------------
    # Píxeles exteriores
    # ----------------------------------------

    if exterior is not None:

        ax.scatter(
            exterior["x"],
            exterior["y"],
            s=22,
            alpha=0.30,
            color="0.65",
            edgecolors="none",
            label="Outside compact regions",
            zorder=1,
        )


    # ----------------------------------------
    # Región compacta
    # ----------------------------------------

    ax.scatter(
        resultado["x"],
        resultado["y"],
        s=32,
        alpha=0.75,
        color="tab:blue",
        edgecolors="none",
        label=nombre_region,
        zorder=3,
    )

    # ----------------------------------------
    # Límites generales
    # ----------------------------------------

    xmin = np.min(
        resultado["x"]
    )

    xmax = np.max(
        resultado["x"]
    )

    ymin = np.min(
        resultado["y"]
    )

    ymax = np.max(
        resultado["y"]
    )

    rango_x = xmax - xmin
    rango_y = ymax - ymin

    margen_x = (
        0.05 * rango_x
        if rango_x > 0
        else 0.05
    )

    margen_y = (
        0.05 * rango_y
        if rango_y > 0
        else 0.05
    )

    # ----------------------------------------
    # Recta ODR
    # ----------------------------------------

    x_modelo = np.linspace(
        xmin - margen_x,
        xmax + margen_x,
        300,
    )

    y_modelo, sigma_modelo = banda_lineal(
        x_modelo,
        resultado["slope"],
        resultado["intercept"],
        resultado["cov_beta"],
        )

    ax.plot(
        x_modelo,
        y_modelo,
        linewidth=2.5,
        label="Linear fit",
        zorder=5,
    )

    # Banda 1 sigma
    ax.fill_between(
        x_modelo,
        y_modelo - sigma_modelo,
        y_modelo + sigma_modelo,
        alpha=0.18,
        zorder=2,
    )

    # ----------------------------------------
    # Estadísticas
    # ----------------------------------------

    texto = (
        rf"$N_{{\rm pix}}$ = "
        f"{resultado['npix']}\n\n"

        rf"Pearson $r$ = "
        f"{resultado['pearson']:.3f}\n"

        rf"Spearman $\rho$ = "
        f"{resultado['spearman']:.3f}\n\n"

        rf"Slope = "
        f"{resultado['slope']:.3f}"
        rf" $\pm$ "
        f"{resultado['slope_err']:.3f}\n"

        rf"Intercept = "
        f"{resultado['intercept']:.3f}"
        rf" $\pm$ "
        f"{resultado['intercept_err']:.3f}\n\n"

        rf"$\sigma_{{\rm res}}$ = "
        f"{resultado['sigma_res']:.3f} dex\n"

        rf"$\sigma_{{\rm meas}}$ = "
        f"{resultado['sigma_meas']:.3f} dex\n"

        rf"$\sigma_{{\rm int}}$ = "
        f"{resultado['sigma_int']:.3f} dex\n\n"

        rf"$R^2$ = "
        f"{resultado['r2']:.3f}"
    )

    ax.text(
        0.03,
        0.97,
        texto,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        bbox=dict(
            boxstyle="square,pad=0.5",
            facecolor="white",
            alpha=0.9,
        ),
        zorder=10,
    )

    # ----------------------------------------
    # Labels
    # ----------------------------------------

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
        f"{nombre_region}: "
        f"{mol1} vs {mol2}",
        fontsize=16,
    )

    ax.tick_params(
        axis="both",
        labelsize=13,
    )

    ax.legend(
        loc="lower right",
        fontsize=11,
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    # ----------------------------------------
    # Guardar
    # ----------------------------------------

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

    return resultado

# ============================================================
# GENERAR TODAS LAS COMPARACIONES Ncol vs Ncol
# ============================================================

resultados_ncol_ncol = {}


print(
    "\n"
    + "=" * 70
)

print(
    "GENERANDO log(Nmol1) vs log(Nmol2)"
)

print(
    "=" * 70
)


for nombre_region in datos_regiones:

    print(
        f"\n>>> {nombre_region}"
    )

    ruta_salida = (
        ruta_salida_base
        / REGION
        / nombre_region
        / "Ncol_vs_Ncol"
    )

    resultados_ncol_ncol[
        nombre_region
    ] = {}

    # Todas las parejas únicas
    for mol1, mol2 in combinations(
        moleculas,
        2,
    ):

        print(
            f"    {mol1} "
            f"vs {mol2}"
        )

        resultado = plot_ncol_vs_ncol(
            nombre_region,
            mol1,
            mol2,
            ruta_salida,
        )

        resultados_ncol_ncol[
            nombre_region
        ][
            (mol1, mol2)
        ] = resultado
