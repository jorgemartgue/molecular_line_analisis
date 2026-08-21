#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:17:04 2026

@author: jorge
"""

"""
Módulo de alto nivel para calcular/cargar resultados de diagramas rotacionales.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import TFM_config as cfg
from astropy.io import fits
from TFM_rotational_diagram import diagrama_rotacional
from TFM_runtime import _resolve_name
from TFM_storage import load_diagrot_results, save_diagrot_results
import numpy as np


def seleccionar_fila_molecula(tab_mol_config, molecula):
    """
    Selecciona la fila de una molécula en moleculas_config.ecsv.
    """

    mask = tab_mol_config["nombre"] == molecula
    fila = tab_mol_config[mask]

    if len(fila) == 0:
        raise KeyError(
            f"La molécula {molecula} no está en moleculas_config.ecsv."
        )

    if len(fila) > 1:
        raise ValueError(
            f"La molécula {molecula} aparece más de una vez en "
            "moleculas_config.ecsv."
        )

    return fila[0]


def obtener_freq_noconsid_diagrot(region_name, molecula):
    """
    Devuelve las frecuencias excluidas del diagrama rotacional
    para una molécula y una región concretas.
    """

    if region_name not in cfg.REGION_LINE_CONFIG:
        raise KeyError(
            f"No existe configuración de líneas para {region_name}. "
            f"Regiones disponibles: "
            f"{list(cfg.REGION_LINE_CONFIG.keys())}"
        )

    config_region = cfg.REGION_LINE_CONFIG[region_name]

    if "diagrot_no_considerar" not in config_region:
        raise KeyError(
            f"La región {region_name} no contiene el bloque "
            "'diagrot_no_considerar'."
        )

    dict_noconsid = config_region["diagrot_no_considerar"]

    if molecula not in dict_noconsid:
        raise KeyError(
            f"No hay configuración de líneas excluidas del diagrama "
            f"rotacional para {molecula} en {region_name}."
        )

    return dict_noconsid[molecula]

def path_diagrot_region(region_name, base_dir=None):
    """
    Ruta de la tabla resumen de diagramas rotacionales para una región.

    Ejemplo:
        tables/diagrot/MF2/diagrot_resultados.ecsv
    """

    if base_dir is None:
        base_dir = cfg.rutatablas

    path_dir = Path(base_dir) / "diagrot" / region_name
    path_dir.mkdir(parents=True, exist_ok=True)

    return path_dir / "diagrot_resultados.ecsv"


def cargar_resultados_diagrot_region(region_name, base_dir=None):
    """
    Carga resultados de diagrama rotacional de una región si existen.
    Si no existen, devuelve diccionario vacío.
    """

    path = path_diagrot_region(region_name, base_dir=base_dir)

    if path.exists():
        return load_diagrot_results(path)

    return {}


def guardar_resultados_diagrot_region(resultados, region_name, base_dir=None):
    """
    Guarda la tabla resumen de diagrama rotacional de una región.
    """

    path = path_diagrot_region(region_name, base_dir=base_dir)
    save_diagrot_results(resultados, path)

    print(f"[diagrot] Tabla actualizada: {path}")


def path_fig_diagrot(region_name, molecula):
    """
    Ruta del PDF del diagrama rotacional del espectro medio.
    """

    path_dir = cfg.rutafig_diagrot / region_name
    path_dir.mkdir(parents=True, exist_ok=True)

    return path_dir / f"{molecula}_diagrot.pdf"


def _cat_mol_from_id_cat_name(id_cat_name):
    """
    Decide si la función de partición debe usar CDMS o JPL
    a partir del nombre simbólico del id_cat.
    """

    id_cat_name = str(id_cat_name)

    if id_cat_name.startswith("id_JPL"):
        return "JPL"

    if id_cat_name.startswith("id_cdms"):
        return "CDMS"

    raise ValueError(
        f"No sé inferir cat_mol a partir de id_cat_name={id_cat_name}"
    )


def calcular_diagrot_molecula(
        molecula,
        region_name,
        tab_mol_config,
        tab_filtrada,
        guardar_pdf=True,
        plot_Q=True, plots = False):
    """
    Calcula el diagrama rotacional de una molécula usando su tabla filtrada.
    Si guardar_pdf=True, guarda el PDF del diagrama.
    """

    fila = seleccionar_fila_molecula(tab_mol_config, molecula)

    id_cat_name = str(fila["id_cat"])

    id_cat = _resolve_name(fila["id_cat"])
    B0 = _resolve_name(fila["B0"])
    cat_mol = _cat_mol_from_id_cat_name(id_cat_name)
    freq_noconsid = obtener_freq_noconsid_diagrot(region_name=region_name,
                                                  molecula=molecula)

    save_path = None

    if guardar_pdf:
        save_path = path_fig_diagrot(region_name, molecula)

    (
        T_ex,
        deltaTex,
        tab_usada,
        N_col,
        deltaN_col,
        pol,
        QTex,
    ) = diagrama_rotacional(
        molecula,
        id_cat,
        tab_filtrada,
        B0,
        cat_mol,
        freq_noconsid=freq_noconsid,
        plot_Q=plot_Q,
        save_path=save_path, plots = plots
    )

    resultado = {
        "T_ex": T_ex,
        "Delta_Tex": deltaTex,
        "N_col": N_col,
        "Delta_Ncol": deltaN_col,
        "pendiente": pol[0],
        "ordenada": pol[1],
        "QTex": QTex,
    }

    return resultado, tab_usada

def cargar_o_calcular_diagrot_molecula_medio(
        molecula,
        region_name,
        tab_mol_config,
        tab_filtrada,
        recalcular=False,
        guardar_pdf=True,
        plot_Q=True,
        base_dir=None, plots = False):
    """
    Carga o calcula el diagrama rotacional del espectro promedio.

    Si ya existe el resultado en la tabla resumen y recalcular=False,
    carga el resultado. Si no existe o recalcular=True, calcula el diagrama,
    actualiza la tabla resumen y guarda el PDF si guardar_pdf=True.
    """

    resultados = cargar_resultados_diagrot_region(
        region_name,
        base_dir=base_dir,
    )

    if molecula in resultados and not recalcular:
        print(f"[diagrot] Cargando resultado existente para {molecula}")
        return resultados[molecula]

    print(f"[diagrot] Calculando diagrama rotacional para {molecula}")

    resultado, tab_usada = calcular_diagrot_molecula(
        molecula=molecula,
        region_name=region_name,
        tab_mol_config=tab_mol_config,
        tab_filtrada=tab_filtrada,
        guardar_pdf=guardar_pdf,
        plot_Q=plot_Q, plots = plots
    )
    
    resultados[molecula] = resultado

    guardar_resultados_diagrot_region(
        resultados,
        region_name,
        base_dir=base_dir,
    )

    return resultado


def path_maps_diagrot(region_name, molecula):
    """
    Carpeta donde se guardan los mapas FITS de T_ex y N_col.
    """

    path_dir = cfg.rutamaps_diagrot / region_name / molecula
    path_dir.mkdir(parents=True, exist_ok=True)

    return path_dir


def save_diagrot_maps_fits(
        region_name,
        molecula,
        T_ex_map,
        N_col_map,
        Delta_Tex_map,
        Delta_Ncol_map,
        header_2d=None):
    """
    Guarda los mapas de T_ex y N_col en FITS.
    """

    path_dir = path_maps_diagrot(region_name, molecula)

    path_T = path_dir / f"{molecula}_Tex.fits"
    path_N = path_dir / f"{molecula}_Ncol.fits"
    path_deltaT = path_dir / f"{molecula}_Delta_Tex.fits"
    path_deltaN = path_dir / f"{molecula}_Delta_Ncol.fits"

    if header_2d is None:
        header_2d = fits.Header()

    hdu_T = fits.PrimaryHDU(data=T_ex_map, header=header_2d)
    hdu_T.header["BUNIT"] = "K"
    hdu_T.header["MOLEC"] = molecula
    hdu_T.header["TYPE"] = "T_EX"

    hdu_N = fits.PrimaryHDU(data=N_col_map, header=header_2d)
    hdu_N.header["BUNIT"] = "cm-2"
    hdu_N.header["MOLEC"] = molecula
    hdu_N.header["TYPE"] = "N_COL"

    hdu_deltaT = fits.PrimaryHDU(data=Delta_Tex_map, header=header_2d,)
    hdu_deltaT.header["BUNIT"] = "K"
    hdu_deltaT.header["MOLEC"] = molecula
    hdu_deltaT.header["TYPE"] = "DELTA_T_EX"

    hdu_deltaN = fits.PrimaryHDU(data=Delta_Ncol_map, header=header_2d)
    hdu_deltaN.header["BUNIT"] = "cm-2"
    hdu_deltaN.header["MOLEC"] = molecula
    hdu_deltaN.header["TYPE"] = "DELTA_N_COL"

    hdu_T.writeto(path_T, overwrite=True)
    hdu_N.writeto(path_N, overwrite=True)
    hdu_deltaT.writeto(path_deltaT, overwrite=True)
    hdu_deltaN.writeto(path_deltaN, overwrite=True)

    print(f"[diagrot_pix] Guardado: {path_T}")
    print(f"[diagrot_pix] Guardado: {path_N}")
    print(f"[diagrot_pix] Guardado: {path_deltaT}")
    print(f"[diagrot_pix] Guardado: {path_deltaN}")
    
def corregir_mapas_T_N(
        T_ex_map,
        N_col_map,
        Delta_Tex_map,
        Delta_Ncol_map,
        radio_deteccion=1,
        radio_busqueda=3,
        n_sigma=5.0,
        umbral_relativo=1.0,
        min_vecinos=3,
        limitar_incertidumbres=True,
        T_min=3.0,
        T_max=800.0,
        N_max=1e19,
        max_iter_correccion=5):
    """
    Detecta y corrige soluciones anómalas del diagrama rotacional.

    Se marca un píxel cuando:
    - T_ex no es finita o está fuera de [T_min, T_max].
    - N_col no es finita o está fuera de (0, N_max].
    - T_ex es un outlier local.
    - log10(N_col) es un outlier local.

    Los píxeles marcados se sustituyen iterativamente mediante
    la mediana de vecinos válidos. Los que no puedan repararse
    se dejan temporalmente como NaN.
    
    Los cuatro parámetros del píxel se reemplazan conjuntamente:
    T_ex, N_col, Delta_Tex y Delta_Ncol.

    Returns
    -------
    T_corr, N_corr, dT_corr, dN_corr : ndarray
        Mapas corregidos.

    mask_corregidos : ndarray de bool
        True en los píxeles sustituidos.
    """

    mapas = [
        np.asarray(T_ex_map, dtype=float),
        np.asarray(N_col_map, dtype=float),
        np.asarray(Delta_Tex_map, dtype=float),
        np.asarray(Delta_Ncol_map, dtype=float),
    ]

    shape = mapas[0].shape

    if any(mapa.ndim != 2 or mapa.shape != shape for mapa in mapas):
        raise ValueError(
            "Los cuatro mapas deben ser 2D y tener la misma forma."
        )

    if radio_deteccion < 1 or radio_busqueda < 1:
        raise ValueError(
            "Los radios deben ser mayores o iguales que 1."
        )

    if min_vecinos < 1:
        raise ValueError(
            "min_vecinos debe ser mayor o igual que 1."
        )
    
    if T_min >= T_max:
        raise ValueError(
            "T_min debe ser menor que T_max."
        )

    if N_max <= 0:
        raise ValueError(
            "N_max debe ser positivo."
        )

    if max_iter_correccion < 1:
        raise ValueError(
            "max_iter_correccion debe ser mayor "
            "o igual que 1."
        )

    (
        T_original,
        N_original,
        dT_original,
        dN_original,
    ) = mapas

    ny, nx = shape
    
    # Todos los píxeles del array deben analizarse,
    # incluidos aquellos donde T y N sean NaN.
    footprint = np.ones(
        shape,
        dtype=bool,
    )

    # --------------------------------------------------------
    # Máscaras de valores físicamente inválidos
    # --------------------------------------------------------

    mask_T_invalida = footprint & (
        ~np.isfinite(T_original)
        | (T_original < T_min)
        | (T_original > T_max)
    )

    mask_N_invalida = footprint & (
        ~np.isfinite(N_original)
        | (N_original <= 0)
        | (N_original > N_max)
    )

    mask_nan_TN = (
        ~np.isfinite(T_original)
        | ~np.isfinite(N_original)
    )

    mask_nan_completo = (
        ~np.isfinite(T_original)
        & ~np.isfinite(N_original)
    )

    mask_outliers_T = np.zeros(
        shape,
        dtype=bool,
    )

    mask_outliers_logN = np.zeros(
        shape,
        dtype=bool,
    )

    # --------------------------------------------------------
    # Detectar outliers locales de T y log10(N)
    # --------------------------------------------------------

    pixeles_validos = np.argwhere(
        footprint
        & np.isfinite(T_original)
        & (T_original >= T_min)
        & (T_original <= T_max)
        & np.isfinite(N_original)
        & (N_original > 0)
        & (N_original <= N_max)
    )

    logN_original = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    mask_N_positiva = (
        np.isfinite(N_original)
        & (N_original > 0)
        & (N_original <= N_max)
    )

    logN_original[mask_N_positiva] = np.log10(
        N_original[mask_N_positiva]
    )

    for y, x in pixeles_validos:

        y0 = max(0, y - radio_deteccion)
        y1 = min(ny, y + radio_deteccion + 1)

        x0 = max(0, x - radio_deteccion)
        x1 = min(nx, x + radio_deteccion + 1)

        # ----------------------------------------------------
        # Outlier de temperatura
        # ----------------------------------------------------

        entorno_T = T_original[y0:y1, x0:x1]

        validos_T = (
            np.isfinite(entorno_T)
            & (entorno_T >= T_min)
            & (entorno_T <= T_max)
        )

        validos_T[y - y0, x - x0] = False

        vecinos_T = entorno_T[validos_T]

        if vecinos_T.size >= min_vecinos:

            mediana_T = np.median(vecinos_T)

            mad_T = np.median(
                np.abs(vecinos_T - mediana_T)
            )

            sigma_T = 1.4826 * mad_T

            limite_T = max(
                n_sigma * sigma_T,
                umbral_relativo * abs(mediana_T),
            )

            if (
                abs(T_original[y, x] - mediana_T)
                > limite_T
            ):
                mask_outliers_T[y, x] = True

        # ----------------------------------------------------
        # Outlier de densidad de columna en log10(N)
        # ----------------------------------------------------

        entorno_logN = logN_original[
            y0:y1,
            x0:x1,
        ]

        validos_logN = np.isfinite(
            entorno_logN
        )

        validos_logN[y - y0, x - x0] = False

        vecinos_logN = entorno_logN[
            validos_logN
        ]

        if vecinos_logN.size >= min_vecinos:

            mediana_logN = np.median(
                vecinos_logN
            )

            mad_logN = np.median(
                np.abs(
                    vecinos_logN
                    - mediana_logN
                )
            )

            sigma_logN = 1.4826 * mad_logN

            # umbral_relativo no tiene una interpretación útil
            # en log10(N). Añadimos un corte mínimo en dex.
            limite_logN = max(
                n_sigma * sigma_logN,
                0.5,
            )

            if (
                abs(
                    logN_original[y, x]
                    - mediana_logN
                )
                > limite_logN
            ):
                mask_outliers_logN[y, x] = True

    # --------------------------------------------------------
    # Máscara final
    # --------------------------------------------------------

    mask_corregidos = (
        mask_T_invalida
        | mask_N_invalida
        | mask_outliers_T
        | mask_outliers_logN
    )

    # --------------------------------------------------------
    # Copias que iremos corrigiendo iterativamente
    # --------------------------------------------------------

    T_corr = T_original.copy()
    N_corr = N_original.copy()
    dT_corr = dT_original.copy()
    dN_corr = dN_original.copy()

    # Píxeles que todavía necesitan ser reparados.
    mask_pendientes = mask_corregidos.copy()

    # Píxeles que han sido sustituidos correctamente.
    mask_reparados = np.zeros(
        shape,
        dtype=bool,
    )

    # --------------------------------------------------------
    # Corrección iterativa mediante medianas locales
    # --------------------------------------------------------

    for iteracion in range(
            max_iter_correccion):

        n_pendientes_inicio = np.count_nonzero(
            mask_pendientes
        )

        if n_pendientes_inicio == 0:
            break

        # Los donantes deben cumplir todos los límites.
        mask_donantes = (
            footprint
            & ~mask_pendientes
            & np.isfinite(T_corr)
            & (T_corr >= T_min)
            & (T_corr <= T_max)
            & np.isfinite(N_corr)
            & (N_corr > 0)
            & (N_corr <= N_max)
        )

        # Copias fijas para que todos los píxeles de esta
        # iteración utilicen la misma referencia.
        T_referencia = T_corr.copy()
        N_referencia = N_corr.copy()
        dT_referencia = dT_corr.copy()
        dN_referencia = dN_corr.copy()

        mask_donantes_referencia = (
            mask_donantes.copy()
        )

        correcciones_iteracion = []

        for y, x in np.argwhere(
                mask_pendientes):

            reemplazo_encontrado = False

            for radio in range(
                    1,
                    radio_busqueda + 1):

                y0 = max(0, y - radio)
                y1 = min(
                    ny,
                    y + radio + 1,
                )

                x0 = max(0, x - radio)
                x1 = min(
                    nx,
                    x + radio + 1,
                )

                buenos_locales = (
                    mask_donantes_referencia[
                        y0:y1,
                        x0:x1,
                    ]
                )

                if (
                    np.count_nonzero(
                        buenos_locales
                    )
                    < min_vecinos
                ):
                    continue

                entorno_T = T_referencia[
                    y0:y1,
                    x0:x1,
                ]

                entorno_N = N_referencia[
                    y0:y1,
                    x0:x1,
                ]

                entorno_dT = dT_referencia[
                    y0:y1,
                    x0:x1,
                ]

                entorno_dN = dN_referencia[
                    y0:y1,
                    x0:x1,
                ]

                mediana_T = np.median(
                    entorno_T[buenos_locales]
                )

                mediana_N = np.median(
                    entorno_N[buenos_locales]
                )

                validos_dT = (
                    buenos_locales
                    & np.isfinite(entorno_dT)
                    & (entorno_dT >= 0)
                )

                validos_dN = (
                    buenos_locales
                    & np.isfinite(entorno_dN)
                    & (entorno_dN >= 0)
                )

                if np.any(validos_dT):
                    mediana_dT = np.median(
                        entorno_dT[validos_dT]
                    )
                else:
                    mediana_dT = np.nan

                if np.any(validos_dN):
                    mediana_dN = np.median(
                        entorno_dN[validos_dN]
                    )
                else:
                    mediana_dN = np.nan

                # Comprobación adicional de seguridad.
                reemplazo_valido = (
                    np.isfinite(mediana_T)
                    and (
                        T_min
                        <= mediana_T
                        <= T_max
                    )
                    and np.isfinite(mediana_N)
                    and (
                        0
                        < mediana_N
                        <= N_max
                    )
                )

                if reemplazo_valido:

                    correcciones_iteracion.append(
                        (
                            y,
                            x,
                            mediana_T,
                            mediana_N,
                            mediana_dT,
                            mediana_dN,
                        )
                    )

                    reemplazo_encontrado = True
                    break

            if not reemplazo_encontrado:
                continue

        # Aplicar simultáneamente las correcciones.
        mask_reparados_iteracion = np.zeros(
            shape,
            dtype=bool,
        )

        for (
            y,
            x,
            mediana_T,
            mediana_N,
            mediana_dT,
            mediana_dN,
        ) in correcciones_iteracion:

            T_corr[y, x] = mediana_T
            N_corr[y, x] = mediana_N
            dT_corr[y, x] = mediana_dT
            dN_corr[y, x] = mediana_dN

            mask_reparados_iteracion[
                y,
                x,
            ] = True

        mask_reparados |= (
            mask_reparados_iteracion
        )

        # Volver a comprobar los límites después de corregir.
        mask_fuera_limites = footprint & (
            ~np.isfinite(T_corr)
            | (T_corr < T_min)
            | (T_corr > T_max)
            | ~np.isfinite(N_corr)
            | (N_corr <= 0)
            | (N_corr > N_max)
        )

        # Los outliers iniciales dejan de estar pendientes
        # cuando ya han sido sustituidos.
        mask_pendientes = (
            (
                mask_pendientes
                & ~mask_reparados_iteracion
            )
            | mask_fuera_limites
        )

        n_reparados_iteracion = (
            np.count_nonzero(
                mask_reparados_iteracion
            )
        )

        print(
            f"[mapas_T_N] Iteración "
            f"{iteracion + 1}: "
            f"{n_pendientes_inicio} pendientes, "
            f"{n_reparados_iteracion} reparados, "
            f"{np.count_nonzero(mask_pendientes)} "
            "todavía pendientes."
        )

        # Si no se ha reparado ninguno, las siguientes
        # iteraciones tampoco progresarán.
        if n_reparados_iteracion == 0:
            break

    # --------------------------------------------------------
    # Píxeles que no se pudieron reparar localmente
    # --------------------------------------------------------

    mask_sin_reparar = mask_pendientes.copy()

    # De momento se dejan como NaN.
    # Después añadiremos aquí el respaldo del espectro promedio.
    T_corr[mask_sin_reparar] = np.nan
    N_corr[mask_sin_reparar] = np.nan
    dT_corr[mask_sin_reparar] = np.nan
    dN_corr[mask_sin_reparar] = np.nan
    
    # --------------------------------------------------------
    # Limitar incertidumbres mayores que el propio resultado
    # --------------------------------------------------------

    n_deltaT_limitadas = 0
    n_deltaN_limitadas = 0

    if limitar_incertidumbres:

        mask_deltaT_excesiva = (
            np.isfinite(T_corr)
            & (T_corr > 0)
            & np.isfinite(dT_corr)
            & (dT_corr > T_corr)
        )

        mask_deltaN_excesiva = (
            np.isfinite(N_corr)
            & (N_corr > 0)
            & np.isfinite(dN_corr)
            & (dN_corr > N_corr)
        )

        # La incertidumbre máxima permitida es el 100 %.
        dT_corr[mask_deltaT_excesiva] = (
            T_corr[mask_deltaT_excesiva]
        )

        dN_corr[mask_deltaN_excesiva] = (
            N_corr[mask_deltaN_excesiva]
        )

        n_deltaT_limitadas = np.count_nonzero(
            mask_deltaT_excesiva
        )

        n_deltaN_limitadas = np.count_nonzero(
            mask_deltaN_excesiva
        )

    print(
        "[mapas_T_N] Corrección robusta:\n"
        f"    T fuera de [{T_min}, {T_max}] K: "
        f"{np.count_nonzero(mask_T_invalida)}\n"
        f"    N fuera de (0, {N_max:.2e}] cm-2: "
        f"{np.count_nonzero(mask_N_invalida)}\n"
        f"    Outliers locales de T: "
        f"{np.count_nonzero(mask_outliers_T)}\n"
        f"    Outliers locales de log10(N): "
        f"{np.count_nonzero(mask_outliers_logN)}\n"
        f"    Píxeles con algún NaN en T o N: "
        f"{np.count_nonzero(mask_nan_TN)}\n"
        f"    Píxeles con T y N simultáneamente NaN: "
        f"{np.count_nonzero(mask_nan_completo)}\n"
        f"    Píxeles reparados con mediana: "
        f"{np.count_nonzero(mask_reparados)}\n"
        f"    Píxeles todavía sin reparar: "
        f"{np.count_nonzero(mask_sin_reparar)}\n"
        f"    Delta_T limitadas a T: "
        f"{n_deltaT_limitadas}\n"
        f"    Delta_N limitadas a N: "
        f"{n_deltaN_limitadas}\n"
        f"    Límite de incertidumbres activado: "
        f"{limitar_incertidumbres}"
    )

    return (
        T_corr,
        N_corr,
        dT_corr,
        dN_corr,
        mask_corregidos,
    )

def load_diagrot_maps_fits(region_name, molecula):
    """
    Carga mapas FITS de T_ex y N_col.
    """

    path_dir = path_maps_diagrot(region_name, molecula)

    path_T = path_dir / f"{molecula}_Tex.fits"
    path_N = path_dir / f"{molecula}_Ncol.fits"
    path_deltaT = path_dir / f"{molecula}_Delta_Tex.fits"
    path_deltaN = path_dir / f"{molecula}_Delta_Ncol.fits"

    paths = (path_T, path_N, path_deltaT, path_deltaN)

    if not all(path.exists() for path in paths):
        return None

    with fits.open(path_T) as hdul:
        T_ex_map = hdul[0].data.copy()
        header_2d = hdul[0].header.copy()

    with fits.open(path_N) as hdul:
        N_col_map = hdul[0].data.copy()

    with fits.open(path_deltaT) as hdul:
        Delta_Tex_map = hdul[0].data.copy()

    with fits.open(path_deltaN) as hdul:
        Delta_Ncol_map = hdul[0].data.copy()

    return {"T_ex_map": T_ex_map,"N_col_map": N_col_map,
            "Delta_Tex_map": Delta_Tex_map, "Delta_Ncol_map": Delta_Ncol_map,
            "header": header_2d}


def cargar_o_calcular_diagrot_pixeles(
        molecula,
        region_name,
        tab_mol_config,
        tab_pixeles,
        dict_cubos_comp,
        recalcular=False,
        guardar_fits=True,
        guardar_plots=True, plots = False):
    """
    Carga o calcula mapas de T_ex y N_col píxel a píxel.
    """

    from TFM_rotational_diagram import diagrama_rot_pixeles

    if not recalcular:
        mapas_guardados = load_diagrot_maps_fits(region_name, molecula)

        if mapas_guardados is not None:
            print(
                f"[diagrot_pix] Cargando mapas existentes para "
                f"{molecula}"
            )
            return mapas_guardados

    print(f"[diagrot_pix] Calculando mapas de T_ex y N_col para {molecula}")

    fila = seleccionar_fila_molecula(tab_mol_config, molecula)

    # ------------------------------------------------------------
    # Configuración desde tabla maestra nueva
    # ------------------------------------------------------------
    id_cat_name = str(fila["id_cat"])

    id_cat = _resolve_name(fila["id_cat"])
    B0 = _resolve_name(fila["B0"])
    freq_noconsid = obtener_freq_noconsid_diagrot(region_name=region_name,
                                                  molecula=molecula)

    if id_cat_name.startswith("id_JPL"):
        cat_mol = "JPL"
    elif id_cat_name.startswith("id_cdms"):
        cat_mol = "CDMS"
    else:
        raise ValueError(f"No sé inferir cat_mol desde {id_cat_name}")

    # ------------------------------------------------------------
    # WCS de referencia
    # ------------------------------------------------------------
    first_key = list(dict_cubos_comp.keys())[0]
    cubo_ref = dict_cubos_comp[first_key]["Temp_brillo"]
    header_2d = cubo_ref.wcs.celestial.to_header()
    wcs_ref = cubo_ref.wcs.celestial

    # ------------------------------------------------------------
    # Cálculo de mapas
    # ------------------------------------------------------------
    (T_ex_map, N_col_map, Delta_Tex_map,
     Delta_Ncol_map) = diagrama_rot_pixeles(molecula, id_cat, tab_pixeles,
                                            B0, cat_mol,
                                            freq_noconsid=freq_noconsid,
                                            wcs_ref=wcs_ref, plots = plots)

    # ------------------------------------------------------------
    # Guardar FITS
    # ------------------------------------------------------------
    if guardar_fits:
        save_diagrot_maps_fits(
        region_name=region_name,
        molecula=molecula,
        T_ex_map=T_ex_map,
        N_col_map=N_col_map,
        Delta_Tex_map=Delta_Tex_map,
        Delta_Ncol_map=Delta_Ncol_map,
        header_2d=header_2d,
        )

    # ------------------------------------------------------------
    # Guardar plots PDF
    # ------------------------------------------------------------
    if guardar_plots:
        save_diagrot_map_plot(
            region_name=region_name,
            molecula=molecula,
            data=T_ex_map,
            header=header_2d,
            map_type="Tex",
            unit_label=r"$T_\mathrm{ex}$ (K)"
        )

        save_diagrot_map_plot(
            region_name=region_name,
            molecula=molecula,
            data=N_col_map,
            header=header_2d,
            map_type="Ncol",
            unit_label=r"$N_\mathrm{col}$ (cm$^{-2}$)"
        )

    return {"T_ex_map": T_ex_map, "N_col_map": N_col_map,
            "Delta_Tex_map": Delta_Tex_map, "Delta_Ncol_map": Delta_Ncol_map,
            "header": header_2d}

def path_fig_maps_diagrot(region_name, molecula):
    """
    Carpeta donde se guardan las figuras de los mapas de diagrama rotacional.
    """

    path_dir = cfg.rutafig_maps_diagrot / region_name / molecula
    path_dir.mkdir(parents=True, exist_ok=True)

    return path_dir


def save_diagrot_map_plot(region_name, molecula, data, header,
                          map_type="Tex", unit_label="K",
                          cmap="viridis"):
    """
    Guarda un plot 2D de un mapa de diagrama rotacional solo como PDF.
    """

    from astropy.wcs import WCS
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    path_dir = path_fig_maps_diagrot(region_name, molecula)

    path_pdf = path_dir / f"{molecula}_{map_type}.pdf"

    wcs = WCS(header)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection=wcs)

    norm = simple_norm(data, "linear", percent=99)

    im = ax.imshow(data, origin="lower", norm=norm, cmap=cmap)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(unit_label)

    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_title(f"{molecula} - {map_type}")

    plt.savefig(path_pdf, dpi=300, bbox_inches="tight")

    print(f"[diagrot_pix] Figura guardada: {path_pdf}")

    plt.show()
    plt.close()