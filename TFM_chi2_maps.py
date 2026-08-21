#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:56:00 2026

@author: jorge
"""

import pickle
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits

import TFM_config as cfg
from TFM_line_search import mod_vel
from TFM_runtime import _resolve_name, resolve_intervalo_region
from TFM_synthetic_pipeline import seleccionar_fila_molecula, cat_mol_from_id_cat_name
from TFM_chi2_fit import minimchi2

def ventanas_de_intervalos(intervalos):
    """
    Devuelve las etiquetas tipo B6-SPW0, B6-SPW1, etc. de un intervalo molecular.
    """

    return [nombre_vent for _, _, _, nombre_vent in intervalos]


def crear_dict_espec_pixel(dict_cubos_comp, x, y, intervalos):
    """
    Construye un diccionario de espectros 1D para un píxel concreto.

    Formato compatible con minimchi2() y spec_sint_opacidad():
        dict_espec_pixel["B6-SPW7"]["frecuencia"]
        dict_espec_pixel["B6-SPW7"]["Temp_brillo"]
    """

    dict_espec_pixel = {}

    ventanas_validas = ventanas_de_intervalos(intervalos)

    for ventana in ventanas_validas:

        frec = dict_cubos_comp[ventana]["frecuencia"]
        espec = dict_cubos_comp[ventana]["Temp_brillo"][:, y, x]

        dict_espec_pixel[ventana] = {
            "frecuencia": frec,
            "Temp_brillo": espec,
        }

    return dict_espec_pixel


def crear_dict_mapa_pixel(dict_mapas, x, y, intervalos, unit):
    """
    Extrae valores por píxel de un diccionario de mapas.

    Sirve tanto para T_cont_pix como para sigma_pix.
    """

    dict_pixel = {}

    ventanas_validas = ventanas_de_intervalos(intervalos)

    for ventana in ventanas_validas:

        valor = dict_mapas[ventana][y, x]

        if np.isfinite(valor):
            dict_pixel[ventana] = valor * unit
        else:
            dict_pixel[ventana] = np.nan * unit

    return dict_pixel

def construir_tab_lineas_chi2_pixel(
        molecula,
        tab_pixel,
        v_pik):

    if tab_pixel is None or len(tab_pixel) == 0:
        return None

    if "orderedfreq" not in tab_pixel.colnames:
        raise KeyError(
            "La tabla del píxel no contiene 'orderedfreq'"
        )

    freq_lab = tab_pixel["orderedfreq"].to(u.MHz)

    # minimchi2() aplicará después +abs(v_pik).
    # Aquí aplicamos previamente el desplazamiento contrario.
    freq_entrada = mod_vel(
        -np.abs(v_pik),
        freq_lab,
    )

    return {
        molecula: {
            "freq": list(freq_entrada),
            "label": [molecula] * len(tab_pixel),
        }
    }

def construir_config_chi2_pixel(
        molecula,
        tab_mol_config,
        T_init,
        N_init,
        fwhm_dict,
        v_pik_dict,
        dict_T_cont_pixel,
        intervalos_mol_region,
        deltaT_factor=0.5,
        deltaN_factor=0.7,
        fwhm_pixel=None,
        v_pik_pixel=None,
        deltaT_pixel=None,
        deltaN_pixel=None):
    """
    Construye la config que espera minimchi2() para un único píxel.
    """

    fila = seleccionar_fila_molecula(tab_mol_config, molecula)

    if molecula not in fwhm_dict:
        raise KeyError(f"No hay FWHM calibrado para {molecula}")

    if molecula not in v_pik_dict:
        raise KeyError(f"No hay v_pik calibrada para {molecula}")

    if not np.isfinite(T_init) or T_init <= 0:
        return None

    if not np.isfinite(N_init) or N_init <= 0:
        return None

    T_init = T_init * u.K
    N_init = N_init / u.cm**2

    # Valores de respaldo
    deltaT = max(
        deltaT_factor * T_init,
        5 * u.K,
    )
    deltaN = deltaN_factor * N_init

    # Incertidumbre de temperatura del diagrama rotacional
    if deltaT_pixel is not None:
        deltaT_pixel = u.Quantity(deltaT_pixel).to(u.K)

        if (
            np.isfinite(deltaT_pixel.value)
            and deltaT_pixel.value > 0
        ):
            deltaT = deltaT_pixel

    # Incertidumbre de columna del diagrama rotacional
    if deltaN_pixel is not None:
        deltaN_pixel = u.Quantity(deltaN_pixel).to(1 / u.cm**2)

        if (
            np.isfinite(deltaN_pixel.value)
            and deltaN_pixel.value > 0
        ):
            deltaN = deltaN_pixel

    id_cat_name = str(fila["id_cat"])

    line_filter = getattr(cfg, "SPEC_SINT_LINE_FILTERS", {}).get(molecula, {})

    sijmu2_model = line_filter.get("sijmu2", 0 * u.D**2)
    aij_model = line_filter.get("aij", 0 / u.s)

    # Por defecto utilizamos la calibración global
    fwhm_use = fwhm_dict[molecula]

    # Si existe un valor válido para este píxel, lo sustituimos
    if fwhm_pixel is not None:
        fwhm_pixel = u.Quantity(fwhm_pixel).to(u.km / u.s)

        if (
            np.isfinite(fwhm_pixel.value)
            and fwhm_pixel.value > 0
        ):
            fwhm_use = fwhm_pixel

    # Por defecto utilizamos la velocidad global
    v_pik_use = v_pik_dict[molecula]

    # Si existe un valor válido para este píxel, lo sustituimos
    if v_pik_pixel is not None:
        v_pik_pixel = u.Quantity(v_pik_pixel).to(u.km / u.s)

        if np.isfinite(v_pik_pixel.value):
            v_pik_use = v_pik_pixel

    return {
        molecula: {
            "T_ex": T_init,
            "N_col": N_init,
            "deltaT": deltaT,
            "deltaN": deltaN,

            "mol": str(fila["mol"]),
            "intervalo": resolve_intervalo_region(
                fila["intervalo"],
                intervalos_mol_region=intervalos_mol_region,
            ),
            "id_splat": _resolve_name(fila["id_splat"]),
            "filtro_estructuras": _resolve_name(fila["filtro_estructuras"]),

            "FWHM": fwhm_use,
            "T_cont": dict_T_cont_pixel,

            "f0": fila["f0"],
            "v_pik": v_pik_use,

            "cat_mol": cat_mol_from_id_cat_name(id_cat_name),
            "id_cat": _resolve_name(id_cat_name),

            "sij": sijmu2_model,
            "aij": aij_model,
        }
    }

def ajustar_chi2_pixel(
        molecula,
        tab_mol_config,
        T_init,
        N_init,
        tab_pixel,
        dict_espec_pixel,
        dict_T_cont_pixel,
        dict_sigma_pixel,
        fwhm_dict,
        v_pik_dict,
        intervalos_mol_region,
        n_grid=10,
        max_iter=1,
        tol=0.5,
        dict_lin_noconsid=None,
        debug=False,
        show_best_model=False,
        fwhm_pixel=None,
        v_pik_pixel=None,
        deltaT_pixel=None,
        deltaN_pixel=None):
    """
    Ejecuta ajuste chi2 para un único píxel.
    Sin residuos y sin modelo previo.
    """

    config_pixel = construir_config_chi2_pixel(
        molecula=molecula,
        tab_mol_config=tab_mol_config,
        T_init=T_init,
        N_init=N_init,
        fwhm_dict=fwhm_dict,
        v_pik_dict=v_pik_dict,
        dict_T_cont_pixel=dict_T_cont_pixel,
        intervalos_mol_region=intervalos_mol_region,
        fwhm_pixel=fwhm_pixel,
        v_pik_pixel=v_pik_pixel,
        deltaT_pixel=deltaT_pixel,
        deltaN_pixel=deltaN_pixel,
    )

    if config_pixel is None:
        return None

    v_pik_use = config_pixel[molecula]["v_pik"]

    tab_lineas_pixel = construir_tab_lineas_chi2_pixel(
        molecula=molecula,
        tab_pixel=tab_pixel,
        v_pik=v_pik_use,
    )

    if tab_lineas_pixel is None:
        return None

    if len(tab_lineas_pixel[molecula]["freq"]) < 2:
        return None

    intervalos = config_pixel[molecula]["intervalo"]

    config_actual = config_pixel

    T0 = config_actual[molecula]["T_ex"]
    N0 = config_actual[molecula]["N_col"]

    resultado_final = None

    for i in range(max_iter):

        modelo_fit, dict_TN = minimchi2(
            config_actual,
            dict_sigma_pixel,
            intervalos,
            n_grid,
            tab_lineas_pixel,
            dict_resol_espec=None,
            dict_especchi=dict_espec_pixel,
            dict_lin_noconsid=dict_lin_noconsid,
            debug=debug,
            model_sint=None,
            dictTcont=dict_T_cont_pixel,
            residuos=False,
            show_plots=show_best_model,
            devolver_tabla_opacidad=True,
        )

        T1 = dict_TN[molecula]["T_fit"]
        N1 = dict_TN[molecula]["N_fit"]

        resultado_final = dict_TN[molecula]

        dif_T = np.abs(T1 - T0).to(u.K)

        config_actual[molecula]["T_ex"] = T1
        config_actual[molecula]["N_col"] = N1
        config_actual[molecula]["deltaT"] = dict_TN[molecula]["deltaT"]
        config_actual[molecula]["deltaN"] = dict_TN[molecula]["deltaN"]

        if dif_T <= tol * u.K:
            break

        T0 = T1
        N0 = N1

    return resultado_final

def path_chi2_maps(region_name, molecula):
    """
    Carpeta de salida de los mapas chi2.
    """

    path_dir = (
        cfg.rutamaps_chi2
        / region_name
        / molecula
    )

    path_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path_dir


def save_chi2_maps_fits(region_name, molecula, mapas, header_2d=None):
    """
    Guarda mapas FITS del ajuste chi2 por píxel.
    """

    path_dir = path_chi2_maps(region_name, molecula)

    if header_2d is None:
        header_2d = fits.Header()

    outputs = {
    "Tex_chi2": ("T_fit_map", "K"),
    "Ncol_chi2": ("N_fit_map", "cm-2"),
    "deltaTex_chi2": ("deltaT_map", "K"),
    "deltaNcol_chi2": ("deltaN_map", "cm-2"),
    "chi2_min": ("chi2_min_map", ""),
    "tau_linea_fuerte_chi2": (
        "tau_linea_fuerte_map",
        "1",
    ),
}

    for suffix, (key, bunit) in outputs.items():

        path = path_dir / f"{molecula}_{suffix}.fits"

        hdu = fits.PrimaryHDU(data=mapas[key], header=header_2d)
        hdu.header["MOLEC"] = molecula
        hdu.header["TYPE"] = suffix

        if bunit != "":
            hdu.header["BUNIT"] = bunit

        hdu.writeto(path, overwrite=True)

        print(f"[chi2_pix] Guardado: {path}")
        
def cargar_o_calcular_chi2_pixeles(
        molecula,
        region_name,
        tab_mol_config,
        mapas_diagrot,
        tab_pixeles,
        dict_cubos_comp,
        dict_T_cont_pix,
        dict_sigma_pix,
        fwhm_dict,
        v_pik_dict,
        fwhm_map=None,
        v_pik_map=None,
        intervalos_mol_region=None,
        header_2d=None,
        recalcular=False,
        n_grid=10,
        max_iter=1,
        tol=0.5,
        dict_lin_noconsid=None,
        debug=False,
        show_modelo=False):
    """
    Calcula mapas chi2 de T_ex y N_col píxel a píxel.
    """

    path_dir = path_chi2_maps(region_name, molecula)

    path_T = path_dir / f"{molecula}_Tex_chi2.fits"
    path_N = path_dir / f"{molecula}_Ncol_chi2.fits"
    path_deltaT = path_dir / f"{molecula}_deltaTex_chi2.fits"
    path_deltaN = path_dir / f"{molecula}_deltaNcol_chi2.fits"
    path_chi2 = path_dir / f"{molecula}_chi2_min.fits"
    path_tau = (path_dir / f"{molecula}_tau_linea_fuerte_chi2.fits")
    paths_mapas = {
        "T_fit_map": path_T,
        "N_fit_map": path_N,
        "deltaT_map": path_deltaT,
        "deltaN_map": path_deltaN,
        "chi2_min_map": path_chi2,
        "tau_linea_fuerte_map": path_tau,
    }

    if not recalcular:

        paths_faltantes = [
            path
            for path in paths_mapas.values()
            if not path.exists()
        ]

        if not paths_faltantes:

            print(
                f"[chi2_pix] Cargando mapas existentes para {molecula}"
            )

            with fits.open(path_T) as hdul:
                T_fit_map = hdul[0].data.copy()
                header = hdul[0].header.copy()

            with fits.open(path_N) as hdul:
                N_fit_map = hdul[0].data.copy()

            with fits.open(path_deltaT) as hdul:
                deltaT_map = hdul[0].data.copy()

            with fits.open(path_deltaN) as hdul:
                deltaN_map = hdul[0].data.copy()

            with fits.open(path_chi2) as hdul:
                chi2_min_map = hdul[0].data.copy()
                
            with fits.open(path_tau) as hdul:
                tau_linea_fuerte_map = (
                    hdul[0].data.copy()
                )

            return {
                "T_fit_map": T_fit_map,
                "N_fit_map": N_fit_map,
                "deltaT_map": deltaT_map,
                "deltaN_map": deltaN_map,
                "chi2_min_map": chi2_min_map,
                "tau_linea_fuerte_map": tau_linea_fuerte_map,
                "header": header,
            }

        print(
            "[chi2_pix] Faltan mapas χ² guardados. "
            "Se recalcularán:\n    "
            + "\n    ".join(
                str(path)
                for path in paths_faltantes
            )
        )

    T_init_map = mapas_diagrot["T_ex_map"]
    N_init_map = mapas_diagrot["N_col_map"]
    Delta_Tex_map = mapas_diagrot.get("Delta_Tex_map")
    Delta_Ncol_map = mapas_diagrot.get("Delta_Ncol_map")

    ny, nx = T_init_map.shape

    T_fit_map = np.full((ny, nx), np.nan)
    N_fit_map = np.full((ny, nx), np.nan)
    deltaT_map = np.full((ny, nx), np.nan)
    deltaN_map = np.full((ny, nx), np.nan)
    chi2_min_map = np.full((ny, nx), np.nan)
    tau_linea_fuerte_map = np.full((ny, nx), np.nan)

    # Intervalo real de la molécula en la región activa
    fila = seleccionar_fila_molecula(tab_mol_config, molecula)
    intervalos = resolve_intervalo_region(
        fila["intervalo"],
        intervalos_mol_region=intervalos_mol_region,
    )

    print(f"[chi2_pix] Calculando mapas chi2 para {molecula}")
    print(f"[chi2_pix] Tamaño mapa: ny={ny}, nx={nx}")

    for y in range(ny):
        print(f"[chi2_pix] Fila {y + 1}/{ny}")

        for x in range(nx):

            T_init = T_init_map[y, x]
            N_init = N_init_map[y, x]
            deltaT_pixel = None
            deltaN_pixel = None

            if not np.isfinite(T_init) or not np.isfinite(N_init):
                continue

            if Delta_Tex_map is not None:
                deltaT_pixel = Delta_Tex_map[y, x] * u.K

            if Delta_Ncol_map is not None:
                deltaN_pixel = (
                    Delta_Ncol_map[y, x] / u.cm**2
                )

            tab_pixel = tab_pixeles[y, x]

            if tab_pixel is None or len(tab_pixel) < 2:
                continue

            try:
                dict_espec_pixel = crear_dict_espec_pixel(
                    dict_cubos_comp=dict_cubos_comp,
                    x=x,
                    y=y,
                    intervalos=intervalos,
                )

                dict_T_cont_pixel = crear_dict_mapa_pixel(
                    dict_mapas=dict_T_cont_pix,
                    x=x,
                    y=y,
                    intervalos=intervalos,
                    unit=u.K,
                )

                dict_sigma_pixel = crear_dict_mapa_pixel(
                    dict_mapas=dict_sigma_pix,
                    x=x,
                    y=y,
                    intervalos=intervalos,
                    unit=u.K,
                )

                fwhm_pixel = None
                v_pik_pixel = None

                if fwhm_map is not None:
                    fwhm_pixel = (
                        fwhm_map[y, x] * u.km / u.s
                    )

                if v_pik_map is not None:
                    v_pik_pixel = (
                        v_pik_map[y, x] * u.km / u.s
                    )

                resultado = ajustar_chi2_pixel(
                    molecula=molecula,
                    tab_mol_config=tab_mol_config,
                    T_init=T_init,
                    N_init=N_init,
                    tab_pixel=tab_pixel,
                    dict_espec_pixel=dict_espec_pixel,
                    dict_T_cont_pixel=dict_T_cont_pixel,
                    dict_sigma_pixel=dict_sigma_pixel,
                    fwhm_dict=fwhm_dict,
                    v_pik_dict=v_pik_dict,
                    intervalos_mol_region=intervalos_mol_region,
                    n_grid=n_grid,
                    max_iter=max_iter,
                    tol=tol,
                    dict_lin_noconsid=dict_lin_noconsid,
                    debug=debug,
                    show_best_model=show_modelo,
                    fwhm_pixel=fwhm_pixel,
                    v_pik_pixel=v_pik_pixel,
                    deltaT_pixel=deltaT_pixel,
                    deltaN_pixel=deltaN_pixel,
                )


                if resultado is None:
                    continue

                T_fit_map[y, x] = resultado["T_fit"].to_value(u.K)
                N_fit_map[y, x] = resultado["N_fit"].to_value(1 / u.cm**2)
                deltaT_map[y, x] = resultado["deltaT"].to_value(u.K)
                deltaN_map[y, x] = resultado["deltaN"].to_value(1 / u.cm**2)
                chi2_min_map[y, x] = resultado["chi_min"]
                tab_opacidad = resultado["tab_lineas_opacidad"]

                indice_linea_fuerte = np.nanargmax(
                    np.asarray(
                        tab_opacidad["sijmu2"],
                        dtype=float,
                    )
                )

                tau_linea_fuerte_map[y, x] = (
                    tab_opacidad["tau"][indice_linea_fuerte])

            except Exception as e:
                print(f"[chi2_pix] Fallo en píxel x={x}, y={y}: {e}")
                continue

    mapas = {
    "T_fit_map": T_fit_map,
    "N_fit_map": N_fit_map,
    "deltaT_map": deltaT_map,
    "deltaN_map": deltaN_map,
    "chi2_min_map": chi2_min_map,
    "tau_linea_fuerte_map": tau_linea_fuerte_map,
    "header": header_2d,
}

    save_chi2_maps_fits(
        region_name=region_name,
        molecula=molecula,
        mapas=mapas,
        header_2d=header_2d,
    )

    return mapas
