#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:06:59 2026

@author: jorge
"""

"""
Pipeline de ajuste chi2 para el TFM.

Este módulo:
- toma como punto inicial los resultados del diagrama rotacional;
- construye automáticamente una CONFIG_CHI2 para una molécula;
- ejecuta minimchi2() de TFM_chi2_fit.py;
- itera hasta converger;
- guarda los resultados en tables/chi2/<REGION>/chi2_resultados.ecsv;
- genera el modelo sintético final con los parámetros chi2.
"""

import numpy as np
from astropy import units as u

import TFM_config as cfg

from TFM_runtime import _resolve_name, resolve_intervalo_region
from TFM_chi2_fit import minimchi2

from TFM_storage import (
    load_chi2_results,
    save_chi2_results,
    update_chi2_result,
    chi2_result_exists,
    get_chi2_result,
)

from TFM_synthetic_pipeline import (
    seleccionar_fila_molecula,
    cat_mol_from_id_cat_name,
    calcular_modelo_sintetico_molecula,
)

def obtener_dict_noconsid_chi2(region_name):
    """
    Devuelve las frecuencias excluidas del ajuste chi2
    para una región concreta.
    """

    if region_name not in cfg.REGION_LINE_CONFIG:
        raise KeyError(
            f"No existe configuración de líneas para {region_name}. "
            f"Regiones disponibles: "
            f"{list(cfg.REGION_LINE_CONFIG.keys())}"
        )

    config_region = cfg.REGION_LINE_CONFIG[region_name]

    if "chi2_no_considerar" not in config_region:
        raise KeyError(
            f"La región {region_name} no contiene el bloque "
            "'chi2_no_considerar'."
        )

    return config_region["chi2_no_considerar"]

def obtener_config_grid_chi2(molecula, n_grid_default=10):
    """
    Devuelve deltaT, deltaN y n_grid para una molécula desde cfg.CHI2_GRID_CONFIG.

    Si la molécula no está configurada, devuelve None, None y n_grid_default.
    En ese caso el pipeline podrá usar los deltas del diagrama rotacional
    o los que se pasen manualmente.
    """

    grid_config = getattr(cfg, "CHI2_GRID_CONFIG", {})

    if molecula not in grid_config:
        print(
            f"[chi2] No hay CHI2_GRID_CONFIG para {molecula}. "
            "Se usarán deltaT/deltaN del argumento o del diagrot."
        )
        return None, None, n_grid_default

    cfg_mol = grid_config[molecula]

    deltaT = cfg_mol.get("deltaT", None)
    deltaN = cfg_mol.get("deltaN", None)
    n_grid = cfg_mol.get("n_grid", n_grid_default)

    return deltaT, deltaN, n_grid

# ============================================================
# LECTURA DE PUNTO INICIAL
# ============================================================

def obtener_resultado_diagrot_molecula(molecula, resultados_diagrot):
    """
    Devuelve el resultado de diagrama rotacional para una molécula.

    resultados_diagrot normalmente es un diccionario:
        resultados_diagrot[molecula]["T_ex"]
        resultados_diagrot[molecula]["N_col"]
    """

    if isinstance(resultados_diagrot, dict):

        if molecula not in resultados_diagrot:
            raise KeyError(
                f"No hay resultado de diagrama rotacional para {molecula}. "
                f"Disponibles: {list(resultados_diagrot.keys())}"
            )

        return resultados_diagrot[molecula]

    # Por seguridad, también soporta QTable
    if "molecula" not in resultados_diagrot.colnames:
        raise KeyError(
            "resultados_diagrot no tiene columna 'molecula'. "
            f"Columnas disponibles: {resultados_diagrot.colnames}"
        )

    mask = resultados_diagrot["molecula"] == molecula

    if not np.any(mask):
        raise KeyError(f"No hay resultado de diagrama rotacional para {molecula}.")

    return resultados_diagrot[mask][0]


def obtener_T_N_inicial_diagrot(molecula, resultados_diagrot):
    """
    Obtiene T_ex y N_col iniciales desde los resultados del diagrama rotacional.
    """

    fila = obtener_resultado_diagrot_molecula(molecula, resultados_diagrot)

    T_init = fila["T_ex"].to(u.K)
    N_init = fila["N_col"].to(1 / u.cm**2)

    return T_init, N_init


def obtener_deltas_iniciales_diagrot(
        molecula,
        resultados_diagrot,
        deltaT=None,
        deltaN=None,
        deltaT_factor=1.0,
        deltaN_factor=1.0):
    """
    Obtiene deltaT y deltaN iniciales.

    Prioridad:
        1. deltaT/deltaN pasados manualmente.
        2. Delta_Tex/Delta_Ncol del diagrama rotacional.
        3. Fallback: 30% de T y 50% de N.
    """

    fila = obtener_resultado_diagrot_molecula(molecula, resultados_diagrot)

    T_init = fila["T_ex"].to(u.K)
    N_init = fila["N_col"].to(1 / u.cm**2)

    if deltaT is None:
        if "Delta_Tex" in fila:
            deltaT = fila["Delta_Tex"].to(u.K)
        else:
            deltaT = 0.3 * T_init

    if deltaN is None:
        if "Delta_Ncol" in fila:
            deltaN = fila["Delta_Ncol"].to(1 / u.cm**2)
        else:
            deltaN = 0.5 * N_init

    deltaT = deltaT.to(u.K)
    deltaN = deltaN.to(1 / u.cm**2)

    # Evitar mallas degeneradas si la incertidumbre sale NaN, 0 o negativa
    if (not np.isfinite(deltaT.value)) or deltaT.value <= 0:
        deltaT = 0.3 * T_init

    if (not np.isfinite(deltaN.value)) or deltaN.value <= 0:
        deltaN = 0.5 * N_init

    deltaT = deltaT_factor * deltaT
    deltaN = deltaN_factor * deltaN

    return deltaT.to(u.K), deltaN.to(1 / u.cm**2)


# ============================================================
# CONFIG CHI2
# ============================================================

def construir_config_chi2_molecula(
        molecula,
        tab_mol_config,
        resultados_diagrot,
        fwhm_dict,
        dict_T_cont,
        deltaT=None,
        deltaN=None,
        deltaT_factor=1.0,
        deltaN_factor=1.0,
        v_pik_dict=None,
        intervalos_mol_region=None):
    """
    Construye un diccionario tipo CONFIG_CHI2 para una sola molécula.

    Devuelve:
        {
            "C2H5OH_g": {
                "T_ex": ...,
                "N_col": ...,
                "deltaT": ...,
                "deltaN": ...,
                ...
            }
        }
    """

    fila_cfg = seleccionar_fila_molecula(tab_mol_config, molecula)

    if molecula not in fwhm_dict:
        raise KeyError(
            f"No hay FWHM calibrado para {molecula}. "
            "Calcula primero la calibración de FWHM."
        )

    T_init, N_init = obtener_T_N_inicial_diagrot(
        molecula=molecula,
        resultados_diagrot=resultados_diagrot,
    )

    deltaT_init, deltaN_init = obtener_deltas_iniciales_diagrot(
        molecula=molecula,
        resultados_diagrot=resultados_diagrot,
        deltaT=deltaT,
        deltaN=deltaN,
        deltaT_factor=deltaT_factor,
        deltaN_factor=deltaN_factor,
    )

    id_cat_name = str(fila_cfg["id_cat"])

    line_filter = getattr(cfg, "SPEC_SINT_LINE_FILTERS", {}).get(molecula, {})

    sijmu2_model = line_filter.get("sijmu2", 0 * u.D**2)
    aij_model = line_filter.get("aij", 0 / u.s)

    if v_pik_dict is None or molecula not in v_pik_dict:
        raise KeyError(
            f"No hay v_pik calibrada para {molecula}. "
            "Debes pasar v_pik_dict al ajuste chi2."
        )

    v_pik = v_pik_dict[molecula]

    config_chi2 = {
        molecula: {
            "T_ex": T_init,
            "N_col": N_init,
            "deltaT": deltaT_init,
            "deltaN": deltaN_init,

            "mol": str(fila_cfg["mol"]),
            "intervalo": resolve_intervalo_region(fila_cfg["intervalo"],
                                intervalos_mol_region=intervalos_mol_region),
            "id_splat": _resolve_name(fila_cfg["id_splat"]),
            "filtro_estructuras": _resolve_name(fila_cfg["filtro_estructuras"]),

            "FWHM": fwhm_dict[molecula],
            "T_cont": dict_T_cont,

            "f0": fila_cfg["f0"],
            "v_pik": v_pik,

            "cat_mol": cat_mol_from_id_cat_name(id_cat_name),
            "id_cat": _resolve_name(id_cat_name),

            # minimchi2() espera estas claves:
            "sij": sijmu2_model,
            "aij": aij_model,
        }
    }

    return config_chi2


def obtener_intervalos_molecula(
        tab_mol_config,
        molecula,
        intervalos_mol_region=None):

    fila = seleccionar_fila_molecula(tab_mol_config, molecula)

    return resolve_intervalo_region(
        fila["intervalo"],
        intervalos_mol_region=intervalos_mol_region,
    )


def nombre_intervalo_molecula(tab_mol_config, molecula):
    """
    Devuelve el nombre simbólico de la banda de una molécula:
        'Banda6', 'Banda3', etc.
    """

    fila = seleccionar_fila_molecula(
        tab_mol_config,
        molecula,
    )

    return str(fila["intervalo"]).strip()


def obtener_orden_ajuste_chi2(tab_mol_config, molecula):
    """
    Devuelve el orden de ajuste chi2 correspondiente a la banda
    de la molécula seleccionada.
    """

    banda = nombre_intervalo_molecula(
        tab_mol_config,
        molecula,
    )

    if banda not in cfg.CHI2_FIT_ORDER_BY_BAND:
        raise KeyError(
            f"No hay un orden de ajuste chi2 definido para {banda}. "
            "Bandas disponibles: "
            f"{list(cfg.CHI2_FIT_ORDER_BY_BAND.keys())}"
        )

    orden = list(
        cfg.CHI2_FIT_ORDER_BY_BAND[banda]
    )

    if molecula not in orden:
        raise ValueError(
            f"{molecula} pertenece a {banda}, pero no aparece "
            f"en su orden de ajuste chi2: {orden}"
        )

    return orden

# ============================================================
# AJUSTE CHI2 CONVERGENTE
# ============================================================

def ajustar_chi2_convergente(
        molecula,
        config_chi2,
        dict_sigma_vent,
        intervalos,
        tab_lineas,
        dict_cubos_med,
        dict_lin_noconsid=None,
        dict_resol_espec=None,
        model_sint=None,
        dict_T_cont=None,
        residuos=False,
        n_grid=10,
        tol=1.0,
        max_iter=10,
        debug=False,
        plots=True,
        rutacarp_region=None,
        rutaregion_region=None):
    """
    Ajuste chi2 iterativo usando minimchi2().
    """

    configmol = {molecula: config_chi2[molecula]}

    T0 = configmol[molecula]["T_ex"]
    N0 = configmol[molecula]["N_col"]
    deltaT0 = configmol[molecula]["deltaT"]
    deltaN0 = configmol[molecula]["deltaN"]

    modelo_actual = model_sint
    chi_min = np.nan

    converged = False
    n_iter_real = 0

    for i in range(max_iter):

        print(
            f"[chi2] Iteración {i + 1}/{max_iter} para {molecula}: "
            f"T0={T0:.3g}, N0={N0:.3g}"
        )

        modelo_fit, dict_TN = minimchi2(
            configmol,
            dict_sigma_vent,
            intervalos,
            n_grid,
            tab_lineas,
            dict_resol_espec=dict_resol_espec,
            dict_especchi=dict_cubos_med,
            dict_lin_noconsid=dict_lin_noconsid,
            debug=debug,
            model_sint=modelo_actual,
            dictTcont=dict_T_cont,
            residuos=residuos,
            rutacarp_region=rutacarp_region,
            rutaregion_region=rutaregion_region,
            show_plots=plots)

        T1 = dict_TN[molecula]["T_fit"]
        N1 = dict_TN[molecula]["N_fit"]

        deltaT1 = dict_TN[molecula]["deltaT"]
        deltaN1 = dict_TN[molecula]["deltaN"]
        chi_min = dict_TN[molecula]["chi_min"]

        dif_T = np.abs(T1 - T0).to(u.K)
        dif_N_rel = np.abs((N1 - N0) / N0).decompose().value

        n_iter_real = i + 1

        toca_borde = (
            T1 + deltaT1 >= T0 + deltaT0 or
            T1 - deltaT1 <= T0 - deltaT0 or
            N1 + deltaN1 >= N0 + deltaN0 or
            N1 - deltaN1 <= N0 - deltaN0
        )
        toca_borde_N = (N1 + deltaN1 >= N0 + deltaN0 or
                        N1 - deltaN1 <= N0 - deltaN0)

        toca_borde_T = (T1 + deltaT1 >= T0 + deltaT0 or
                        T1 - deltaT1 <= T0 - deltaT0)

        print(f"[chi2] Resultado iteración {i + 1}:")
        print(f"       T_fit = {T1}")
        print(f"       N_fit = {N1}")
        print(f"       deltaT = {deltaT1}")
        print(f"       deltaN = {deltaN1}")
        print(f"       chi2_min = {chi_min}")
        print(f"       toca_borde = {toca_borde}")
        print(f"       dif_T = {dif_T}")
        print(f"       dif_N_rel = {dif_N_rel:.3g}")

        # Actualizamos centro para la siguiente iteración
        configmol[molecula]["T_ex"] = T1
        configmol[molecula]["N_col"] = N1

        if not toca_borde:
            configmol[molecula]["deltaT"] = deltaT1
            configmol[molecula]["deltaN"] = deltaN1
            deltaT0 = deltaT1
            deltaN0 = deltaN1
        elif not toca_borde_N:
            configmol[molecula]["deltaN"] = deltaN1
            deltaN0 = deltaN1
        elif not toca_borde_T:
            configmol[molecula]["deltaT"] = deltaT1
            deltaT0 = deltaT1
            
        # Si estamos ajustando sobre un modelo acumulado y no sobre residuos,
        # actualizamos el modelo base como hacía tu chi2_conv().
        if modelo_actual is not None and not residuos:
            modelo_actual = modelo_fit

        # ------------------------------------------------------------
        # Criterio de convergencia
        # ------------------------------------------------------------
        # Reproducimos el criterio antiguo de chi2_conv:
        # parar únicamente si la diferencia en temperatura es menor que tol.

        if isinstance(tol, u.Quantity):
            tol_T = tol.to(u.K)
        else:
            tol_T = tol * u.K

        if dif_T <= tol_T:
            converged = True
            T0 = T1
            N0 = N1

            print("[chi2] Convergencia alcanzada:")
            print(f"       dif_T = {dif_T}")
            print(f"       tol_T = {tol_T}")
            print(f"       dif_N_rel = {dif_N_rel:.3e}")
            print("       Criterio usado: solo diferencia en T_ex")

            break
 
        T0 = T1
        N0 = N1

    return {
        "T_fit": T0.to(u.K),
        "N_fit": N0.to(1 / u.cm**2),
        "deltaT": configmol[molecula]["deltaT"].to(u.K),
        "deltaN": configmol[molecula]["deltaN"].to(1 / u.cm**2),
        "chi2_min": chi_min,
        "modelo": modelo_fit,
        "config_final": configmol,
        "n_iter": n_iter_real,
        "converged": converged,
    }


# ============================================================
# FUNCIÓN DE ALTO NIVEL: UNA MOLÉCULA
# ============================================================

def cargar_o_calcular_chi2_molecula(
        molecula,
        region_name,
        tab_mol_config,
        resultados_diagrot,
        fwhm_dict,
        dict_T_cont,
        dict_sigma_vent,
        dict_cubos_med,
        tab_lineas,
        recalcular=False,
        n_grid=10,
        tol=1.0,
        max_iter=10,
        deltaT=None,
        deltaN=None,
        deltaT_factor=1.0,
        deltaN_factor=1.0,
        dict_lin_noconsid=None,
        dict_resol_espec=None,
        model_sint=None,
        residuos=False,
        fuente_init="diagrot",
        show_model=True,
        save_model=True,
        debug=False,
        plots=True,
        v_pik_dict = None,
        dict_cubos_plot=None,
        intervalos_mol_region=None,
        rutacarp_region=None,
        rutaregion_region=None):
    """
    Carga o calcula el ajuste chi2 de una molécula.

    Si ya existe resultado y recalcular=False:
        - carga la fila guardada;
        - muestra valores;
        - genera modelo sintético con fuente_parametros='chi2'.

    Si no existe o recalcular=True:
        - toma T,N iniciales desde diagrot;
        - construye CONFIG_CHI2;
        - ejecuta minimchi2() iterativamente;
        - guarda resultado;
        - genera modelo sintético final con fuente_parametros='chi2'.
    """

    tab_chi2 = load_chi2_results(region_name)

    if dict_cubos_plot is None:
        dict_cubos_plot = dict_cubos_med

    if chi2_result_exists(tab_chi2, molecula) and not recalcular:
        fila = get_chi2_result(tab_chi2, molecula)

        print(f"[chi2] Resultado existente para {molecula}:")
        print(f"       T_fit = {fila['T_fit']}")
        print(f"       N_fit = {fila['N_fit']}")
        print(f"       deltaT = {fila['deltaT']}")
        print(f"       deltaN = {fila['deltaN']}")
        print(f"       chi2_min = {fila['chi2_min']}")
        print(f"       converged = {fila['converged']}")

        modelo_chi2 = None

        if show_model or save_model:
            modelo_chi2 = calcular_modelo_sintetico_molecula(molecula=molecula,
                                                       region_name=region_name,
                                                 tab_mol_config=tab_mol_config,
                                                resultados_parametros=tab_chi2,
                                                           fwhm_dict=fwhm_dict,
                                                       dict_T_cont=dict_T_cont,
                                                dict_cubos_med=dict_cubos_plot,
                                                      fuente_parametros="chi2",
                                                   modelo_radiativo="opacidad",
                                                             plot_lineas=False,
                                                         show_plots=show_model,
                                                         save_plots=save_model,
                                                         v_pik_dict=v_pik_dict,
                                   intervalos_mol_region=intervalos_mol_region,
                                   rutacarp_region=rutacarp_region,
                                   rutaregion_region=rutaregion_region,
                                                         )

        return {
            "molecula": molecula,
            "resultado": fila,
            "modelo": modelo_chi2,
            "calculado": False,
        }

    T_init, N_init = obtener_T_N_inicial_diagrot(
        molecula=molecula,
        resultados_diagrot=resultados_diagrot,
    )

    print(f"[chi2] Calculando ajuste chi2 para {molecula}")
    print(f"[chi2] Punto inicial desde {fuente_init}:")
    print(f"       T_init = {T_init}")
    print(f"       N_init = {N_init}")

    # ------------------------------------------------------------
    # Configuración de malla chi2 desde TFM_config.py
    # ------------------------------------------------------------
    deltaT_cfg, deltaN_cfg, n_grid_cfg = obtener_config_grid_chi2(
        molecula,
        n_grid_default=n_grid,
    )

    if deltaT is None:
        deltaT = deltaT_cfg

    if deltaN is None:
        deltaN = deltaN_cfg

    if n_grid is None:
        n_grid = n_grid_cfg

    print("[chi2] Malla usada:")
    print(f"       deltaT = {deltaT}")
    print(f"       deltaN = {deltaN}")
    print(f"       n_grid = {n_grid}")
    
    config_chi2 = construir_config_chi2_molecula(
        molecula=molecula,
        tab_mol_config=tab_mol_config,
        resultados_diagrot=resultados_diagrot,
        fwhm_dict=fwhm_dict,
        dict_T_cont=dict_T_cont,
        deltaT=deltaT,
        deltaN=deltaN,
        deltaT_factor=deltaT_factor,
        deltaN_factor=deltaN_factor, v_pik_dict= v_pik_dict,
        intervalos_mol_region=intervalos_mol_region)

    intervalos = obtener_intervalos_molecula(tab_mol_config,molecula,
                                intervalos_mol_region=intervalos_mol_region)

    ajuste = ajustar_chi2_convergente(
        molecula=molecula,
        config_chi2=config_chi2,
        dict_sigma_vent=dict_sigma_vent,
        intervalos=intervalos,
        tab_lineas=tab_lineas,
        dict_cubos_med=dict_cubos_med,
        dict_lin_noconsid=dict_lin_noconsid,
        dict_resol_espec=dict_resol_espec,
        model_sint=model_sint,
        dict_T_cont=dict_T_cont,
        residuos=residuos,
        n_grid=n_grid,
        tol=tol,
        max_iter=max_iter,
        debug=debug, plots = plots,
        rutacarp_region=rutacarp_region,
        rutaregion_region=rutaregion_region,
    )

    tab_chi2 = update_chi2_result(
        tab_chi2=tab_chi2,
        molecula=molecula,
        region_name=region_name,
        T_init=T_init,
        N_init=N_init,
        T_fit=ajuste["T_fit"],
        N_fit=ajuste["N_fit"],
        deltaT=ajuste["deltaT"],
        deltaN=ajuste["deltaN"],
        chi2_min=ajuste["chi2_min"],
        n_grid=n_grid,
        n_iter=ajuste["n_iter"],
        converged=ajuste["converged"],
        fuente_init=fuente_init,
    )

    save_chi2_results(tab_chi2, region_name)

    modelo_chi2 = None

    if show_model or save_model:
        modelo_chi2 = calcular_modelo_sintetico_molecula(molecula=molecula,
                                                    region_name=region_name,
                                                tab_mol_config=tab_mol_config,
                                                resultados_parametros=tab_chi2,
                                                fwhm_dict=fwhm_dict,
                                                dict_T_cont=dict_T_cont,
                                                dict_cubos_med=dict_cubos_plot,
                                                fuente_parametros="chi2",
                                                modelo_radiativo="opacidad",
                                                plot_lineas=False,
                                                show_plots=show_model,
                                                save_plots=save_model,
                                                v_pik_dict=v_pik_dict,
                                   intervalos_mol_region=intervalos_mol_region,
                                   rutacarp_region=rutacarp_region,
                                   rutaregion_region=rutaregion_region,)

    return {
        "molecula": molecula,
        "T_init": T_init,
        "N_init": N_init,
        "T_fit": ajuste["T_fit"],
        "N_fit": ajuste["N_fit"],
        "deltaT": ajuste["deltaT"],
        "deltaN": ajuste["deltaN"],
        "chi2_min": ajuste["chi2_min"],
        "n_iter": ajuste["n_iter"],
        "converged": ajuste["converged"],
        "modelo_fit": ajuste["modelo"],
        "modelo_chi2": modelo_chi2,
        "config_final": ajuste["config_final"],
        "calculado": True,
    }

def construir_residuo_base_chi2(
        molecula,
        region_name,
        tab_mol_config,
        resultados_chi2,
        fwhm_dict,
        dict_T_cont,
        dict_cubos_med,
        show_plots=False,
        save_plots=False,
        v_pik_dict=None,
        intervalos_mol_region=None,
        rutacarp_region=None,
        rutaregion_region=None):
    """
    Reconstruye el espectro/residuo base sobre el que debe ajustarse una molécula.

    Para una molécula dada, resta únicamente los modelos chi2 de las moléculas
    anteriores en el orden de ajuste.

    Ejemplo:
        orden = [A, B, C]

        Si molecula = A:
            devuelve dict_cubos_med

        Si molecula = B:
            devuelve dict_cubos_med - modelo_chi2(A)

        Si molecula = C:
            devuelve dict_cubos_med - modelo_chi2(A) - modelo_chi2(B)

    Esto permite recalcular una molécula anterior sin usar residuos que ya
    contienen la resta de ella misma.
    """

    from TFM_synthetic_pipeline import calcular_modelo_sintetico

    orden_moleculas = obtener_orden_ajuste_chi2(
        tab_mol_config=tab_mol_config,
        molecula=molecula,
    )

    idx = orden_moleculas.index(molecula)
    moleculas_previas = orden_moleculas[:idx]

    # Solo podemos restar moléculas previas que ya tengan resultado chi2 guardado.
    if hasattr(resultados_chi2, "colnames"):
        mols_chi2_guardadas = list(resultados_chi2["molecula"])
    elif isinstance(resultados_chi2, dict):
        mols_chi2_guardadas = list(resultados_chi2.keys())
    else:
        mols_chi2_guardadas = []

    moleculas_previas_ajustadas = [
        mol for mol in moleculas_previas
        if mol in mols_chi2_guardadas
    ]

    print("[chi2] Reconstruyendo residuo base")
    print(f"       Molécula actual: {molecula}")
    print(f"       Moléculas previas en el orden: {moleculas_previas}")
    print(f"       Moléculas previas con chi2 guardado: {moleculas_previas_ajustadas}")

    if len(moleculas_previas_ajustadas) == 0:
        print("[chi2] No hay moléculas previas ajustadas. Uso espectro completo.")
        return {
            "dict_espec_base": dict_cubos_med,
            "moleculas_restadas": [],
            "modelo_previas": None,
        }

    modelo_previas = calcular_modelo_sintetico(
        moleculas=moleculas_previas_ajustadas, region_name=region_name,
        tab_mol_config=tab_mol_config,
        resultados_parametros=resultados_chi2, fwhm_dict=fwhm_dict,
        dict_T_cont=dict_T_cont, dict_cubos_med=dict_cubos_med,
        fuente_parametros="chi2", modelo_radiativo="opacidad",
        plot_lineas=False, show_plots=show_plots, save_plots=save_plots,
        guardar_solo_final=True, v_pik_dict=v_pik_dict,
        intervalos_mol_region=intervalos_mol_region,
        rutacarp_region=rutacarp_region,
        rutaregion_region=rutaregion_region,)

    # Si calcular_modelo_sintetico devuelve separado por bandas, intentamos
    # escoger la banda de la molécula actual.
    fila = seleccionar_fila_molecula(tab_mol_config, molecula)
    intervalo_mol = str(fila["intervalo"])

    if (
        isinstance(modelo_previas, dict)
        and intervalo_mol in modelo_previas
        and "residuos" in modelo_previas[intervalo_mol]
    ):
        dict_espec_base = modelo_previas[intervalo_mol]["residuos"]

    elif isinstance(modelo_previas, dict) and "residuos" in modelo_previas:
        dict_espec_base = modelo_previas["residuos"]

    else:
        raise KeyError(
            "No he podido encontrar los residuos del modelo de moléculas previas. "
            f"Claves disponibles: {list(modelo_previas.keys())}"
        )

    return {
        "dict_espec_base": dict_espec_base,
        "moleculas_restadas": moleculas_previas_ajustadas,
        "modelo_previas": modelo_previas,
    }