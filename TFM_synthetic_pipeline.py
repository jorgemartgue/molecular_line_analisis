#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:28:35 2026

@author: jorge
"""

"""
Pipeline de alto nivel para modelos sintéticos.

Permite:
- generar modelo sintético de una molécula individual;
- generar modelo sintético acumulado de una lista de moléculas;
- usar parámetros iniciales del diagrama rotacional;
- usar parámetros refinados del ajuste chi²;
- guardar automáticamente los plots en PDF.
"""

from pathlib import Path
import numpy as np
from astropy import units as u
import TFM_config as cfg

from TFM_runtime import _resolve_name, resolve_intervalo_region
from TFM_synthetic_model import spec_sint_class, spec_sint_opacidad


# ============================================================
# HELPERS GENERALES
# ============================================================

def seleccionar_fila_molecula(tab_mol_config, molecula):
    """
    Selecciona la fila correspondiente a una molécula en moleculas_config.ecsv.
    """

    mask = tab_mol_config["nombre"] == molecula
    fila = tab_mol_config[mask]

    if len(fila) == 0:
        raise KeyError(
            f"La molécula {molecula} no está en moleculas_config.ecsv. "
            f"Moléculas disponibles: {list(tab_mol_config['nombre'])}"
        )

    if len(fila) > 1:
        raise ValueError(
            f"La molécula {molecula} aparece más de una vez en "
            "moleculas_config.ecsv."
        )

    return fila[0]


def cat_mol_from_id_cat_name(id_cat_name):
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


def normalizar_lista_moleculas(moleculas):
    """
    Permite pasar una molécula como string o una lista/tupla de moléculas.

    Returns
    -------
    mols : list
        Lista de moléculas.

    es_lista : bool
        False si el usuario pasó un string.
        True si el usuario pasó una lista/tupla.
    """

    if isinstance(moleculas, str):
        return [moleculas], False

    return list(moleculas), True


def obtener_T_N_molecula(
        molecula,
        resultados_parametros,
        fuente_parametros="diagrot"):
    """
    Obtiene T_ex y N_col para una molécula desde una fuente de parámetros.

    fuente_parametros:
        - 'diagrot': puede venir como dict o QTable.
        - 'chi2': normalmente viene como QTable con T_fit y N_fit.
    """

    # ============================================================
    # CASO 1: diccionario
    # ============================================================
    if isinstance(resultados_parametros, dict):

        if molecula not in resultados_parametros:
            raise KeyError(
                f"No hay parámetros para {molecula}. "
                f"Disponibles: {list(resultados_parametros.keys())}"
            )

        fila = resultados_parametros[molecula]

        if fuente_parametros == "diagrot":
            T_ex = fila["T_ex"].to(u.K)
            N_col = fila["N_col"].to(1 / u.cm**2)

        elif fuente_parametros == "chi2":
            T_ex = fila["T_fit"].to(u.K)
            N_col = fila["N_fit"].to(1 / u.cm**2)

        else:
            raise ValueError(
                f"fuente_parametros='{fuente_parametros}' no reconocido. "
                "Usa 'diagrot' o 'chi2'."
            )

        return T_ex, N_col

    # ============================================================
    # CASO 2: QTable
    # ============================================================
    if "molecula" not in resultados_parametros.colnames:
        raise KeyError(
            "La tabla de parámetros no tiene columna 'molecula'. "
            f"Columnas disponibles: {resultados_parametros.colnames}"
        )

    mask = resultados_parametros["molecula"] == molecula

    if not np.any(mask):
        raise KeyError(
            f"No hay parámetros guardados para {molecula} "
            f"en fuente_parametros='{fuente_parametros}'."
        )

    fila = resultados_parametros[mask][0]

    if fuente_parametros == "diagrot":
        T_ex = fila["T_ex"].to(u.K)
        N_col = fila["N_col"].to(1 / u.cm**2)

    elif fuente_parametros == "chi2":
        T_ex = fila["T_fit"].to(u.K)
        N_col = fila["N_fit"].to(1 / u.cm**2)

    else:
        raise ValueError(
            f"fuente_parametros='{fuente_parametros}' no reconocido. "
            "Usa 'diagrot' o 'chi2'."
        )

    return T_ex, N_col


# ============================================================
# RUTAS
# ============================================================

def path_synthetic_fig_dir(region_name, moleculas,
                           fuente_parametros="diagrot",
                           modelo_radiativo="delgado"):
    """
    Devuelve la carpeta donde guardar los PDFs del modelo sintético.

    modelo_radiativo:
        - 'delgado': aproximación ópticamente delgada
        - 'opacidad': modelo con opacidad
    """

    mols, es_lista = normalizar_lista_moleculas(moleculas)

    if es_lista:
        folder_name = "complete_model"
    else:
        folder_name = mols[0]

    if modelo_radiativo == "delgado":
        modelo_folder = "opticamente_delgado"
    elif modelo_radiativo == "opacidad":
        modelo_folder = "opacidad"
    else:
        raise ValueError(
            f"modelo_radiativo='{modelo_radiativo}' no reconocido. "
            "Usa 'delgado' u 'opacidad'."
        )

    path_dir = (
        cfg.rutafig_synthetic
        / region_name
        / fuente_parametros
        / modelo_folder
        / folder_name
    )

    path_dir.mkdir(parents=True, exist_ok=True)

    return path_dir


# ============================================================
# CONSTRUCCIÓN DE PARÁMETROS
# ============================================================

def construir_parametros_modelo_molecula(
        molecula,
        tab_mol_config,
        resultados_parametros,
        fwhm_dict,
        dict_T_cont,
        fuente_parametros="diagrot",
        v_pik_dict=None,
        intervalos_mol_region=None):
    """
    Construye el diccionario de parámetros necesario para spec_sint_class().

    T_ex y N_col pueden venir de:
        - diagrama rotacional
        - ajuste chi²
    """

    fila = seleccionar_fila_molecula(tab_mol_config, molecula)

    if molecula not in fwhm_dict:
        raise KeyError(
            f"No hay FWHM calibrado para {molecula}. "
            "Calcula primero la calibración de FWHM."
        )

    T_ex, N_col = obtener_T_N_molecula(
        molecula=molecula,
        resultados_parametros=resultados_parametros,
        fuente_parametros=fuente_parametros,
    )

    id_cat_name = str(fila["id_cat"])

    f0 = fila["f0"]
    
    if v_pik_dict is None or molecula not in v_pik_dict:
        raise KeyError(
            f"No hay v_pik calibrada para {molecula}. "
            "Debes pasar v_pik_dict al modelo sintético."
        )

    v_pik = v_pik_dict[molecula]
    
    line_filter = getattr(cfg, "SPEC_SINT_LINE_FILTERS", {}).get(molecula, {})

    sijmu2_model = line_filter.get("sijmu2", 0 * u.D**2)
    aij_model = line_filter.get("aij", 0 / u.s)
    
    filtros_modelo = getattr(cfg, "SPEC_SINT_STRUCTURE_FILTERS", {})

    if molecula in filtros_modelo:
        filtro_estructuras_modelo = filtros_modelo[molecula]
    else:
        filtro_estructuras_modelo = _resolve_name(fila["filtro_estructuras"])
    
    params = {
        "T_ex": T_ex,
        "N_col": N_col,

        "mol": str(fila["mol"]),
        "intervalo": resolve_intervalo_region(fila["intervalo"],
                                intervalos_mol_region=intervalos_mol_region),
        "id_splat": _resolve_name(fila["id_splat"]),
        "filtro_estructuras": filtro_estructuras_modelo,

        "FWHM": fwhm_dict[molecula],
        "T_cont": dict_T_cont,

        "f0": f0,
        "v_pik": v_pik,

        "name_mol": molecula,
        "cat_mol": cat_mol_from_id_cat_name(id_cat_name),
        "id_cat": _resolve_name(id_cat_name),

        "sijmu2": sijmu2_model,
        "aij": aij_model,
    }

    return params


# ============================================================
# MODELO DE UNA MOLÉCULA
# ============================================================

def calcular_modelo_sintetico_molecula(
        molecula,
        region_name,
        tab_mol_config,
        resultados_parametros,
        fwhm_dict,
        dict_T_cont,
        dict_cubos_med,
        fuente_parametros="diagrot",
        modelo_radiativo="delgado",
        modeloin=None,
        tab_lineas_mol=None,
        plot_lineas=False,
        show_plots=True,
        save_plots=True,
        save_dir=None,
        plot_prefix=None,
        v_pik_dict=None,
        dict_sigma = None,
        nsigma_lineas = 1.0,
        intervalos_mol_region=None, 
        rutacarp_region = None,
        rutaregion_region = None):
    """
    Calcula el modelo sintético de una molécula.

    Puede usarse:
    - de forma aislada;
    - dentro de un modelo acumulado pasando modeloin y tab_lineas_mol.
    """

    params = construir_parametros_modelo_molecula(
        molecula=molecula,
        tab_mol_config=tab_mol_config,
        resultados_parametros=resultados_parametros,
        fwhm_dict=fwhm_dict,
        dict_T_cont=dict_T_cont,
        fuente_parametros=fuente_parametros, v_pik_dict= v_pik_dict,
        intervalos_mol_region= intervalos_mol_region
    )

    if save_dir is None:
        save_dir = path_synthetic_fig_dir(
            region_name=region_name,
            moleculas=molecula,
            fuente_parametros=fuente_parametros,
            modelo_radiativo=modelo_radiativo,
        )

    if plot_prefix is None:
        plot_prefix = molecula

    print(
        f"[spec_sint] Calculando modelo sintético para {molecula} "
        f"usando fuente_parametros='{fuente_parametros}'"
    )

    if modelo_radiativo == "delgado":

        print(
        f"[spec_sint] Calculando modelo sintético ópticamente delgado "
        f"para {molecula} usando fuente_parametros='{fuente_parametros}'"
        )

        tab_lineas, modelo, tab_lineas_plot, residuos = spec_sint_class(
        params["T_ex"],
        params["N_col"],
        params["mol"],
        params["intervalo"],
        params["id_splat"],
        params["filtro_estructuras"],
        params["FWHM"],
        params["T_cont"],
        params["f0"],
        params["v_pik"],
        params["name_mol"],
        params["cat_mol"],
        params["id_cat"],
        sijmu2=params["sijmu2"],
        aij=params["aij"],
        modeloin=modeloin,
        dict_espec=dict_cubos_med,
        plot_lineas=plot_lineas,
        tab_lineas_mol=tab_lineas_mol,
        show_plots=show_plots,
        save_plots=save_plots,
        save_dir=save_dir,
        plot_prefix=plot_prefix,
        rutacarp_region= rutacarp_region,
        rutaregion_region = rutaregion_region
    )

    elif modelo_radiativo == "opacidad":

        print(
        f"[spec_sint] Calculando modelo sintético con opacidad "
        f"para {molecula} usando fuente_parametros='{fuente_parametros}'"
        )

        tab_lineas, modelo, tab_lineas_plot, residuos = spec_sint_opacidad(
        params["T_ex"],
        params["N_col"],
        params["mol"],
        params["intervalo"],
        params["id_splat"],
        params["filtro_estructuras"],
        params["FWHM"],
        params["T_cont"],
        params["v_pik"],
        params["name_mol"],
        params["cat_mol"],
        params["id_cat"],
        sijmu2=params["sijmu2"],
        aij=params["aij"],
        modeloin=modeloin,
        dict_espec=dict_cubos_med,
        dict_sigma=dict_sigma,
        nsigma_lineas=nsigma_lineas,
        plot_lineas=plot_lineas,
        tab_lineas_mol=tab_lineas_mol,
        show_plots=show_plots,
        rutacarp_region= rutacarp_region,
        rutaregion_region= rutaregion_region
    )

    else:
        raise ValueError(
            f"modelo_radiativo='{modelo_radiativo}' no reconocido. "
            "Usa 'delgado' u 'opacidad'."
        )
    
    return {
        "molecula": molecula,
        "tab_lineas": tab_lineas,
        "modelo": modelo,
        "tab_lineas_plot": tab_lineas_plot,
        "residuos": residuos,
        "parametros": params,
        "save_dir": save_dir,
        "modelo_radiativo": modelo_radiativo,
    }


# ============================================================
# MODELO DE UNA LISTA DE MOLÉCULAS
# ============================================================

def calcular_modelo_sintetico_lista(
        moleculas,
        region_name,
        tab_mol_config,
        resultados_parametros,
        fwhm_dict,
        dict_T_cont,
        dict_cubos_med,
        fuente_parametros="diagrot",
        modelo_radiativo="delgado",
        plot_lineas=False,
        show_plots=True,
        save_plots=True,
        guardar_solo_final=True,
        nombre_grupo=None,
        v_pik_dict = None,
        dict_sigma=None,
        nsigma_lineas=1.0,
        intervalos_mol_region=None,
        rutacarp_region = None,
        rutaregion_region = None):
    """
    Calcula un modelo sintético acumulado de varias moléculas.

    Parameters
    ----------
    moleculas : list
        Lista de moléculas internas.

    guardar_solo_final : bool
        Si True, solo guarda/enseña plots al añadir la última molécula.
        Si False, guarda también modelos parciales.
    """

    moleculas, es_lista = normalizar_lista_moleculas(moleculas)

    if len(moleculas) == 0:
        raise ValueError("La lista de moléculas está vacía.")

    if nombre_grupo is None:
        save_dir = path_synthetic_fig_dir(
            region_name=region_name,
            moleculas=moleculas,
            fuente_parametros=fuente_parametros,
            modelo_radiativo=modelo_radiativo,
            )
    else:
        save_dir = path_synthetic_fig_dir(
                   region_name=region_name,
                   moleculas=f"complete_model_{nombre_grupo}",
                   fuente_parametros=fuente_parametros,
                   modelo_radiativo=modelo_radiativo,
                   )

    plot_prefix = "complete_model"

    modelo_actual = None
    tab_lineas_plot_actual = None
    residuos_actual = None

    resultados_por_molecula = {}

    for i, mol in enumerate(moleculas):

        es_ultima = i == len(moleculas) - 1

        if guardar_solo_final:
            show_plots_i = show_plots and es_ultima
            save_plots_i = save_plots and es_ultima
        else:
            show_plots_i = show_plots
            save_plots_i = save_plots

        print(
            f"[spec_sint] Añadiendo {mol} al modelo "
            f"({i + 1}/{len(moleculas)})"
        )

        res = calcular_modelo_sintetico_molecula(
            molecula=mol,
            region_name=region_name,
            tab_mol_config=tab_mol_config,
            resultados_parametros=resultados_parametros,
            fwhm_dict=fwhm_dict,
            dict_T_cont=dict_T_cont,
            dict_cubos_med=dict_cubos_med,
            fuente_parametros=fuente_parametros,
            modeloin=modelo_actual,
            modelo_radiativo=modelo_radiativo,
            tab_lineas_mol=tab_lineas_plot_actual,
            plot_lineas=plot_lineas,
            show_plots=show_plots_i,
            save_plots=save_plots_i,
            save_dir=save_dir,
            plot_prefix=plot_prefix, v_pik_dict= v_pik_dict,
            dict_sigma= dict_sigma, nsigma_lineas= nsigma_lineas,
            intervalos_mol_region= intervalos_mol_region,
            rutacarp_region= rutacarp_region, 
            rutaregion_region= rutaregion_region)

        modelo_actual = res["modelo"]
        tab_lineas_plot_actual = res["tab_lineas_plot"]
        residuos_actual = res["residuos"]

        resultados_por_molecula[mol] = res

    return {
        "moleculas": moleculas,
        "modelo": modelo_actual,
        "tab_lineas_plot": tab_lineas_plot_actual,
        "residuos": residuos_actual,
        "resultados_por_molecula": resultados_por_molecula,
        "save_dir": save_dir,
        "fuente_parametros": fuente_parametros,
        "modelo_radiativo": modelo_radiativo,
    }


# ============================================================
# FUNCIÓN GENERAL
# ============================================================

def calcular_modelo_sintetico(
        moleculas,
        region_name,
        tab_mol_config,
        resultados_parametros,
        fwhm_dict,
        dict_T_cont,
        dict_cubos_med,
        fuente_parametros="diagrot",
        modelo_radiativo="delgado",
        plot_lineas=False,
        show_plots=True,
        save_plots=True,
        guardar_solo_final=False,
        v_pik_dict=None,
        dict_sigma=None,
        nsigma_lineas=1.0,
        intervalos_mol_region=None, rutacarp_region = None,
        rutaregion_region = None):
    """
    Calcula modelo sintético para una molécula o una lista de moléculas.

    Si recibe una lista con moléculas de distintas bandas/intervalos,
    separa automáticamente por la columna 'intervalo' de la tabla maestra
    y calcula un modelo independiente para cada banda.
    """

    mols, es_lista = normalizar_lista_moleculas(moleculas)

    if not es_lista:
        return calcular_modelo_sintetico_molecula(
            molecula=mols[0],
            region_name=region_name,
            tab_mol_config=tab_mol_config,
            resultados_parametros=resultados_parametros,
            fwhm_dict=fwhm_dict,
            dict_T_cont=dict_T_cont,
            dict_cubos_med=dict_cubos_med,
            fuente_parametros=fuente_parametros,
            plot_lineas=plot_lineas,
            show_plots=show_plots,
            save_plots=save_plots, v_pik_dict= v_pik_dict,
            modelo_radiativo=modelo_radiativo,
            dict_sigma=dict_sigma,
            nsigma_lineas=nsigma_lineas,
            intervalos_mol_region= intervalos_mol_region,
            rutacarp_region= rutacarp_region, 
            rutaregion_region= rutaregion_region
        )

    grupos = agrupar_moleculas_por_intervalo(
        moleculas=mols,
        tab_mol_config=tab_mol_config,
    )

    if len(grupos) == 1:
        return calcular_modelo_sintetico_lista(
            moleculas=moleculas,
            region_name=region_name,
            tab_mol_config=tab_mol_config,
            resultados_parametros=resultados_parametros,
            fwhm_dict=fwhm_dict,
            dict_T_cont=dict_T_cont,
            dict_cubos_med=dict_cubos_med,
            fuente_parametros=fuente_parametros,
            modelo_radiativo=modelo_radiativo,
            plot_lineas=plot_lineas,
            show_plots=show_plots,
            save_plots=save_plots,
            guardar_solo_final=guardar_solo_final,
            v_pik_dict=v_pik_dict,
            dict_sigma=dict_sigma,
            nsigma_lineas=nsigma_lineas,
            intervalos_mol_region= intervalos_mol_region,
            rutacarp_region= rutacarp_region, 
            rutaregion_region= rutaregion_region
            )        
    
    print("[spec_sint] Detectadas moléculas en varios intervalos/bandas:")
    for intervalo, mols_intervalo in grupos.items():
        print(f"  - {intervalo}: {mols_intervalo}")

    resultados_por_intervalo = {}

    for intervalo, mols_intervalo in grupos.items():

        print(
            f"[spec_sint] Calculando modelo completo para {intervalo}: "
            f"{mols_intervalo}"
        )

        resultados_por_intervalo[intervalo] = calcular_modelo_sintetico_lista(
            moleculas=mols_intervalo,
            region_name=region_name,
            tab_mol_config=tab_mol_config,
            resultados_parametros=resultados_parametros,
            fwhm_dict=fwhm_dict,
            dict_T_cont=dict_T_cont,
            dict_cubos_med=dict_cubos_med,
            fuente_parametros=fuente_parametros,
            modelo_radiativo=modelo_radiativo,
            plot_lineas=plot_lineas,
            show_plots=show_plots,
            save_plots=save_plots,
            guardar_solo_final=guardar_solo_final,
            nombre_grupo=intervalo,
            v_pik_dict=v_pik_dict,
            dict_sigma=dict_sigma,
            nsigma_lineas=nsigma_lineas,
            intervalos_mol_region= intervalos_mol_region,
            rutacarp_region= rutacarp_region, 
            rutaregion_region= rutaregion_region
        )

    return resultados_por_intervalo

def obtener_intervalo_molecula(tab_mol_config, molecula):
    """
    Devuelve el nombre simbólico del intervalo de una molécula.
    Ejemplo: 'Banda6' o 'Banda3'.
    """

    fila = seleccionar_fila_molecula(tab_mol_config, molecula)
    return str(fila["intervalo"]).strip()


def agrupar_moleculas_por_intervalo(moleculas, tab_mol_config):
    """
    Agrupa moléculas por la columna 'intervalo' de la tabla maestra.

    Returns
    -------
    grupos : dict
        Diccionario tipo:
        {
            "Banda6": [...],
            "Banda3": [...]
        }
    """

    grupos = {}

    for mol in moleculas:
        intervalo = obtener_intervalo_molecula(tab_mol_config, mol)

        if intervalo not in grupos:
            grupos[intervalo] = []

        grupos[intervalo].append(mol)

    return grupos