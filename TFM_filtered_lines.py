#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:49:39 2026

@author: jorge
"""

"""
Módulo para cargar o calcular tablas filtradas de líneas moleculares.

Una tabla filtrada contiene las líneas de una molécula ya pasadas por:
- consulta a Splatalogue,
- filtros de estructura,
- filtros físicos,
- detección en mapas,
- ajuste de línea,
- intensidad integrada, deltaW, FWHM, vlin y continuo.

Estas tablas son la entrada directa del diagrama rotacional.
"""

from pathlib import Path
import pickle

from astropy import units as u
from astropy.table import Table

import TFM_config as cfg

from TFM_filtering import filtrador, filtrador_por_pixel
from TFM_runtime import build_config_filtrador_from_table
from TFM_storage import ensure_dir, load_table, save_table


def seleccionar_tabla_molecula(tab_mol_config, molecula):
    """
    Selecciona de la tabla maestra de moléculas solo la fila de una molécula.

    Parameters
    ----------
    tab_mol_config : QTable
        Tabla maestra tables/config/moleculas_config.ecsv.

    molecula : str
        Nombre interno de la molécula, por ejemplo 'C2H5CN'.

    Returns
    -------
    tab_mol : QTable
        Tabla con una única fila.
    """

    mask = tab_mol_config["nombre"] == molecula
    tab_mol = tab_mol_config[mask]

    if len(tab_mol) == 0:
        raise KeyError(
            f"La molécula {molecula} no está en moleculas_config.ecsv. "
            f"Moléculas disponibles: {list(tab_mol_config['nombre'])}"
        )

    if len(tab_mol) > 1:
        raise ValueError(
            f"La molécula {molecula} aparece más de una vez en "
            "moleculas_config.ecsv."
        )

    return tab_mol


def path_tabla_filtrada(region_name, molecula, base_dir=None):
    """
    Devuelve la ruta donde se guarda la tabla filtrada de una molécula.

    Se guarda por región porque la tabla filtrada depende del espectro,
    del continuo y de los ajustes de esa región.

    Ejemplo:
        tables/filtradas/MF2/C2H5CN.ecsv
    """

    if base_dir is None:
        base_dir = cfg.rutatablas

    path_dir = Path(base_dir) / "filtradas" / region_name
    ensure_dir(path_dir)

    return path_dir / f"{molecula}.ecsv"


def ejecutar_filtrador_desde_config(
        molecula,
        config_filtrador,
        plots=True,
        rutacarp_region=None,
        rutaregion_region=None,
        tab_catalogo=None):
    """
    Ejecuta filtrador() usando una entrada de CONFIG_FILTRADOR.
    """

    if molecula not in config_filtrador:
        raise KeyError(
            f"{molecula} no está en CONFIG_FILTRADOR. "
            f"Claves disponibles: {list(config_filtrador.keys())}"
        )

    c = config_filtrador[molecula]

    tab_filtrada = filtrador(
        c["mol"],
        c["intervalo"],
        c["E_max"],
        c["aij_min"],
        c["sijmu2_min"],
        id_splat1=c.get("id_splat1"),
        filtro_estructurasf=c.get("filtro_estructurasf"),
        list_freq_nofilt=c.get("list_freq_nofilt"),
        filt_inter=c["filt_inter"],
        Tcontf=c.get("Tcont"),
        anch_fix=c.get("anch_fix"),
        linelistf=c["linelist"],
        dict_especf=c.get("dict_espec"), plots = plots,
        v_filt=c.get("v_filt", 63 * u.km / u.s),
        rutacarp_region= rutacarp_region,
        rutaregion_region= rutaregion_region,
        tab_catalogo=tab_catalogo,
    )

    return tab_filtrada


def cargar_o_calcular_tabla_filtrada_molecula(
        molecula,
        region_name,
        tab_mol_config,
        dict_cubos_med,
        dict_T_cont,
        fwhm_dict,
        recalcular=False,
        base_dir=None,
        plots=True,
        v_filt_override=None,
        rutacarp_region=None,
        rutaregion_region=None,
        intervalos_mol_region=None,
        tab_catalogo=None):
    """
    Carga o calcula la tabla filtrada de una molécula.

    Parameters
    ----------
    molecula : str
        Nombre interno, por ejemplo 'C2H5CN'.

    region_name : str
        Nombre de la región, por ejemplo 'MF2'.

    tab_mol_config : QTable
        Tabla maestra de configuración de moléculas.

    dict_cubos_med : dict
        Diccionario de espectros promedio cargados con load_mean_cubes().

    dict_T_cont : dict
        Diccionario de continuo por ventana.

    fwhm_dict : dict
        Diccionario de FWHM por molécula.

    recalcular : bool
        Si False, carga la tabla si existe.
        Si True, fuerza recalcular aunque exista.

    base_dir : Path or None
        Carpeta base de tablas. Por defecto cfg.rutatablas.

    Returns
    -------
    tab_filtrada : QTable
        Tabla filtrada lista para diagrama_rotacional().
    """

    path = path_tabla_filtrada(region_name, molecula, base_dir=base_dir)

    if path.exists() and not recalcular:
        print(f"[filtrado] Cargando tabla filtrada existente: {path}")
        tab_filtrada = load_table(path)

    else:
        print(f"[filtrado] Calculando tabla filtrada para {molecula}")

        tab_mol = seleccionar_tabla_molecula(tab_mol_config, molecula)

        config_filtrador = build_config_filtrador_from_table(tab_mol,
                           region_name=region_name,
                           dict_cubos_med=dict_cubos_med,
                           dict_T_contv2=dict_T_cont,
                           fwhm=fwhm_dict,
                           intervalos_mol_region=intervalos_mol_region)

        if v_filt_override is not None:
            config_filtrador[molecula]["v_filt"] = v_filt_override

        tab_filtrada = ejecutar_filtrador_desde_config(
            molecula,
            config_filtrador,
            plots=plots,
            rutacarp_region=rutacarp_region,
            rutaregion_region=rutaregion_region,
            tab_catalogo=tab_catalogo,
        )

        if len(tab_filtrada) == 0:
            raise ValueError(
                f"La tabla filtrada de {molecula} está vacía. "
                "Revisa id_splat, filtros, FWHM, continuo o frecuencias "
                "no consideradas."
            )

        save_table(tab_filtrada, path)
        print(f"[filtrado] Tabla filtrada guardada en: {path}")

    print(f"[filtrado] Nº de líneas filtradas para {molecula}: {len(tab_filtrada)}")

    return tab_filtrada

def cargar_o_calcular_tablas_filtradas_pixeles(
        molecula,
        region_name,
        tab_mol_config,
        dict_cubos_med,
        dict_T_cont_med,
        dict_cubos_comp,
        dict_T_cont_pix,
        fwhm_dict,
        v_pik_map,
        fwhm_map,
        recalcular=False,
        base_dir=None,
        plots=True,
        v_filt_override=None,
        rutacarp_region=None,
        rutaregion_region=None,
        intervalos_mol_region=None,
        tab_catalogo=None):
    """
    Carga o calcula las tablas filtradas píxel a píxel.

    Usa la tabla filtrada media como tab_info y luego mide las líneas
    píxel a píxel con filtrador_por_pixel().
    """



    if base_dir is None:
        base_dir = cfg.rutatablas

    path_dir = Path(base_dir) / "filtradas_pixeles" / region_name
    ensure_dir(path_dir)

    path = path_dir / f"{molecula}.pkl"

    if path.exists() and not recalcular:
        print(f"[filtrado_pix] Cargando tablas por píxel existentes: {path}")

        with open(path, "rb") as f:
            tab_pixeles = pickle.load(f)

        return tab_pixeles

    print(f"[filtrado_pix] Calculando tablas filtradas por píxel para {molecula}")

    tab_mol = seleccionar_tabla_molecula(
        tab_mol_config,
        molecula,
    )

    config_filtrador = build_config_filtrador_from_table(
        tab_mol,
        region_name=region_name,
        dict_cubos_med=dict_cubos_comp,
        dict_T_contv2=dict_T_cont_pix,
        fwhm=fwhm_dict,
        intervalos_mol_region=intervalos_mol_region,
    )

    if molecula not in config_filtrador:
        raise KeyError(f"{molecula} no está en CONFIG_FILTRADOR.")

    c = config_filtrador[molecula]

    # Tabla de líneas seleccionadas que se usará como base para medir píxel a píxel
    tab_info = cargar_o_calcular_tabla_filtrada_molecula(
        molecula=molecula,
        region_name=region_name,
        tab_mol_config=tab_mol_config,
        dict_cubos_med=dict_cubos_med,
        dict_T_cont=dict_T_cont_med,
        fwhm_dict=fwhm_dict,
        recalcular=recalcular,
        base_dir=base_dir,
        plots=plots,
        rutacarp_region=rutacarp_region,
        rutaregion_region=rutaregion_region,
        intervalos_mol_region=intervalos_mol_region,
        tab_catalogo=tab_catalogo)

    tab_pixeles = filtrador_por_pixel(
        tab_info,
        c["intervalo"],
        v_busc=v_filt_override,
        long_int=20 * u.km / u.s,
        Tcontf=dict_T_cont_pix,
        anch_fix=c.get("anch_fix"),
        v_pik_map=v_pik_map,
        fwhm_map=fwhm_map,
        dict_especf=dict_cubos_comp,
        plots=plots,
    )

    with open(path, "wb") as f:
        pickle.dump(tab_pixeles, f)

    print(f"[filtrado_pix] Tablas por píxel guardadas en: {path}")

    return tab_pixeles