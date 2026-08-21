#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:49:15 2026

@author: jorge
"""

import TFM_config as cfg

from TFM_io_cubes import load_mean_cubes, load_full_cubes

def seleccionar_region(nombre_region):
    """
    Devuelve la configuración asociada a una región.
    """

    if nombre_region not in cfg.regiones:
        raise ValueError(
            f"Región '{nombre_region}' no disponible. "
            f"Opciones válidas: {list(cfg.regiones.keys())}"
        )

    config_region = cfg.regiones[nombre_region]

    print("[region] Región seleccionada:")
    print(f"  {nombre_region}")

    print("[region] Archivo DS9:")
    print(f"  {config_region['ruta']}")

    print("[region] Modo de carga:")
    print(f"  cargar_promedio = {config_region['cargar_promedio']}")
    print(f"  cargar_completo = {config_region['cargar_completo']}")

    return config_region


def cargar_datos_region(config_region):
    """
    Carga los espectros promedio y/o cubos completos según la configuración
    de la región seleccionada.
    """

    rutaregion = config_region["ruta"]

    if "rutacarp" not in config_region:
        raise KeyError(
            "La región seleccionada no tiene definida la clave 'rutacarp'."
        )

    if "ventanas_obs" not in config_region:
        raise KeyError(
            "La región seleccionada no tiene definida la clave 'ventanas_obs'."
        )

    rutacarp_region = config_region["rutacarp"]
    ventanas_region = config_region["ventanas_obs"]

    cargar_promedio = config_region["cargar_promedio"]
    cargar_completo = config_region["cargar_completo"]

    print("[cubos] Carpeta de datos usada:")
    print(f"  {rutacarp_region}")

    print("[cubos] Archivo de región usado:")
    print(f"  {rutaregion}")

    print("[cubos] Ventanas usadas:")
    print(f"  {[v[3] for v in ventanas_region]}")

    dict_cubos_med = None
    dict_resol_esp = None
    dict_cubos_comp = None

    if cargar_promedio:
        print("\n[cubos] Cargando espectros promedio...")

        dict_cubos_med, dict_resol_esp = load_mean_cubes(
            rutacarp_region,
            rutaregion,
            ventanas_region,
        )

        print("[cubos] Espectros promedio cargados:")
        print(f"  {list(dict_cubos_med.keys())}")

    if cargar_completo:
        print("\n[cubos] Cargando cubos completos...")

        dict_cubos_comp = load_full_cubes(
            rutacarp_region,
            rutaregion,
            ventanas_region,
        )

        print("[cubos] Cubos completos cargados:")
        print(f"  {list(dict_cubos_comp.keys())}")

    if not cargar_promedio and not cargar_completo:
        raise ValueError(
            "La región seleccionada no tiene ningún modo de carga activo."
        )

    return dict_cubos_med, dict_resol_esp, dict_cubos_comp