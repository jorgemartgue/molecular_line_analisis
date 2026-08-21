#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:00:08 2026

@author: jorge
"""

from astropy import units as u
from regions import Regions
from spectral_cube import SpectralCube

def read_spectral_cube(ruta_cubo, ruta_region, T_unit=u.K, spec_unit=u.GHz, 
                 promedio = True):
    '''
    Esta función lee el cubo con ruta (ruta_cubo) y devuelve el eje 
    espectral en spec_unit y el espectro promedio en T_unit.
    '''

    # Se lee el cubo
    cube = SpectralCube.read(ruta_cubo)
    header = cube.header
    spec_resol = header['CDELT3'] * u.Hz

    # Se lee la región
    region = Regions.read(ruta_region, format='ds9')

    # Recortamos el cubou propia figura
    subcube = cube.subcube_from_regions(region)
    subcube.allow_huge_operations = True

    # Pasamos a las unidades solicitadas
    subcubeK = subcube.to(T_unit)
    subcubeK = subcubeK.with_spectral_unit(spec_unit)

    if promedio:
        
        # Calculamos el espectro promedio
        espectro = subcubeK.mean(axis=(1, 2))
        return subcubeK.spectral_axis, espectro, spec_resol
        
    
    return subcubeK.spectral_axis, subcubeK, spec_resol

def load_mean_cubes(rutacarp, rutaregion, ventanas_obs,
                    T_unit=u.K, spec_unit=u.GHz):
    """
    Carga todos los cubos de ventanas_obs recortados a la región indicada
    y devuelve los espectros promedio.

    Returns
    -------
    dict_cubos_med : dict
        Diccionario con frecuencia y temperatura promedio por ventana.

    dict_resol_esp : dict
        Diccionario con resolución espectral por ventana.
    """

    dict_cubos_med = {}
    dict_resol_esp = {}

    for ventana, fmin, fmax, nombre_vent in ventanas_obs:
        nombre = ventana.strip()

        archivos = list(rutacarp.glob(f"*{nombre}*"))

        if len(archivos) == 0:
            raise FileNotFoundError(
                f"No se encontró ningún cubo para la ventana {nombre_vent} "
                f"con patrón '*{nombre}*' en {rutacarp}"
            )

        ruta_cubo = archivos[0]

        frec, espec, spec_resol = read_spectral_cube(
            ruta_cubo,
            rutaregion,
            T_unit=T_unit,
            spec_unit=spec_unit,
            promedio=True
        )

        dict_cubos_med[nombre_vent] = {
            "frecuencia": frec.to(u.MHz),
            "Temp_brillo": espec,
        }

        dict_resol_esp[nombre_vent] = spec_resol

    return dict_cubos_med, dict_resol_esp


def load_full_cubes(rutacarp, rutaregion, ventanas_obs,
                    T_unit=u.K, spec_unit=u.GHz):
    """
    Carga todos los cubos de ventanas_obs recortados a la región indicada
    sin promediar espacialmente.

    Returns
    -------
    dict_cubos_comp : dict
        Diccionario con frecuencia y cubo completo por ventana.
    """

    dict_cubos_comp = {}

    for ventana, fmin, fmax, nombre_vent in ventanas_obs:
        nombre = ventana.strip()

        archivos = list(rutacarp.glob(f"*{nombre}*"))

        if len(archivos) == 0:
            raise FileNotFoundError(
                f"No se encontró ningún cubo para la ventana {nombre_vent} "
                f"con patrón '*{nombre}*' en {rutacarp}"
            )

        ruta_cubo = archivos[0]

        frec, espec, spec_resol = read_spectral_cube(
            ruta_cubo,
            rutaregion,
            T_unit=T_unit,
            spec_unit=spec_unit,
            promedio=False
        )

        dict_cubos_comp[nombre_vent] = {
            "frecuencia": frec.to(u.MHz),
            "Temp_brillo": espec,
        }

    return dict_cubos_comp