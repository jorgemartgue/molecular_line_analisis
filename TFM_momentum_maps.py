#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:37:21 2026

@author: jorge
"""

from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np

from regions import Regions
from spectral_cube import SpectralCube

import TFM_config as cfg
from TFM_line_search import mod_frec

def map_intens_int(frec_busc, v_busc, intervalos, molecula,
                   long_int=10*u.km/u.s, mapa=True,
                   rutacarp_region=None,
                   rutaregion_region=None):
    '''
Parameters
----------
frec_busc : u.Quantity
    Frequency to search for. Must have units convertible to Hz.

v_busc : u.Quantity
    Velocity of the source. Must have units convertible to km/s.

intervalos : dict
    Dictionary containing the possible frequency intervals where the line
    may be located. Each entry must follow the structure:
    ('file_name', nu_min, nu_max, 'short_label'),
    where nu_min and nu_max are frequencies in Hz or convertible units.

molecula : str
    Name of the molecule.

long_int : u.Quantity, optional
    Velocity interval used to compute the moments. Must have units
    convertible to km/s. Default is 10 * u.km / u.s.

mapa : bool, optional
    If True, plots the moment maps. Default is True.

Returns
-------
momento0 : ndarray
    Array representing moment 0 for each pixel in the region.

momento1 : ndarray
    Array representing moment 1 for each pixel in the region.

subcubeslab_sinc_mask : spectral_cube.SpectralCube
    Subcube obtained after applying a 3-sigma detection filter.

Notes
-----
If `mapa=True`, two plots are generated:
    - Moment 0 map
    - Moment 1 map
    '''
    
    if rutacarp_region is None:
        rutacarp_region = cfg.rutacarp

    if rutaregion_region is None:
        rutaregion_region = cfg.rutaregion
        
    if not isinstance(
            frec_busc, u.Quantity) or not frec_busc.unit.is_equivalent(u.MHz):
        raise u.UnitConversionError(
            f"'frec_busc' debe tener unidades de frecuencia. "
            f"Recibido: {frec_busc.unit}"
        )

    if not isinstance(v_busc, u.Quantity) or not v_busc.unit.is_equivalent(
            u.km/u.s):
        raise u.UnitConversionError(
            f"'v_busc' debe tener unidades de velocidad. "
            f"Recibido: {v_busc.unit}"
        )

    if not isinstance(
            long_int, u.Quantity) or not long_int.unit.is_equivalent(u.km/u.s):

        raise u.UnitConversionError(
            f"'long_int' debe tener unidades de velocidad. "
            f"Recibido: {long_int.unit}"
        )

    # Aquí terminan los sistemas de seguridad
    frec_busc = frec_busc.to(u.MHz)

    nombre = None

    # Primero vamos a ver en que ventana está
    for ventana, fmin, fmax, name_window in intervalos:
        if fmin <= frec_busc <= fmax:
            # Esta variable nos ayudará a abrir el cubo correspondiente
            nombre = ventana.strip()
            nombre_window = name_window

    if nombre is None:
        return print(
            f'La frecuencia buscada, {frec_busc.value}{frec_busc.unit} '
            'no se encuentra en ninguna ventana')

    # Representación de la línea preseleccionada

    archivos = sorted(rutacarp_region.glob(f"*{nombre}*"))

    if len(archivos) == 0:
        raise FileNotFoundError(
            f"No se encontró ningún cubo para {nombre} en {rutacarp_region}"
        )

    ruta_cubo = archivos[0]

    cube = SpectralCube.read(ruta_cubo)

    # Se lee la región
    region = Regions.read(rutaregion_region, format='ds9')

    # Recortamos el cubo su propia figura
    subcube = cube.subcube_from_regions(region)
    subcube.allow_huge_operations = True

    # Pasamos a las unidades solicitadas
    subcubeK = subcube.to(u.K)
    subcubeK = subcubeK.with_spectral_unit(u.km/u.s,
                                           velocity_convention='radio',
                                           rest_value=frec_busc)

    subcubeslab = subcubeK.spectral_slab(v_busc-long_int/2, v_busc+long_int/2)

    # esto de ahora es para restar el continuo

    off1 = subcubeK.spectral_slab(v_busc-3*long_int, v_busc-1.5*long_int)
    off2 = subcubeK.spectral_slab(v_busc+1.5*long_int, v_busc+3*long_int)

    cont1 = off1.median(axis=0)
    cont2 = off2.median(axis=0)

    continuo = 0.5 * (cont1 + cont2)

    # Calculo el ruido
    sig1 = off1.std(axis=0)
    sig2 = off2.std(axis=0)
    dispersion = np.sqrt(0.5 * (sig1**2 + sig2**2))

    subcubeslab_sincont = subcubeslab - continuo

    subcubeslab_sinc_mask3 = subcubeslab_sincont.with_mask(
        subcubeslab_sincont > 2*dispersion)

    subcubeslab_sinc_mask2 = subcubeslab_sincont.with_mask(
        subcubeslab_sincont > 2*dispersion)

    momento0 = subcubeslab_sinc_mask3.moment(order=0)
    momento1 = subcubeslab_sinc_mask2.moment(order=1)

    if mapa is True:

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection=momento0.wcs.celestial)

        im = ax.imshow(momento0.value, origin="lower", cmap="inferno")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(str(momento0.unit))

        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")

        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')

        ax.coords[0].set_ticks(spacing=0.2*u.arcsec)
        ax.coords[1].set_ticks(spacing=0.2*u.arcsec)

        ax.coords[0].display_minor_ticks(True)
        ax.coords[1].display_minor_ticks(True)

        ax.coords[0].set_major_formatter('hh:mm:ss.ss')
        ax.coords[1].set_major_formatter('dd:mm:ss.s')

        ax.coords.grid(True, color='white', ls=':', alpha=0.5)

        ax.set_title(f"M0 – {molecula} - {frec_busc} - {name_window}")

        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection=momento1.wcs.celestial)

        im = ax.imshow(momento1.value, origin="lower", cmap="inferno",
                       vmin=59, vmax=63)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(str(momento1.unit))

        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")

        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')

        ax.coords[0].set_ticks(spacing=0.2*u.arcsec)
        ax.coords[1].set_ticks(spacing=0.2*u.arcsec)

        ax.coords[0].display_minor_ticks(True)
        ax.coords[1].display_minor_ticks(True)

        ax.coords[0].set_major_formatter('hh:mm:ss.ss')
        ax.coords[1].set_major_formatter('dd:mm:ss.s')

        ax.coords.grid(True, color='white', ls=':', alpha=0.5)

        ax.set_title(f"M1 – {molecula} - {frec_busc} - {nombre_window}")

        plt.tight_layout()
        plt.show()
    return momento0, momento1, subcubeslab_sinc_mask3