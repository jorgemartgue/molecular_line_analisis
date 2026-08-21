#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:02:35 2026

@author: jorge
"""

from astropy import units as u
from astropy import constants as c
from astropy.modeling import models, fitting
from astropy.table import QTable, Table, unique, vstack
from astroquery.splatalogue import Splatalogue
from astroquery.linelists.cdms import CDMS
from astroquery.jplspec import JPLSpec
from pathlib import Path
import re
from regions import Regions
from spectral_cube import SpectralCube
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import os

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

# Función para ajustar las gaussianas
def gaussiana(x, A, x0, sigma, C):
    return A*np.exp((x-x0)**2 / (2*sigma**2))+C

# Función para usar mod vel


def mod_vel(v, nu_obs):
    nu0 = nu_obs*(1+v/c.c)
    return nu0


def mod_frec(frec_lab, frec_obs):

    v = c.c*(frec_lab/frec_obs - 1)

    return v

# Buscador de líneas


def buscador_lin(frec_busc, intervalos, long_int='tot', ajuste=False,
                 v=None, dict_espec=None):
    '''
    Function used for searching a frecuency of interest in a list of possible
    interval.

    Parameters
    ----------
    frec_busc : u.Quantity
        This is the frecuency you are searching for. It must be on frecuency 
        units.

    intervalos : Posible intervals your frecuency may be. They are thought to 
        be defined as [('name_interval', frec_min,frec_max)].

    long_int : u.Quantity or string

        This is the lenght of the interval in frecuency you want to represent.
        If you want to represent all of the spectrum, don't include this input
        or stablish it as 'tot'. The default for this input is 'tot'.

    ajuste: Boolean

        If you want to create a gaussian fit to your interval introduce True

    v: u.Quantity or None

        If you want to modify the frecuency because the doppler effect, 
        introduce a velocity in velocity units for the script to adjust it
    Returns
    -------

    As a return it will provide a plot with the spectrum of the cube that 
    contains the frecuency you are searching for.


MEJORAS QUE HAY QUE HACER:

  --HACER FUNCIONAR EL AJUSTE GAUSSIANO
  --HACER LO DE CONVERSION A VELOCIDADES
    '''
    # Comprobaciones de seguridad para que la función no se rompa
    if not isinstance(
            frec_busc, u.Quantity) or not frec_busc.unit.is_equivalent(u.MHz):
        raise u.UnitConversionError(
            f"'frec_busc' debe tener unidades de frecuencia. "
            f"Recibido: {frec_busc.unit}"
        )

    if not long_int == 'tot':
        if not isinstance(
                long_int, u.Quantity) or not long_int.unit.is_equivalent(u.MHz):

            raise u.UnitConversionError(
                f"'long_int' debe tener unidades de frecuencia. "
                f"Recibido: {long_int.unit}"
            )
        long_int = long_int.to(u.MHz)

    if ajuste and long_int == 'tot':
        raise ValueError(
            "Si ajuste=True, debes proporcionar long_int distinto de 'tot'")

    if not v is None:
        if not isinstance(v, u.Quantity) or not v.unit.is_equivalent(u.km/u.s):
            raise u.UnitConversionError(
                f"'v' debe tener unidades de velocidad. "
                f"Recibido: {v.unit}"
            )

    # Aquí terminan los sistemas de seguridad
    frec_busc = frec_busc.to(u.MHz)

    nombre = None

    # Primero vamos a ver en que ventana está
    for ventana, fmin, fmax, name_window in intervalos:
        if fmin <= frec_busc <= fmax:
            # Esta variable nos ayudará a abrir el cubo correspondiente
            nombre = ventana.strip()
            nombre_ventana = name_window

    if nombre is None:
        return print(
            f'La frecuencia buscada, {frec_busc.value}{frec_busc.unit} '
            'no se encuentra en ninguna ventana')

    # Representación de la línea preseleccionada

    if dict_espec is None:

        archivos = list(rutacarp.glob(f"*{nombre}*"))

        ruta_cubo = archivos[0]

        frec, espec, res = read_spectral_cube(ruta_cubo, rutaregion1)

        frec = frec.to(u.MHz)

    else:

        frec = dict_espec[nombre_ventana]['frecuencia']
        espec = dict_espec[nombre_ventana]['Temp_brillo']

    if not v is None:

        frec = mod_vel(v, frec)

    # Ajuste de las gaussianas
    if ajuste is True:
        # máscara que nos ayudará a coger solo los valores del intervalo
        # estudiado

        mask = (frec.value >= frec_busc.value-long_int.value/2) & (
            frec.value <= frec_busc.value+long_int.value/2)

        xdat0 = frec.value[mask]
        ydat0 = espec.value[mask]

        # Vamos a buscar ahora el pico de la línea

        ymax = np.max(ydat0)
        mask2 = (ydat0 == ymax)
        xmax = xdat0[mask2]

        # Creamos nuevo intervalo alrededor de la línea

        mask3 = (frec.value >= xmax-long_int.value/2) & (frec.value <=
                                                         xmax+long_int.value/2)

        xdat1 = frec.value[mask3]
        ydat1 = espec.value[mask3]

        # Ahora vamos a intentar sacar el valor del continuo

        Tcont = np.median(ydat1)
        cont = models.Const1D(Tcont)

        # Realizamos el ajuste gaussiano ATENCION, el valor inicial de stddev
        # es completamente arbitrario, he probado con algunas líneas y funciona
        # pero no sé si en un futuro se va a romper

        gaus0 = models.Gaussian1D(amplitude=ymax, mean=xmax, stddev=1)+cont

        gfit = fitting.LevMarLSQFitter()

        gausfit = gfit(gaus0, xdat1, ydat1)

        Tcont = gausfit.amplitude_1 * u.K
        Tmax = gausfit.amplitude_0 * u.K
        freclin = gausfit.mean_0 * u.MHz
        sigma = gausfit.stddev_0 * u.MHz
        FWHM = 2*np.sqrt(2*np.log(2))*sigma

        int_integrada = Tmax * sigma * np.sqrt(2*np.pi)

        frec_busc = freclin

    # Representación:

    plt.figure()

    plt.plot(frec, espec.value)

    plt.axvline(frec_busc.value, linestyle='--', color='g',
                label=f'Línea buscada: \n{frec_busc.value:.3f}'
                f'{frec_busc.unit}')

    plt.xlabel(str(frec.unit))
    plt.ylabel(str(espec.unit))

    if ajuste is True:

        xfit = np.linspace(np.min(xdat1), np.max(xdat1), 10000)
        plt.plot(xfit, gausfit(xfit), 'r',
                 label='ajuste gaussiano')

    if not long_int == 'tot':

        plt.xlim(frec_busc.value-long_int.value/2,
                 frec_busc.value+long_int.value/2)

    plt.title(f'Espectro promedio del cubo {nombre_ventana}')
    plt.legend(loc='upper right', fontsize=8)
    plt.gca().ticklabel_format(useOffset=False)
    plt.show()

    if ajuste is True:

        return Tmax, freclin, Tcont, sigma, FWHM, int_integrada
    
def busc_mult_lin(list_frec, interval, list_long, fit=False, v_mult=None,
                  tab=False, dict_especm=None):
    '''
    This function is the vectorizacion of the function buscador_lin.

    Parameters
    ----------
    list_frec : list of u.Quantity
        This is the list of frecuencies you are searching for.
        It must be on frecuency units.

    interval : Posible intervals your frecuency may be. They are thought to 
        be defined as [('name_interval', frec_min,frec_max)].

    list_long: list of u.Quantity or string

        This is the list of the lenghts of the intervals in frecuency you want 
        to represent.
        If you want to represent all of the spectrum, instead of a lenght in 
        frecuency insert 'tot'.

    fit: Boolean

       If you want to do a gaussian fit of all the frecuencies set it as True

    v_mult: u.Quantity or None

        If you want to modify the frecuency because the doppler effect, 
        introduce a velocity in velocity units for the script to adjust it

    tab: Boolean

       If you want to generate a table with the results of the fit, set it as 
       True. The table will have 6 colums: frecuency of the line, T of the 
       line, T of the continuum, sigma, FWHM and the integrated intensity

    Returns
    -------

    As a return the function will provide a sort of plots with the spectrum of
    the cubes that contains the frecuencies you are searching for. 

    If the tab input was setted as True, the function also returns a QTable 
    with the colums described before.

    '''
    # Seguridad para comprobar que len(list_frec) == len()

    if not len(list_frec) == len(list_long):
        raise ValueError('Debes introducir dos listas (list_frec y list_long) '
                         'de igual tamaño')

    if tab is True:

        Tab = QTable(
            names=('frecuencia', 'T de la línea', 'T cont', 'sigma',
                   'FWHM', 'intensidad integrada'),
            units=(u.MHz, u.K, u.K, u.MHz, u.MHz, u.K*u.MHz)
        )
    for fb, li in zip(list_frec, list_long):
        print(f'Buscando la frecuencia {fb.value}{fb.unit}....')

        if tab is True:

            Tlin, frec, Tcont, sigma, FWHM, int_integ = buscador_lin(fb,
                                                        interval, long_int=li,
                                                        ajuste=fit, v=v_mult,
                                                        dict_espec=dict_especm)

            Tab.add_row((frec, Tlin, Tcont, sigma, FWHM, int_integ))

        else:

            buscador_lin(fb, interval, long_int=li, ajuste=fit, v=v_mult,
                         dict_espec=dict_especm)

        print(f'Proceso de busqueda de la frecuencia {fb.value}{fb.unit} '
              'finalizado.')

    if tab is True:
        return Tab
    
    
def buscador_lin_cubo_compl(frec_busc_c, intervalos_c, long_int_c='tot', 
                            ajuste_c=False, v_c=None, dict_espec_c=None):
    
    # Comprobaciones de seguridad para que la función no se rompa
    if not isinstance(
            frec_busc_c, u.Quantity) or not frec_busc_c.unit.is_equivalent(u.MHz):
        raise u.UnitConversionError(
            f"'frec_busc_c' debe tener unidades de frecuencia. "
            f"Recibido: {frec_busc_c.unit}"
        )

    if not long_int_c == 'tot':
        if not isinstance(
                long_int_c, u.Quantity) or not long_int_c.unit.is_equivalent(u.MHz):

            raise u.UnitConversionError(
                f"'long_int' debe tener unidades de frecuencia. "
                f"Recibido: {long_int_c.unit}"
            )
        long_int_c = long_int_c.to(u.MHz)

    if ajuste_c and long_int_c == 'tot':
        raise ValueError(
            "Si ajuste=True, debes proporcionar long_int distinto de 'tot'")

    if not v_c is None:
        if not isinstance(v_c, u.Quantity) or not v_c.unit.is_equivalent(u.km/u.s):
            raise u.UnitConversionError(
                f"'v' debe tener unidades de velocidad. "
                f"Recibido: {v_c.unit}"
            )    

     #Primero vamos a ver en que ventana está la frecuencia buscadda
     
    frec_busc_c = frec_busc_c.to(u.MHz)
     
    nombre = None

     # Primero vamos a ver en que ventana está
    for ventana, fmin, fmax, name_window in intervalos_c:
        if fmin <= frec_busc_c <= fmax:
            # Esta variable nos ayudará a abrir el cubo correspondiente
            nombre = ventana.strip()
            nombre_ventana = name_window

    if nombre is None:
        return print(
             f'La frecuencia buscada, {frec_busc_c.value}{frec_busc_c.unit} '
             'no se encuentra en ninguna ventana')
     
    #Cargamos el cubo donde se encuentra la línea
    
    if dict_espec_c is None:
        
        archivos = list(rutacarp.glob(f"*{nombre}*"))

        ruta_cubo = archivos[0]

        frec, espec, res = read_spectral_cube(ruta_cubo, rutaregion1, 
                                              promedio=False)

        frec = frec.to(u.MHz)

    else:

        frec = dict_espec_c[nombre_ventana]['frecuencia']
        espec = dict_espec_c[nombre_ventana]['Temp_brillo']
        
        
    #Ahora vamos con la búsqueda de la línea en cada píxel
    
    nchan, ny, nx = espec.shape
    if ajuste_c:
        
        Tmax_pix = np.full((ny, nx), np.nan)
        freclin_pix = np.full((ny, nx), np.nan)
        Tcont_pix = np.full((ny, nx), np.nan)
        sigma_pix = np.full((ny, nx), np.nan)
        FWHM_pix = np.full((ny, nx), np.nan)
        int_integrada_pix = np.full((ny, nx), np.nan)

    for y in range(ny):
        for x in range(nx):
            espec_pixel = espec[:, y, x]

            cubo_pixel = {
                nombre_ventana: {
                    'frecuencia': frec,
                    'Temp_brillo': espec_pixel
                }
            }
           
            if ajuste_c:
                
                (Tmax, freclin, Tcont, sigma, 
                 FWHM, int_integrada) = buscador_lin(frec_busc_c, intervalos_c,
                                                     long_int=long_int_c, 
                                                     ajuste=ajuste_c, v= v_c,
                                                     dict_espec=cubo_pixel)
                
                Tmax_pix[y,x] = Tmax.value
                freclin_pix[y,x] = freclin.value
                Tcont_pix[y,x] = Tcont.value
                sigma_pix[y,x] = sigma.value
                FWHM_pix[y,x] = FWHM.value
                int_integrada_pix[y,x] = int_integrada.value
                
            else:
                
                buscador_lin(frec_busc_c, intervalos_c,long_int=long_int_c,
                             ajuste=ajuste_c, v=v_c, dict_espec=cubo_pixel)
                
    if ajuste_c:
        
        return (Tmax_pix, freclin_pix, Tcont_pix, sigma_pix, FWHM_pix, 
                int_integrada_pix)
    
def buscador_lin_vel(frec_busc, v_busc, intervalos, long_int='tot',
                     ajuste=False, Tcont_fix=None, anch_lin=None,
                     dict_espec=None):
    '''
    Function used for searching a frecuency of interest in a list of possible
    interval.

    Parameters
    ----------
    frec_busc : u.Quantity
        This is the frecuency you are searching for. It must be on frecuency 
        units.

    v_busc: u.Quantity
        This is the velocity of your 

    intervalos : Posible intervals your frecuency may be. They are thought to 
        be defined as [('name_interval', frec_min,frec_max)].

    long_int : u.Quantity or string

        This is the lenght of the interval in velocity you want to represent.
        If you want to represent all of the spectrum, don't include this input
        or stablish it as 'tot'. The default for this input is 'tot'.

    ajuste: Boolean

        If you want to create a gaussian fit to your interval introduce True

    Tcont_fix: u.Quantity

        If you want to set the continuum temperature for the gaussian fit set 
        this parameter with the continuum temperature in u.K.

        Recomendation: Use this code with calibration lines for detecting wich 
        is the continuum temperature, and then use this value for the rest of 
        your fits.

    anch_lin: u.Quantity

        If you want to set the FWHM of the lines for the gaussian fit, set this
        parameter with the FWHM you want in u.km/u.sK


    Returns
    -------

    As a return it will provide a plot with the spectrum of the cube that 
    contains the frecuency you are searching for and if you had set the fit in 
    True, it returns the fitting values.

    '''
    # Comprobaciones de seguridad para que la función no se rompa
    if not isinstance(
            frec_busc, u.Quantity) or not frec_busc.unit.is_equivalent(u.MHz):
        raise u.UnitConversionError(
            f"'frec_busc' debe tener unidades de frecuencia. "
            f"Recibido: {frec_busc.unit}"
        )

    if not long_int == 'tot':
        if not isinstance(
                long_int, u.Quantity) or not long_int.unit.is_equivalent(u.km/u.s):

            raise u.UnitConversionError(
                f"'long_int' debe tener unidades de velocidad. "
                f"Recibido: {long_int.unit}"
            )
        long_int = long_int.to(u.km/u.s)

    if ajuste and long_int == 'tot':
        raise ValueError(
            "Si ajuste=True, debes proporcionar long_int distinto de 'tot'")

    if not isinstance(v_busc, u.Quantity) or not v_busc.unit.is_equivalent(
            u.km/u.s):
        raise u.UnitConversionError(
            f"'v_busc' debe tener unidades de velocidad. "
            f"Recibido: {v_busc.unit}"
        )

    # Aquí terminan los sistemas de seguridad
    frec_busc = frec_busc.to(u.MHz)

    nombre = None

    # Primero vamos a ver en que ventana está
    for ventana, fmin, fmax, name_window in intervalos:
        if fmin <= frec_busc <= fmax:
            # Esta variable nos ayudará a abrir el cubo correspondiente
            nombre = ventana.strip()

            nombre_ventana = name_window

    if nombre is None:
        return print(
            f'La frecuencia buscada, {frec_busc.value}{frec_busc.unit} '
            'no se encuentra en ninguna ventana')

    # Representación de la línea preseleccionada

    if dict_espec is None:

        archivos = list(rutacarp.glob(f"*{nombre}*"))

        ruta_cubo = archivos[0]

        frec, espec, res = read_spectral_cube(ruta_cubo, rutaregion1)

        frec = frec.to(u.MHz)

    else:

        frec = dict_espec[nombre_ventana]['frecuencia']
        espec = dict_espec[nombre_ventana]['Temp_brillo']

    v = mod_frec(frec_busc, frec).to(u.km/u.s)

    # Ajuste de las gaussianas
    if ajuste is True:
        # máscara que nos ayudará a coger solo los valores del intervalo
        # estudiado

        mask = (v.value >= v_busc.value-long_int.value/2) & (
            v.value <= v_busc.value+long_int.value/2)

        xdat0 = v.value[mask]
        ydat0 = espec.value[mask]

        # Vamos a buscar ahora el pico de la línea

        ymax = np.max(ydat0)
        xmax = xdat0[np.argmax(ydat0)]

        # Creamos nuevo intervalo alrededor de la línea

        mask3 = (v.value >= xmax-long_int.value/2) & (v.value <=
                                                      xmax+long_int.value/2)

        xdat1 = v.value[mask3]
        ydat1 = espec.value[mask3]

        if Tcont_fix is None:

            # Ahora vamos a intentar sacar el valor del continuo

            Tcont = np.median(ydat1)
            cont = models.Const1D(Tcont)

        elif isinstance(Tcont_fix, u.Quantity):

            Tcont = Tcont_fix
            cont = models.Const1D(Tcont.value)
            cont.amplitude.fixed = True

        else:

            Tcont = Tcont_fix[nombre_ventana]
            cont = models.Const1D(Tcont.value)
            cont.amplitude.fixed = True

        # Realizamos el ajuste gaussiano ATENCION, el valor inicial de stddev
        # es completamente arbitrario, he probado con algunas líneas y funciona
        # pero no sé si en un futuro se va a romper

        gaus0 = models.Gaussian1D(amplitude=ymax, mean=xmax, stddev=1)+cont

        if anch_lin is not None:

            sigmaf = anch_lin.value/(2*np.sqrt(2*np.log(2)))

            gaus0 = models.Gaussian1D(amplitude=ymax,
                                      mean=xmax, stddev=sigmaf)+cont
            gaus0.stddev_0.fixed = True

        gfit = fitting.LevMarLSQFitter(calc_uncertainties=True)

        gausfit = gfit(gaus0, xdat1, ydat1)

        cov = gfit.fit_info['param_cov']

        Tcont = gausfit.amplitude_1 * u.K

        if cov is not None:

            incertid = np.sqrt(np.diag(cov))
            cov_id = 0

            delta_Tmax = incertid[cov_id] * u.K
            cov_id += 1

            delta_vlin = incertid[cov_id] * u.km/u.s
            cov_id += 1

            if anch_lin is None:

                delta_sigma = incertid[cov_id] * u.km/u.s
                cov_id += 1

            else:

                delta_sigma = 0.0 * u.km/u.s

            if Tcont_fix is None:

                delta_Tcont = incertid[cov_id] * u.K
                cov_id += 1

            else:

                delta_Tcont = 0.0 * u.K

        else:

            delta_Tmax = np.nan * u.K
            delta_vlin = np.nan * u.km/u.s
            delta_sigma = np.nan * u.km/u.s
            delta_Tcont = np.nan * u.K
            print("Warning: Covariance matrix is None, uncertainties are NaN")

        Tmax = gausfit.amplitude_0 * u.K
        vlin = gausfit.mean_0 * u.km/u.s
        sigma = gausfit.stddev_0 * u.km/u.s

        FWHM = 2*np.sqrt(2*np.log(2))*sigma
        int_integrada = Tmax * sigma * np.sqrt(2*np.pi)

        dW_dT = sigma * np.sqrt(2*np.pi)
        dW_dsigma = Tmax * np.sqrt(2*np.pi)

        deltaW1 = np.sqrt((dW_dT * delta_Tmax)**2 +
                          (dW_dsigma * delta_sigma)**2)

        # vamos a calcular el ruido de la señal
        mask_line = ((v.value >= vlin.value - 2*FWHM.value) &
                     (v.value <= vlin.value + 2*FWHM.value))

        maskoff = ((v.value < vlin.value-long_int.value/2) &
                   (v.value > vlin.value-long_int.value)) | (
                       (v.value > vlin.value+long_int.value/2) &
                       (v.value < vlin.value+long_int.value))

        freqoff = espec.value[maskoff]

        rms = 1.4826 * np.median(np.abs(freqoff - np.median(freqoff))) * u.K

        dv = np.abs(np.median(np.diff(v.value))) * u.km/u.s
        N = np.count_nonzero(mask_line)
        deltaW_rms = rms * dv * np.sqrt(N)

        deltaW = np.sqrt(0.5*(deltaW1**2+deltaW_rms**2))
        # Calculemos los momentos con nuestro ajuste

        gaus_int = models.Gaussian1D(amplitude=Tmax, mean=vlin,
                                     stddev=sigma)

        v_int = np.linspace(vlin-5*sigma, vlin+5*sigma, 100000)

        T_int = gaus_int(v_int)

        M0 = np.trapezoid(T_int.value, v_int.value)*(u.K*u.km/u.s)

        M1 = np.trapezoid(v_int.value*T_int.value,
                          v_int.value)/M0.value*u.km/u.s

        M2 = np.sqrt(np.trapezoid(T_int.value*(
            v_int.value-M1.value)**2, v_int.value) / M0.value) * (u.km/u.s)

        dif_v = np.abs(vlin-v_busc)

        if dif_v > 3*u.km/u.s:

            return (Tmax, vlin, Tcont, sigma, FWHM, int_integrada, M0, M1, M2,
                    deltaW)

        v_busc = vlin

    # Representación:

    plt.figure()

    plt.plot(v, espec.value)

    plt.axvline(v_busc.value, linestyle='--', color='g',
                label=f'Línea buscada: \n{frec_busc.value:.3f}'
                f'{frec_busc.unit}')

    plt.xlabel(str(v.unit))
    plt.ylabel(str(espec.unit))

    if ajuste is True:
        v_fit = np.linspace(v_busc-long_int/2, v_busc+long_int/2, 1000)
        plt.plot(v_fit.value, gausfit(v_fit.value), 'r',
                 label='ajuste gaussiano')

    if not long_int == 'tot':

        plt.xlim(v_busc.value-long_int.value/2,
                 v_busc.value+long_int.value/2)

    plt.title(f'Espectro promedio del cubo {nombre_ventana}')
    plt.legend(loc='upper right', fontsize=8)
    plt.gca().ticklabel_format(useOffset=False)
    plt.show()

    if ajuste is True:
        return (Tmax, vlin, Tcont, sigma, FWHM, int_integrada,
                M0, M1, M2, deltaW)
    
def busc_mult_lin_v(list_frec, v_busc, interval, list_long, list_Tcont,
                    fit=False, tab=False, dict_especm=None):
    '''
    This function is the vectorizacion of the function buscador_lin.

    Parameters
    ----------
    list_frec : list of u.Quantity
        This is the list of frecuencies you are searching for.
        It must be on frecuency units.

    v_busc: u.Quantity or None

        This is the velocity of your system. It must be in velocity units.

    interval : Posible intervals your frecuency may be. They are thought to 
        be defined as [('name_interval', frec_min,frec_max)].

    list_long: list of u.Quantity or string

        This is the list of the lenghts of the intervals in frecuency you want 
        to represent.
        If you want to represent all of the spectrum, instead of a lenght in 
        frecuency insert 'tot'.

    fit: Boolean

       If you want to do a gaussian fit of all the frecuencies set it as True

    tab: Boolean

        If you want to generate a table with the results of the fit, set it as 
        True. The table will have 6 colums: velocity of the line, T of the 
        line, T of the continuum, sigma, FWHM and the integrated intensity

    Returns
    -------

    As a return the function will provide a sort of plots with the spectrum of
    the cubes that contains the frecuencies you are searching for.

    If the tab input was setted as True, the function also returns a QTable 
    with the colums described before.
    '''
    # Seguridad para comprobar que len(list_frec) == len()

    if not len(list_frec) == len(list_long):
        raise ValueError('Debes introducir dos listas (list_frec y list_long) '
                         'de igual tamaño')

    if tab is True:

        Tab = QTable(
            names=('velocidad linea', 'frecuencia', 'T de la línea', 'T cont',
                   'sigma', 'FWHM', 'intensidad integrada', 'deltaW',
                   'Momento 0', 'Momento 1', 'Momento2'),
            dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                   'f8', 'f8'),
            units=(u.km/u.s, u.MHz, u.K, u.K, u.km/u.s, u.km/u.s, u.K*u.km/u.s,
                   u.K*u.km/u.s, u.K*u.km/u.s, u.km/u.s, u.km/u.s)
        )
    for fb, li in zip(list_frec, list_long):
        print(f'Buscando la frecuencia {fb.value}{fb.unit}....')

        if tab is True:

            (Tlin, vlin, Tcont, sigma, FWHM, int_integ, M0,
             M1, M2, deltaW) = buscador_lin_vel(fb, v_busc, interval,
                                                long_int=li, ajuste=fit,
                                                dict_espec=dict_especm)

            Tab.add_row((vlin, fb, Tlin, Tcont, sigma, FWHM, int_integ, deltaW,
                         M0, M1, M2))

        else:

            buscador_lin_vel(fb, v_busc, interval, long_int=li, ajuste=fit,
                             dict_espec=dict_especm)

        print(f'Proceso de busqueda de la frecuencia {fb.value}{fb.unit} '
              'finalizado.')

    if tab is True:
        return Tab

def buscador_lin_vel_cubo_comp(frec_busc_c, v_busc_c, intervalos_c,
                               long_int_c='tot',
                     ajuste_c=False, Tcont_fix_c=None, anch_lin_c=None,
                     dict_espec_c=None):
    
    # Comprobaciones de seguridad para que la función no se rompa
    if not isinstance(
            frec_busc_c, u.Quantity) or not frec_busc_c.unit.is_equivalent(u.MHz):
        raise u.UnitConversionError(
            f"'frec_busc' debe tener unidades de frecuencia. "
            f"Recibido: {frec_busc_c.unit}"
        )

    if not long_int_c == 'tot':
        if not isinstance(
                long_int_c, u.Quantity) or not long_int_c.unit.is_equivalent(u.km/u.s):

            raise u.UnitConversionError(
                f"'long_int' debe tener unidades de velocidad. "
                f"Recibido: {long_int_c.unit}"
            )
        long_int_c = long_int_c.to(u.km/u.s)

    if ajuste_c and long_int_c == 'tot':
        raise ValueError(
            "Si ajuste=True, debes proporcionar long_int distinto de 'tot'")

    if not isinstance(v_busc_c, u.Quantity) or not v_busc_c.unit.is_equivalent(
            u.km/u.s):
        raise u.UnitConversionError(
            f"'v_busc' debe tener unidades de velocidad. "
            f"Recibido: {v_busc_c.unit}"
        )

     #Primero vamos a ver en que ventana está la frecuencia buscadda
     
    frec_busc_c = frec_busc_c.to(u.MHz)
     
    nombre = None

     # Primero vamos a ver en que ventana está
    for ventana, fmin, fmax, name_window in intervalos_c:
        if fmin <= frec_busc_c <= fmax:
            # Esta variable nos ayudará a abrir el cubo correspondiente
            nombre = ventana.strip()
            nombre_ventana = name_window

    if nombre is None:
        return print(
             f'La frecuencia buscada, {frec_busc_c.value}{frec_busc_c.unit} '
             'no se encuentra en ninguna ventana')
     
    #Cargamos el cubo donde se encuentra la línea
    
    if dict_espec_c is None:
        
        archivos = list(rutacarp.glob(f"*{nombre}*"))

        ruta_cubo = archivos[0]

        frec, espec, res = read_spectral_cube(ruta_cubo, rutaregion1, 
                                              promedio=False)

        frec = frec.to(u.MHz)

    else:

        frec = dict_espec_c[nombre_ventana]['frecuencia']
        espec = dict_espec_c[nombre_ventana]['Temp_brillo']
        
        
    #Ahora vamos con la búsqueda de la línea en cada píxel
    
    nchan, ny, nx = espec.shape
    if ajuste_c:
        
        Tmax_pix = np.full((ny, nx), np.nan)
        freclin_pix = np.full((ny, nx), np.nan)
        Tcont_pix = np.full((ny, nx), np.nan)
        sigma_pix = np.full((ny, nx), np.nan)
        FWHM_pix = np.full((ny, nx), np.nan)
        int_integrada_pix = np.full((ny, nx), np.nan)
        M0_pix = np.full((ny,nx), np.nan)
        M1_pix = np.full((ny,nx), np.nan)
        M2_pix = np.full((ny,nx), np.nan)
        deltaW_pix = np.full((ny,nx), np.nan)

    for y in range(ny):
        for x in range(nx):
            espec_pixel = espec[:, y, x]

            cubo_pixel = {
                nombre_ventana: {
                    'frecuencia': frec,
                    'Temp_brillo': espec_pixel
                }
            }
           
            if ajuste_c:
                
            
                (Tmax, freclin, Tcont, sigma, 
                 FWHM, int_integrada, M0, M1,
                 M2, deltaW) = buscador_lin_vel(frec_busc_c, v_busc_c,
                                                         intervalos_c, 
                                                         long_int_c, ajuste_c,
                                                         Tcont_fix_c, 
                                                         anch_lin_c, 
                                                         cubo_pixel)
        
                                                
                Tmax_pix[y,x] = Tmax.value
                freclin_pix[y,x] = freclin.value
                Tcont_pix[y,x] = Tcont.value
                sigma_pix[y,x] = sigma.value
                FWHM_pix[y,x] = FWHM.value
                int_integrada_pix[y,x] = int_integrada.value
                M0_pix[y,x] = M0.value
                M1_pix[y,x] = M1.value
                M2_pix[y,x] = M2.value
                deltaW_pix[y,x] = deltaW.value
                
                
            else:
                
                buscador_lin_vel(frec_busc_c, v_busc_c,
                                        intervalos_c, 
                                        long_int_c, ajuste_c,
                                        Tcont_fix_c, 
                                        anch_lin_c, 
                                        cubo_pixel)
                
    if ajuste_c:
        
        return (Tmax_pix, freclin_pix, Tcont_pix, sigma_pix, FWHM_pix, 
                int_integrada_pix, M0_pix, M1_pix, M2_pix, deltaW_pix)
    
def busc_mult_lin_v_cubo(list_frec, v_busc, interval, list_long,
                         fit=False, list_Tcont=None, list_anch=None,
                         dict_espec_c=None):

    dict_result = {}

    for i, fb in enumerate(list_frec):
        long_i = list_long[i]

        Tcont_i = None if list_Tcont is None else list_Tcont[i]
        anch_i = None if list_anch is None else list_anch[i]

        (Tmax_pix, freclin_pix, Tcont_pix, sigma_pix, FWHM_pix, 
                int_integrada_pix, M0_pix, M1_pix,
                M2_pix, deltaW_pix) = buscador_lin_vel_cubo_comp(
            fb, v_busc, interval,
            long_int_c=long_i,
            ajuste_c=fit,
            Tcont_fix_c=Tcont_i,
            anch_lin_c=anch_i,
            dict_espec_c=dict_espec_c
        )

        dict_result[f'linea_{i}'] = {
            'frecuencia_buscada': fb,
            'Tmax_pix': Tmax_pix,
            'freclin_pix': freclin_pix,
            'Tcont_pix': Tcont_pix,
            'sigma_pix': sigma_pix,
            'FWHM_pix': FWHM_pix,
            'int_integrada_pix': int_integrada_pix,
            'M0_pix': M0_pix,
            'M1_pix': M1_pix,
            'M2_pix': M2_pix,
            'deltaW_pix': deltaW_pix
        }

    return dict_result

def det_Tcont(fmin, fmax, intervalos, vfuent=None, dict_espec=None):

    if not isinstance(
            fmin, u.Quantity) or not fmin.unit.is_equivalent(u.MHz):
        raise u.UnitConversionError(
            f"'fmin' debe tener unidades de frecuencia. "
            f"Recibido: {fmin.unit}"
        )

    if not isinstance(
            fmax, u.Quantity) or not fmax.unit.is_equivalent(u.MHz):
        raise u.UnitConversionError(
            f"'fmax' debe tener unidades de frecuencia. "
            f"Recibido: {fmax.unit}"
        )

    nombre = None

    # Primero vamos a ver en que ventana está
    for ventana, fmin_vent, fmax_vent, name_window in intervalos:
        if fmin_vent <= fmin <= fmax_vent:
            # Esta variable nos ayudará a abrir el cubo correspondiente
            nombre = ventana.strip()

            nombre_ventana = name_window

    if nombre is None:
        return print(
            'El intervalo buscado está fuera de las ventanas que buscas ')

    # Representación de la línea preseleccionada
    if dict_espec is None:

        archivos = list(rutacarp.glob(f"*{nombre}*"))

        ruta_cubo = archivos[0]

        frec, espec, res = read_spectral_cube(ruta_cubo, rutaregion1)

        frec = frec.to(u.MHz)

    else:

        frec = dict_espec[nombre_ventana]['frecuencia']
        espec = dict_espec[nombre_ventana]['Temp_brillo']

    if not vfuent is None:

        frec = mod_vel(vfuent, frec)

    plt.figure()
    plt.plot(frec.value, espec.value)
    plt.axvspan(fmin.value, fmax.value, alpha=0.25, color='orange')
    plt.show()
    mask = (fmin <= frec) & (frec <= fmax)

    espec_cont = espec[mask]

    mask_fin = np.isfinite(espec_cont.value)

    espec_cont = espec_cont[mask_fin]

    if len(espec_cont) < 3:
        return nombre_ventana, np.nan * u.K, np.nan * u.K

    Tcont = np.nanmedian(espec_cont)
    sigma = np.nanstd(espec_cont)

    return nombre_ventana, Tcont.to(u.K), sigma.to(u.K)

def cont_pixeles(list_interval_cont, intervalos_c, vfuent=None, 
                 dict_espec_c=None):
    
    T_cont = {}
    sigma = {}

    for fmin, fmax in list_interval_cont:
        
        nombre = None
        nombre_ventana = None

        for ventana, fmin_i, fmax_i, name_window in intervalos_c:
            if fmin_i <= fmin <= fmax_i:
                nombre = ventana.strip()
                nombre_ventana = name_window
                break

        if nombre is None:
            print(
                f'La frecuencia buscada, {fmin.value}{fmin.unit} '
                'no se encuentra en ninguna ventana'
            )
            continue
        
        if dict_espec_c is None:
            archivos = list(rutacarp.glob(f"*{nombre}*"))
            ruta_cubo = archivos[0]

            frec, espec, res = read_spectral_cube(
                ruta_cubo, rutaregion1, promedio=False
            )
            frec = frec.to(u.MHz)

        else:
            frec = dict_espec_c[nombre_ventana]['frecuencia']
            espec = dict_espec_c[nombre_ventana]['Temp_brillo']

        nchan, ny, nx = espec.shape
        T_cont_inter = np.full((ny, nx), np.nan)
        sigma_inter = np.full((ny, nx), np.nan)

        for y in range(ny):
            for x in range(nx):
                espec_pixel = espec[:, y, x]

                cubo_pixel = {
                    nombre_ventana: {
                        'frecuencia': frec,
                        'Temp_brillo': espec_pixel
                    }
                }
                
                try:
                    name, T_vent, sigma_vent = det_Tcont(
                        fmin, fmax, intervalos_c, vfuent, cubo_pixel
                    )
                    
                    T_cont_inter[y, x] = T_vent.value
                    sigma_inter[y, x] = sigma_vent.value

                except Exception:
                    continue
        
        T_cont[nombre_ventana] = T_cont_inter
        sigma[nombre_ventana] = sigma_inter
        
    return T_cont, sigma

def map_intens_int(frec_busc, v_busc, intervalos, molecula,
                   long_int=10*u.km/u.s, mapa=True):
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

    archivos = list(rutacarp.glob(f"*{nombre}*"))

    ruta_cubo = archivos[0]

    cube = SpectralCube.read(ruta_cubo)

    # Se lee la región
    region = Regions.read(rutaregion1, format='ds9')

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

def configmapas(nombre_molecula):

    name = nombre_molecula.strip()

    if name not in CONFIG_MAPAS:
        raise ValueError(f'Molécula {name} no está disponible no está '
                         'disponible en CONFIG_MAPAS.')

    config = CONFIG_MAPAS[name]

    return map_intens_int(frec_busc=config['frec_busc'],
                          v_busc=config['v_busc'],
                          intervalos=config['intervalos'],
                          molecula=f'{name}',
                          long_int=config['long_int'])

def aplica_filtros(qn, filtros):
    s = _pat_html.sub('', str(qn))  # limpia html

    for f in filtros:
        if isinstance(f, str):
            if f in s:
                return False
        else:
            # regex compilada (re.Pattern)
            if f.search(s):
                return False
    return True

def buscador_splatalogue_cdms(elemento, intervalo, E_max, id_splat=None,
                              columnas=None, filtro_estructuras=None,
                              linelist=['CDMS']):
    """
Parameters
----------
elemento : str
    Name of the molecule as registered in Splatalogue.

intervalos : dict
    Dictionary containing the possible frequency intervals where the line
    may be located. Each entry must follow the structure:
    ('file_name', nu_min, nu_max, 'short_label'),
    where nu_min and nu_max are frequencies in Hz or convertible units.

E_max : u.Quantity
    Maximum upper state energy of the transitions to be considered.
    Must have units convertible to K.

id_splat : int, optional
    Identifier used by Splatalogue to label the molecule (column `species_id`).
    If None (default), no filtering by species_id is applied.

columnas : array-like of str, optional
    Columns to retrieve from Splatalogue. If None (default), the following
    columns are used:
        ('species_id', 'name', 'resolved_QNs', 'orderedfreq',
         'aij', 'sijmu2', 'upper_state_energy_K', 'upperStateDegen',
         'ventana_obs', 'linelist').

filtro_estructuras : list of re.Pattern, optional
    List of compiled regular expressions used to filter out specific
    structures in the quantum numbers.

    Example:
        filtro_CH3OCHO_quitar_A = [re.compile(r'\\sA$')]

    This would remove all transitions labeled as 'A' for CH3OCHO.
    Default is None.

linelist : array-like of str, optional
    Spectroscopic catalogs to query (e.g., CDMS, JPL). Default is ['CDMS'].

    To use both CDMS and JPL:
        ['CDMS', 'JPL']

Returns
-------
tab : QTable
    Table containing all the transitions that satisfy the applied filters,
    including the selected columns.
    """

    tab = Table()

    if columnas is None:

        columnas = ('species_id', 'name', 'resolved_QNs', 'orderedfreq',
                    'aij', 'sijmu2', 'upper_state_energy_K', 'upperStateDegen',
                    'ventana_obs', 'linelist')

    for name, nu_min, nu_max, name_window in intervalo:

        t = Splatalogue.query_lines(nu_min, nu_max, chemical_name=elemento,
                                    energy_max=E_max.value, energy_type='eu_k',
                                    line_strengths=['Aij'],
                                    line_lists=linelist)

        # si la tabla está vacía, salta
        if len(t) == 0:
            continue

        if id_splat is not None:
            if 'species_id' in t.colnames:

                t = t[t['species_id'] == id_splat]
            elif 'moleculeTag' in t.colnames:

                t = t[t['moleculeTag'] == id_splat]
            else:
                continue

        if len(t) == 0:
            continue

        t['ventana_obs'] = [name] * len(t)

    # selecciona solo columnas disponibles
        cols_ok = [c for c in columnas if c in t.colnames]
        if len(cols_ok) == 0:
            continue
        t = t[cols_ok]

        if len(tab) == 0:

            tab = t

        else:

            tab = vstack([tab, t])

    if filtro_estructuras is not None and 'resolved_QNs' in tab.colnames:

        mask = [aplica_filtros(q,
                               filtro_estructuras) for q in tab['resolved_QNs']]
        tab = tab[mask]
        
    return tab

def filtrador(elemento_busc, intervalo_busc, E_max_busc, aij_min, sijmu2_min,
              id_splat1=None, filtro_estructurasf=None,
              list_freq_nofilt=None, filt_inter=0.5, Tcontf=None,
              anch_fix=None, linelistf=['CDMS'], dict_especf=None):
    """
Parameters
----------
elemento : str
    Name of the molecule as registered in Splatalogue.

intervalos : dict
    Dictionary containing the possible frequency intervals where the line
    may be located. Each entry must follow the structure:
    ('file_name', nu_min, nu_max, 'short_label'),
    where nu_min and nu_max are frequencies in Hz or convertible units.

E_max : u.Quantity
    Maximum upper state energy of the transitions to be considered.
    Must have units convertible to K.

aij_min : u.Quantity
    Minimum Einstein coefficient (Aij) of the transitions to be considered.
    Must have units convertible to 1/s.

sijmu2_min : u.Quantity
    Minimum line strength (Sijμ²) of the transitions to be considered.
    Must have units convertible to Debye² (D²).

id_splat1 : int, optional
    Identifier used by Splatalogue to label the molecule (column `species_id`).
    If None (default), no filtering by species_id is applied.

filtro_estructuras : list of re.Pattern, optional
    List of compiled regular expressions used to filter out specific
    structures in the quantum numbers.

    Example:
        filtro_CH3OCHO_quitar_A = [re.compile(r'\\sA$')]

    This removes all transitions labeled as 'A' for CH3OCHO.
    Default is None.

list_freq_nofilt : array-like of u.Quantity, optional
    Array of frequencies that should NOT be removed even if they do not
    satisfy the filtering criteria. Default is None.

filt_inter : float, optional
    Filter based on the difference between integrated intensities computed
    over 20 km/s and 10 km/s intervals. This parameter sets the maximum
    allowed difference between both values.

    Default is 0.5. To disable this filter, use a very large value (e.g., 1e100).

Tcontf : u.Quantity, optional
    Continuum temperature. If provided, this value is fixed during the analysis.
    Must have units convertible to K. Default is None.

anch_fix : u.Quantity, optional
    Fixed FWHM for the spectral lines. Must have units convertible to velocity
    (e.g., m/s or km/s). Default is None.

linelistf : array-like of str, optional
    Spectroscopic catalogs to query (e.g., CDMS, JPL). Default is ['CDMS'].

    To use both CDMS and JPL:
        ['CDMS', 'JPL']

Returns
-------
tab : QTable
    Table containing all the transitions that satisfy the applied filters.
    """

    tabla_lineas = buscador_splatalogue_cdms(elemento_busc,
                                             intervalo_busc,
                                             E_max_busc, id_splat=id_splat1,
                                             filtro_estructuras=filtro_estructurasf,
                                             linelist=linelistf)

    filas = []
    ints = []
    Tcontv = []
    anch = []
    deltaWv = []
    vpik = []

    for row in tabla_lineas:
        freq = row['orderedfreq'] * u.MHz

        if list_freq_nofilt is not None:

            nofiltrar = any(np.isclose(freq.value,
                                       list_freq_nofilt.value, atol=1))
        else:
            nofiltrar = False

        Aij_row = 10**row['aij']/u.s

        sijmu2 = row['sijmu2']*u.D**2

        if nofiltrar or (Aij_row >= aij_min and sijmu2 >= sijmu2_min):

            m0, m1, slab = map_intens_int(freq,
                                          63*u.km/u.s, ventanas_obs,
                                          molecula=elemento_busc, mapa=False)

            if nofiltrar or np.any(~np.isnan(m0)):

                (Tmax, vlin, Tcont, sigma, FWHM, int_integrada,
                 M0, M1, M2, deltaW) = buscador_lin_vel(freq, 63*u.km/u.s,
                                                        ventanas_obs,
                                                        long_int=20*u.km/u.s, ajuste=True,
                                                        Tcont_fix=Tcontf,
                                                        anch_lin=anch_fix,
                                                        dict_espec=dict_especf)

                (Tmax2, vlin2, Tcont2, sigma2, FWHM2, int_integrada2,
                 M02, M12, M22, deltaW2) = buscador_lin_vel(freq, 63*u.km/u.s,
                                                            ventanas_obs,
                                                            long_int=10*u.km/u.s,
                                                            ajuste=True,
                                                            Tcont_fix=Tcontf,
                                                            anch_lin=anch_fix,
                                                            dict_espec=dict_especf)

                dif_int = np.abs(int_integrada - int_integrada2)/np.abs(
                    int_integrada)

                # prop_incert = deltaW / int_integrada
                dif_vel = np.abs(vlin-63*u.km/u.s)

                # Change the dif_int for including more lines
                if (nofiltrar or (dif_vel < 3*u.km/u.s and
                                  (dif_int < filt_inter))) and int_integrada > 0:

                    filas.append(Table([row]))

                    ints.append(int_integrada.value)
                    Tcontv.append(Tcont.value)
                    anch.append(FWHM.value)
                    deltaWv.append(deltaW.value)
                    vpik.append(vlin.value)

    tabla_filtrada = vstack(filas)

    # Ponemos las columnas bonitas y con las unidades bien

    tabla_filtrada['intensidad_integrada'] = ints*u.K*u.km/u.s
    tabla_filtrada['deltaW'] = deltaWv * u.K * u.km/u.s
    tabla_filtrada['vlin'] = vpik * u.km/u.s
    tabla_filtrada['Temp_continuo'] = Tcontv * u.K
    tabla_filtrada['FWHM'] = anch * u.km/u.s
    tabla_filtrada['orderedfreq'] = tabla_filtrada['orderedfreq']*u.MHz
    tabla_filtrada['upper_state_energy_K'] = tabla_filtrada[
        'upper_state_energy_K']*u.K
    tabla_filtrada['aij'] = 10**(tabla_filtrada['aij']) / u.s

    tabla_filtrada = unique(tabla_filtrada, keys='resolved_QNs')

    return tabla_filtrada

def configfiltrador(nombre_molecula):

    name = nombre_molecula.strip()

    if name not in CONFIG_FILTRADOR:
        raise ValueError(f'Molécula {name} no está disponible no está '
                         'disponible en CONFIG_FILTRADOR.')

    config = CONFIG_FILTRADOR[name]

    return filtrador(config['mol'],
                     config['intervalo'],
                     config['E_max'],
                     config['aij_min'],
                     config['sijmu2_min'],
                     id_splat1=config.get('id_splat1', None),
                     filtro_estructurasf=config.get('filtro_estructurasf',
                                                    None),
                     list_freq_nofilt=config.get('list_freq_nofilt', None),
                     filt_inter=config['filt_inter'],
                     Tcontf=config.get('Tcont', None),
                     anch_fix=config.get('anch_fix', None),
                     linelistf=config['linelist'],
                     dict_especf=config.get('dict_espec', None))

def ventanas_de_intervalos(intervalos):
    return [nombre_vent for _, _, _, nombre_vent in intervalos]

def crear_dict_Tcont_pixel(map_Tcont, x, y, intervalos):
    
    dict_Tcont_pix = {}
    ventanas_validas = ventanas_de_intervalos(intervalos)
    
    for ventana in ventanas_validas:
        
        mapa = map_Tcont[ventana]
        Tcont_pix = mapa[y,x]
        dict_Tcont_pix[ventana] = Tcont_pix * u.K
        
    return dict_Tcont_pix

def crear_dict_espec_pixel(dict_espec, x, y, intervalos):
    
    dict_espec_pix = {}
    ventanas_validas = ventanas_de_intervalos(intervalos)
    
    for ventana in ventanas_validas:
        
        frec = dict_espec[ventana]['frecuencia']
        T_brillo = dict_espec[ventana]['Temp_brillo'][:,y,x]
        
        dict_espec_pix[ventana] = {'frecuencia': frec, 'Temp_brillo': T_brillo}
        
    return dict_espec_pix

def filtrador_por_pixel(tab_info, intervalos, v_busc=63*u.km/u.s,
                        long_int=20*u.km/u.s, Tcontf=None,
                        anch_fix=None, dict_especf=None):

    if dict_especf is None:
        
        dict_especf = {}

        for ventana, fmin, fmax, nombre_vent in intervalos:

             nombre = ventana.strip()
             archivos = list(rutacarp.glob(f'*{nombre}*'))
             
             ruta_cubo = archivos[0]
             
             frec_c, espec_c, spec_resol = read_spectral_cube(ruta_cubo, rutaregion1,
                                                              promedio = False)
             
             frec_c  = frec_c.to(u.MHz)
             
             dict_especf[nombre_vent] = dict(frecuencia = frec_c, 
                                                 Temp_brillo = espec_c)
             
        primera_ventana = list(dict_especf.keys())[0]
        cubo_ref = dict_especf[primera_ventana]['Temp_brillo']
        nchan, ny, nx = cubo_ref.shape
        
        
    else:
        
        ventanas_validas = [nombre_vent for _, _, _, nombre_vent in intervalos]

        dict_especf = {
            k: v for k, v in dict_especf.items()
            if k in ventanas_validas
        }

        primera_ventana = list(dict_especf.keys())[0]
        cubo_ref = dict_especf[primera_ventana]['Temp_brillo']
        nchan, ny, nx = cubo_ref.shape
        
        
    tab_pixeles = np.empty((ny,nx), dtype = object)  
    
    for y in range(ny):
        for x in range(nx):
            
            tab_pixel = tab_info.copy()

            frec = tab_pixel['orderedfreq'].to(u.MHz)
            
            if Tcontf is not None:
                 dict_Tcont_pix = crear_dict_Tcont_pixel(Tcontf, x, y,
                                                         intervalos)
            else:
                dict_Tcont_pix = None
                
            if dict_especf is not None:
                dict_espec_pix = crear_dict_espec_pixel(dict_especf, x, y,
                                                        intervalos)
            else:
                dict_espec_pix = None
            
            int_integ_col = []
            deltaW_col = []
            vlin_col = []
            T_cont_col = []
            FWHM_col = []
            
            for f in frec:
                
                try:
                    (Tmax, vlin, Tcont, sigma, FWHM, int_integrada,
                     M0, M1, M2, deltaW) = buscador_lin_vel(
                                           f, v_busc, intervalos,
                                           long_int=long_int,
                                           ajuste=True,
                                           Tcont_fix=dict_Tcont_pix,
                                           anch_lin=anch_fix,
                                           dict_espec=dict_espec_pix
                                           )

                    int_integ_col.append(int_integrada)
                    deltaW_col.append(deltaW)
                    vlin_col.append(vlin)
                    T_cont_col.append(Tcont)
                    FWHM_col.append(FWHM)

                except Exception:
                    int_integ_col.append(np.nan * u.K * u.km/u.s)
                    deltaW_col.append(np.nan * u.K * u.km/u.s)
                    vlin_col.append(np.nan * u.km/u.s)
                    T_cont_col.append(np.nan * u.K)
                    FWHM_col.append(np.nan * u.km/u.s)
            

            tab_pixel['intensidad_integrada'] = int_integ_col
            tab_pixel['deltaW'] = deltaW_col
            tab_pixel['vlin'] = vlin_col
            tab_pixel['Temp_continuo'] = T_cont_col
            tab_pixel['FWHM'] = FWHM_col

            mask_valid = (
               np.isfinite(tab_pixel['intensidad_integrada'].value) &
               (tab_pixel['intensidad_integrada'] > 0)
               )

            tab_pixeles[y, x] = tab_pixel[mask_valid]
            
    return tab_pixeles

def configfiltrador_Pixeles(nombre_molecula):
    name = nombre_molecula.strip()

    if name not in CONFIG_FILTRADOR_PIXELES:
        raise ValueError(f'Molécula {name} no está disponible no está '
                         'disponible en CONFIG_FILTRADOR_PIXELES.')

    config = CONFIG_FILTRADOR_PIXELES[name]

    return filtrador_por_pixel(config['tab_info'], config['intervalos'], 
                               v_busc = 63*u.km/u.s, long_int = 20 * u.km/u.s,
                               Tcontf = config['Tcontf'], 
                               anch_fix= config['anch_fix'], 
                               dict_especf= config['dict_especf'])

def filtrador_tablas(tab_no_filt, filt_sijmu, filt_E_max, filt_aij,
                     filt_estructuras=None, duplicados=False):
    '''


    Parameters
    ----------
    tab_no_filt : u.QTable
        The table with the lines you want to filter.

    filt_sijmu : u.Quantity
    Minimum line strength (Sijμ²) of the transitions to be considered.
    Must have units convertible to Debye² (D²).

    filt_E_max : u.Quantity
    Maximum upper state energy of the transitions to be considered.
    Must have units convertible to K.

    filt_aij: u.Quantity
        Minimum Einstein coefficient (Aij) of the transitions to be considered.
        Must have units convertible to 1/s.

filtro_estructuras : list of re.Pattern, optional
    List of compiled regular expressions used to filter out specific
    structures in the quantum numbers.

    Example:
        filtro_CH3OCHO_quitar_A = [re.compile(r'\\sA$')]

    This removes all transitions labeled as 'A' for CH3OCHO.
    Default is None.


    Returns
    -------
tab_filt : QTable
    Table containing all the transitions that satisfy the applied filters.

    '''
    mascara = (
        (tab_no_filt['aij'] >= filt_aij.value) &
        (tab_no_filt['sijmu2'] >= filt_sijmu.value) &
        (tab_no_filt['upper_state_energy_K'] <= filt_E_max.value)
    )

    tab_filt = tab_no_filt[mascara]

    if filt_estructuras is not None:

        mask = [aplica_filtros(q,
                               filt_estructuras) for q in tab_filt['resolved_QNs']]

        tab_filt = tab_filt[mask]
    if duplicados is True:

        tab_filt = unique(tab_filt, keys='orderedfreq', keep='first')

    return tab_filt

def Q(T_ex, cat_mol, id_cat, deltaTex=None, inc=False, plot = True):

    if deltaTex == None and inc is True:

        raise ValueError('If you want uncertities, you need to provide a ' +
                         'Delta_T_ex.')
    # cálculo de la función de partición

    if cat_mol == 'CDMS':

        tab_cdms = CDMS.get_species_table()
        mol = tab_cdms[tab_cdms['tag'] == id_cat]

        q_cols = [c for c in mol.colnames if c.startswith('lg(Q(')]

        T_cols = []
        lgQ_vals = []

        for col in q_cols:
            m = re.search(r'lg\(Q\(([\d.]+)\)\)', col)
            if m is None:
                continue
            T_cols.append(float(m.group(1)))
            lgQ_vals.append(float(mol[col][0]))

        T_cols = np.array(T_cols, dtype=float)
        lgQ_vals = np.array(lgQ_vals, dtype=float)

        mask = np.isfinite(T_cols) & np.isfinite(lgQ_vals)
        T_cols = T_cols[mask]
        lgQ_vals = lgQ_vals[mask]

        order = np.argsort(T_cols)
        T_cols = T_cols[order]
        lgQ_vals = lgQ_vals[order]

    elif cat_mol == 'JPL':

        tab_JPL = JPLSpec.get_species_table()
        mol = tab_JPL[tab_JPL['TAG'] == id_cat]

        T_cols = tab_JPL.meta['Temperature (K)']

        q_cols = [c for c in mol.colnames if c.startswith('QLOG')]
        lgQ_vals = np.array([mol[c][0] for c in q_cols], dtype=float)

    Q_vals = 10**lgQ_vals

    # lgQ_Tex = np.interp(T_ex.to_value(u.K), T_cols, lgQ_vals)

    coefQ, covQ = np.polyfit(T_cols, Q_vals, deg=4, cov=True)

    Q_Tex_poli = np.polyval(coefQ, T_ex.value)

    Q_Tex_interp = np.interp(T_ex.value, T_cols, Q_vals)
    if plot:
        plt.figure()

        plt.plot(T_cols, Q_vals, '+')
        plt.plot(np.linspace(0, np.max(T_cols), 1000),
             np.polyval(coefQ, np.linspace(0, np.max(T_cols), 1000)), 'r')

        plt.show()

    if inc:

        dQ_da = T_ex.value**2
        dQ_db = T_ex.value
        dQ_dx = 2*coefQ[0]*T_ex.value + coefQ[1]

        deltQ_Tex = np.sqrt((dQ_da * np.sqrt(covQ[0, 0]))**2 +
                            (dQ_db * np.sqrt(covQ[1, 1]))**2 +
                            covQ[2, 2] + (dQ_dx * deltaTex.value)**2)

        return Q_Tex_interp, deltQ_Tex

    return Q_Tex_interp

def to_float(col):
    # MaskedColumn -> rellena con nan
    if hasattr(col, "filled"):
        col = col.filled(np.nan)
    return np.array(col, dtype=float)


def diagrama_rotacional(elemento, id_cat, tab_filtrada1, B0, cat_mol,
                        freq_noconsid=None, plot_Q = True):
    """
Parameters
----------
elemento : str
    Name of the molecule for which the rotational diagram is computed.

id_cat : int
    Identifier of the molecule in the selected spectroscopic catalog.
    Ensure that the ID corresponds to the chosen catalog (cat_mol).

tab_filtrada1 : QTable
    Table of spectral lines obtained with the function `filtrador()`.

    Recommended usage:
        Use `filtrador()` or `filtrador_tablas()` to generate this table,
        as they ensure the correct format required by this function.

B0 : u.Quantity
    Rotational constant B0 obtained from the molecular catalog.
    Must have units convertible to Hz.

cat_mol : str
    Name of the spectroscopic catalog (e.g., 'CDMS' or 'JPL').

freq_noconsid : array-like of u.Quantity, optional
    List of frequencies to exclude from the rotational diagram.
    Must have units convertible to Hz. Default is None.

Returns
-------
T_ex : u.Quantity
    Excitation temperature derived from the rotational diagram.

tab_filtrada1 : QTable
    Table containing the spectral lines used in the analysis.

N_col : u.Quantity
    Column density, typically expressed in cm⁻².

pol : ndarray
    Coefficients of the linear fit. 
    pol[0] corresponds to the intercept (n) and pol[1] to the slope (m).

Q_Tex : float
    Partition function evaluated at the excitation temperature.CH3CHO_v0
    """

    freq = tab_filtrada1['orderedfreq']

    if freq_noconsid is not None:
        freq_noconsid = u.Quantity(freq_noconsid).to(u.MHz)

        match = np.isclose(freq[:, None].value, freq_noconsid[None, :].value,
                           atol=1)

        mask_excluir = match.any(axis=1)

        tab_filtrada1 = tab_filtrada1[~mask_excluir]
        freq = tab_filtrada1['orderedfreq'].to(u.MHz)

    freq = freq.to(u.Hz)

    Aij = tab_filtrada1['aij']

    Aij = Aij.to(1/u.s)

    Eu = tab_filtrada1['upper_state_energy_K']

    Eu = Eu.to(u.K)

    gu = to_float(tab_filtrada1['upperStateDegen'])

    W = tab_filtrada1['intensidad_integrada']

    W = W.to(u.K * u.m/u.s)

    deltaW = tab_filtrada1['deltaW']

    deltaW = deltaW.to(u.K * u.m / u.s)

    gammau = (4*np.pi*c.k_B*freq**2)/(c.h*c.c**3*Aij)

    arg = gammau*W/gu

    arg = arg.to(u.cm**(-2))

    # Con y me refiero a ln(gamma_u*W/gu )

    y = np.log(arg.value)

    deltay = deltaW.value/W.value
    # weight = 1/deltay

    pol, cov = np.polyfit(Eu.value, y, deg=1,  cov=True)

    x = np.linspace(np.min(Eu.value), np.max(Eu.value), 10000)

    recta = np.polyval(pol, x)

    T_ex = -1/pol[0]*u.K
    deltaTex = (1/pol[0]**2) * np.sqrt(cov[0, 0]) * u.K

    # # cálculo de la función de partición

    Q_Tex, deltQ_Tex = Q(T_ex, cat_mol, id_cat, deltaTex, True, plot_Q)
    N_col = Q_Tex*np.exp(pol[1])/u.cm**2

    # dN_dQ = np.exp(pol[1])
    # dN_dpol = Q_Tex * np.exp(pol[1])

    # deltaN_col = np.sqrt((dN_dQ * deltQ_Tex)**2 +
    #                    (dN_dpol * np.sqrt(cov[1,1]))**2)/u.cm**2

    sigma_b = np.sqrt(cov[1, 1])

    deltaN_col = np.sqrt((Q_Tex * np.exp(pol[1]) * sigma_b)**2)/u.cm**2

    # Incertidumbre del ajuste

    var_yfit = (x**2)*cov[0, 0] + cov[1, 1] + 2*x*cov[0, 1]
    sig_yfit = np.sqrt(var_yfit)

    k = 3  # el numero de sigmas que tenemos en cuenta en el ajuste

    rectalo = recta - k*sig_yfit
    rectahi = recta + k*sig_yfit

    # Representación

    plt.figure()

    plt.errorbar(Eu.value, y, yerr=deltay, fmt='.r',
                 label='experimental data', capsize=3)
    plt.plot(Eu.value, y, '.r', label='experimental data')
    plt.plot(x, recta, '--b', label='lineal fit')
    plt.fill_between(x, rectalo, rectahi, alpha=0.25, label=f'{k:.2g}σ band')

    plt.plot([], [], ' ', label=fr'$T_{{ex}} = {T_ex.value:.1f} \pm'
             fr' {deltaTex.value:.1f}\,$K')
    plt.plot([], [], ' ', label=fr'$N = {N_col.to_value(1/u.cm**2):.2e} \pm{deltaN_col.value:.2e}'
             fr'\,\mathrm{{cm^{{-2}}}}$')

    plt.xlabel('Eu/K')
    plt.ylabel(r'$\ln(\gamma_u W/g_u)$')

    plt.grid('on')

    plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))

    plt.title('Rotational diagram for ' + elemento)

    plt.show()

    return T_ex, tab_filtrada1, N_col.to(1/u.cm**2), pol, Q_Tex

def configdiagrot(nombre_molecula):

    name = nombre_molecula.strip()

    if name not in CONFIG_DIAGROT:
        raise ValueError(f'Molécula {name} no está disponible no está '
                         'disponible en CONFIG_DIAGROT.')

    config = CONFIG_DIAGROT[name]

    return diagrama_rotacional(f' {name} ',
                               config['id_cat'],
                               config['tab_filtrada1'],
                               config['B0'],
                               config['cat_mol'],
                               config['freq_noconsid'])

def diagrama_rot_pixeles(elemento, id_cat, tab_pixeles, B0, cat_mol,
                         freq_noconsid=None, wcs_ref=None):
    
    ny, nx = tab_pixeles.shape

    N_col_map = np.full((ny, nx), np.nan)
    T_ex_map = np.full((ny, nx), np.nan)
    
    for y in range(ny):
        for x in range(nx):
    
            tab_indiv = tab_pixeles[y, x]

            if tab_indiv is None or len(tab_indiv) < 2:
                continue

            try:
                (T_extot, tab_filtrada2, N_coltot, pol, QTex) = diagrama_rotacional(
                    elemento, id_cat, tab_indiv, B0, cat_mol,
                    freq_noconsid=freq_noconsid, plot_Q = False
                )

                N_col_map[y, x] = N_coltot.value
                T_ex_map[y, x] = T_extot.value

            except Exception:
                continue

    # --- PLOTS ---
    fig = plt.figure(figsize=(12, 5))

    fig.suptitle(
        f'Mapas de Temperatura y Densidad de Columna para {elemento}',
        fontsize=14
    )

    N_plot = np.where(N_col_map > 0, N_col_map, np.nan)

    if wcs_ref is not None:

    # Mapa de T_ex con RA/Dec
        ax1 = fig.add_subplot(1, 2, 1, projection=wcs_ref)
        im1 = ax1.imshow(T_ex_map, origin='lower')

        ax1.set_title(r'$T_{ex}$ (K)')
        ax1.coords[0].set_axislabel('RA')
        ax1.coords[1].set_axislabel('Dec')
        ax1.coords[0].set_major_formatter('hh:mm:ss.s')
        ax1.coords[1].set_major_formatter('dd:mm:ss.s')

        # Más ticks
        ax1.coords[0].set_ticks(spacing=0.5*u.arcsec)
        ax1.coords[1].set_ticks(spacing=0.25*u.arcsec)
        ax1.coords[0].display_minor_ticks(True)
        ax1.coords[1].display_minor_ticks(True)

        cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
        cbar1.set_label('K')

        # Mapa de N_col con RA/Dec
        ax2 = fig.add_subplot(1, 2, 2, projection=wcs_ref)
        im2 = ax2.imshow(np.log10(N_plot), origin='lower')

        ax2.set_title(r'$\log_{10}(N_{col})$ (cm$^{-2}$)')
        ax2.coords[0].set_axislabel('RA')
        ax2.coords[1].set_axislabel('Dec')
        ax2.coords[0].set_major_formatter('hh:mm:ss.s')
        ax2.coords[1].set_major_formatter('dd:mm:ss.s')

        # Más ticks
        ax2.coords[0].set_ticks(spacing=0.5*u.arcsec)
        ax2.coords[1].set_ticks(spacing=0.25*u.arcsec)
        ax2.coords[0].display_minor_ticks(True)
        ax2.coords[1].display_minor_ticks(True)

        cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
        cbar2.set_label(r'$\log_{10}$(cm$^{-2}$)')

    else:

        # Mapa de T_ex sin WCS
        ax1 = fig.add_subplot(1, 2, 1)
        im1 = ax1.imshow(T_ex_map, origin='lower')
        ax1.set_title(r'$T_{ex}$ (K)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, label='K')

        # Mapa de N_col sin WCS
        ax2 = fig.add_subplot(1, 2, 2)
        im2 = ax2.imshow(np.log10(N_plot), origin='lower')
        ax2.set_title(r'$\log_{10}(N_{col})$ (cm$^{-2}$)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, label=r'$\log_{10}$(cm$^{-2}$)')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    return T_ex_map, N_col_map

def configdiagrot_pixeles(nombre_molecula):
 
    name = nombre_molecula.strip()

    if name not in CONFIG_DIAGROT_PIXELES:
        raise ValueError(f'Molécula {name} no está disponible no está '
                         'disponible en CONFIG_DIAGROT_PIXELES.')

    config = CONFIG_DIAGROT_PIXELES[name]

    return diagrama_rot_pixeles(f' {name} ', id_cat = config['id_cat'],
                                tab_pixeles = config['tab_filtrada1'],
                                B0 = config['B0'], cat_mol = config['cat_mol'],
                                freq_noconsid = config['freq_noconsid'], 
                                wcs_ref = config['wcs_ref'])


def spec_sint(n, m, molecula, intervalo1, id_splat1, filtro_estructuras1,
              anch_lin, Tcont, f0, dict_espec=None):
    """
   Parameters
   ----------
   n : float
       Intercept of the rotational diagram.

   m : float
       Slope of the rotational diagram.

   molecula : str
       Name of the molecule as registered in Splatalogue.

   intervalo1 : dict
       Dictionary containing the possible frequency intervals where the line
       may be located. Each entry must follow the structure:
       ('file_name', nu_min, nu_max, 'short_label'),
       where nu_min and nu_max are frequencies in Hz or convertible units.

   id_splat1 : int, optional
       Identifier used by Splatalogue to label the molecule (column
       `species_id`). If None, no filtering by species_id is applied.

   filtro_estructuras1 : list of re.Pattern, optional
       List of compiled regular expressions used to filter out specific
       structures in the quantum numbers.

       Example:
           filtro_CH3OCHO_quitar_A = [re.compile(r'\\sA$')]

       This removes all transitions labeled as 'A' for CH3OCHO.
       Default is None.

   anch_lin : u.Quantity
       FWHM of the lines considered by the model. Must have units
       convertible to velocity (e.g., m/s or km/s).

   Tcont : u.Quantity
       Continuum temperature of the model. Must have units convertible to K.

   f0 : u.Quantity
       Reference frequency used to apply the Doppler effect. Must have units
       convertible to Hz.

   Returns
   -------

       Generates plots of the different spectral windows, showing both the
       observed spectrum and the synthetic spectrum.
   """
    sigma = anch_lin/(2*np.sqrt(2*np.log(2)))
    sigma_freq = (f0 * sigma / c.c.to(u.km/u.s)).to(u.MHz)

    recta = Polynomial([n, m])

    tab_lineas = buscador_splatalogue_cdms(molecula, intervalo1, 1000*u.K,
                                           id_splat1,
                                           filtro_estructuras=filtro_estructuras1)

    Energias = tab_lineas['upper_state_energy_K']
    freq = tab_lineas['orderedfreq'] * u.MHz
    Aij = 10**tab_lineas['aij'] / u.s
    g = to_float(tab_lineas['upperStateDegen'])

    ln_gamWg = recta(Energias)
    gammau = (4*np.pi*c.k_B*freq**2)/(c.h*c.c**3*Aij)
    gammau = gammau.to(u.s/(u.K*u.m**3))
    gamWg = np.exp(ln_gamWg) / u.cm**2

    W = (gamWg * g) / gammau

    W = W.to(u.K*u.km/u.s)

    T_lin = W/anch_lin * (np.sqrt(4*np.log(2)/np.pi))

    for ventana, fmin, fmax, name_window in intervalo1:

        mask_lin_int = (fmin <= freq) & (freq <= fmax)
        T_lin_inter = T_lin[mask_lin_int]
        freq_inter = freq[mask_lin_int]

        modelo = models.Const1D(amplitude=Tcont)

        for f, T in zip(freq_inter, T_lin_inter):

            modelo += models.Gaussian1D(amplitude=T,
                                        mean=f,
                                        stddev=sigma_freq)

        if dict_espec is None:

            archivos = list(rutacarp.glob(f"*{ventana}*"))

            ruta_cubo = archivos[0]

            frec_med, espec_med, res = read_spectral_cube(ruta_cubo, rutaregion1)

            frec_med = frec_med.to(u.MHz)

        else:

            frec_med = dict_espec[name_window]['frecuencia']
            espec_med = dict_espec[name_window]['Temp_brillo']

        espec_sint = modelo(frec_med)

        plt.figure()

        plt.plot(frec_med, espec_med, 'b')
        plt.plot(frec_med, espec_sint, 'r')

        plt.xlim(np.min(frec_med.value)-10, np.max(frec_med.value)+10)

        plt.title(f'Modelo sintético para {name_window}')

        plt.show()

# Comprobación de que la recta es la que queremos y que los puntos los coge bien
    x = np.linspace(np.min(Energias), np.max(Energias), 1000)
    plt.figure()

    plt.plot(x, recta(x), '--')
    plt.plot(Energias, ln_gamWg, '+')

    plt.xlabel('Eu/K')
    plt.ylabel(r'$\ln(\gamma_u W/g_u)$')

    plt.grid('on')

    plt.show()
    
    
def spec_sint_class(T_ex, N_col, molecula, intervalo1, id_splat1, filtro_estructuras1,
                    anch_lin, Tcont, f0, vpic, name_mol, cat_mol, id_cat,
                    sijmu2=0*u.D**2, aij=0/u.s, modeloin=None,
                    dict_espec=None, plot_lineas=False, tab_lineas_mol=None,
                    show_plots=True):
    """
    Parameters
    ----------
    T: u.Quantity
        Excitation temperature of the rotational diagram. Must be in K or 
        convertible.

    N : u.Quantity
        Column density of the rotational diagram. Must be in 1/cm**2 or 
        convertible.

    molecula : str
        Name of the molecule as registered in Splatalogue.

    intervalo1 : dict
        Dictionary containing the possible frequency intervals where the line
        may be located. Each entry must follow the structure:
        ('file_name', nu_min, nu_max, 'short_label'),
        where nu_min and nu_max are frequencies in Hz or convertible units.

    id_splat1 : int, optional
        Identifier used by Splatalogue to label the molecule (column
        `species_id`). If None, no filtering by species_id is applied.

    filtro_estructuras1 : list of re.Pattern, optional
        List of compiled regular expressions used to filter out specific
        structures in the quantum numbers.

        Example:
            filtro_CH3OCHO_quitar_A = [re.compile(r'\\sA$')]

        This removes all transitions labeled as 'A' for CH3OCHO.
        Default is None.

    anch_lin : u.Quantity
        FWHM of the lines considered by the model. Must have units
        convertible to velocity (e.g., m/s or km/s).

    Tcont : u.Quantity
        Continuum temperature of the model. Must have units convertible to K.

    f0 : u.Quantity
        Reference frequency used to apply the Doppler effect. Must have units
        convertible to Hz.

    QT: float

        Partition function evaluated for T. 
        Advise: 
            Use the function "diagrama_rotational()" for obtaining this value

    vpic: u.Quantity

        Measure velocity of the lines. Must be in m/s or convertible.

    name_mol: str

        Name you want to give to the molecule for the plot title.


    Returns
    -------
        Generates plots of the different spectral windows, showing both the
        observed spectrum and the synthetic spectrum.
    """
    QT = Q(T_ex, cat_mol, id_cat, plot = False)

    sigma = anch_lin/(2*np.sqrt(2*np.log(2)))
    sigma_freq = (f0 * sigma / c.c.to(u.km/u.s)).to(u.MHz)

    m = -1/T_ex.value
    n = np.log(N_col.value/QT)

    recta = Polynomial([n, m])

    tab_lineas = buscador_splatalogue_cdms(molecula, intervalo1, 1000*u.K,
                                           id_splat1,
                                           filtro_estructuras=filtro_estructuras1,
                                           linelist=[cat_mol])

    tab_lineas['aij'] = 10**tab_lineas['aij']
    tab_lineas = filtrador_tablas(tab_lineas, sijmu2, 1000*u.K, aij)

    tab_lineas['Freq_corrected_v'] = mod_vel(vpic, tab_lineas['orderedfreq'])
    tab_lineas['Compuesto'] = name_mol

    Energias = tab_lineas['upper_state_energy_K']
    freq = tab_lineas['orderedfreq'] * u.MHz
    Aij = tab_lineas['aij']/u.s
    g = to_float(tab_lineas['upperStateDegen'])

    ln_gamWg = recta(Energias)
    gammau = (4*np.pi*c.k_B*freq**2)/(c.h*c.c**3*Aij)
    gammau = gammau.to(u.s/(u.K*u.m**3))
    gamWg = np.exp(ln_gamWg) / u.cm**2

    W = (gamWg * g) / (gammau)

    W = W.to(u.K*u.km/u.s)

    tab_lineas['W_sint'] = W

    T_lin = W/anch_lin * (np.sqrt(4*np.log(2)/np.pi))

    tab_lineas['Tlin'] = T_lin

    # filtrado de las líneas más intensas

    mask_lin = (5 <= tab_lineas['W_sint'].value)

    tab_lin_intens = tab_lineas[mask_lin]

    modelo = {}
    tab_lineas_plot = {name_mol: dict(freq=[], label=[])}

    tab_lineas_plot[name_mol]['freq'].extend(
        list(tab_lin_intens['Freq_corrected_v']))

    tab_lineas_plot[name_mol]['label'].extend(
        list(tab_lin_intens['Compuesto']))

    if tab_lineas_mol is not None:

        for m in tab_lineas_mol:

            tab_lineas_plot[m] = {'freq': [], 'label': []}
            tab_lineas_plot[m]['freq'] = tab_lineas_mol[m]['freq']
            tab_lineas_plot[m]['label'] = tab_lineas_mol[m]['label']

    freq_total = []
    label_total = []

    for m in tab_lineas_plot:
        freq_total.extend(tab_lineas_plot[m]['freq'])
        label_total.extend(tab_lineas_plot[m]['label'])

    freq_arr = np.array([
        x.value if hasattr(x, 'value') else x
        for x in freq_total
    ], dtype=float)

    label_arr = np.array(label_total, dtype=object)
    
    residuos = {}
    
    for ventana, fmin, fmax, name_window in intervalo1:

        mask_lin_inter = (fmin <= freq) & (freq <= fmax)
        T_lin_inter = T_lin[mask_lin_inter]
        freq_inter = freq[mask_lin_inter]

        if modeloin is None:

            if isinstance(Tcont, u.Quantity):

                modelo[name_window] = models.Const1D(amplitude=Tcont)

            else:
                Tcont_window = Tcont[name_window]

                modelo[name_window] = models.Const1D(amplitude=Tcont_window)

        else:

            modelo[name_window] = modeloin[name_window]

        for f, T in zip(freq_inter, T_lin_inter):

            modelo[name_window] += models.Gaussian1D(amplitude=T,
                                                     mean=f,
                                                     stddev=sigma_freq)
        if dict_espec is None:

            archivos = list(rutacarp.glob(f"*{ventana}*"))

            ruta_cubo = archivos[0]

            frec_med, espec_med, res = read_spectral_cube(ruta_cubo, rutaregion1)

            frec_med = frec_med.to(u.MHz)

        else:

            frec_med = dict_espec[name_window]['frecuencia']
            espec_med = dict_espec[name_window]['Temp_brillo']

        frec_sint = np.linspace(fmin, fmax, 10000)
        espec_sint = modelo[name_window](frec_sint)

        frec_sint = mod_vel(vpic, frec_sint)

        ymin_espec = np.min(espec_med)
        ymax_espec = np.max(espec_med)

        mask_f_dentro = (np.min(frec_med).value <= freq_arr) & (
            freq_arr <= np.max(frec_med).value)

        freq_dentro = freq_arr[mask_f_dentro]
        label_dentro = label_arr[mask_f_dentro]
        
        #Cálculo de los residuos:
        
        if isinstance(Tcont, u.Quantity):
                
            continuo_mod = models.Const1D(amplitude=Tcont)

        else:
            Tcont_window = Tcont[name_window]

            continuo_mod = models.Const1D(amplitude=Tcont_window)
        
        Tab_espectros = []
        Tab_espectros.append(frec_med)
        Tab_espectros.append(espec_med)
        
        frec_resi = mod_vel(-vpic, frec_med)
        espec_sint_alineado = modelo[name_window](frec_resi)
        
        Tab_espectros.append(espec_sint_alineado)
        
        continuo = continuo_mod(frec_med)
        
        Tab_espectros.append(espec_med - espec_sint_alineado + continuo)
        
        residuos[name_window] = {'frecuencia': Tab_espectros[0], 
                                 'Temp_brillo': Tab_espectros[3]}
        
        if plot_lineas and show_plots:

            for f, label in zip(freq_dentro, label_dentro):

                mask_linea = (
                    f-10 <= frec_med.value) & (frec_med.value <= f+10)

                espec_linea_med = espec_med[mask_linea]
                frec_linea_med = frec_med[mask_linea]

                if len(espec_linea_med) == 0:
                    continue

                plt.figure()

                plt.plot([], [], ' ', label=f'{name_window}//{name_mol}')
                plt.plot(frec_linea_med.value, espec_linea_med.value, 'b',
                         drawstyle='steps-mid', label='Espectro observado')
                plt.plot(frec_sint, espec_sint, '--r',
                         label='Modelo sintético')
                plt.plot([], [], ' ', label=fr'T_ex = {T_ex.value:.1f} K')
                plt.plot([], [], ' ', label=fr'N_col = {
                         N_col.value:.2e} 1/cm²')

                ymax = np.max(espec_linea_med).value
                ymin = np.min(espec_linea_med).value

                for f2, label2 in zip(freq_dentro, label_dentro):
                    if f2 >= f-10 and f2 <= f + 10:

                        if ymax >= 35:

                            plt.axvline(f2, color='k', linestyle=':',
                                        alpha=0.5)

                            plt.text(f2-1, ymax + 0.5*(ymax-ymin), label2,
                                     rotation=90, fontsize=8, ha='center',
                                     va='top')

                            plt.xlim(f-10, f+10)
                            plt.ylim(np.min(espec_sint).value-1, ymax+7)

                        else:
                            plt.axvline(f2, color='k', linestyle=':',
                                        alpha=0.5)

                            plt.text(f2-1, 45, label2, rotation=90,
                                     fontsize=8, ha='center', va='top')

                            plt.xlim(f-10, f+10)
                            plt.ylim(np.min(espec_sint).value-2, 50)

                plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                           borderaxespad=0)

                plt.xlabel('Frecuencia (MHz)')
                plt.ylabel('Temperatura de brillo (K)')

                # plt.savefig(os.path.join('/home/jorge/TFM/Plots',
                #                 f'{name_window}_{f:.1f}_mol_{name_mol}.jpg'),
                #             dpi = 200, bbox_inches = 'tight')

                plt.show()
                plt.close()

        elif show_plots:
            if name_window == 'B6-SPW7':

                intervalos_zoom = [
                    (231450, 231750),
                    (231725, 232050),
                    (232025, 232350),
                    (232325, 232650),
                    (232625, 233000),
                    (232975, 233350)
                ]

                for xmin, xmax in intervalos_zoom:
                    plt.figure()

                    plt.plot([], [], ' ', label=f'{
                             name_window}//Modelo completo')
                    plt.plot(frec_med, espec_med, 'b', drawstyle='steps-mid',
                             label='Espectro observado')
                    plt.plot(frec_sint, espec_sint, '--r',
                             label='Modelo sintético')

                    for f2, label2 in zip(freq_dentro, label_dentro):

                        if xmin <= f2 and f2 <= xmax:

                            plt.axvline(f2, color='k', linestyle=':',
                                        alpha=0.5)

                            plt.text(f2-1, 80, label2, rotation=90,
                                     fontsize=8, ha='center',
                                     va='top')

                    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                               borderaxespad=0)

                    plt.ylim(ymin_espec.value, 85)
                    plt.xlim(xmin, xmax)

                    plt.xlabel('Frecuencia (MHz)')
                    plt.ylabel('Temperatura de brillo (K)')

                    plt.show()
                    plt.close()

            else:
                plt.figure()

                plt.plot([], [], ' ', label=f'{name_window}//Modelo Completo')
                plt.plot(frec_med, espec_med, drawstyle='steps-mid',
                         color='b', label='Espectro observado')
                plt.plot(frec_sint, espec_sint, '--r',
                         label='Modelo sintético')

                for f2, label2 in zip(freq_dentro, label_dentro):

                    plt.axvline(f2, color='k', linestyle=':',
                                alpha=0.5)

                    plt.text(f2-1, ymax_espec.value+20, label2, rotation=90,
                             fontsize=8, ha='center',
                             va='top')

                plt.xlim(np.min(frec_med.value)-10, np.max(frec_med.value)+10)
                plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                           borderaxespad=0)

                plt.ylim(ymin_espec.value, ymax_espec.value+30)
                plt.xlabel('Frecuencia (MHz)')
                plt.ylabel('Temperatura de brillo (K)')

                plt.show()
                plt.close()

# Comprobación de que la recta es la que queremos y que los puntos los coge bien
    # x = np.linspace(np.min(Energias), np.max(Energias), 1000)
    # plt.figure()

    # plt.plot(x, recta(x), '--')
    # plt.plot(Energias, ln_gamWg, '+')

    # plt.xlabel('Eu/K')
    # plt.ylabel(r'$\ln(\gamma_u W/g_u)$')

    # plt.grid('on')

    # plt.show()

    return tab_lineas, modelo, tab_lineas_plot, residuos

def configspecsint_TN(nombre_molecula):

    if isinstance(nombre_molecula, str):

        nombres = [nombre_molecula]

    else:
        nombres = nombre_molecula

    modelo = None
    tab_plot = None

    for nombre_mol in nombres:

        name = nombre_mol.strip()

        if name not in CONFIG_SPEC_SINT_TN:
            raise ValueError(f'Molécula {name} no está disponible no está '
                             'disponible en CONFIG_SPEC_SINT_TN.')

        config = CONFIG_SPEC_SINT_TN[name]

        if isinstance(nombre_molecula, str):

            lineas = True

        else:

            lineas = False

        tab, modelo, tab_plot, residuos = spec_sint_class(config['T_ex'], 
                                                          config['N_col'],
                                                config['mol'],
                                                config['intervalo'],
                                                config['id_splat'],
                                                config['filtro_estructuras'], 
                                                config['FWHM'],
                                                config['T_cont'], config['f0'],
                                                config['v_pik'], f'{
                                                    name}', config['cat_mol'],
                                                config['id_cat'],
                                                config.get('sij', 0*u.D**2),
                                                config.get('aij', 0/u.s),
                                                modeloin=modelo,
                                                dict_espec=config.get(
                                                    'dict_especf', None),
                                                plot_lineas=lineas,
                                                tab_lineas_mol=tab_plot)

    return tab, modelo, tab_plot, residuos

def minimchi2(list_molec, sigma, intervalos, n, tab_lineas,
              dict_resol_espec=None,
              dict_especchi=None, list_lin_noconsid=None, debug=False,
              model_sint=None, dictTcont = None, residuos = False):
    
    if residuos and dictTcont is None:
        raise ValueError("Si residuos=True debes introducir dictTcont.")
    
    if model_sint is None:
        modeloinchi = None
    else:
        modelo_total = model_sint.copy()
        
    dict_TN_fit = {}

    for mol in list_molec:

        freq_lineas = np.array([x.value if hasattr(x, 'value') else x
                                for x in tab_lineas[mol]['freq']], dtype=float)

        param = list_molec[mol]

        freq_lineas = mod_vel(np.abs(param['v_pik']), freq_lineas)

        if not list_lin_noconsid is None:

            list_lin_noconsid = mod_vel(np.abs(param['v_pik']),
                                        list_lin_noconsid)

        T_ex = param['T_ex'].value
        N_col = param['N_col'].value
        deltaT = param['deltaT'].value
        deltaN = param['deltaN'].value
        FWHMvel = param['FWHM'].value

        vect_T = np.linspace(T_ex-deltaT, T_ex+deltaT, n)
        logN_min = np.log10(N_col) - np.log10(1 + deltaN / N_col)
        logN_max = np.log10(N_col) + np.log10(1 + deltaN / N_col)

        vect_N = np.logspace(logN_min, logN_max, n)

        chi2_map = np.zeros((len(vect_T), len(vect_N)))      

        if model_sint is not None:
            if dictTcont is None:
               raise ValueError("Debes introducir un diccionario con la "
                                "Temperatura del continuo") 
            tab0, modelomol, tab_l0, resi0 = spec_sint_class(T_ex*u.K, 
                                                   N_col/u.cm**2,
                                                   param['mol'],
                                                   param['intervalo'],
                                                   param['id_splat'],
                                                   param['filtro_estructuras'],
                                                   param['FWHM'],
                                                   param['T_cont'],
                                                   param['f0'], param['v_pik'],
                                                   mol, param['cat_mol'],
                                                   param['id_cat'],
                                                   modeloin=None,
                                                   dict_espec=dict_especchi,
                                                   plot_lineas=False,
                                                   show_plots=False) 
            
            modelsinmol = {}
            for ventana, fmin_vent, fmax_vent, name_window in intervalos:
                T_cont_vent = dictTcont[name_window]
                Continuo = models.Const1D(amplitude=T_cont_vent)
                modelsinmol[name_window] = ((modelo_total[name_window] - 
                                            modelomol[name_window]) + Continuo)
            modelo_ajustar = modelsinmol

        else:
            if residuos:
                modelo_ajustar = None
            else:
                modelo_ajustar = modeloinchi
        
        if residuos:
            (tab_resi, modelo_resimol, 
             tab_l0, resi0) = spec_sint_class(T_ex*u.K, 
                                              N_col/u.cm**2,
                                              param['mol'],
                                              param['intervalo'],
                                              param['id_splat'],
                                              param['filtro_estructuras'],
                                              param['FWHM'],
                                              param['T_cont'],
                                              param['f0'], param['v_pik'],
                                              mol, param['cat_mol'],
                                              param['id_cat'],
                                              modeloin=None,
                                              dict_espec=dict_especchi,
                                              plot_lineas=False,
                                              show_plots=False)
        
        for i, T in enumerate(vect_T):
            for j, N in enumerate(vect_N):

                tab, modelo, tab_l, resil = spec_sint_class(T*u.K, N/u.cm**2,
                                                     param['mol'],
                                                     param['intervalo'],
                                                     param['id_splat'],
                                                     param['filtro_estructuras'],
                                                     param['FWHM'],
                                                     param['T_cont'],
                                                     param['f0'], param['v_pik'],
                                                     mol, param['cat_mol'],
                                                     param['id_cat'],
                                                     modeloin=modelo_ajustar,
                                                     dict_espec=dict_especchi,
                                                     plot_lineas=False,
                                                     show_plots=False)
                    
                for ventana, fmin_vent, fmax_vent, name_window in intervalos:

                    if isinstance(sigma, u.Quantity):

                        sigma_vent = sigma

                    else:

                        sigma_vent = sigma[name_window]

                    if dict_especchi is None:

                        nombre = ventana.strip()

                        archivos = list(rutacarp.glob(f"*{nombre}*"))

                        ruta_cubo = archivos[0]

                        frec, espec, res = read_spectral_cube(ruta_cubo, rutaregion1)

                        frec = frec.to(u.MHz)

                    elif residuos:

                        frec = dict_especchi[name_window]['frecuencia']
                        espec = dict_especchi[name_window]['Temp_brillo']
                        
                        frec_resid = mod_vel(-param['v_pik'], frec)
                        
                        espec_mol = modelo_resimol[name_window](frec_resid)
                        
                        T_cont = dictTcont[name_window]
                        
                        Continuo = models.Const1D(amplitude=T_cont)
                        
                        espec = espec + espec_mol - Continuo(frec)
                
                    else:
                        
                        frec = dict_especchi[name_window]['frecuencia']
                        espec = dict_especchi[name_window]['Temp_brillo']

                    frec = mod_vel(np.abs(param['v_pik']), frec)
                    mask_lineas_vent = ((fmin_vent.value <= freq_lineas) &
                                        (freq_lineas <= fmax_vent.value))

                    freq_lineas_mol_vent = freq_lineas[mask_lineas_vent]

                    chi2_vent = 0
                    for f in freq_lineas_mol_vent:

                        sigma_vel = FWHMvel/(2*np.sqrt(2*np.log(2)))
                        sigma_freq = (param['f0'] * sigma_vel * (u.km/u.s)
                                      / c.c.to(u.km/u.s)).to(u.MHz)

                        FWHMfreq = 2 * np.sqrt(2*np.log(2)) * sigma_freq

                        if (not list_lin_noconsid is None):

                            if np.any(np.isclose(
                                    f, list_lin_noconsid, atol=0.1)):

                                continue

                        mask_alred_lin = ((f-FWHMfreq.value*2/2 <= frec.value) &
                                          (f+FWHMfreq.value*2/2 >= frec.value))

                        f_alred_lin = frec[mask_alred_lin]

                        espec_model = modelo[name_window](f_alred_lin)

                        espec_alred_lin = espec[mask_alred_lin]

                        chi2_lin = np.sum((espec_alred_lin - espec_model)**2
                                          / sigma_vent**2)

                        chi2_vent += chi2_lin
                        if debug:
                            plt.figure()

                            plt.plot(f_alred_lin, espec_alred_lin, 'r',
                                     drawstyle='steps-mid')

                            plt.plot(f_alred_lin, espec_model, 'k',
                                     drawstyle='steps-mid')

                            plt.axvline(f)

                            plt.show()

                    chi2_map[i, j] += chi2_vent

        i_min, j_min = np.unravel_index(np.argmin(chi2_map), chi2_map.shape)

        T_min = vect_T[i_min]*u.K
        N_min = vect_N[j_min]/u.cm**2
        chi_min = np.min(chi2_map)
        dict_TN_fit[mol] = dict(T_fit=T_min, N_fit=N_min)

        tab, modeloinchi, tab_lin4, resi4 = spec_sint_class(T_min, N_min,
                                                     param['mol'],
                                                     param['intervalo'],
                                                     param['id_splat'],
                                                     param['filtro_estructuras'],
                                                     param['FWHM'],
                                                     param['T_cont'],
                                                     param['f0'], param['v_pik'],
                                                     mol, param['cat_mol'],
                                                     param['id_cat'],
                                                     modeloin=modelo_ajustar)
        if model_sint is not None:
            
            modelo_total = modeloinchi.copy()
    deltchi2_map = chi2_map - chi_min
    ratio_map = chi_min / chi2_map

    x = np.log10(vect_N)
    y = vect_T

    X, Y = np.meshgrid(x, y)

    mask_1 = deltchi2_map <= 2.30
    mask_2 = deltchi2_map <= 6.17
    mask_3 = deltchi2_map <= 11.8

    # Cálculo de las incertidumbres en T y N

    T_vals_1sigma = vect_T[np.any(mask_1, axis=1)]
    T_inf = T_vals_1sigma.min()
    T_sup = T_vals_1sigma.max()
    deltT_sup = np.abs(T_min.value - T_sup)
    deltT_inf = np.abs(T_min.value - T_inf)

    if deltT_sup > deltT_inf:
        deltT = deltT_sup
    else:
        deltT = deltT_inf

    N_vals_1sigma = vect_N[np.any(mask_1, axis=0)]
    N_inf = N_vals_1sigma.min()
    N_sup = N_vals_1sigma.max()
    deltN_sup = np.abs(N_min.value - N_sup)
    deltN_inf = np.abs(N_min.value - N_inf)

    if deltN_sup > deltN_inf:
        deltN = deltN_sup
    else:
        deltN = deltN_inf

    if deltN == 0.0 or deltT == 0.0:

        deltT = np.abs(vect_T[0] - vect_T[1])
        deltN = np.abs(vect_N[0] - vect_N[1])

    plt.figure()

    pcm = plt.pcolormesh(x, y, ratio_map, shading='auto')#Mirar lo de contourf y tal para el suavizad
    plt.colorbar(pcm, label=r'$\chi^2_{\min}/\chi^2$')

    plt.scatter(X[mask_3], Y[mask_3], marker='s', s=300, alpha=0.8,
                label=r'$3\sigma$')
    plt.scatter(X[mask_2], Y[mask_2], marker='s', s=220, alpha=0.8,
                label=r'$2\sigma$')
    plt.scatter(X[mask_1], Y[mask_1], marker='s', s=180, alpha=0.8,
                label=r'$1\sigma$')

    plt.scatter(np.log10(N_min.value), T_min.value, color='red',
                label='Best fit')

    plt.scatter(np.log10(N_col), T_ex, color='green', label='Initial fit')

    plt.xlabel(r'$\log_{10}(N\ /\ \mathrm{cm^{-2}})$')
    plt.ylabel(r'$T\ (\mathrm{K})$')

    labels_legend = [
        Patch(facecolor='blue', edgecolor='blue', linewidth=2,
              label=r'$3\sigma$'),
        Patch(facecolor='orange', edgecolor='orange', linewidth=2,
              label=r'$2\sigma$'),
        Patch(facecolor='green', edgecolor='green', linewidth=2,
              label=r'$1\sigma$'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='red', markersize=8, label='Best fit'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='green', markersize=8, label='Initial fit')]

    plt.title(fr'Mapa de $\chi^2$ para la molécula {mol}')

    plt.legend(handles=labels_legend)
    plt.show()

    return modeloinchi, dict_TN_fit, deltT * u.K, deltN / u.cm**2, chi_min

def get_config(config_dict, key):
    configmol = config_dict[key]
    return {key: configmol}


def chi2_conv(mol, tol, config, list_noconsid, dict_espec, preguntar=False, 
              model_sintc = None, dictT = None, residuos = False):        
        
    configmol = get_config(config, mol)

    T0 = configmol[mol]['T_ex']
    N0 = configmol[mol]['N_col']
    deltT0 = configmol[mol]['deltaT']
    deltN0 = configmol[mol]['deltaN']

    dif = 1000
    mod, dictTN, deltT1, deltN1, chimin = minimchi2(configmol, dict_sigma_vent,
                                                    Banda6, 10, tab_plot,
                                            dict_resol_espec=dict_resol_esp,
                                            dict_especchi=dict_espec,
                                            list_lin_noconsid=list_noconsid, 
                                            debug=False, 
                                            model_sint= model_sintc,
                                            dictTcont= dictT, 
                                            residuos= residuos)

    T1 = dictTN[mol]['T_fit']
    N1 = dictTN[mol]['N_fit']

    if (T1+deltT1 >= T0 + deltT0/1 or T1-deltT1 <= T0 - deltT0/1 or
            N1 + deltN1 >= N0 + deltN0/1 or N1 - deltN1 <= N0 - deltN0/1):

        configmol[mol]['T_ex'] = T1
        configmol[mol]['N_col'] = N1

        T0 = T1
        N0 = N1

    else:

        configmol[mol]['T_ex'] = T1
        configmol[mol]['N_col'] = N1
        configmol[mol]['deltaT'] = deltT1
        configmol[mol]['deltaN'] = deltN1

        T0 = T1
        N0 = N1
        deltT0 = deltT1
        deltN0 = deltN1
    if model_sintc is not None:
        model_sintc = mod
    i = 0
    while dif > tol and i < 10:
        mod, dictTN, deltT1, deltN1, chimin = minimchi2(configmol,
                                                        dict_sigma_vent,
                                                        Banda6, 10,
                                                        tab_plot, 
                                            dict_resol_espec=dict_resol_esp,
                                            dict_especchi=dict_espec,
                                            list_lin_noconsid=list_noconsid, 
                                            debug=False, 
                                            model_sint= model_sintc,
                                            dictTcont= dictT,
                                            residuos= residuos)

        T1 = dictTN[mol]['T_fit']
        N1 = dictTN[mol]['N_fit']

        dif = np.abs(T1-T0).value
        if (T1+deltT1 >= T0 + deltT0/1 or T1-deltT1 <= T0 - deltT0/1 or
                N1 + deltN1 >= N0 + deltN0/1 or N1 - deltN1 <= N0 - deltN0/1):

            configmol[mol]['T_ex'] = T1
            configmol[mol]['N_col'] = N1
            T0 = T1
            N0 = N1

        else:

            configmol[mol]['T_ex'] = T1
            configmol[mol]['N_col'] = N1
            configmol[mol]['deltaT'] = deltT1
            configmol[mol]['deltaN'] = deltN1

            T0 = T1
            N0 = N1
            deltT0 = deltT1
            deltN0 = deltN1
            
        if model_sintc is not None and not residuos:
            model_sintc = mod
            
        i += 1
        if preguntar:
            continuar = input(f'Valor de chi²_min = {chimin}¿Continuar? (Y/N)')

            if continuar == 'N':
                T_fit = T1
                N_fit = N1
                return T_fit, N_fit, deltT1, deltN1

    T_fit = T1
    N_fit = N1
    return T_fit, N_fit, deltT1, deltN1

