#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:04:58 2026

@author: jorge
"""

from astropy import units as u
from astropy import constants as c
from astropy.modeling import models, fitting
from astropy.table import QTable
import matplotlib.pyplot as plt
import numpy as np

from TFM_io_cubes import read_spectral_cube

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

def abrir_espectro_region(nombre, nombre_ventana,
                          dict_espec=None,
                          rutacarp_region=None,
                          rutaregion_region=None,
                          promedio=True):
    """
    Devuelve frecuencia y espectro para una ventana.

    Prioridad:
        1. Si dict_espec existe, usa el cubo/espectro ya cargado.
        2. Si no, abre el cubo desde rutacarp_region y rutaregion_region.
    """

    if dict_espec is not None:
        frec = dict_espec[nombre_ventana]["frecuencia"]
        espec = dict_espec[nombre_ventana]["Temp_brillo"]
        return frec.to(u.MHz), espec

    if rutacarp_region is None or rutaregion_region is None:
        raise ValueError(
            "No se ha proporcionado dict_espec ni rutas de región. "
            "Debes pasar dict_espec o bien rutacarp_region/rutaregion_region."
        )

    archivos = sorted(rutacarp_region.glob(f"*{nombre}*"))

    if len(archivos) == 0:
        raise FileNotFoundError(
            f"No se encontró ningún cubo para {nombre} en {rutacarp_region}"
        )

    ruta_cubo = archivos[0]

    frec, espec, res = read_spectral_cube(
        ruta_cubo,
        rutaregion_region,
        promedio=promedio,
    )

    return frec.to(u.MHz), espec

# Buscador de líneas


def buscador_lin(frec_busc, intervalos, long_int='tot', ajuste=False,
                 v=None, dict_espec=None,
                 rutacarp_region=None,
                 rutaregion_region=None):
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

    frec, espec = abrir_espectro_region(nombre=nombre,
                                        nombre_ventana=nombre_ventana,
                                        dict_espec=dict_espec,
                                        rutacarp_region=rutacarp_region,
                                        rutaregion_region=rutaregion_region,
                                        promedio=True)

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
                  tab=False, dict_especm=None, rutacarp_region=None,
                  rutaregion_region=None):
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
                                                        dict_espec=dict_especm,
                                            rutacarp_region= rutacarp_region,
                                        rutaregion_region= rutaregion_region)

            Tab.add_row((frec, Tlin, Tcont, sigma, FWHM, int_integ))

        else:

            buscador_lin(fb, interval, long_int=li, ajuste=fit, v=v_mult,
                         dict_espec=dict_especm)

        print(f'Proceso de busqueda de la frecuencia {fb.value}{fb.unit} '
              'finalizado.')

    if tab is True:
        return Tab
    
    
def buscador_lin_cubo_compl(frec_busc_c, intervalos_c, long_int_c='tot', 
                            ajuste_c=False, v_c=None, dict_espec_c=None,
                            rutacarp_region=None,
                            rutaregion_region=None):
    
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
    
    frec, espec = abrir_espectro_region(nombre=nombre,
                                        nombre_ventana=nombre_ventana,
                                        dict_espec=dict_espec_c,
                                        rutacarp_region=rutacarp_region,
                                        rutaregion_region=rutaregion_region,
                                        promedio=False)
        
        
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
                     dict_espec=None, plots=True,
                     rutacarp_region=None,
                     rutaregion_region=None):
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

    frec, espec = abrir_espectro_region(nombre=nombre,
                                        nombre_ventana=nombre_ventana,
                                        dict_espec=dict_espec,
                                        rutacarp_region=rutacarp_region,
                                        rutaregion_region=rutaregion_region,
                                        promedio=True)

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
    if plots:
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
    
def busc_mult_lin_v(list_frec, v_busc, interval, list_long,
                    fit=False, tab=False, dict_especm=None, plots=True,
                    rutacarp_region=None,
                    rutaregion_region=None):
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
                                                dict_espec=dict_especm,
                                                plots=plots,
                                               rutacarp_region=rutacarp_region,
                                           rutaregion_region=rutaregion_region)

            Tab.add_row((vlin, fb, Tlin, Tcont, sigma, FWHM, int_integ, deltaW,
                         M0, M1, M2))

        else:

            buscador_lin_vel(fb, v_busc, interval, long_int=li, ajuste=fit,
                             dict_espec=dict_especm, plots=plots,
                             rutacarp_region=rutacarp_region,
                             rutaregion_region=rutaregion_region)

        print(f'Proceso de busqueda de la frecuencia {fb.value}{fb.unit} '
              'finalizado.')

    if tab is True:
        return Tab

def buscador_lin_vel_cubo_comp(frec_busc_c, v_busc_c, intervalos_c,
                               long_int_c='tot',
                               ajuste_c=False, Tcont_fix_c=None,
                               anch_lin_c=None,
                               dict_espec_c=None,
                               plots=True,
                               rutacarp_region=None,
                               rutaregion_region=None):
    
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
    
    frec, espec = abrir_espectro_region(nombre=nombre,
                                        nombre_ventana=nombre_ventana,
                                        dict_espec=dict_espec_c,
                                        rutacarp_region=rutacarp_region,
                                        rutaregion_region=rutaregion_region,
                                        promedio=False)
        
        
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

            # No intentamos ajustar píxeles vacíos
            if np.count_nonzero(np.isfinite(espec_pixel.value)) < 5:
                continue

            cubo_pixel = {
                nombre_ventana: {
                    "frecuencia": frec,
                    "Temp_brillo": espec_pixel,
                }
            }

            if ajuste_c:

                try:
                    resultado = buscador_lin_vel(
                        frec_busc_c,
                        v_busc_c,
                        intervalos_c,
                        long_int_c,
                        ajuste_c,
                        Tcont_fix_c,
                        anch_lin_c,
                        cubo_pixel,
                        plots=plots,
                    )

                    if resultado is None:
                        continue

                    (
                        Tmax,
                        freclin,
                        Tcont,
                        sigma,
                        FWHM,
                        int_integrada,
                        M0,
                        M1,
                        M2,
                        deltaW,
                    ) = resultado

                except (
                    ValueError,
                    TypeError,
                    IndexError,
                    RuntimeError,
                    np.linalg.LinAlgError,
                ):
                    continue

                # Guardamos el resultado del ajuste de este píxel
                Tmax_pix[y, x] = Tmax.value
                freclin_pix[y, x] = freclin.value
                Tcont_pix[y, x] = Tcont.value
                sigma_pix[y, x] = sigma.value
                FWHM_pix[y, x] = FWHM.value
                int_integrada_pix[y, x] = int_integrada.value
                M0_pix[y, x] = M0.value
                M1_pix[y, x] = M1.value
                M2_pix[y, x] = M2.value
                deltaW_pix[y, x] = deltaW.value

            else:

                buscador_lin_vel(
                    frec_busc_c,
                    v_busc_c,
                    intervalos_c,
                    long_int_c,
                    ajuste_c,
                    Tcont_fix_c,
                    anch_lin_c,
                    cubo_pixel,
                    plots=plots,
                )
                
    if ajuste_c:
        
        return (Tmax_pix, freclin_pix, Tcont_pix, sigma_pix, FWHM_pix, 
                int_integrada_pix, M0_pix, M1_pix, M2_pix, deltaW_pix)
    
def busc_mult_lin_v_cubo(list_frec, v_busc, interval, list_long,
                         fit=False, list_Tcont=None, list_anch=None,
                         dict_espec_c=None, plots = True):

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
            dict_espec_c=dict_espec_c, plots = plots
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