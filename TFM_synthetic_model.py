#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:09:20 2026

@author: jorge
"""
from astropy import units as u
from astropy import constants as c
from astropy.modeling import models
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import os
from pathlib import Path
from TFM_line_search import mod_vel, abrir_espectro_region
from TFM_splatalogue_tools import buscador_splatalogue_cdms
from TFM_filtering import filtrador_tablas
from TFM_rotational_diagram import Q, to_float

def J_nu(nu, T):
    
    return (c.h * nu / c.k_B) / (np.exp((c.h * nu / (c.k_B * T))) - 1)

def tau_lineprofile(frec, T_ex, N_col, func_parti, deg, f0, Eu, Aij):
    
    Nu = N_col * deg / func_parti * np.exp(-Eu / ( T_ex))
    
    tau_phi = (c.c**2 / (8 * np.pi * frec ** 2) * Aij * Nu * 
               (np.exp((c.h * frec) / (c.k_B * T_ex)) - 1))

    return tau_phi
    

def spec_sint(n, m, molecula, intervalo1, id_splat1, filtro_estructuras1,
              anch_lin, Tcont, f0, dict_espec=None, rutacarp_region=None,
              rutaregion_region=None):
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

        frec_med, espec_med = abrir_espectro_region(
            nombre=ventana.strip(),
            nombre_ventana=name_window,
            dict_espec=dict_espec,
            rutacarp_region=rutacarp_region,
            rutaregion_region=rutaregion_region,
            promedio=True)

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
                    show_plots=True, save_plots=False, save_dir=None,
                    plot_prefix=None, rutacarp_region = None, 
                    rutaregion_region = None):
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
    
    if save_plots:
        if save_dir is None:
            raise ValueError(
                "Si save_plots=True, debes proporcionar save_dir."
            )

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if plot_prefix is None:
        plot_prefix = name_mol

    if str(plot_prefix) == "complete_model":
        modelo_label = "Modelo completo"
    else:
        modelo_label = name_mol

    def _safe_name(text):
        """
        Limpia strings para usarlos como nombre de archivo.
        """
        text = str(text)
        for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]"]:
            text = text.replace(ch, "_")
        return text

    def _save_current_fig(filename):
        """
        Guarda la figura actual como PDF si save_plots=True.
        """
        if save_plots:
            out = save_dir / filename
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[spec_sint] Figura guardada en: {out}")
    
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
    gammau = (8*np.pi*c.k_B*freq**2)/(c.h*c.c**3*Aij)
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
            if name_window in modeloin:
                modelo[name_window] = modeloin[name_window]
            else:
                if isinstance(Tcont, u.Quantity):
                    modelo[name_window] = models.Const1D(amplitude=Tcont)
                else:
                    Tcont_window = Tcont[name_window]
                    modelo[name_window] = models.Const1D(amplitude=Tcont_window)

        for f, T in zip(freq_inter, T_lin_inter):

            modelo[name_window] += models.Gaussian1D(amplitude=T,
                                                     mean=f,
                                                     stddev=sigma_freq)
        frec_med, espec_med = abrir_espectro_region(
            nombre=ventana.strip(),
            nombre_ventana=name_window,
            dict_espec=dict_espec,
            rutacarp_region=rutacarp_region,
            rutaregion_region=rutaregion_region,
            promedio=True,)

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
                         label='Modelo sintético ópticamente delgado')
                plt.plot([], [], ' ', label=fr'T_ex = {T_ex.value:.1f} K')
                plt.plot([], [], ' ', label=fr'N_col = {N_col.value:.2e} 1/cm²')

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

                filename = (
                    f"{_safe_name(plot_prefix)}_"
                    f"{_safe_name(name_window)}_"
                    f"{f:.1f}_line_delgado.pdf"
                    )

                _save_current_fig(filename)

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

                    plt.plot([], [], ' ', 
                             label=f'{name_window}//{modelo_label}')
                    plt.plot(frec_med, espec_med, 'b', drawstyle='steps-mid',
                             label='Espectro observado')
                    plt.plot(frec_sint, espec_sint, '--r',
                             label='Modelo sintético ópticamente delgado')

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
                    
                    filename = (
                        f"{_safe_name(plot_prefix)}_"
                        f"{_safe_name(name_window)}_"
                        f"{int(xmin)}_{int(xmax)}_model_delgado.pdf"
                    )

                    _save_current_fig(filename)

                    plt.show()
                    plt.close()

            else:
                plt.figure()

                plt.plot([], [], ' ', label=f'{name_window}//{modelo_label}')
                plt.plot(frec_med, espec_med, drawstyle='steps-mid',
                         color='b', label='Espectro observado')
                plt.plot(frec_sint, espec_sint, '--r',
                         label='Modelo sintético ópticamente delgado')

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
                
                filename = (
                    f"{_safe_name(plot_prefix)}_"
                    f"{_safe_name(name_window)}_model_delgado.pdf"
                )

                _save_current_fig(filename)

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

def spec_sint_opacidad(T_ex, N_col, molecula, intervalo1, id_splat1, 
                       filtro_estructuras1,
                       anch_lin, Tcont, vpic, name_mol, cat_mol, id_cat,
                       sijmu2=0*u.D**2, aij=0/u.s, modeloin=None,
                       dict_espec=None, plot_lineas=False, tab_lineas_mol=None,
                       show_plots=True, dict_sigma=None,
                       nsigma_lineas=1.0,
                       rutacarp_region=None,
                       rutaregion_region=None):
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

    sigma_v = anch_lin.cgs / (2*np.sqrt(2*np.log(2)))

    tab_lineas = buscador_splatalogue_cdms(molecula, intervalo1, 1000*u.K,
                                           id_splat1,
                                           filtro_estructuras=filtro_estructuras1,
                                           linelist=[cat_mol])

    tab_lineas['aij'] = 10**tab_lineas['aij'] 
    tab_lineas = filtrador_tablas(tab_lineas, sijmu2, 1000*u.K, aij)

    tab_lineas['Freq_corrected_v'] = mod_vel(vpic, tab_lineas['orderedfreq'])
    tab_lineas['Compuesto'] = name_mol

    Energias = tab_lineas['upper_state_energy_K'] * u.K
    freq = tab_lineas['orderedfreq'] * u.MHz
    Aij = tab_lineas['aij']/u.s
    g = to_float(tab_lineas['upperStateDegen'])

    # Cada posición corresponde directamente a una fila
    # concreta de tab_lineas.
    tau_col = np.full(
        len(tab_lineas),
        np.nan,
        dtype=float,
    )

    descrp_trans = np.full(
        len(tab_lineas),
        "No calculada",
        dtype="U32",
    )
    
    if modeloin is None:
        modelo = {}
        
    else:
        modelo = modeloin

    #Esto es para guardar ls líneas intensas
    if tab_lineas_mol is None:

        tab_lineas_plot = {name_mol: dict(freq=[], label=[])}
        
    else:
        tab_lineas_plot = tab_lineas_mol
        tab_lineas_plot[name_mol] = dict(freq=[], label=[])
    
    residuos = {}
    
    for ventana, fmin, fmax, name_window in intervalo1:
        
        mask_line_spw = (
            (fmin <= freq)
            & (freq <= fmax)
        )

        # Índices originales dentro de tab_lineas.
        # Así cada tau se asigna después a su transición.
        indices_line_spw = np.flatnonzero(
            np.asarray(
                mask_line_spw,
                dtype=bool,
            )
        )

        f0_in = freq[mask_line_spw]
        Eu_in = Energias[mask_line_spw]
        Aij_in = Aij[mask_line_spw]
        g_in = g[mask_line_spw]

        frec_vent, espec_vent = abrir_espectro_region(
            nombre=ventana.strip(),
            nombre_ventana=name_window,
            dict_espec=dict_espec,
            rutacarp_region=rutacarp_region,
            rutaregion_region=rutaregion_region,
            promedio=True)

        frec_synt = mod_vel(-vpic, frec_vent)
        model_tau = np.zeros(len(frec_vent))

        for (indice_linea, f0, Eu, aij, deg,) in zip(indices_line_spw, f0_in,
                                                     Eu_in, Aij_in, g_in,):
        
            tau_phi = tau_lineprofile(frec_synt, T_ex, N_col, QT, deg, f0, Eu,
                                      aij)
        
            sigma_freq = sigma_v / c.c * f0
            
            phi_nu = (1 / (np.sqrt(2*np.pi) * sigma_freq) * 
                      np.exp(-(frec_synt - f0)**2 / (2 * sigma_freq ** 2)) )
        
            tau_profile = tau_phi * phi_nu
            
            tau_profile = tau_profile.cgs
            
            tau_max = np.nanmax(
                tau_profile
            )

            tau_max_value = u.Quantity(
                tau_max
            ).to_value(
                u.dimensionless_unscaled
            )

            # Guardar en la fila original de tab_lineas.
            tau_col[indice_linea] = (
                tau_max_value
            )

            if tau_max_value < 0.3:

                descrp_trans[
                    indice_linea
                ] = "Ópticamente delgada"

            elif tau_max_value < 1:

                descrp_trans[
                    indice_linea
                ] = "Opacidad moderada"

            else:

                descrp_trans[
                    indice_linea
                ] = "Ópticamente gruesa"
            
            model_tau += tau_profile
            
        if isinstance(Tcont, u.Quantity):
                    
            T_cont_vent = Tcont

        else:
            T_cont_vent = Tcont[name_window]
        
        if dict_sigma is None:
            sigma_vent = 1.0 * u.K

        elif isinstance(dict_sigma, u.Quantity):
            sigma_vent = dict_sigma.to(u.K)

        else:
            sigma_vent = dict_sigma[name_window].to(u.K)
        
        J_line = J_nu(frec_synt, T_ex).to(u.K)
        J_cont = J_nu(frec_synt, T_cont_vent).to(u.K) #ESTO HAY QUE VER SI HAY 
                                             #PONER LA T_CONT O EL 2.7 DEL CMB
        
        if isinstance(model_tau, u.Quantity):
            tau_val = model_tau.to_value(u.dimensionless_unscaled)
        else:
            tau_val = np.asarray(model_tau, dtype=float)

        tau_val = np.where(np.isfinite(tau_val), tau_val, np.nan)

        tau_val = np.clip(tau_val, 0.0, 100.0)

        factor_tau = 1.0 - np.exp(-tau_val)

        if modeloin is None:
    
            molec_model = (J_line - T_cont_vent) * factor_tau
                      
            modelo[name_window] = molec_model + T_cont_vent

        else:
    
            molec_model = (J_line - J_cont) * factor_tau
            modelo[name_window] += molec_model
                               
       # Guardamos las líneas intensas.

        for f0 in f0_in:
           
            f_obs = mod_vel(vpic, f0)
    
            mask_linea = (
               (f_obs - 1 * u.MHz < frec_vent)
               & (frec_vent < f_obs + 1 * u.MHz)
           )
    
            if np.sum(mask_linea) == 0:
                continue

            molec_model_line = molec_model[mask_linea]
       
            if np.all(~np.isfinite(molec_model_line.value)):
                continue

            pico_modelo = np.nanmax(molec_model_line).to(u.K)

            if pico_modelo > nsigma_lineas * sigma_vent:
        
                tab_lineas_plot[name_mol]["freq"].append(f_obs)
                tab_lineas_plot[name_mol]["label"].append(name_mol)
        
        #Cálculo de residuos
        
        residuos[name_window] = {"frecuencia": frec_vent,
                "Temp_brillo": espec_vent - modelo[name_window] + T_cont_vent,}
        
        #Representación
        
        if show_plots and plot_lineas:
            
            for f0 in tab_lineas_plot[name_mol]['freq']:
                
                mask_linea = (f0 - 20 * u.MHz < frec_vent) & (f0
                                                    + 20 * u.MHz > frec_vent)
                
                frec_line = frec_vent[mask_linea]
                line_med = espec_vent[mask_linea]
                model_line = modelo[name_window][mask_linea]
                
                if np.sum(mask_linea) == 0:
                    continue
                
                plt.figure()
                
                plt.plot([],[], ' ', label = fr'{name_window} // {name_mol}')
                
                plt.plot(frec_line, line_med, 'b', drawstyle='steps-mid',
                         label = 'espectro observado')
                plt.plot(frec_line, model_line, 'r', drawstyle='steps-mid',
                         label = 'espectro sintético')
                
                plt.plot([], [], ' ', label=fr'T_ex = {T_ex.value:.1f} K')
                plt.plot([], [], ' ', 
                         label=fr'N_col = {N_col.value:.2e} 1/cm²')
                
                plt.xlabel('Frecuencia (MHz)')
                plt.ylabel('Temperatura de brillo (K)')
                
                plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                           borderaxespad=0)
                
                plt.show()
            
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
                    
                    mask_zoom = ((xmin <= frec_vent.value) & 
                                 (frec_vent.value <= xmax))
                    frec_zoom = frec_vent[mask_zoom]
                    espec_zoom = espec_vent[mask_zoom]
                    model_zoom = modelo[name_window][mask_zoom]
                    
                    plt.figure()
                    
                    plt.plot([],[], ' ',
                             label = f'{name_window} // Modelo Completo')
                    
                    plt.plot(frec_zoom, espec_zoom, 'b', drawstyle='steps-mid',
                             label = 'Espectro observado')
                    
                    plt.plot(frec_zoom, model_zoom, 'r', drawstyle='steps-mid',
                             label = 'Espectro sintético')
                    
                    for mol in tab_lineas_plot:
                        
                        for flin, labellin in zip(tab_lineas_plot[mol]['freq'],
                                                tab_lineas_plot[mol]['label']):
                            
                            if xmin <= flin.value <= xmax:
                                
                                plt.axvline(flin.value, color = 'k', 
                                            linestyle = ':', alpha = 0.5)
                                
                                plt.text(flin.value - 2, 
                                         np.max(espec_zoom.value), 
                                         labellin, rotation = 90, fontsize = 8,
                                         ha = 'center', va =  'top')
                    
                    plt.xlabel('Frecuencia (MHz)')
                    plt.ylabel('Temperatura de brillo (K)')
                    
                    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                               borderaxespad=0)
                    
                    plt.show()
                    
            else:
                
                plt.figure()
        
                plt.plot([],[], ' ',
                         label = f'{name_window} // Modelo Completo')
                
                plt.plot(frec_vent, espec_vent, 'b',drawstyle='steps-mid',
                         label = 'Espectro observado')
                plt.plot(frec_vent, modelo[name_window], 'r',
                         drawstyle='steps-mid', label = 'Espectro sintético')
        
                for mol in tab_lineas_plot:
                    
                    for flin, labellin in zip(tab_lineas_plot[mol]['freq'],
                                            tab_lineas_plot[mol]['label']):
                        
                        if (np.min(frec_vent.value) <= flin.value 
                            <=np.max(frec_vent.value)):
                            
                            plt.axvline(flin.value, color = 'k', 
                                        linestyle = ':', alpha = 0.5)
                            
                            plt.text(flin.value - 2, np.max(espec_vent.value), 
                                     labellin, rotation = 90, fontsize = 8,
                                     ha = 'center', va =  'top')
                            
                plt.xlabel('Frecuencia (MHz)')
                plt.ylabel('Temperatura de brillo (K)')
                
                plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                           borderaxespad=0)
                
                plt.show()

    tab_lineas['tau'] = tau_col
    tab_lineas['Descripcion'] = descrp_trans

    return tab_lineas, modelo, tab_lineas_plot, residuos