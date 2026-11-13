#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:15:47 2026

@author: jorge
"""

from astropy import units as u
from astropy import constants as c
from astropy.modeling import models
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from TFM_line_search import mod_vel, abrir_espectro_region
from TFM_synthetic_model import spec_sint_opacidad

def minimchi2(
        list_molec,
        sigma,
        intervalos,
        n,
        tab_lineas,
        dict_resol_espec=None,
        dict_especchi=None,
        dict_lin_noconsid=None,
        debug=False,
        model_sint=None,
        dictTcont=None,
        residuos=False,
        rutacarp_region=None,
        rutaregion_region=None,
        show_plots=True,
        devolver_tabla_opacidad=False):
    
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
 
        list_lin_noconsid = None

        if dict_lin_noconsid is not None:

            lista_mol = dict_lin_noconsid.get(mol, [])

            list_lin_noconsid = np.asarray([
                x.to_value(u.MHz)
                if isinstance(x, u.Quantity)
                else float(x)
                for x in lista_mol
            ])

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
            tab0, modelomol, tab_l0, resi0 = spec_sint_opacidad(T_ex*u.K, 
                                                   N_col/u.cm**2,
                                                   param['mol'],
                                                   param['intervalo'],
                                                   param['id_splat'],
                                                   param['filtro_estructuras'],
                                                   param['FWHM'],
                                                   param['T_cont'],
                                                   param['v_pik'],
                                                   mol, param['cat_mol'],
                                                   param['id_cat'],
                                                   modeloin=None,
                                                   dict_espec=dict_especchi,
                                                   plot_lineas=False,
                                                   show_plots=False,
                                                   rutacarp_region=rutacarp_region,
                                                   rutaregion_region=rutaregion_region,) 
            
            modelsinmol = {}
            for ventana, fmin_vent, fmax_vent, name_window in intervalos:
                T_cont_vent = dictTcont[name_window]
                modelsinmol[name_window] = ((modelo_total[name_window] - 
                                            modelomol[name_window]) + T_cont_vent)
            modelo_ajustar = modelsinmol

        else:
            if residuos:
                modelo_ajustar = None
            else:
                modelo_ajustar = modeloinchi
        
        if residuos:
            (tab_resi, modelo_resimol, 
             tab_l0, resi0) = spec_sint_opacidad(T_ex*u.K, 
                                              N_col/u.cm**2,
                                              param['mol'],
                                              param['intervalo'],
                                              param['id_splat'],
                                              param['filtro_estructuras'],
                                              param['FWHM'],
                                              param['T_cont'],
                                              param['v_pik'],
                                              mol, param['cat_mol'],
                                              param['id_cat'],
                                              modeloin=None,
                                              dict_espec=dict_especchi,
                                              plot_lineas=False,
                                              show_plots=False,
                                              rutacarp_region=rutacarp_region,
                                              rutaregion_region=rutaregion_region,)
        
        for i, T in enumerate(vect_T):
            for j, N in enumerate(vect_N):
                if modelo_ajustar is not None:
                    modelo_dentro = modelo_ajustar.copy()
                else:
                    modelo_dentro = modelo_ajustar
                    
                tab, modelo, tab_l, resil = spec_sint_opacidad(T*u.K, N/u.cm**2,
                                                     param['mol'],
                                                     param['intervalo'],
                                                     param['id_splat'],
                                                     param['filtro_estructuras'],
                                                     param['FWHM'],
                                                     param['T_cont'],
                                                     param['v_pik'],
                                                     mol, param['cat_mol'],
                                                     param['id_cat'],
                                                     modeloin=modelo_dentro,
                                                     dict_espec=dict_especchi,
                                                     plot_lineas=False,
                                                     show_plots=False,
                                                     rutacarp_region=rutacarp_region,
                                                     rutaregion_region=rutaregion_region,)
                    
                for ventana, fmin_vent, fmax_vent, name_window in intervalos:

                    if isinstance(sigma, u.Quantity):

                        sigma_vent = sigma

                    else:

                        sigma_vent = sigma[name_window]

                    if residuos:

                        frec, espec = abrir_espectro_region(
                            nombre=ventana.strip(),
                            nombre_ventana=name_window,
                            dict_espec=dict_especchi,
                            rutacarp_region=rutacarp_region,
                            rutaregion_region=rutaregion_region,
                            promedio=True,
                        )

                        espec_mol = modelo_resimol[name_window]

                        T_cont = dictTcont[name_window]

                        Continuo = models.Const1D(amplitude=T_cont)

                        espec = espec + espec_mol - Continuo(frec)

                    else:

                        frec, espec = abrir_espectro_region(
                            nombre=ventana.strip(),
                            nombre_ventana=name_window,
                            dict_espec=dict_especchi,
                            rutacarp_region=rutacarp_region,
                            rutaregion_region=rutaregion_region,
                            promedio=True,
                        )

                    frec = mod_vel(np.abs(param['v_pik']), frec)
                    mask_lineas_vent = ((fmin_vent.value <= freq_lineas) &
                                        (freq_lineas <= fmax_vent.value))

                    freq_lineas_mol_vent = freq_lineas[mask_lineas_vent]

                    mask_chi2_vent = np.zeros(len(frec), dtype=bool)

                    sigma_vel = FWHMvel / (2 * np.sqrt(2 * np.log(2)))

                    for f in freq_lineas_mol_vent:

                        if list_lin_noconsid is not None:
                            if np.any(np.isclose(f, list_lin_noconsid, atol=0.1)):
                                continue

                        sigma_freq = (
                            f * u.MHz * sigma_vel * (u.km / u.s)
                            / c.c.to(u.km / u.s)
                        ).to(u.MHz)

                        FWHMfreq = 2 * np.sqrt(2 * np.log(2)) * sigma_freq

                        # Máscara alrededor de la línea.
                        # De momento usamos ±1 FWHM en frecuencia.
                        ancho_chi2 = FWHMfreq.to_value(u.MHz)
                    
                        mask_linea = (
                            (frec.to_value(u.MHz) >= f - ancho_chi2) &
                            (frec.to_value(u.MHz) <= f + ancho_chi2)
                        )

                        mask_chi2_vent |= mask_linea

                    # Si no hay canales útiles en esta ventana, no suma nada
                    if np.sum(mask_chi2_vent) == 0:
                        continue

                    espec_model = modelo[name_window][mask_chi2_vent]
                    espec_obs = espec[mask_chi2_vent]

                    residuo = espec_obs - espec_model

                    # Convertimos a valores adimensionales para evitar problemas con Quantity
                    chi2_vals = ((residuo / sigma_vent) ** 2).decompose().value

                    # Quitamos NaN/inf
                    chi2_vals = chi2_vals[np.isfinite(chi2_vals)]

                    if len(chi2_vals) == 0:
                        continue

                    chi2_vent = np.sum(chi2_vals)

                    chi2_map[i, j] += chi2_vent

                    if debug:
                        plt.figure()
                        plt.plot(
                            frec[mask_chi2_vent],
                            espec_obs,
                            "r",
                            drawstyle="steps-mid",
                            label="Observado",
                        )
                        plt.plot(
                            frec[mask_chi2_vent],
                            espec_model,
                            "k",
                            drawstyle="steps-mid",
                            label="Modelo",
                        )

                        for f in freq_lineas_mol_vent:
                            plt.axvline(f)
                    
                        plt.legend()
                        plt.show()
                    

        i_min, j_min = np.unravel_index(np.argmin(chi2_map), chi2_map.shape)

        T_min = vect_T[i_min]*u.K
        N_min = vect_N[j_min]/u.cm**2
        chi_min = np.min(chi2_map)

        if modelo_ajustar is not None:
            modelo_dentro = modelo_ajustar.copy()
        else:
            modelo_dentro = modelo_ajustar

        tab_opacidad, modeloinchi, tab_lin4, resi4 = (spec_sint_opacidad(T_min,
                                                      N_min,
                                                      param["mol"],
                                                      param["intervalo"],
                                                      param["id_splat"],
                                                      param["filtro_estructuras"],
                                                      param["FWHM"],
                                                      param["T_cont"],
                                                      param["v_pik"],
                                                      mol,
                                                      param["cat_mol"],
                                                      param["id_cat"],
                                                      modeloin=modelo_dentro,
                                                      dict_espec=dict_especchi,
                                                      plot_lineas=False,
                                                      show_plots=show_plots,
                                                      rutacarp_region=rutacarp_region,
                                                      rutaregion_region=rutaregion_region,
                                                      ))
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

        resultado_mol = {
            "T_fit": T_min,
            "N_fit": N_min,
            "deltaT": deltT * u.K,
            "deltaN": deltN / u.cm**2,
            "chi_min": chi_min,
            "chi2_map": chi2_map,
        }

        if devolver_tabla_opacidad:
            resultado_mol["tab_lineas_opacidad"] = (
                tab_opacidad.copy()
            )

        dict_TN_fit[mol] = resultado_mol

        plt.figure()
#Mirar lo de contourf y tal para el suavizad
        pcm = plt.pcolormesh(x, y, ratio_map, shading='auto')
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



    return modeloinchi, dict_TN_fit

def get_config(config_dict, key):
    configmol = config_dict[key]
    return {key: configmol}


def chi2_conv_fino(mol, tol, config, dict_noconsid, dict_espec,
                   sigma, intervalos, tab_lineas,
                   dict_resol_espec=None,
                   preguntar=False, model_sintc=None, dictT=None,
                   residuos=False,
                   n_grid=10, max_iter=10, debug=False,
                   rutacarp_region=None,
                   rutaregion_region=None):

    configmol = get_config(config, mol)

    T0 = configmol[mol]["T_ex"]
    N0 = configmol[mol]["N_col"]
    deltT0 = configmol[mol]["deltaT"]
    deltN0 = configmol[mol]["deltaN"]

    dif = 1000

    mod, dictTN = minimchi2(
        configmol,
        sigma,
        intervalos,
        n_grid,
        tab_lineas,
        dict_resol_espec=dict_resol_espec,
        dict_especchi=dict_espec,
        dict_lin_noconsid=dict_noconsid,
        debug=False,
        model_sint=model_sintc,
        dictTcont=dictT,
        residuos=residuos,
        rutacarp_region=rutacarp_region,
        rutaregion_region=rutaregion_region,
    )

    T1 = dictTN[mol]["T_fit"]
    N1 = dictTN[mol]["N_fit"]

    deltT1 = dictTN[mol]["deltaT"]
    deltN1 = dictTN[mol]["deltaN"]
    chimin = dictTN[mol]["chi_min"]

    if (
        T1 + deltT1 >= T0 + deltT0
        or T1 - deltT1 <= T0 - deltT0
        or N1 + deltN1 >= N0 + deltN0
        or N1 - deltN1 <= N0 - deltN0
    ):

        configmol[mol]["T_ex"] = T1
        configmol[mol]["N_col"] = N1

        T0 = T1
        N0 = N1

    else:

        configmol[mol]["T_ex"] = T1
        configmol[mol]["N_col"] = N1
        configmol[mol]["deltaT"] = deltT1
        configmol[mol]["deltaN"] = deltN1

        T0 = T1
        N0 = N1
        deltT0 = deltT1
        deltN0 = deltN1

    if model_sintc is not None and not residuos:
        model_sintc = mod

    i = 0

    while dif > tol and i < max_iter:

        mod, dictTN = minimchi2(
            configmol,
            sigma,
            intervalos,
            n_grid,
            tab_lineas,
            dict_resol_espec=dict_resol_espec,
            dict_especchi=dict_espec,
            dict_lin_noconsid=dict_noconsid,
            debug=False,
            model_sint=model_sintc,
            dictTcont=dictT,
            residuos=residuos,
            rutacarp_region=rutacarp_region,
            rutaregion_region=rutaregion_region,
        )

        T1 = dictTN[mol]["T_fit"]
        N1 = dictTN[mol]["N_fit"]

        deltT1 = dictTN[mol]["deltaT"]
        deltN1 = dictTN[mol]["deltaN"]
        chimin = dictTN[mol]["chi_min"]

        dif = np.abs(T1 - T0).value

        if (
            T1 + deltT1 >= T0 + deltT0
            or T1 - deltT1 <= T0 - deltT0
            or N1 + deltN1 >= N0 + deltN0
            or N1 - deltN1 <= N0 - deltN0
        ):

            configmol[mol]["T_ex"] = T1
            configmol[mol]["N_col"] = N1

            T0 = T1
            N0 = N1

        else:

            configmol[mol]["T_ex"] = T1
            configmol[mol]["N_col"] = N1
            configmol[mol]["deltaT"] = deltT1
            configmol[mol]["deltaN"] = deltN1

            T0 = T1
            N0 = N1
            deltT0 = deltT1
            deltN0 = deltN1

        if model_sintc is not None and not residuos:
            model_sintc = mod

        i += 1

        if preguntar:
            continuar = input(
                f"Valor de chi²_min = {chimin}. ¿Continuar? (Y/N)"
            )

            if continuar == "N":
                return T1, N1, deltT1, deltN1

    return T1, N1, deltT1, deltN1