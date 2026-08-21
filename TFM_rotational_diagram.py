#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:01:23 2026

@author: jorge
"""

from astropy import units as u
from astropy import constants as c
from astroquery.linelists.cdms import CDMS
from astroquery.jplspec import JPLSpec
import matplotlib.pyplot as plt
import numpy as np
import re

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

        if len(mol) == 0:
            raise ValueError(f"No se encontró ninguna molécula JPL con TAG={id_cat}")

        if len(mol) > 1:
            raise ValueError(f"Hay más de una molécula JPL con TAG={id_cat}")

        T_cols = np.array(tab_JPL.meta['Temperature (K)'], dtype=float)

        q_cols = [f'QLOG{i}' for i in range(1, len(T_cols) + 1)]
        lgQ_vals = np.array([mol[c][0] for c in q_cols], dtype=float)

        mask = np.isfinite(T_cols) & np.isfinite(lgQ_vals)
        T_cols = T_cols[mask]
        lgQ_vals = lgQ_vals[mask]

    order = np.argsort(T_cols)
    T_cols = T_cols[order]
    lgQ_vals = lgQ_vals[order]

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
                        freq_noconsid=None, plot_Q=True,
                        save_path=None, plots = True):
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
    pol[0] corresponds to the slope m.
    pol[1] corresponds to the intercept b.

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

    if len(tab_filtrada1) == 2:
        
        pol = np.polyfit(Eu.value, y, deg = 1)
    else:
        pol, cov = np.polyfit(Eu.value, y, deg=1,  cov=True)

    x = np.linspace(np.min(Eu.value), np.max(Eu.value), 10000)

    recta = np.polyval(pol, x)

    T_ex = -1/pol[0]*u.K
    
    if len(tab_filtrada1) == 2:
        
        deltaTex = 0.1 * T_ex
    else:
        
        deltaTex = (1/pol[0]**2) * np.sqrt(cov[0, 0]) * u.K

    #  cálculo de la función de partición

    Q_Tex, deltQ_Tex = Q(T_ex, cat_mol, id_cat, deltaTex, True, plot_Q)
    N_col = Q_Tex*np.exp(pol[1])/u.cm**2

    # dN_dQ = np.exp(pol[1])
    # dN_dpol = Q_Tex * np.exp(pol[1])

    # deltaN_col = np.sqrt((dN_dQ * deltQ_Tex)**2 +
    #                    (dN_dpol * np.sqrt(cov[1,1]))**2)/u.cm**2

    if len(tab_filtrada1) > 2:
        
        sigma_b = np.sqrt(cov[1, 1])

        deltaN_col = np.sqrt((Q_Tex * np.exp(pol[1]) * sigma_b)**2)/u.cm**2
    else:
        deltaN_col = 0.1 * N_col
    # Incertidumbre del ajuste

    if len(tab_filtrada1) > 2:

        var_yfit = (x**2)*cov[0, 0] + cov[1, 1] + 2*x*cov[0, 1]
        sig_yfit = np.sqrt(var_yfit)

        k = 3  # el numero de sigmas que tenemos en cuenta en el ajuste

        rectalo = recta - k*sig_yfit
        rectahi = recta + k*sig_yfit

    # ============================================================
    # Representación
    # ============================================================

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Datos experimentales con barras de error
    ax.errorbar(
        Eu.value,
        y,
        yerr=deltay,
        fmt='o',
        color='red',
        markersize=4,
        capsize=3,
        label='Experimental data'
    )

    # Ajuste lineal
    ax.plot(
        x,
        recta,
        '--',
        color='blue',
        linewidth=1.5,
        label='Linear fit'
    )

    # Banda de incertidumbre
    if len(tab_filtrada1) > 2:
        ax.fill_between(
            x,
            rectalo,
            rectahi,
            alpha=0.25,
            label=fr'{k:.2g}$\sigma$ band'
        )
    
    # Resultados del ajuste en la leyenda
    ax.plot(
        [],
        [],
        ' ',
        label=(
            fr'$T_{{\rm ex}} = {T_ex.value:.1f} '
            fr'\pm {deltaTex.value:.1f}\,$K'
        )
    )
    
    ax.plot(
        [],
        [],
        ' ',
        label=(
            fr'$N = {N_col.to_value(1/u.cm**2):.2e} '
            fr'\pm {deltaN_col.to_value(1/u.cm**2):.2e}'
            fr'\,\mathrm{{cm^{{-2}}}}$'
        )
    )

    # ============================================================
    # Ejes
    # ============================================================
    
    ax.set_xlabel(
        r'$E_u$ [K]',
        fontsize=13
    )
    
    ax.set_ylabel(
        r'$\ln(\gamma_u W/g_u\,[\mathrm{cm}^{-2}])$',
        fontsize=13
    )
    
    ax.tick_params(
        axis='both',
        labelsize=11
    )

    # ============================================================
    # Título y rejilla
    # ============================================================

    ax.set_title(
        'Rotational diagram for ' + elemento,
        fontsize=14
    )

    ax.grid(
        True,
        alpha=0.4
    )

    # ============================================================
    # Leyenda
    # ============================================================

    ax.legend(
        loc='lower left',
        fontsize=10,
        frameon=True
    )

    fig.tight_layout()
    # ============================================================
    # Guardar
    # ============================================================

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"[diagrot] Figura guardada en: {save_path}"
        )



    plt.show()

    if not plots:
        plt.close(fig)

    return (T_ex,deltaTex, tab_filtrada1, N_col.to(1/u.cm**2), 
            deltaN_col.to(1/u.cm**2), pol, Q_Tex)

def diagrama_rot_pixeles(elemento, id_cat, tab_pixeles, B0, cat_mol,
                         freq_noconsid=None, wcs_ref=None, plots = True):
    
    ny, nx = tab_pixeles.shape

    N_col_map = np.full((ny, nx), np.nan)
    T_ex_map = np.full((ny, nx), np.nan)

    Delta_Ncol_map = np.full((ny, nx), np.nan)
    Delta_Tex_map = np.full((ny, nx), np.nan)
    
    for y in range(ny):
        for x in range(nx):
    
            tab_indiv = tab_pixeles[y, x]

            if tab_indiv is None or len(tab_indiv) < 2:
                continue

            try:
                (T_extot, deltaTex, tab_filtrada2, N_coltot, deltaN_col, pol,
                 QTex,) = diagrama_rotacional(elemento, id_cat, tab_indiv, B0,
                                              cat_mol,
                                              freq_noconsid=freq_noconsid,
                                              plot_Q=False, save_path=None,
                                              plots = plots)

                N_col_map[y, x] = N_coltot.to_value(1 / u.cm**2)
                T_ex_map[y, x] = T_extot.to_value(u.K)
                Delta_Ncol_map[y, x] = deltaN_col.to_value(1 / u.cm**2)
                Delta_Tex_map[y, x] = deltaTex.to_value(u.K)
                

            except Exception:
                continue

    if plots:
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
        
    return (T_ex_map, N_col_map, Delta_Tex_map, Delta_Ncol_map)

def _cat_mol_from_id_cat_name(id_cat_name):
    if id_cat_name.startswith("id_JPL"):
        return "JPL"
    if id_cat_name.startswith("id_cdms"):
        return "CDMS"