#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:45:55 2026

@author: jorge
"""

from astropy import units as u
from astropy.table import Table, unique, vstack
import numpy as np

from TFM_line_search import buscador_lin_vel
from TFM_momentum_maps import map_intens_int
from TFM_splatalogue_tools import (
    buscador_splatalogue_cdms,
    aplica_filtros,
)

def filtrador(elemento_busc, intervalo_busc, E_max_busc, aij_min, sijmu2_min,
              id_splat1=None, filtro_estructurasf=None,
              list_freq_nofilt=None, filt_inter=0.5, Tcontf=None,
              anch_fix=None, linelistf=("CDMS",), dict_especf=None,
              v_filt=63*u.km/u.s, plots=True,
              rutacarp_region=None,
              rutaregion_region=None,
              tab_catalogo=None):
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

    if tab_catalogo is None:

        # Comportamiento antiguo: consultar Splatalogue
        tabla_lineas = buscador_splatalogue_cdms(
            elemento_busc,
            intervalo_busc,
            E_max_busc,
            id_splat=id_splat1,
            filtro_estructuras=filtro_estructurasf,
            linelist=linelistf,
        )

    else:

        # Nuevo comportamiento: trabajar con el catálogo local
        tabla_lineas = tab_catalogo.copy()

        # Aplicar localmente el filtro de energía
        if "upper_state_energy_K" not in tabla_lineas.colnames:
            raise KeyError(
                "El catálogo local no contiene "
                "'upper_state_energy_K'."
            )

        energia = tabla_lineas["upper_state_energy_K"]

        if getattr(energia, "unit", None) is None:
            energia = energia * u.K
        else:
            energia = energia.to(u.K)

        tabla_lineas = tabla_lineas[
            energia <= E_max_busc
        ]

        # Aplicar localmente los filtros de estructura
        if (
            filtro_estructurasf is not None
            and "resolved_QNs" in tabla_lineas.colnames
        ):
            mask_estructura = [
                aplica_filtros(qn, filtro_estructurasf)
                for qn in tabla_lineas["resolved_QNs"]
            ]

            tabla_lineas = tabla_lineas[mask_estructura]

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

            m0, m1, slab = map_intens_int(freq, v_filt, intervalo_busc,
                                          molecula=elemento_busc, mapa=False,
                                          rutacarp_region=rutacarp_region,
                                          rutaregion_region=rutaregion_region)

            if nofiltrar or np.any(~np.isnan(m0)):

                (Tmax, vlin, Tcont, sigma, FWHM, int_integrada,
                 M0, M1, M2, deltaW) = buscador_lin_vel(freq, v_filt,
                                                        intervalo_busc,
                                                        long_int=20*u.km/u.s,
                                                        ajuste=True,
                                                        Tcont_fix=Tcontf,
                                                        anch_lin=anch_fix,
                                                        dict_espec=dict_especf,
                                                        plots=plots,
                                                        rutacarp_region=rutacarp_region,
                                                        rutaregion_region=rutaregion_region)

                (Tmax2, vlin2, Tcont2, sigma2, FWHM2, int_integrada2,
                 M02, M12, M22, deltaW2) = buscador_lin_vel(
                     freq,
                     v_filt,
                     intervalo_busc, long_int=10*u.km/u.s,
                     ajuste=True,
                     Tcont_fix=Tcontf,
                     anch_lin=anch_fix,
                     dict_espec=dict_especf,
                     plots=plots,
                     rutacarp_region=rutacarp_region,
                     rutaregion_region=rutaregion_region)

                dif_int = np.abs(int_integrada - int_integrada2)/np.abs(
                    int_integrada)

                # prop_incert = deltaW / int_integrada
                dif_vel = np.abs(vlin- v_filt)

                # Change the dif_int for including more lines
                if (nofiltrar or (dif_vel < 3*u.km/u.s and
                                  (dif_int < filt_inter))) and int_integrada > 0:

                    filas.append(Table([row]))

                    ints.append(int_integrada.value)
                    Tcontv.append(Tcont.value)
                    anch.append(FWHM.value)
                    deltaWv.append(deltaW.value)
                    vpik.append(vlin.value)

    if len(filas) == 0:
        return Table()
    
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

def filtrador_por_pixel(
        tab_info,
        intervalos,
        v_busc=63 * u.km / u.s,
        long_int=20 * u.km / u.s,
        Tcontf=None,
        anch_fix=None,
        v_pik_map=None,
        fwhm_map=None,
        dict_especf=None,
        plots=True):
    """
    Mide las líneas de tab_info en cada píxel.

    Si se proporcionan v_pik_map y fwhm_map, utiliza la velocidad
    y la FWHM calibradas para cada píxel. Los valores v_busc y
    anch_fix quedan como respaldo para píxeles sin calibración válida.

    Notes
    -----
    v_pik_map usa el convenio del modelo sintético:

        v_pik = -v_cal

    Por ello, la velocidad física enviada a buscador_lin_vel()
    es -v_pik_map[y, x].
    """

    if dict_especf is None:
        raise ValueError(
            "dict_especf es None. En el pipeline por regiones debes pasar "
            "dict_cubos_comp ya cargado desde la región activa."
        )

    ventanas_validas = [nombre_vent for _, _, _, nombre_vent in intervalos]

    dict_especf = {
        k: v for k, v in dict_especf.items()
        if k in ventanas_validas
    }

    primera_ventana = list(dict_especf.keys())[0]
    cubo_ref = dict_especf[primera_ventana]["Temp_brillo"]
    nchan, ny, nx = cubo_ref.shape
        
    
    if v_pik_map is not None:
        v_pik_map = np.asarray(v_pik_map, dtype=float)

        if v_pik_map.shape != (ny, nx):
            raise ValueError(
                f"v_pik_map tiene forma {v_pik_map.shape}, "
                f"pero los cubos tienen forma espacial {(ny, nx)}."
            )

    if fwhm_map is not None:
        fwhm_map = np.asarray(fwhm_map, dtype=float)

        if fwhm_map.shape != (ny, nx):
            raise ValueError(
                f"fwhm_map tiene forma {fwhm_map.shape}, "
                f"pero los cubos tienen forma espacial {(ny, nx)}."
            )
        
    tab_pixeles = np.empty((ny,nx), dtype = object)  
    
    for y in range(ny):
        for x in range(nx):
            
            tab_pixel = tab_info.copy()

            # --------------------------------------------------
            # Calibración correspondiente a este píxel
            # --------------------------------------------------

            v_busc_pixel = v_busc
            anch_fix_pixel = anch_fix

            if (
                v_pik_map is not None
                and np.isfinite(v_pik_map[y, x])
            ):
                # v_pik_map utiliza el convenio del modelo:
                # v_pik = -v_cal
                v_busc_pixel = (
                    -v_pik_map[y, x] * u.km / u.s
                )

            if (
                fwhm_map is not None
                and np.isfinite(fwhm_map[y, x])
                and fwhm_map[y, x] > 0
            ):
                anch_fix_pixel = (
                    fwhm_map[y, x] * u.km / u.s
                )

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
                    (Tmax, vlin, Tcont, sigma, FWHM, int_integrada, M0, M1, M2,
                     deltaW,) = buscador_lin_vel(f, v_busc_pixel,  intervalos,
                                                 long_int=long_int,
                                                 ajuste=True,
                                                 Tcont_fix=dict_Tcont_pix,
                                                 anch_lin=anch_fix_pixel,
                                                 dict_espec=dict_espec_pix, 
                                                 plots=plots,)

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