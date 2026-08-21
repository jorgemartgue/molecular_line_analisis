#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:21:03 2026

@author: jorge
"""
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from astropy.io import fits
from TFM_storage import save_continuos, load_continuos, save_table, load_table
from astropy.table import QTable
from TFM_io_cubes import read_spectral_cube
from TFM_line_search import mod_vel, abrir_espectro_region


def det_Tcont(fmin, fmax, intervalos, vfuent=None, dict_espec=None, 
              plots = True, rutacarp_region = None, rutaregion_region = None):

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
    frec, espec = abrir_espectro_region(nombre=nombre,
                                        nombre_ventana=nombre_ventana,
                                        dict_espec=dict_espec,
                                        rutacarp_region=rutacarp_region,
                                        rutaregion_region=rutaregion_region,
                                        promedio=True,
                                        )

    if not vfuent is None:

        frec = mod_vel(vfuent, frec)
    
    if plots:
        
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
                 dict_espec_c=None, plots = True, rutacarp_region = None,
                 rutaregion_region = None):
    
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
        
        frec, espec = abrir_espectro_region(nombre=nombre,
                                            nombre_ventana=nombre_ventana,
                                            dict_espec=dict_espec_c,
                                            rutacarp_region=rutacarp_region,
                                            rutaregion_region=rutaregion_region,
                                            promedio=False)

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
                        fmin, fmax, intervalos_c, vfuent, cubo_pixel, 
                        plots = plots
                    )
                    
                    T_cont_inter[y, x] = T_vent.value
                    sigma_inter[y, x] = sigma_vent.value

                except Exception:
                    continue
        
        T_cont[nombre_ventana] = T_cont_inter
        sigma[nombre_ventana] = sigma_inter
        
    return T_cont, sigma




def calcular_continuos(intervalos_Tcont, ventanas_obs, vfuent, dict_espec,
                       plots=True,
                       rutacarp_region=None,
                       rutaregion_region=None):
    dict_T_cont = {}
    dict_sigma = {}

    for fmin, fmax in intervalos_Tcont:
        nombre_vent, T_vent, sigma_vent = det_Tcont(fmin, fmax, ventanas_obs,
                                                    vfuent,
                                                    dict_espec=dict_espec,
                                                    plots=plots,
                                                    rutacarp_region=rutacarp_region,
                                                    rutaregion_region=rutaregion_region)

        dict_T_cont[nombre_vent] = T_vent
        dict_sigma[nombre_vent] = sigma_vent

    return dict_T_cont, dict_sigma

def det_Tcont_percent(percen_min, percen_max, intervalos, SPW,
                      dict_espec=None, plots=True, rutacarp_region = None,
                      rutaregion_region = None):
    
    if percen_min >= percen_max:
        raise ValueError(
            f"percen_min debe ser menor que percen_max. "
            f"Recibido: {percen_min}, {percen_max}"
        )

    if percen_min < 0 or percen_max > 100:
        raise ValueError(
            f"Los percentiles deben estar entre 0 y 100. "
            f"Recibido: {percen_min}, {percen_max}"
        )

    SPW_disponibles = [ventana[3] for ventana in intervalos]
    
    if SPW not in SPW_disponibles:
        raise KeyError(
            f"La SPW '{SPW}' no está disponible.\n"
            f"SPW disponibles: {SPW_disponibles}"
        )

    nombre = None

    for ventana, fmin, fmax, name_window in intervalos:
        if name_window == SPW:
            nombre = ventana.strip()
            break

    if nombre is None:
        raise KeyError(f"No se ha encontrado la ventana asociada a {SPW}")

    frec, espec = abrir_espectro_region(nombre=nombre,
                                        nombre_ventana=SPW,
                                        dict_espec=dict_espec,
                                        rutacarp_region=rutacarp_region,
                                        rutaregion_region=rutaregion_region,
                                        promedio=True)

    if hasattr(espec, "unit"):
        espec_q = espec.to(u.K)
        espec_val = espec_q.value
    else:
        espec_val = np.asarray(espec, dtype=float)
        espec_q = espec_val * u.K

    mask_fin = np.isfinite(espec_val)

    if np.sum(mask_fin) < 3:
        return np.nan * u.K, np.nan * u.K

    fl_low = np.nanpercentile(espec_val[mask_fin], percen_min)
    fl_high = np.nanpercentile(espec_val[mask_fin], percen_max)

    mask_percent = (
        mask_fin
        & (espec_val >= fl_low)
        & (espec_val <= fl_high)
    )

    if np.sum(mask_percent) < 3:
        return np.nan * u.K, np.nan * u.K

    frec_continuo = frec[mask_percent]
    espec_continuo = espec_q[mask_percent]

    Tcont = np.nanmedian(espec_continuo).to(u.K)
    sigma = np.nanstd(espec_continuo).to(u.K)

    if plots:
        plt.figure()
        
        plt.plot(
            frec.value,
            espec_q.value,
            "b",
            drawstyle="steps-mid",
            label="Espectro observado",
        )

        plt.plot(
            frec_continuo.value,
            espec_continuo.value,
            ".r",
            markersize=5,
            label="espec para Tcont",
        )

        plt.axhline(
            y=Tcont.value,
            color="r",
            linestyle="--",
            label="Valor del continuo",
        )

        plt.xlabel("Frecuencia (MHz)")
        plt.ylabel("Temperatura de brillo (K)")
        plt.title(f"Continuo por percentiles - {SPW}")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return Tcont, sigma

def calcular_continuos_percent(percentiles, ventanas_obs,
                               dict_cubos_med=None, SPWs=None,
                               plots=True,
                               rutacarp_region=None,
                               rutaregion_region=None):
    """
    Calcula T_cont y sigma usando percentiles específicos por SPW.

    Parameters
    ----------
    percentiles : dict
        Diccionario tipo:
        {
            "B3-SPW0": (5, 60),
            "B3-SPW1": (5, 60),
            ...
        }

    ventanas_obs : list
        Lista de ventanas observadas.

    dict_cubos_med : dict
        Diccionario de espectros promedio.

    SPWs : list or None
        Lista de SPWs a calcular. Si None, calcula todas.

    plots : bool
        Si True, muestra los plots.
    """

    dict_T_cont = {}
    dict_sigma_vent = {}

    SPWs_disponibles = [ventana[3] for ventana in ventanas_obs]

    if SPWs is None:
        SPWs = SPWs_disponibles

    for spw in SPWs:

        if spw not in SPWs_disponibles:
            raise KeyError(
                f"La SPW '{spw}' no está en ventanas_obs. "
                f"SPWs disponibles: {SPWs_disponibles}"
            )

        if spw not in percentiles:
            raise KeyError(
                f"No has definido percentiles para {spw}. "
                f"Claves disponibles: {list(percentiles.keys())}"
            )

        percen_min, percen_max = percentiles[spw]

        print(
            f"[continuo percent] {spw}: "
            f"percentiles ({percen_min}, {percen_max})"
        )

        T_vent, sigma_vent = det_Tcont_percent(percen_min=percen_min,
                                               percen_max=percen_max,
                                               intervalos=ventanas_obs,
                                               SPW=spw,
                                               dict_espec=dict_cubos_med,
                                               plots=plots,
                                               rutacarp_region=rutacarp_region,
                                               rutaregion_region=rutaregion_region)

        dict_T_cont[spw] = T_vent
        dict_sigma_vent[spw] = sigma_vent

    return dict_T_cont, dict_sigma_vent

def path_continuo_region(region_name, modo, base_dir):
    """
    Devuelve la ruta de la tabla de continuo para una región y modo.

    Parameters
    ----------
    region_name : str
        Nombre de la región, por ejemplo 'MF2'.

    modo : str
        'medio' o 'pixeles'.

    base_dir : Path
        Ruta base de tablas, normalmente cfg.rutatablas.
    """

    return Path(base_dir) / "continuos" / f"continuo_{region_name}_{modo}.ecsv"


def cargar_o_calcular_continuo_medio(
        region_name,
        intervalos_Tcont,
        ventanas_obs,
        vfuent,
        dict_cubos_med,
        base_dir,
        recalcular=False,
        plots=True,
        rutacarp_region=None,
        rutaregion_region=None):
    """
    Carga o calcula el continuo medio clásico usando intervalos_Tcont.
    """

    path = path_continuo_region(region_name, "medio", base_dir)

    if path.exists() and not recalcular:
        print(f"[continuo] Cargando continuo medio: {path}")
        dict_T_cont, dict_sigma = load_continuos(path)

    else:
        print(f"[continuo] Calculando continuo medio para región {region_name}")

        dict_T_cont, dict_sigma = calcular_continuos(
            intervalos_Tcont=intervalos_Tcont,
            ventanas_obs=ventanas_obs,
            vfuent=vfuent,
            dict_espec=dict_cubos_med,
            plots=plots,
            rutacarp_region=rutacarp_region,
            rutaregion_region=rutaregion_region,
        )

        save_continuos(dict_T_cont, dict_sigma, path)
        print(f"[continuo] Continuo medio guardado en: {path}")

    return dict_T_cont, dict_sigma

def _tag_percent(percen_min, percen_max):
    """
    Tag seguro para nombres de archivo/carpeta.
    """

    pmin = str(percen_min).replace(".", "p")
    pmax = str(percen_max).replace(".", "p")

    return f"percent_p{pmin}_{pmax}"

def _tag_percent_dict():
    """
    Tag para el caso en el que cada SPW tiene sus propios percentiles.
    """

    return "percent_por_spw"

def cargar_o_calcular_continuo_medio_percent(
        region_name,
        percentiles,
        ventanas_obs,
        dict_cubos_med,
        base_dir,
        SPWs=None,
        recalcular=False,
        plots=True,
        rutacarp_region=None,
        rutaregion_region=None):
    """
    Carga o calcula continuo medio por percentiles específicos de cada SPW.
    """

    tag = _tag_percent_dict()
    modo = f"medio_{tag}"

    path = path_continuo_region(region_name, modo, base_dir)

    if path.exists() and not recalcular:
        print(f"[continuo_percent] Cargando continuo medio: {path}")
        dict_T_cont, dict_sigma = load_continuos(path)

    else:
        print(
            f"[continuo_percent] Calculando continuo medio para {region_name} "
            "con percentiles específicos por SPW"
        )

        dict_T_cont, dict_sigma = calcular_continuos_percent(
            percentiles=percentiles,
            ventanas_obs=ventanas_obs,
            dict_cubos_med=dict_cubos_med,
            SPWs=SPWs,
            plots=plots,
            rutacarp_region=rutacarp_region,
            rutaregion_region=rutaregion_region,
        )

        save_continuos(dict_T_cont, dict_sigma, path)
        print(f"[continuo_percent] Continuo medio guardado en: {path}")

    return dict_T_cont, dict_sigma


def cont_pixeles_percent(percentiles, intervalos_c, SPWs=None,
                         dict_espec_c=None, rutacarp_region = None,
                         rutaregion_region=None):
    """
    Calcula continuo y sigma por píxel usando percentiles específicos
    para cada SPW.
    """

    SPW_disponibles = [ventana[3] for ventana in intervalos_c]

    if SPWs is None:
        SPWs = SPW_disponibles

    elif isinstance(SPWs, str):
        SPWs = [SPWs]

    T_cont = {}
    sigma = {}

    for SPW in SPWs:

        if SPW not in SPW_disponibles:
            raise KeyError(
                f"La SPW '{SPW}' no está disponible.\n"
                f"SPW disponibles: {SPW_disponibles}"
            )

        if SPW not in percentiles:
            raise KeyError(
                f"No has definido percentiles para {SPW}. "
                f"Claves disponibles: {list(percentiles.keys())}"
            )

        percen_min, percen_max = percentiles[SPW]

        if percen_min >= percen_max:
            raise ValueError(
                f"percen_min debe ser menor que percen_max para {SPW}. "
                f"Recibido: {percen_min}, {percen_max}"
            )

        if percen_min < 0 or percen_max > 100:
            raise ValueError(
                f"Los percentiles deben estar entre 0 y 100 para {SPW}. "
                f"Recibido: {percen_min}, {percen_max}"
            )

        print(
            f"[continuo percent píxeles] {SPW}: "
            f"percentiles ({percen_min}, {percen_max})"
        )

        nombre = None
        nombre_ventana = None

        for ventana, fmin_i, fmax_i, name_window in intervalos_c:
            if name_window == SPW:
                nombre = ventana.strip()
                nombre_ventana = name_window
                break

        if nombre is None:
            print(f"No se ha encontrado la ventana asociada a {SPW}")
            continue

        frec, espec = abrir_espectro_region(nombre=nombre,
                                            nombre_ventana=nombre_ventana,
                                            dict_espec=dict_espec_c,
                                            rutacarp_region=rutacarp_region,
                                            rutaregion_region=rutaregion_region,
                                            promedio=False)

        if hasattr(espec, "filled_data"):
            espec_val = espec.to(u.K).filled_data[:].value

        elif hasattr(espec, "unit"):
            espec_val = espec.to(u.K).value

        else:
            espec_val = np.asarray(espec, dtype=float)

        if espec_val.ndim != 3:
            raise ValueError(
                f"Para {SPW}, espec debe tener forma (nchan, ny, nx). "
                f"Forma recibida: {espec_val.shape}"
            )

        mask_fin = np.isfinite(espec_val)

        fl_low = np.nanpercentile(espec_val, percen_min, axis=0)
        fl_high = np.nanpercentile(espec_val, percen_max, axis=0)

        mask_percent = (
            mask_fin
            & (espec_val >= fl_low[None, :, :])
            & (espec_val <= fl_high[None, :, :])
        )

        espec_cont = np.where(mask_percent, espec_val, np.nan)

        n_validos = np.sum(np.isfinite(espec_cont), axis=0)

        T_cont_inter = np.nanmedian(espec_cont, axis=0)
        sigma_inter = np.nanstd(espec_cont, axis=0)

        T_cont_inter[n_validos < 3] = np.nan
        sigma_inter[n_validos < 3] = np.nan

        T_cont[nombre_ventana] = T_cont_inter
        sigma[nombre_ventana] = sigma_inter

    return T_cont, sigma

def path_continuo_pixeles_region(region_name, base_dir):
    """
    Devuelve la carpeta donde se guardan los mapas FITS de continuo
    por píxel de una región.
    """

    path_dir = Path(base_dir) / "continuos" / region_name
    path_dir.mkdir(parents=True, exist_ok=True)

    return path_dir


def save_continuos_pixeles_fits(dict_T_cont_pix, dict_sigma_pix,
                                dict_cubos_comp, region_name, base_dir):
    """
    Guarda mapas 2D de continuo y sigma en FITS.

    Parameters
    ----------
    dict_T_cont_pix : dict
        Diccionario:
            dict_T_cont_pix["B6-SPW7"] = mapa 2D de T_cont en K

    dict_sigma_pix : dict
        Diccionario:
            dict_sigma_pix["B6-SPW7"] = mapa 2D de sigma en K

    dict_cubos_comp : dict
        Diccionario de cubos completos. Se usa para obtener el WCS/header
        espacial de referencia.

    region_name : str
        Nombre de la región, por ejemplo "MF2".

    base_dir : str or Path
        Carpeta base de tablas, normalmente cfg.rutatablas.
    """

    path_dir = path_continuo_pixeles_region(region_name, base_dir)

    for ventana in dict_T_cont_pix:

        mapa_T = dict_T_cont_pix[ventana]
        mapa_sigma = dict_sigma_pix[ventana]

        cubo_ref = dict_cubos_comp[ventana]["Temp_brillo"]

        # El cubo completo es SpectralCube. Usamos su WCS celeste.
        header_2d = cubo_ref.wcs.celestial.to_header()

        hdu_T = fits.PrimaryHDU(data=mapa_T, header=header_2d)
        hdu_T.header["BUNIT"] = "K"
        hdu_T.header["REGION"] = region_name
        hdu_T.header["WINDOW"] = ventana
        hdu_T.header["TYPE"] = "TCONT"

        hdu_sigma = fits.PrimaryHDU(data=mapa_sigma, header=header_2d)
        hdu_sigma.header["BUNIT"] = "K"
        hdu_sigma.header["REGION"] = region_name
        hdu_sigma.header["WINDOW"] = ventana
        hdu_sigma.header["TYPE"] = "SIGMA"

        path_T = path_dir / f"{ventana}_Tcont.fits"
        path_sigma = path_dir / f"{ventana}_sigma.fits"

        hdu_T.writeto(path_T, overwrite=True)
        hdu_sigma.writeto(path_sigma, overwrite=True)

        print(f"[continuo] Guardado: {path_T}")
        print(f"[continuo] Guardado: {path_sigma}")

def load_continuos_pixeles_fits(region_name, base_dir):
    """
    Carga mapas FITS de continuo por píxel.

    Returns
    -------
    dict_T_cont_pix : dict
        Diccionario de mapas 2D de T_cont.

    dict_sigma_pix : dict
        Diccionario de mapas 2D de sigma.
    """

    path_dir = path_continuo_pixeles_region(region_name, base_dir)

    dict_T_cont_pix = {}
    dict_sigma_pix = {}

    for path_T in sorted(path_dir.glob("*_Tcont.fits")):

        ventana = path_T.name.replace("_Tcont.fits", "")
        path_sigma = path_dir / f"{ventana}_sigma.fits"

        if not path_sigma.exists():
            raise FileNotFoundError(
                f"Existe {path_T}, pero no existe su sigma asociado: "
                f"{path_sigma}"
            )

        with fits.open(path_T) as hdul:
            dict_T_cont_pix[ventana] = hdul[0].data

        with fits.open(path_sigma) as hdul:
            dict_sigma_pix[ventana] = hdul[0].data

    if len(dict_T_cont_pix) == 0:
        raise FileNotFoundError(
            f"No se encontraron mapas de continuo FITS en {path_dir}"
        )

    return dict_T_cont_pix, dict_sigma_pix

def cargar_o_calcular_continuo_pixeles(region_name, intervalos_Tcont,
                                       ventanas_obs, vfuent, dict_cubos_comp,
                                       base_dir, recalcular=False, 
                                       plots = True):
    """
    Carga o calcula el continuo por píxel para una región y lo guarda en FITS.

    Returns
    -------
    dict_T_cont_pix : dict
        Diccionario de mapas 2D de T_cont.

    dict_sigma_pix : dict
        Diccionario de mapas 2D de sigma.
    """

    path_dir = path_continuo_pixeles_region(region_name, base_dir)
    existen_fits = len(list(path_dir.glob("*_Tcont.fits"))) > 0

    if existen_fits and not recalcular:
        print(f"[continuo] Cargando continuo por píxeles desde FITS: {path_dir}")

        dict_T_cont_pix, dict_sigma_pix = load_continuos_pixeles_fits(
            region_name,
            base_dir,
        )

    else:
        print(f"[continuo] Calculando continuo por píxeles para {region_name}")

        if dict_cubos_comp is None:
            raise ValueError(
                "Para calcular continuo por píxeles necesitas cargar cubos completos."
            )

        dict_T_cont_pix, dict_sigma_pix = cont_pixeles(
            intervalos_Tcont,
            ventanas_obs,
            vfuent=vfuent,
            dict_espec_c=dict_cubos_comp, plots = plots
        )

        save_continuos_pixeles_fits(
            dict_T_cont_pix,
            dict_sigma_pix,
            dict_cubos_comp,
            region_name,
            base_dir,
        )

    return dict_T_cont_pix, dict_sigma_pix

def cargar_o_calcular_continuo_pixeles_percent(
        region_name,
        percentiles,
        ventanas_obs,
        dict_cubos_comp,
        base_dir,
        SPWs=None,
        recalcular=False,
        plots=True,
        rutacarp_region=None,
        rutaregion_region=None):
    """
    Carga o calcula continuo por píxeles usando percentiles específicos
    para cada SPW.
    """

    tag = _tag_percent_dict()
    region_tag = f"{region_name}_{tag}"

    path_dir = path_continuo_pixeles_region(region_tag, base_dir)
    existen_fits = len(list(path_dir.glob("*_Tcont.fits"))) > 0

    if existen_fits and not recalcular:
        print(
            f"[continuo_percent] Cargando continuo por píxeles desde FITS: "
            f"{path_dir}"
        )

        dict_T_cont_pix, dict_sigma_pix = load_continuos_pixeles_fits(
            region_tag,
            base_dir,
        )

    else:
        print(
            f"[continuo_percent] Calculando continuo por píxeles para "
            f"{region_name} con percentiles específicos por SPW"
        )

        if dict_cubos_comp is None:
            raise ValueError(
                "Para calcular continuo por píxeles necesitas cargar cubos completos."
            )

        dict_T_cont_pix, dict_sigma_pix = cont_pixeles_percent(
            percentiles=percentiles,
            intervalos_c=ventanas_obs,
            SPWs=SPWs,
            dict_espec_c=dict_cubos_comp,
            rutacarp_region=rutacarp_region,
            rutaregion_region=rutaregion_region,
        )

        save_continuos_pixeles_fits(
            dict_T_cont_pix,
            dict_sigma_pix,
            dict_cubos_comp,
            region_tag,
            base_dir,
        )

    return dict_T_cont_pix, dict_sigma_pix