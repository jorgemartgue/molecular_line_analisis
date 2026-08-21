#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:02:32 2026

@author: jorge
"""

"""
Este módulo:
- define qué líneas se usan para calibrar cada molécula;
- carga la tabla de calibración si ya existe;
- si no existe, la calcula con busc_mult_lin_v();
- calcula el FWHM medio;
- actualiza tables/fwhm_lineas.ecsv.
"""

import warnings
import numpy as np

from astropy import units as u
from astropy.io import fits
from astropy.table import QTable

import TFM_config as cfg

from TFM_line_search import (
    busc_mult_lin_v,
    busc_mult_lin_v_cubo,
)

from TFM_storage import (
    ensure_dir,
    save_table,
    load_table,
    save_fwhm_table,
    load_fwhm_table,
)


VF_SYS_DEFAULT = 63 * u.km / u.s


def get_calibration_plan(region_name):
    """
    Devuelve el plan de calibración correspondiente a una región.
    """

    if region_name not in cfg.REGION_LINE_CONFIG:
        raise KeyError(
            f"No existe configuración de líneas para {region_name}. "
            f"Regiones disponibles: "
            f"{list(cfg.REGION_LINE_CONFIG.keys())}"
        )

    config_region = cfg.REGION_LINE_CONFIG[region_name]

    if "calibracion" not in config_region:
        raise KeyError(
            f"La región {region_name} no contiene el bloque "
            "'calibracion'."
        )

    calibration_plan = config_region["calibracion"]

    for molecula, config_mol in calibration_plan.items():

        if "freqs" not in config_mol:
            raise KeyError(
                f"Falta 'freqs' en la calibración de "
                f"{molecula} para {region_name}."
            )

        if "longs" not in config_mol:
            raise KeyError(
                f"Falta 'longs' en la calibración de "
                f"{molecula} para {region_name}."
            )

        if len(config_mol["freqs"]) != len(config_mol["longs"]):
            raise ValueError(
                f"En {region_name}, la calibración de {molecula} "
                "tiene distinto número de frecuencias y ventanas: "
                f"{len(config_mol['freqs'])} frecuencias y "
                f"{len(config_mol['longs'])} ventanas."
            )

    return calibration_plan

def path_tabla_calibracion(region_name, molecula):
    """
    Devuelve la ruta donde se guarda la tabla de calibración
    de una molécula para una región concreta.
    """

    ruta_calibracion = cfg.rutatablas / "calibracion" / region_name
    ensure_dir(ruta_calibracion)

    return ruta_calibracion / f"{molecula}.ecsv"


def path_resumen_calibracion(region_name):
    """
    Tabla resumen de calibración por región.

    Guarda:
        molecula, FWHM, v_cal, v_pik
    """

    ruta_calibracion = cfg.rutatablas / "calibracion" / region_name
    ensure_dir(ruta_calibracion)

    return ruta_calibracion / "calibracion_resultados.ecsv"


def crear_tabla_resumen_calibracion():
    """
    Crea una tabla vacía con los resultados de calibración.
    """

    return QTable(
        names=["molecula", "FWHM", "v_cal", "v_pik"],
        dtype=["U100", "f8", "f8", "f8"],
        units=[None, u.km / u.s, u.km / u.s, u.km / u.s],
    )


def load_resumen_calibracion(region_name):
    """
    Carga la tabla resumen de calibración de una región.
    Si no existe, devuelve tabla vacía.
    """

    path = path_resumen_calibracion(region_name)

    if path.exists():
        return load_table(path)

    return crear_tabla_resumen_calibracion()


def save_resumen_calibracion(tab, region_name):
    """
    Guarda la tabla resumen de calibración.
    """

    path = path_resumen_calibracion(region_name)
    save_table(tab, path)

    print(f"[calibración] Resumen actualizado: {path}")

    return path


def actualizar_resumen_calibracion(
        region_name,
        molecula,
        fwhm_medio,
        v_cal,
        v_pik):
    """
    Añade o actualiza una molécula en la tabla resumen de calibración.
    """

    tab = load_resumen_calibracion(region_name)

    mask = tab["molecula"] == molecula

    if len(tab) > 0 and np.any(mask):
        idx = np.where(mask)[0][0]

        tab["FWHM"][idx] = fwhm_medio.to(u.km / u.s)
        tab["v_cal"][idx] = v_cal.to(u.km / u.s)
        tab["v_pik"][idx] = v_pik.to(u.km / u.s)

    else:
        tab.add_row((
            molecula,
            fwhm_medio.to(u.km / u.s),
            v_cal.to(u.km / u.s),
            v_pik.to(u.km / u.s),
        ))

    save_resumen_calibracion(tab, region_name)

    return tab


def dicts_calibracion_region(region_name):
    """
    Devuelve fwhm_dict y v_pik_dict desde la tabla resumen
    de calibración de una región.
    """

    tab = load_resumen_calibracion(region_name)

    fwhm_dict = {}
    v_pik_dict = {}

    for row in tab:
        mol = str(row["molecula"])
        fwhm_dict[mol] = row["FWHM"].to(u.km / u.s)
        v_pik_dict[mol] = row["v_pik"].to(u.km / u.s)

    return fwhm_dict, v_pik_dict


def obtener_columna_velocidad_calibracion(tab_cali):
    """
    Busca el nombre de la columna de velocidad en la tabla de calibración.
    """

    posibles = [
        "velocidad linea",
        "velocidad línea",
        "vlin",
        "v_linea",
        "v línea",
    ]

    for col in posibles:
        if col in tab_cali.colnames:
            return col

    raise KeyError(
        "No encuentro columna de velocidad en la tabla de calibración. "
        f"Columnas disponibles: {tab_cali.colnames}"
    )


def cargar_o_calcular_tabla_calibracion(
        molecula,
        region_name,
        dict_cubos_med,
        recalcular=False,
        v_sys=VF_SYS_DEFAULT,
        ventanas_obs=None,
        plots=True,
        rutacarp_region=None,
        rutaregion_region=None):
    """
    Carga la tabla de calibración de una molécula si existe.
    Si no existe, la calcula usando busc_mult_lin_v().

    Parameters
    ----------
    molecula : str
        Nombre interno de la molécula, por ejemplo 'C2H5CN'.

    dict_cubos_med : dict
        Diccionario de espectros promedio.

    recalcular : bool
        Si True, fuerza recalcular aunque exista la tabla.

    v_sys : Quantity
        Velocidad sistémica usada para buscar las líneas.

    Returns
    -------
    tab_cali : QTable
        Tabla con los ajustes de líneas de calibración.

    fwhm_medio : Quantity
        FWHM medio calculado a partir de la tabla.
    """

    calibration_plan = get_calibration_plan(region_name)

    if molecula not in calibration_plan:
        raise KeyError(
            f"No hay plan de calibración definido para {molecula}. "
            f"Opciones disponibles: {list(calibration_plan.keys())}"
        )

    path_cali = path_tabla_calibracion(region_name, molecula)

    if path_cali.exists() and not recalcular:
        print(f"[calibración] Cargando tabla existente: {path_cali}")
        tab_cali = load_table(path_cali)

    else:
        print(f"[calibración] Calculando líneas de calibración para {molecula}")

        freqs = calibration_plan[molecula]["freqs"]
        longs = calibration_plan[molecula]["longs"]

        if len(freqs) == 0:
            raise ValueError(
                f"La molécula {molecula} no tiene frecuencias de calibración."
            )
        if ventanas_obs is None:
            raise ValueError(
                "No se ha proporcionado ventanas_obs. "
                "Debes pasar VENTANAS_REGION desde el main."
            )
            
        tab_cali = busc_mult_lin_v(list_frec=freqs, v_busc=v_sys,
                                   interval=ventanas_obs, list_long=longs,
                                   fit=True, tab=True,
                                   dict_especm=dict_cubos_med, plots=plots,
                                   rutacarp_region=rutacarp_region,
                                   rutaregion_region=rutaregion_region)

        save_table(tab_cali, path_cali)
        print(f"[calibración] Tabla guardada en: {path_cali}")

    fwhm_medio = np.nanmean(tab_cali["FWHM"]).to(u.km / u.s)

    col_vel = obtener_columna_velocidad_calibracion(tab_cali)

    v_cal = np.nanmedian(tab_cali[col_vel]).to(u.km / u.s)


    v_pik_model = -v_cal

    print(f"[calibración] FWHM medio para {molecula}: {fwhm_medio:.2f}")
    print(f"[calibración] Velocidad calibrada para {molecula}: {v_cal:.2f}")
    print(f"[calibración] v_pik_model para {molecula}: {v_pik_model:.2f}")

    return tab_cali, fwhm_medio, v_cal, v_pik_model


def cargar_o_calcular_calibracion_molecula(
        molecula,
        region_name,
        dict_cubos_med,
        recalcular=False,
        v_sys=VF_SYS_DEFAULT,
        ventanas_obs=None,
        plots=True,
        rutacarp_region=None,
        rutaregion_region=None):
    """
    Función de alto nivel para una molécula:

    1. Carga o calcula la tabla de calibración.
    2. Calcula FWHM medio.
    3. Calcula v_pik_model desde la velocidad calibrada.
    4. Actualiza la tabla resumen de calibración por región.

    Returns
    -------
    tab_cali : QTable
    fwhm_medio : Quantity
    v_pik_model : Quantity
    fwhm_dict : dict
    v_pik_dict : dict
    """

    tab_cali, fwhm_medio, v_cal, v_pik_model = cargar_o_calcular_tabla_calibracion(
    molecula=molecula,
    region_name=region_name,
    dict_cubos_med=dict_cubos_med,
    recalcular=recalcular,
    v_sys=v_sys,
    plots=plots,
    ventanas_obs=ventanas_obs,
    rutacarp_region=rutacarp_region,
    rutaregion_region=rutaregion_region,
)

    actualizar_resumen_calibracion(
        region_name=region_name,
        molecula=molecula,
        fwhm_medio=fwhm_medio,
        v_cal=v_cal,
        v_pik=v_pik_model,
    )

    fwhm_dict, v_pik_dict = dicts_calibracion_region(region_name)

    return tab_cali, fwhm_medio, v_pik_model, fwhm_dict, v_pik_dict


# ============================================================================
# CALIBRACIÓN PÍXEL A PÍXEL
# ============================================================================

def path_calibracion_pixeles(region_name, molecula):
    """
    Devuelve las rutas de los mapas de v_pik y FWHM
    de una molécula y una región.
    """

    ruta = (
        cfg.rutatablas
        / "calibracion_pixeles"
        / region_name
        / molecula
    )

    ensure_dir(ruta)

    path_v_pik = ruta / f"{molecula}_v_pik.fits"
    path_fwhm = ruta / f"{molecula}_FWHM.fits"

    return path_v_pik, path_fwhm


def _footprint_calibracion(dict_cubos_comp, ventanas_obs, freqs):
    """
    Construye una máscara con los píxeles realmente observados
    en los cubos que contienen líneas de calibración.

    True  -> píxel perteneciente a la región observada.
    False -> píxel exterior o sin datos.
    """

    footprint = None

    for freq in freqs:

        for _, fmin, fmax, nombre_ventana in ventanas_obs:

            # Comprobamos en qué ventana cae la línea
            if (
                fmin <= freq <= fmax
                and nombre_ventana in dict_cubos_comp
            ):
                cubo = dict_cubos_comp[nombre_ventana]["Temp_brillo"]

                # Quitamos las unidades únicamente para comprobar los NaN
                if hasattr(cubo, "value"):
                    datos = np.asarray(cubo.value)
                else:
                    datos = np.asarray(cubo)

                # Un píxel es válido si contiene algún canal finito
                validos = np.any(np.isfinite(datos), axis=0)

                if footprint is None:
                    footprint = validos
                else:
                    footprint = footprint | validos

                break

    if footprint is None:
        raise ValueError(
            "Ninguna línea de calibración cae dentro de "
            "los cubos disponibles."
        )

    return footprint

def _rellenar_mapa_calibracion(
        mapa,
        footprint,
        valor_global,
        radio_max=2):
    """
    Rellena los píxeles interiores no válidos utilizando la mediana
    de los vecinos válidos.

    Si no encuentra ningún vecino válido hasta radio_max, utiliza
    el valor global. Los píxeles exteriores permanecen como NaN.

    Parameters
    ----------
    mapa : ndarray
        Mapa 2D con los ajustes válidos y NaN en los rechazados.
    footprint : ndarray de bool
        Máscara de los píxeles pertenecientes a la región observada.
    valor_global : float o Quantity
        Valor de la calibración global.
    radio_max : int
        Radio máximo, en píxeles, para buscar vecinos válidos.
    """

    mapa = np.asarray(mapa, dtype=float)
    footprint = np.asarray(footprint, dtype=bool)

    if mapa.shape != footprint.shape:
        raise ValueError(
            "El mapa y el footprint deben tener la misma forma."
        )

    # Convertimos el valor global a número, si tiene unidades
    if hasattr(valor_global, "value"):
        valor_global = valor_global.value

    valor_global = float(valor_global)

    # Copia que contendrá el mapa final
    mapa_relleno = mapa.copy()

    # Fuera de la región observada siempre conservamos NaN
    mapa_relleno[~footprint] = np.nan

    ny, nx = mapa.shape

    # Píxeles interiores cuyo ajuste fue rechazado
    pixeles_invalidos = np.argwhere(
        footprint & ~np.isfinite(mapa)
    )

    for y, x in pixeles_invalidos:

        valor_relleno = None

        for radio in range(1, radio_max + 1):

            y0 = max(0, y - radio)
            y1 = min(ny, y + radio + 1)
            x0 = max(0, x - radio)
            x1 = min(nx, x + radio + 1)

            vecinos = mapa[y0:y1, x0:x1]
            footprint_vecinos = footprint[y0:y1, x0:x1]

            validos = (
                footprint_vecinos
                & np.isfinite(vecinos)
            )

            if np.any(validos):
                valor_relleno = np.nanmedian(vecinos[validos])
                break

        # Si no hay vecinos válidos, usamos la calibración global
        if valor_relleno is None:
            valor_relleno = valor_global

        mapa_relleno[y, x] = valor_relleno

    return mapa_relleno

def _guardar_mapa_calibracion(
        path,
        mapa,
        molecula,
        tipo,
        valor_global,
        header_2d=None):
    """
    Guarda un mapa de calibración en formato FITS.

    Parameters
    ----------
    path : Path
        Ruta del archivo de salida.
    mapa : ndarray
        Mapa bidimensional que se guardará.
    molecula : str
        Nombre de la molécula.
    tipo : str
        Tipo de mapa: 'v_pik' o 'FWHM'.
    valor_global : float o Quantity
        Valor de la calibración global.
    header_2d : fits.Header, opcional
        Cabecera espacial del cubo.
    """

    if hasattr(valor_global, "value"):
        valor_global = valor_global.value

    # Utilizamos una copia para no modificar la cabecera original
    if header_2d is None:
        header = fits.Header()
    else:
        header = header_2d.copy()

    header["MOLEC"] = molecula
    header["TYPE"] = tipo
    header["BUNIT"] = "km/s"
    header["GLOBAL"] = float(valor_global)

    hdu = fits.PrimaryHDU(
        data=np.asarray(mapa, dtype=float),
        header=header,
    )

    hdu.writeto(path, overwrite=True)

    print(f"[calibración pix] Mapa guardado en: {path}")
    
def cargar_o_calcular_calibracion_pixeles_molecula(
        molecula,
        region_name,
        dict_cubos_comp,
        ventanas_obs,
        fwhm_global,
        v_pik_global,
        recalcular=False,
        delta_v_max=5 * u.km / u.s,
        fwhm_factor_min=0.5,
        fwhm_factor_max=2.0,
        radio_vecinos=2,
        header_2d=None):
    """
    Calcula o carga los mapas de v_pik y FWHM píxel a píxel.

    Para cada píxel:
        1. Ajusta todas las líneas de calibración.
        2. Calcula la mediana de v_pik y FWHM.
        3. Rechaza ambos valores si alguno difiere demasiado
           de la calibración global.
        4. Rellena los rechazados con vecinos válidos o con
           la calibración global.

    Returns
    -------
    v_pik_map : ndarray
        Mapa de velocidades en km/s.

    fwhm_map : ndarray
        Mapa de FWHM en km/s.
    """

    path_v_pik, path_fwhm = path_calibracion_pixeles(
        region_name,
        molecula,
    )

    # ------------------------------------------------------------------
    # 1. Cargar mapas existentes
    # ------------------------------------------------------------------

    if (
        path_v_pik.exists()
        and path_fwhm.exists()
        and not recalcular
    ):
        print(
            f"[calibración pix] Cargando mapas existentes "
            f"para {molecula}"
        )

        with fits.open(path_v_pik) as hdul:
            v_pik_map = hdul[0].data.astype(float)

        with fits.open(path_fwhm) as hdul:
            fwhm_map = hdul[0].data.astype(float)

        return v_pik_map, fwhm_map

    # ------------------------------------------------------------------
    # 2. Obtener las líneas de calibración
    # ------------------------------------------------------------------

    calibration_plan = get_calibration_plan(region_name)

    if molecula not in calibration_plan:
        raise KeyError(
            f"No hay plan de calibración definido para {molecula}."
        )

    freqs = calibration_plan[molecula]["freqs"]
    longs = calibration_plan[molecula]["longs"]

    if len(freqs) == 0:
        raise ValueError(
            f"La molécula {molecula} no tiene frecuencias "
            "de calibración."
        )

    # Aseguramos que los valores globales tengan las unidades correctas
    v_pik_global = u.Quantity(
        v_pik_global,
        u.km / u.s,
    ).to(u.km / u.s)

    fwhm_global = u.Quantity(
        fwhm_global,
        u.km / u.s,
    ).to(u.km / u.s)

    delta_v_max = u.Quantity(
        delta_v_max,
        u.km / u.s,
    ).to(u.km / u.s)

    print(
        f"[calibración pix] Ajustando {molecula} píxel a píxel"
    )

    # ------------------------------------------------------------------
    # 3. Ajustar todas las líneas en todos los píxeles
    # ------------------------------------------------------------------

    resultados = busc_mult_lin_v_cubo(
        list_frec=freqs,
        v_busc=-v_pik_global,
        interval=ventanas_obs,
        list_long=longs,
        fit=True,
        dict_espec_c=dict_cubos_comp,
        plots=False,
    )

    velocidades = []
    anchuras = []
    amplitudes = []

    for resultado_linea in resultados.values():

        # Aunque se llama freclin_pix, contiene la velocidad
        # calibrada de la línea en km/s
        velocidades.append(
            np.asarray(
                resultado_linea["freclin_pix"],
                dtype=float,
            )
        )

        anchuras.append(
            np.asarray(
                resultado_linea["FWHM_pix"],
                dtype=float,
            )
        )

        amplitudes.append(
            np.asarray(
                resultado_linea["Tmax_pix"],
                dtype=float,
            )
        )

    if len(velocidades) == 0:
        raise ValueError(
            f"No se ha podido ajustar ninguna línea de {molecula}."
        )

    # ------------------------------------------------------------------
    # 4. Combinar las líneas mediante la mediana
    # ------------------------------------------------------------------

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )

        v_cal_map = np.nanmedian(
            np.stack(velocidades),
            axis=0,
        )

        fwhm_raw = np.nanmedian(
            np.stack(anchuras),
            axis=0,
        )

        amp_raw = np.nanmedian(
            np.stack(amplitudes),
            axis=0,
        )

    # El modelo utiliza el convenio v_pik = -v_cal
    v_pik_raw = -v_cal_map

    # Valores globales sin unidades para trabajar con los arrays
    v_global_value = v_pik_global.to_value(u.km / u.s)
    fwhm_global_value = fwhm_global.to_value(u.km / u.s)
    delta_v_value = delta_v_max.to_value(u.km / u.s)

    # ------------------------------------------------------------------
    # 5. Identificar los píxeles observados
    # ------------------------------------------------------------------

    footprint = _footprint_calibracion(
        dict_cubos_comp=dict_cubos_comp,
        ventanas_obs=ventanas_obs,
        freqs=freqs,
    )

    # ------------------------------------------------------------------
    # 6. Aplicar los criterios de aceptación
    # ------------------------------------------------------------------

    ajuste_valido = (
        footprint
        & np.isfinite(v_pik_raw)
        & np.isfinite(fwhm_raw)
        & np.isfinite(amp_raw)
        & (amp_raw > 0)
        & (
            np.abs(v_pik_raw - v_global_value)
            <= delta_v_value
        )
        & (
            fwhm_raw
            >= fwhm_factor_min * fwhm_global_value
        )
        & (
            fwhm_raw
            <= fwhm_factor_max * fwhm_global_value
        )
    )

    # Si falla v_pik o FWHM, rechazamos ambos
    v_pik_filtrado = np.where(
        ajuste_valido,
        v_pik_raw,
        np.nan,
    )

    fwhm_filtrado = np.where(
        ajuste_valido,
        fwhm_raw,
        np.nan,
    )

    # ------------------------------------------------------------------
    # 7. Rellenar los ajustes rechazados
    # ------------------------------------------------------------------

    v_pik_map = _rellenar_mapa_calibracion(
        mapa=v_pik_filtrado,
        footprint=footprint,
        valor_global=v_global_value,
        radio_max=radio_vecinos,
    )

    fwhm_map = _rellenar_mapa_calibracion(
        mapa=fwhm_filtrado,
        footprint=footprint,
        valor_global=fwhm_global_value,
        radio_max=radio_vecinos,
    )

    n_pixeles = int(np.count_nonzero(footprint))
    n_aceptados = int(np.count_nonzero(ajuste_valido))

    print(
        f"[calibración pix] Ajustes aceptados para {molecula}: "
        f"{n_aceptados}/{n_pixeles}"
    )

    # ------------------------------------------------------------------
    # 8. Guardar los dos mapas
    # ------------------------------------------------------------------

    _guardar_mapa_calibracion(
        path=path_v_pik,
        mapa=v_pik_map,
        molecula=molecula,
        tipo="v_pik",
        valor_global=v_global_value,
        header_2d=header_2d,
    )

    _guardar_mapa_calibracion(
        path=path_fwhm,
        mapa=fwhm_map,
        molecula=molecula,
        tipo="FWHM",
        valor_global=fwhm_global_value,
        header_2d=header_2d,
    )

    return v_pik_map, fwhm_map