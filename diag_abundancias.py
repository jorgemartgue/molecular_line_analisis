#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:39:51 2026

@author: jorge
"""

"""
Lectura de los resultados de chi² de todas las regiones.

Estructura generada:
    dict_chi2_regiones[region][molecula][parametro]
"""

from pathlib import Path
from astropy.table import Table
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. Rutas
# ============================================================

RUTA_CHI2 = Path.home() / 'TFM' / 'tables' / 'chi2'

NOMBRE_ARCHIVO_CHI2 = 'chi2_resultados.ecsv'

# ============================================================
# 2. Cargar resultados de chi²
# ============================================================

def cargar_resultados_chi2_regiones(
    ruta_chi2,
    nombre_archivo='chi2_resultados.ecsv',
    verbose=True,
):
    """
    Lee los archivos de resultados de chi² guardados en las
    carpetas de cada región.

    Se espera una estructura como:

        tables/chi2/
            MF2/
                chi2_resultados.ecsv
            MM31/
                chi2_resultados.ecsv

    Parameters
    ----------
    ruta_chi2 : str o pathlib.Path
        Ruta principal que contiene las carpetas de regiones.

    nombre_archivo : str
        Nombre del archivo ECSV de resultados.

    verbose : bool
        Si es True, muestra un resumen de los resultados cargados.

    Returns
    -------
    dict
        Diccionario con estructura:

        resultados[region][molecula][parametro] = valor
    """

    ruta_chi2 = Path(ruta_chi2)

    if not ruta_chi2.exists():
        raise FileNotFoundError(
            f'No existe la carpeta de resultados de chi²:\n'
            f'{ruta_chi2}'
        )

    dict_resultados = {}

    # Cada subcarpeta de ruta_chi2 se interpreta como una región.
    carpetas_regiones = sorted(
        ruta
        for ruta in ruta_chi2.iterdir()
        if ruta.is_dir()
    )

    for ruta_region in carpetas_regiones:

        region = ruta_region.name
        ruta_archivo = ruta_region / nombre_archivo

        if not ruta_archivo.exists():
            if verbose:
                print(
                    f'[chi2] {region}: no existe '
                    f'{nombre_archivo}.'
                )
            continue

        try:
            tabla = Table.read(
                ruta_archivo,
                format='ascii.ecsv',
            )

        except Exception as error:
            print(
                f'[chi2] Error leyendo {ruta_archivo}:\n'
                f'       {error}'
            )
            continue

        # Comprobamos que exista la columna que identifica
        # a cada molécula.
        if 'molecula' not in tabla.colnames:
            print(
                f'[chi2] {region}: el archivo no contiene '
                f'la columna "molecula".'
            )
            continue

        dict_resultados[region] = {}

        for fila in tabla:

            molecula = str(fila['molecula']).strip()

            # Convertimos la fila en un diccionario.
            # Las unidades de Astropy se conservan.
            resultado_molecula = {
                columna: fila[columna]
                for columna in tabla.colnames
            }

            dict_resultados[region][molecula] = resultado_molecula

        if verbose:
            moleculas = list(dict_resultados[region].keys())

            print(
                f'[chi2] {region}: '
                f'{len(moleculas)} moléculas cargadas.'
            )
            print(f'       {moleculas}')

    return dict_resultados

# ============================================================
# 3. Leer todos los resultados
# ============================================================

dict_chi2_regiones = cargar_resultados_chi2_regiones(
    ruta_chi2=RUTA_CHI2,
    nombre_archivo=NOMBRE_ARCHIVO_CHI2,
    verbose=True,
)



def extraer_densidades_columna(
    dict_chi2_regiones,
    solo_convergidos=True,
    columna_error='deltaN',
):
    """
    Extrae N_fit y su incertidumbre para cada molécula y región.

    Returns
    -------
    dict
        dict_Ncol[region][molecula] = {
            'valor': N_fit,
            'error': error_N_fit,
        }
    """

    dict_Ncol = {}

    for region, resultados_region in dict_chi2_regiones.items():

        dict_Ncol[region] = {}

        for molecula, resultado in resultados_region.items():

            if solo_convergidos:
                converged = bool(resultado.get('converged', True))

                if not converged:
                    continue

            N_fit = resultado['N_fit']
            error_N_fit = resultado.get(columna_error, np.nan)

            if isinstance(N_fit, u.Quantity):
                N_fit = N_fit.to_value(u.cm**-2)
            else:
                N_fit = float(N_fit)

            if isinstance(error_N_fit, u.Quantity):
                error_N_fit = error_N_fit.to_value(u.cm**-2)
            else:
                error_N_fit = float(error_N_fit)

            if not np.isfinite(N_fit) or N_fit <= 0:
                continue

            if not np.isfinite(error_N_fit) or error_N_fit < 0:
                error_N_fit = np.nan

            dict_Ncol[region][molecula] = {
                'valor': N_fit,
                'error': error_N_fit,
            }

    return dict_Ncol

def extraer_temperaturas_excitacion(
    dict_chi2_regiones,
    solo_convergidos=True,
    columna_error='deltaT',
):
    """
    Extrae T_fit y su incertidumbre para cada molécula y región.

    Returns
    -------
    dict
        dict_Tex[region][molecula] = {
            'valor': T_fit,
            'error': error_T_fit,
        }
    """

    dict_Tex = {}

    for region, resultados_region in dict_chi2_regiones.items():

        dict_Tex[region] = {}

        for molecula, resultado in resultados_region.items():

            if solo_convergidos:
                converged = bool(resultado.get('converged', True))

                if not converged:
                    continue

            T_fit = resultado['T_fit']
            error_T_fit = resultado.get(columna_error, np.nan)

            if isinstance(T_fit, u.Quantity):
                T_fit = T_fit.to_value(u.K)
            else:
                T_fit = float(T_fit)

            if isinstance(error_T_fit, u.Quantity):
                error_T_fit = error_T_fit.to_value(u.K)
            else:
                error_T_fit = float(error_T_fit)

            if not np.isfinite(T_fit) or T_fit <= 0:
                continue

            if not np.isfinite(error_T_fit) or error_T_fit < 0:
                error_T_fit = np.nan

            dict_Tex[region][molecula] = {
                'valor': T_fit,
                'error': error_T_fit,
            }

    return dict_Tex

dict_Ncol = extraer_densidades_columna(
    dict_chi2_regiones,
    solo_convergidos=False,
)

dict_Tex = extraer_temperaturas_excitacion(
    dict_chi2_regiones,
    solo_convergidos=False,
)

dict_logN_col = {}

for region, resultados_region in dict_Ncol.items():

    dict_logN_col[region] = {}

    referencia = resultados_region.get('CH3OH_v0')

    if referencia is None:
        print(
            f'[abundancias] {region}: '
            'no existe CH3OH_v0.'
        )
        continue

    N_CH3OH = referencia['valor']
    error_CH3OH = referencia['error']

    for molecula, resultado in resultados_region.items():

        N_col = resultado['valor']
        error_N_col = resultado['error']

        ratio_log = np.log10(N_CH3OH / N_col)

        if (
            np.isfinite(error_CH3OH)
            and np.isfinite(error_N_col)
        ):
            error_ratio_log = (
                1 / np.log(10)
            ) * np.sqrt(
                (error_CH3OH / N_CH3OH)**2
                + (error_N_col / N_col)**2
            )
        else:
            error_ratio_log = np.nan

        dict_logN_col[region][molecula] = {
            'valor': ratio_log,
            'error': error_ratio_log,
        }
    
# dict_logNcol = {
#     region: {
#         molecula: np.log10(N_col)
#         for molecula, N_col in resultados_region.items()
#     }
#     for region, resultados_region in dict_Ncol.items()
# }




def plot_logN_por_region(dict_Ncol):
    """
    Representa log10(N_fit / cm^-2) con barras de error.
    """

    regiones = list(dict_Ncol.keys())

    moleculas = sorted({
        molecula
        for resultados_region in dict_Ncol.values()
        for molecula in resultados_region
    })

    x = np.arange(len(moleculas))

    n_regiones = len(regiones)
    ancho = 0.8 / n_regiones

    fig, ax = plt.subplots(
        figsize=(max(11, len(moleculas) * 0.85), 6.5)
    )

    for i, region in enumerate(regiones):

        valores = []
        errores = []

        for molecula in moleculas:

            resultado = dict_Ncol[region].get(molecula)

            if resultado is None:
                valores.append(np.nan)
                errores.append(np.nan)
                continue

            N_col = resultado['valor']
            error_N_col = resultado['error']

            if np.isfinite(N_col) and N_col > 0:

                valores.append(
                    np.log10(N_col)
                )

                if np.isfinite(error_N_col):

                    error_logN = (
                        error_N_col
                        / (N_col * np.log(10))
                    )

                else:
                    error_logN = np.nan

                errores.append(error_logN)

            else:

                valores.append(np.nan)
                errores.append(np.nan)

        desplazamiento = (
            i - (n_regiones - 1) / 2
        ) * ancho

        ax.bar(
            x + desplazamiento,
            valores,
            width=ancho,
            yerr=errores,
            capsize=4,
            error_kw={
                'elinewidth': 1.2,
                'capthick': 1.2,
                'ecolor': 'black',
                'alpha': 0.9,
            },
            label=region,
        )

    ax.set_ylim(13, 19)

    ax.set_xticks(x)

    ax.set_xticklabels(
        moleculas,
        rotation=45,
        ha='right',
        fontsize=15
    )

    ax.set_ylabel(
        r'$\log_{10}\left(N_{\mathrm{mol}}\,[\mathrm{cm}^{-2}]\right)$',
        fontsize=18
    )

    ax.set_xlabel(
        'Molecule',
        fontsize=18
    )

    ax.tick_params(
        axis='y',
        labelsize=14
    )

    ax.legend(
        title='Region',
        loc='upper right',
        fontsize=10,
        title_fontsize=15,
        frameon=True
    )

    ax.grid(
        axis='y',
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        '/home/jorge/TFM/figures/diag_abundancias/'
        'abundancias_absolutas.pdf',
        bbox_inches='tight'
    )

    plt.show()
                
    
plot_logN_por_region(dict_Ncol)

def plot_logN_CH3OH_por_region(dict_logNcol):
    """
    Representa log10[N(CH3OH)/N(mol)] con barras de error.
    """

    regiones = list(dict_logNcol.keys())

    moleculas = sorted({
        molecula
        for resultados_region in dict_logNcol.values()
        for molecula in resultados_region
        if molecula != "CH3OH_v0"
    })

    x = np.arange(len(moleculas))

    n_regiones = len(regiones)
    ancho = 0.8 / n_regiones

    fig, ax = plt.subplots(
        figsize=(max(11, len(moleculas) * 0.85), 6.5)
    )

    for i, region in enumerate(regiones):

        valores = []
        errores = []

        for molecula in moleculas:

            resultado = dict_logNcol[region].get(molecula)

            if resultado is None:
                valores.append(np.nan)
                errores.append(np.nan)
                continue

            valores.append(
                resultado['valor']
            )

            errores.append(
                resultado['error']
            )

        desplazamiento = (
            i - (n_regiones - 1) / 2
        ) * ancho

        ax.bar(
            x + desplazamiento,
            valores,
            width=ancho,
            yerr=errores,
            capsize=4,
            error_kw={
                'elinewidth': 1.2,
                'capthick': 1.2,
                'ecolor': 'black',
                'alpha': 0.9,
            },
            label=region,
        )

    ax.set_ylim(-1, 4)

    ax.set_xticks(x)

    ax.set_xticklabels(
        moleculas,
        rotation=45,
        ha='right',
        fontsize=15
    )

    ax.set_ylabel(
        r'$\log_{10}\left('
        r'N(\mathrm{CH_3OH}\,v=0)'
        r'/N_{\mathrm{mol}}'
        r'\right)$',
        fontsize=18
    )

    ax.set_xlabel(
        'Molecule',
        fontsize=18
    )

    ax.tick_params(
        axis='y',
        labelsize=14
    )

    ax.legend(
        title='Region',
        loc='best',
        fontsize=10,
        title_fontsize=15,
        frameon=True
    )

    ax.grid(
        axis='y',
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        '/home/jorge/TFM/figures/diag_abundancias/'
        'abundancias_relativas.pdf',
        bbox_inches='tight'
    )

    plt.show()

plot_logN_CH3OH_por_region(dict_logN_col)

def plot_Tex_por_region(dict_Tex):
    """
    Representa T_ex con barras de error.
    """

    regiones = list(dict_Tex.keys())

    moleculas = sorted({
        molecula
        for resultados_region in dict_Tex.values()
        for molecula in resultados_region
    })

    x = np.arange(len(moleculas))

    n_regiones = len(regiones)
    ancho = 0.8 / n_regiones

    fig, ax = plt.subplots(
        figsize=(max(11, len(moleculas) * 0.85), 6.5)
    )

    for i, region in enumerate(regiones):

        valores = []
        errores = []

        for molecula in moleculas:

            resultado = dict_Tex[region].get(molecula)

            if resultado is None:
                valores.append(np.nan)
                errores.append(np.nan)
                continue

            valores.append(
                resultado['valor']
            )

            errores.append(
                resultado['error']
            )

        desplazamiento = (
            i - (n_regiones - 1) / 2
        ) * ancho

        ax.bar(
            x + desplazamiento,
            valores,
            width=ancho,
            yerr=errores,
            capsize=4,
            error_kw={
                'elinewidth': 1.2,
                'capthick': 1.2,
                'ecolor': 'black',
                'alpha': 0.9,
            },
            label=region,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        moleculas,
        rotation=45,
        ha='right',
        fontsize=15
    )

    ax.set_ylabel(
        r'$T_{\mathrm{ex}}$ [K]',
        fontsize=18
    )

    ax.set_xlabel(
        'Molecule',
        fontsize=18
    )

    ax.tick_params(
        axis='y',
        labelsize=15
    )

    ax.legend(
        title='Region',
        loc='upper right',
        fontsize=10,
        title_fontsize=15,
        frameon=True
    )

    ax.grid(
        axis='y',
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        '/home/jorge/TFM/figures/diag_abundancias/'
        'temperaturas.pdf',
        bbox_inches='tight'
    )

    plt.show()

plot_Tex_por_region(dict_Tex)

