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

def etiqueta_region(region):
    etiquetas = {
        'MF2': 'd2',
        'MM14': 'mm14',
        'MM24': 'mm24',
        'MM31': 'mm31',
        'MM35': 'mm35',
        'NORTH': 'north',
        'e2e': 'e2e',
        'e2w': 'e2w',
        'e8mm': 'e8mm',
    }

    return etiquetas.get(region, region)

# ============================================================
# Ratio 12C/13C a partir de CH3OH / 13CH3OH
# ============================================================

R_GC = 6.3  # kpc

# Milam et al. (2005)
PENDIENTE_C = 6.21
ERROR_PENDIENTE_C = 1.00

ORDENADA_C = 18.71
ERROR_ORDENADA_C = 7.37


ratio_C_esperado = (
    PENDIENTE_C * R_GC
    + ORDENADA_C
)

error_ratio_C_esperado = np.sqrt(
    (R_GC * ERROR_PENDIENTE_C)**2
    + ERROR_ORDENADA_C**2
)


print('\n' + '=' * 70)
print('RATIO 12C/13C')
print('=' * 70)

print(
    f'Valor esperado para R_GC = {R_GC:.1f} kpc: '
    f'{ratio_C_esperado:.1f} ± {error_ratio_C_esperado:.1f}'
)

print('-' * 70)


for region, resultados_region in dict_Ncol.items():

    ch3oh = resultados_region.get('CH3OH_v0')
    ch3oh_13 = resultados_region.get('C-13-H3OH')

    if ch3oh is None or ch3oh_13 is None:
        print(
            f'{etiqueta_region(region):>6s}: '
            'no están disponibles ambas especies.'
        )
        continue

    N_12 = ch3oh['valor']
    err_12 = ch3oh['error']

    N_13 = ch3oh_13['valor']
    err_13 = ch3oh_13['error']

    ratio_obs = N_12 / N_13

    # Propagación de errores
    if (
        np.isfinite(err_12)
        and np.isfinite(err_13)
        and N_12 > 0
        and N_13 > 0
    ):
        error_ratio_obs = ratio_obs * np.sqrt(
            (err_12 / N_12)**2
            + (err_13 / N_13)**2
        )
    else:
        error_ratio_obs = np.nan

    diferencia = ratio_obs - ratio_C_esperado
    factor = ratio_obs / ratio_C_esperado

    print(
        f'{etiqueta_region(region):>6s}: '
        f'12C/13C = {ratio_obs:6.1f} '
        f'± {error_ratio_obs:5.1f} | '
        f'esperado = {ratio_C_esperado:5.1f} | '
        f'obs/esp = {factor:.2f}'
    )

import copy

dict_Ncol_corr = copy.deepcopy(dict_Ncol)

# ============================================================
# Ratios isotópicos esperados
# ============================================================

RATIO_C_ESPERADO = ratio_C_esperado

PENDIENTE_O = 58.8
ERROR_PENDIENTE_O = 11.8

ORDENADA_O = 37.1
ERROR_ORDENADA_O = 82.6

ratio_O_esperado = (
    PENDIENTE_O * R_GC
    + ORDENADA_O
)

error_ratio_O_esperado = np.sqrt(
    (R_GC * ERROR_PENDIENTE_O)**2
    + ERROR_ORDENADA_O**2
)


print('\n' + '=' * 100)
print('CORRECCIÓN DE N(CH3OH v=0) POR ISOTOPÓLOGOS')
print('=' * 100)

print(
    f'12C/13C esperado = '
    f'{RATIO_C_ESPERADO:.1f}'
)

print(
    f'16O/18O esperado = '
    f'{ratio_O_esperado:.1f}'
)

print('-' * 100)


for region, resultados_region in dict_Ncol.items():

    ch3oh = resultados_region.get('CH3OH_v0')
    ch3oh_13 = resultados_region.get('C-13-H3OH')
    ch3oh_18 = resultados_region.get('CH3O-18-H')

    if ch3oh is None:
        print(
            f'{etiqueta_region(region):>6s}: '
            'no existe CH3OH_v0.'
        )
        continue

    N_CH3OH = ch3oh['valor']
    error_CH3OH = ch3oh['error']

    # ========================================================
    # Opción 1: 13CH3OH
    # ========================================================

    if ch3oh_13 is not None:

        N_iso = ch3oh_13['valor']

        ratio_obs = N_CH3OH / N_iso

        factor_corr = (
            RATIO_C_ESPERADO
            / ratio_obs
        )

        isotopologo = '13CH3OH'
        ratio_esperado = RATIO_C_ESPERADO

    # ========================================================
    # Opción 2: CH3-18OH
    # ========================================================

    elif ch3oh_18 is not None:

        N_iso = ch3oh_18['valor']

        ratio_obs = N_CH3OH / N_iso

        factor_corr = (
            ratio_O_esperado
            / ratio_obs
        )

        isotopologo = 'CH3-18OH'
        ratio_esperado = ratio_O_esperado

    # ========================================================
    # Ningún isotopólogo disponible
    # ========================================================

    else:

        print(
            f'{etiqueta_region(region):>6s}: '
            'no hay isotopólogo disponible.'
        )

        continue

    # ========================================================
    # Corregir CH3OH
    # ========================================================

    N_CH3OH_corr = (
        N_CH3OH * factor_corr
    )

    if np.isfinite(error_CH3OH):

        error_CH3OH_corr = (
            error_CH3OH * factor_corr
        )

    else:

        error_CH3OH_corr = np.nan

    dict_Ncol_corr[region]['CH3OH_v0']['valor'] = (
        N_CH3OH_corr
    )

    dict_Ncol_corr[region]['CH3OH_v0']['error'] = (
        error_CH3OH_corr
    )

    print(
        f'{etiqueta_region(region):>6s}: '
        f'{isotopologo:>9s} | '
        f'ratio obs = {ratio_obs:7.2f} | '
        f'esperado = {ratio_esperado:7.1f} | '
        f'factor = {factor_corr:6.2f} | '
        f'N = {N_CH3OH:.3e} -> '
        f'{N_CH3OH_corr:.3e}'
    )


dict_logN_col = {}

for region, resultados_region in dict_Ncol_corr.items():

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

# ============================================================
# Estilo común de las figuras
# ============================================================

# Patrones distintos para cada región.
# Se combinan con los colores para que la figura siga siendo
# interpretable en escala de grises y por personas daltónicas.


def etiqueta_molecula(molecula):
    """
    Convierte los nombres internos de las moléculas en etiquetas
    con notación química adecuada para las figuras.
    """

    etiquetas = {

        'Acetona':
            r'$\mathrm{(CH_3)_2CO}$',

        'C-13-H3CN':
            r'$\mathrm{^{13}CH_3CN}$',

        'C-13-H3OH':
            r'$\mathrm{^{13}CH_3OH}$',

        'C2H5CN':
            r'$\mathrm{C_2H_5CN}$',

        'C2H5OH_anti':
            r'$\mathrm{a\!-\!C_2H_5OH}$',

        'C2H5OH_g':
            r'$\mathrm{g\!-\!C_2H_5OH}$',

        'CH3CCH_v0':
            r'$\mathrm{CH_3CCH}\;(v=0)$',

        'CH3CHO_v0':
            r'$\mathrm{CH_3CHO}\;(v=0)$',

        'CH3CN':
            r'$\mathrm{CH_3CN}$',

        'CH3NCO':
            r'$\mathrm{CH_3NCO}$',

        'CH3NCO_B3':
            r'$\mathrm{CH_3NCO}\;(\mathrm{B3})$',

        'CH3O-18-H':
            r'$\mathrm{CH_3^{18}OH}$',

        'CH3OCH3':
            r'$\mathrm{CH_3OCH_3}$',

        'CH3OCHO_v0':
            r'$\mathrm{CH_3OCHO}\;(v=0)$',

        'CH3OCHO_v1':
            r'$\mathrm{CH_3OCHO}\;(v=1)$',

        'CH3OH_v0':
            r'$\mathrm{CH_3OH}\;(v=0)$',

        'CH3OH_v1':
            r'$\mathrm{CH_3OH}\;(v=1)$',

        'OC-13-S':
            r'$\mathrm{O^{13}CS}$',
    }

    return etiquetas.get(molecula, molecula)



FS_TITULO = 38
FS_YLABEL = 35
FS_XTICKS = 28
FS_YTICKS = 28
FS_LEGEND = 31

ESPACIO_MOLECULAS = 1.40


REGIONES_PANEL_1 = [
    'MF2',
    'MM14',
    'MM24',
    'MM31',
    'MM35',
]

REGIONES_PANEL_2 = [
    'NORTH',
    'e2e',
    'e2w',
    'e8mm',
]

HATCH_POR_REGION = {
    'MF2': '',
    'MM14': '//',
    'MM24': '\\\\',
    'MM31': 'xx',
    'MM35': '..',
    'NORTH': '+',
    'e2e': '//',
    'e2w': '..',
    'e8mm': 'xx',
}

COLOR_POR_REGION = {
    'MF2':   '#9B7ED1',  # morado medio
    'MM14':  '#F4A261',  # naranja suave
    'MM24':  '#74C69D',  # verde medio
    'MM31':  '#E76F6F',  # rojo/salmón
    'MM35':  '#6EC5D8',  # cyan con más cuerpo
    'NORTH': '#A9826E',  # marrón suave

    'e2e':   'tab:blue',
    'e2w':   'tab:orange',
    'e8mm':  'tab:green',
}

ESPACIO_MOLECULAS = 1.6

# ============================================================
# Parámetros geométricos comunes
# ============================================================

ESPACIO_MOLECULAS = 2
ANCHO_BARRA = 0.2
SEPARACION_BARRAS = 0.2


def plot_logN_por_region(dict_Ncol):
    """
    Representa log10(N_fit / cm^-2), separando IRS2 y W51-E.
    """

    moleculas = sorted({
        molecula
        for resultados_region in dict_Ncol.values()
        for molecula in resultados_region
    })

    x = np.arange(len(moleculas)) * ESPACIO_MOLECULAS

    fig, axes = plt.subplots(
    2,
    1,
    figsize=(30, 14),
    sharex=True,
    sharey=True,
    gridspec_kw={
    'height_ratios': [1, 1],
    'hspace': 0.035,
})

    grupos = [
        ('', REGIONES_PANEL_1),
        ('', REGIONES_PANEL_2),
        ]

    for ax, (nombre_grupo, regiones) in zip(axes, grupos):

        n_regiones = len(regiones)

        if n_regiones == 5:
            ancho = 0.30
            separacion = 0.32
        else:
            ancho = 0.32
            separacion = 0.34

        for i, region in enumerate(regiones):

            valores = []
            errores = []

            for molecula in moleculas:

                resultado = dict_Ncol.get(
                    region, {}
                ).get(molecula)

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
            ) * separacion

            ax.bar(
                x + desplazamiento,
                valores,
                width=ancho,
                color=COLOR_POR_REGION[region],
                hatch=HATCH_POR_REGION[region],
                edgecolor='black',
                linewidth=1.1,
                yerr=errores,
                capsize=5,
                label=etiqueta_region(region),
                error_kw={
                    'elinewidth': 1.8,
                    'capthick': 1.8,
                    'ecolor': 'black',
                    'alpha': 1.0,
                    },
                )   

        for j in range(len(x) - 1):

            ax.axvline(
                (x[j] + x[j + 1]) / 2,
                color='0.82',
                linewidth=0.8,
                zorder=0,
            )

        # ============================================================
        # Comparación con Sgr B2(N2) - Belloche et al. (2025)
        # ============================================================

        N_CH3OH_SGRB2 = {
    'N2b':    8.0e19,
    'AN02':   3.5e19,
    'AN03':   3.4e19,
    'AN06c2': 6.0e18,
    'AN06':   2.3e19,
}

        if 'CH3OH_v0' in moleculas:
        
            idx_ch3oh = moleculas.index('CH3OH_v0')
            x_ch3oh = x[idx_ch3oh]
        
            offsets_lit = np.linspace(
                -0.65,
                0.65,
                len(N_CH3OH_SGRB2)
            )
        
            for offset, (fuente, N_lit) in zip(
                offsets_lit,
                N_CH3OH_SGRB2.items()
            ):
        
                axes[0].scatter(
                    x_ch3oh + offset,
                    np.log10(N_lit),
                    marker='D',
                    s=140,
                    edgecolor='black',
                    linewidth=1.2,
                    zorder=10,
                    label='Sgr B2(N2), Belloche+25' if fuente == 'N2b' else None
                )

        ax.set_ylim(13, 20.2)
        ax.set_yticks([13, 14, 15, 16, 17, 18, 19, 20]) 
        ax.tick_params(
            axis='y',
            labelsize=FS_YTICKS
        )

        ax.grid(
            axis='y',
            alpha=0.25
        )

        ax.legend(
            loc='upper left',
            bbox_to_anchor=(0.02, 0.93),
            ncol=n_regiones,
            fontsize=FS_LEGEND,
            frameon=False,
        columnspacing=1.5,
        handlelength=2.0,
        )

        ax.set_xlim(
            x[0] - 1.0,
            x[-1] + 1.0
        )

    axes[0].text(
        0.5,
        0.995,
        'a) Column densities',
        transform=axes[0].transAxes,
        ha='center',
        va='top',
        fontsize=FS_TITULO,
        fontweight='bold',
    )

    fig.supylabel(
        r'$\log_{10}\left('
        r'N_{\mathrm{mol}}\,[\mathrm{cm}^{-2}]'
        r'\right)$',
        fontsize=FS_YLABEL,
        x=0.025
    )

    axes[-1].set_xticks(x)

    axes[-1].tick_params(
        axis='x',
        which='both',
        bottom=False,
        labelbottom=False
        )

    ax_top = axes[0].secondary_xaxis('top')

    ax_top.set_xticks(x)

    ax_top.set_xticklabels(
        [etiqueta_molecula(m) for m in moleculas],
        rotation=45,
        ha='left',
        rotation_mode='anchor',
        fontsize=FS_XTICKS
        )

    ax_top.tick_params(
        axis='x',
        pad=8
        )

    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.08,
        top=0.78,
        hspace=0.035
        )

    fig.savefig(
        '/home/jorge/TFM/figures/diag_abundancias/'
        'abundancias_absolutas.pdf',
    )

    plt.show()


plot_logN_por_region(dict_Ncol)
                

def plot_logN_CH3OH_por_region(dict_logNcol):
    """
    Representa log10[N(CH3OH)/N(mol)], separando IRS2 y W51-E.
    """

    moleculas = sorted({
        molecula
        for resultados_region in dict_logNcol.values()
        for molecula in resultados_region
        })

    x = np.arange(len(moleculas)) * ESPACIO_MOLECULAS

    fig, axes = plt.subplots(
    2,
    1,
    figsize=(30, 14),
    sharex=True,
    sharey=True,
    gridspec_kw={
    'height_ratios': [1, 1],
    'hspace': 0.035,
})

    grupos = [
        ('', REGIONES_PANEL_1),
        ('', REGIONES_PANEL_2),
        ]

    for ax, (nombre_grupo, regiones) in zip(axes, grupos):

        n_regiones = len(regiones)

        if n_regiones == 5:
            ancho = 0.30
            separacion = 0.32
        else:
            ancho = 0.32
            separacion = 0.34

        for i, region in enumerate(regiones):

            valores = []
            errores = []

            for molecula in moleculas:

                if molecula == 'CH3OH_v0':
                    valores.append(np.nan)
                    errores.append(np.nan)
                    continue

                resultado = dict_logNcol.get(
                    region, {}
                ).get(molecula)

                if resultado is None:
                    valores.append(np.nan)
                    errores.append(np.nan)
                    continue

                valores.append(resultado['valor'])
                errores.append(resultado['error'])

            desplazamiento = (
                i - (n_regiones - 1) / 2
            ) * separacion

            ax.bar(
                x + desplazamiento,
                valores,
                width=ancho,
                color=COLOR_POR_REGION[region],
                hatch=HATCH_POR_REGION[region],
                edgecolor='black',
                linewidth=1.1,
                yerr=errores,
                capsize=5,
                label=etiqueta_region(region),
                error_kw={
                    'elinewidth': 1.8,
                    'capthick': 1.8,
                    'ecolor': 'black',
                    'alpha': 1.0,
                    },
                )

        for j in range(len(x) - 1):

            ax.axvline(
                (x[j] + x[j + 1]) / 2,
                color='0.82',
                linewidth=0.8,
                zorder=0,
            )

        ax.set_ylim(-1, 4.3)

        ax.tick_params(
            axis='y',
            labelsize=FS_YTICKS
        )

        ax.grid(
            axis='y',
            alpha=0.25
        )

        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 0.93),
            ncol=n_regiones,
            fontsize=FS_LEGEND,
            frameon=False,
            columnspacing=1.5,
            handlelength=2.0,
        )

        ax.set_xlim(
            x[0] - 1.0,
            x[-1] + 1.0
        )

    axes[0].text(
        0.5,
        0.995,
        r'c) $\mathbf{N}(\mathbf{CH}_{3}\mathbf{OH})/'
        r'\mathbf{N}(\mathbf{species})$',
        transform=axes[0].transAxes,
        ha='center',
        va='top',
        fontsize=FS_TITULO,
    )

    fig.supylabel(
        r'$\log_{10}\left('
        r'N(\mathrm{CH_3OH}\,v=0)'
        r'/N_{\mathrm{mol}}'
        r'\right)$',
        fontsize=FS_YLABEL,
        x=0.025
    )

    axes[-1].set_xticks(x)

    axes[-1].set_xticklabels(
        [etiqueta_molecula(m) for m in moleculas],
        rotation=45,
        ha='right',
        rotation_mode='anchor',
        fontsize=FS_XTICKS
        )
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.29,
        top=0.97,
        hspace=0.035
        )

    fig.savefig(
        '/home/jorge/TFM/figures/diag_abundancias/'
        'abundancias_relativas.pdf',
    )

    plt.show()


plot_logN_CH3OH_por_region(dict_logN_col)

def plot_Tex_por_region(dict_Tex):
    """
    Representa T_ex separando IRS2 y W51-E.
    """

    moleculas = sorted({
        molecula
        for resultados_region in dict_Tex.values()
        for molecula in resultados_region
    })

    x = np.arange(len(moleculas)) * ESPACIO_MOLECULAS

    fig, axes = plt.subplots(
    2,
    1,
    figsize=(30, 14),
    sharex=True,
    sharey=True,
    gridspec_kw={
    'height_ratios': [1, 1],
    'hspace': 0.035,
})
    grupos = [
        ('', REGIONES_PANEL_1),
        ('', REGIONES_PANEL_2),
        ]

    for ax, (nombre_grupo, regiones) in zip(axes, grupos):

        n_regiones = len(regiones)

        if n_regiones == 5:
            ancho = 0.30
            separacion = 0.32
        else:
            ancho = 0.32
            separacion = 0.34

        for i, region in enumerate(regiones):

            valores = []
            errores = []

            for molecula in moleculas:

                resultado = dict_Tex.get(
                    region, {}
                ).get(molecula)

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
            ) * separacion

            ax.bar(
                x + desplazamiento,
                valores,
                width=ancho,
                color=COLOR_POR_REGION[region],
                hatch=HATCH_POR_REGION[region],
                edgecolor='black',
                linewidth=1.1,
                yerr=errores,
                capsize=5,
                label=etiqueta_region(region),
                error_kw={
                    'elinewidth': 1.8,
                    'capthick': 1.8,
                    'ecolor': 'black',
                    'alpha': 1.0,
                    },
                )  

        for j in range(len(x) - 1):

            ax.axvline(
                (x[j] + x[j + 1]) / 2,
                color='0.82',
                linewidth=0.8,
                zorder=0,
            )

        ax.set_ylim(-80, 650)
        ax.set_yticks(
            [0, 100, 200, 300, 400, 500, 600]
            )
        ax.tick_params(
            axis='y',
            labelsize=FS_YTICKS
        )

        ax.grid(
            axis='y',
            alpha=0.25
        )

        ax.legend(
            loc='upper left',
            bbox_to_anchor=(0.015, 0.88),
            ncol=2,
            fontsize=FS_LEGEND,
            frameon=False,
            columnspacing=1.4,
            handlelength=2.0,
        )

        ax.set_xlim(
            x[0] - 1.0,
            x[-1] + 1.0
        )

    axes[0].text(
        0.5,
        0.995,
        'b) Excitation temperatures',
        transform=axes[0].transAxes,
        ha='center',
        va='top',
        fontsize=FS_TITULO,
        fontweight='bold',
    )

    fig.supylabel(
        r'$T_{\mathrm{ex}}$ [K]',
        fontsize=FS_YLABEL,
        x=0.025
    )

    axes[-1].set_xticks(x)

    axes[-1].tick_params(
        axis='x',
        which='both',
        bottom=False,
        labelbottom=False
        )

    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.16,
        top=0.995,
        hspace=0.035,
        )

    fig.savefig(
        '/home/jorge/TFM/figures/diag_abundancias/'
        'temperaturas.pdf',
    )

    plt.show()


plot_Tex_por_region(dict_Tex)


dict_ratio_CH3OH = {}

for region, resultados_region in dict_Ncol_corr.items():

    dict_ratio_CH3OH[region] = {}

    referencia = resultados_region.get('CH3OH_v0')

    if referencia is None:
        continue

    N_CH3OH = referencia['valor']
    error_CH3OH = referencia['error']

    for molecula, resultado in resultados_region.items():

        if molecula == 'CH3OH_v0':
            dict_ratio_CH3OH[region][molecula] = {
                'valor': 1.0,
                'error': 0.0,
            }
            continue

        N_col = resultado['valor']
        error_N_col = resultado['error']

        ratio = N_col / N_CH3OH

        if (
            np.isfinite(error_CH3OH)
            and np.isfinite(error_N_col)
        ):
            error_ratio = ratio * np.sqrt(
                (error_N_col / N_col)**2
                + (error_CH3OH / N_CH3OH)**2
            )
        else:
            error_ratio = np.nan

        dict_ratio_CH3OH[region][molecula] = {
            'valor': ratio,
            'error': error_ratio,
        }

def plot_ratio_CH3OH_log(dict_ratio):

    """
    Representa N(X)/N(CH3OH v_t=0) con escala logarítmica
    en dos paneles, siguiendo el mismo formato que T_ex.
    """

    moleculas = sorted({
        molecula
        for resultados_region in dict_ratio.values()
        for molecula in resultados_region
    })

    x = np.arange(len(moleculas)) * ESPACIO_MOLECULAS

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(30, 14),
        sharex=True,
        sharey=True,
        gridspec_kw={
            'height_ratios': [1, 1],
            'hspace': 0.035,
        }
    )

    grupos = [
        REGIONES_PANEL_1,
        REGIONES_PANEL_2,
    ]

    for ax, regiones in zip(axes, grupos):

        n_regiones = len(regiones)

        if n_regiones == 5:
            ancho = 0.30
            separacion = 0.32
        else:
            ancho = 0.32
            separacion = 0.34

        for i, region in enumerate(regiones):

            valores = []
            errores = []

            for molecula in moleculas:

                resultado = dict_ratio.get(
                    region, {}
                ).get(molecula)

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
            ) * separacion

            ax.bar(
                x + desplazamiento,
                valores,
                width=ancho,
                color=COLOR_POR_REGION[region],
                hatch=HATCH_POR_REGION[region],
                edgecolor='black',
                linewidth=1.1,
                yerr=errores,
                capsize=5,
                label=etiqueta_region(region),
                error_kw={
                    'elinewidth': 1.8,
                    'capthick': 1.8,
                    'ecolor': 'black',
                    'alpha': 1.0,
                },
            )

        # Separadores verticales entre moléculas
        for j in range(len(x) - 1):

            ax.axvline(
                (x[j] + x[j + 1]) / 2,
                color='0.82',
                linewidth=0.8,
                zorder=0,
            )

        # Escala logarítmica
        ax.set_yscale('log')

        ax.tick_params(
            axis='y',
            labelsize=FS_YTICKS
        )

        ax.grid(
            axis='y',
            which='both',
            alpha=0.25
        )

        ax.legend(
    loc='upper left',
    bbox_to_anchor=(0.02, 0.91),
    ncol=n_regiones,
    fontsize=FS_LEGEND,
    frameon=False,
    columnspacing=1.2,
    handlelength=1.8,
)

        ax.set_xlim(
            x[0] - 1.0,
            x[-1] + 1.0
        )

    # Título
    axes[0].text(
        0.5,
        0.995,
        r'c) $\mathbf{N}(X)/'
        r'\mathbf{N}(\mathbf{CH_3OH})$',
        transform=axes[0].transAxes,
        ha='center',
        va='top',
        fontsize=FS_TITULO,
        fontweight='bold',
    )

    # Label común y
    fig.supylabel(
        r'$N(X)/N(\mathrm{CH_3OH}\;v_t=0)$',
        fontsize=FS_YLABEL,
        x=0.025,
        y = 0.5
        
    )

    # Etiquetas moleculares solo abajo
    axes[-1].set_xticks(x)

    axes[-1].set_xticklabels(
        [etiqueta_molecula(m) for m in moleculas],
        rotation=45,
        ha='right',
        rotation_mode='anchor',
        fontsize=FS_XTICKS
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.29,
        top=0.97,
        hspace=0.035
    )

    fig.savefig(
        '/home/jorge/TFM/figures/diag_abundancias/'
        'abundancias_relativas_log.pdf',
    )

    plt.show()


plot_ratio_CH3OH_log(dict_ratio_CH3OH)

# ============================================================
# Comparación con modelos químicos - Garrod et al. (2022)
# Table 18
# ============================================================

DATOS_GARROD = {

    'C2H5OH': {
        'IRAS16293B': 2.3e-2,
        'SgrB2N2':    5.0e-2,
        'fast':       6.1e-2,
        'medium':     6.5e-2,
        'slow':       8.0e-2,
    },

    'CH3OCH3': {
        'IRAS16293B': 2.4e-2,
        'SgrB2N2':    5.5e-2,
        'fast':       1.0e-2,
        'medium':     1.7e-2,
        'slow':       2.6e-2,
    },

    'CH3OCHO_v0': {
        'IRAS16293B': 2.6e-2,
        'SgrB2N2':    3.0e-2,
        'fast':       1.7e-2,
        'medium':     1.9e-2,
        'slow':       3.0e-2,
    },

    'CH3CHO_v0': {
        'IRAS16293B': 1.2e-2,
        'SgrB2N2':    1.1e-2,
        'fast':       3.6e-2,
        'medium':     7.3e-2,
        'slow':       3.0e-2,
    },

    'Acetona': {
        'IRAS16293B': 1.7e-3,
        'SgrB2N2':    1.0e-2,
        'fast':       5.9e-4,
        'medium':     6.6e-4,
        'slow':       1.7e-3,
    },

    'CH3NCO': {
        'IRAS16293B': 4.0e-4,
        'SgrB2N2':    5.5e-3,
        'fast':       1.1e-6,
        'medium':     1.1e-6,
        'slow':       4.4e-6,
    },

    'CH3CN': {
        'IRAS16293B': 4.0e-3,
        'SgrB2N2':    5.5e-2,
        'fast':       1.6e-4,
        'medium':     7.8e-4,
        'slow':       8.5e-3,
    },

    'C2H5CN': {
        'IRAS16293B': 3.6e-4,
        'SgrB2N2':    1.6e-1,
        'fast':       9.4e-3,
        'medium':     1.1e-2,
        'slow':       1.8e-2,
    },
}

# ============================================================
# Especies obtenidas sumando varios estados/conformeros
# ============================================================

MOLECULAS_COMBINADAS = {
    'C2H5OH': [
        'C2H5OH_anti',
        'C2H5OH_g',
    ],
}


for region, resultados_region in dict_Ncol_corr.items():

    referencia = resultados_region.get('CH3OH_v0')

    if referencia is None:
        continue

    N_CH3OH = referencia['valor']
    error_CH3OH = referencia['error']

    for molecula_total, componentes in MOLECULAS_COMBINADAS.items():

        # Comprobamos que estén todos los componentes
        if not all(
            componente in resultados_region
            for componente in componentes
        ):
            continue

        # ----------------------------------------------------
        # Sumar densidades de columna
        # ----------------------------------------------------

        N_total = sum(
            resultados_region[componente]['valor']
            for componente in componentes
        )

        # ----------------------------------------------------
        # Error de la suma
        # ----------------------------------------------------

        errores_componentes = [
            resultados_region[componente]['error']
            for componente in componentes
        ]

        if all(
            np.isfinite(error)
            for error in errores_componentes
        ):
            error_N_total = np.sqrt(
                sum(
                    error**2
                    for error in errores_componentes
                )
            )
        else:
            error_N_total = np.nan

        # ----------------------------------------------------
        # Ratio total / CH3OH
        # ----------------------------------------------------

        ratio = N_total / N_CH3OH

        if (
            np.isfinite(error_N_total)
            and np.isfinite(error_CH3OH)
        ):
            error_ratio = ratio * np.sqrt(
                (error_N_total / N_total)**2
                + (error_CH3OH / N_CH3OH)**2
            )
        else:
            error_ratio = np.nan

        # ----------------------------------------------------
        # Guardar como una nueva especie
        # ----------------------------------------------------

        dict_ratio_CH3OH[region][molecula_total] = {
            'valor': ratio,
            'error': error_ratio,
        }

def plot_comparacion_modelos_quimicos(
    dict_ratio,
    datos_modelos,
    ruta_salida,
):
    """
    Compara N(X)/N(CH3OH) en W51 con las observaciones de
    IRAS 16293B y Sgr B2(N2), y con los modelos químicos
    Fast, Medium y Slow de Garrod et al. (2022).

    Se genera una figura independiente para cada molécula.
    """

    ruta_salida = Path(ruta_salida)
    ruta_salida.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Orden deseado para nuestras regiones
    regiones = (
        REGIONES_PANEL_1
        + REGIONES_PANEL_2
    )

    for molecula, datos_lit in datos_modelos.items():

        # ====================================================
        # Comprobar si tenemos esta molécula en alguna región
        # ====================================================

        regiones_disponibles = [
            region
            for region in regiones
            if molecula in dict_ratio.get(region, {})
        ]

        if len(regiones_disponibles) == 0:

            print(
                f'[modelos] {molecula}: '
                'no disponible en W51.'
            )

            continue

        # ====================================================
        # Preparar valores de W51
        # ====================================================

        etiquetas_x = []
        valores = []
        errores = []
        colores = []

        for region in regiones_disponibles:

            resultado = dict_ratio[region][molecula]

            valor = resultado['valor']
            error = resultado['error']

            if (
                not np.isfinite(valor)
                or valor <= 0
            ):
                continue

            etiquetas_x.append(
                etiqueta_region(region)
            )

            valores.append(valor)
            errores.append(error)

            colores.append(
                COLOR_POR_REGION[region]
            )

        # Si después del filtrado no queda nada
        if len(valores) == 0:
            continue

        # ====================================================
        # Añadir fuentes de literatura
        # ====================================================

        n_w51 = len(valores)

        x_w51 = np.arange(n_w51)

        x_iras = n_w51 + 1
        x_sgr = n_w51 + 2

        etiquetas_x.extend([
            'IRAS 16293B',
            'Sgr B2(N2)',
        ])

        # ====================================================
        # Crear figura
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(14, 8)
        )

        # ====================================================
        # W51
        # ====================================================

        for i, (
            x_i,
            valor,
            error,
            color,
            region,
        ) in enumerate(zip(
            x_w51,
            valores,
            errores,
            colores,
            regiones_disponibles,
        )):

            if np.isfinite(error):

                ax.errorbar(
                    x_i,
                    valor,
                    yerr=error,
                    fmt='o',
                    markersize=8,
                    color=color,
                    markeredgecolor='black',
                    markeredgewidth=1.5,
                    elinewidth=1.6,
                    capsize=5,
                    zorder=5,
                )

            else:

                ax.scatter(
                    x_i,
                    valor,
                    s=120,
                    color=color,
                    edgecolor='black',
                    linewidth=1.2,
                    zorder=5,
                )

        # ====================================================
        # IRAS 16293B
        # ====================================================

        ax.scatter(
            x_iras,
            datos_lit['IRAS16293B'],
            marker='D',
            s=150,
            color='black',
            edgecolor='black',
            linewidth=1.2,
            zorder=6,
            label='IRAS 16293B',
        )

        # ====================================================
        # Sgr B2(N2)
        # ====================================================

        ax.scatter(
            x_sgr,
            datos_lit['SgrB2N2'],
            marker='s',
            s=150,
            facecolor='white',
            edgecolor='black',
            linewidth=1.8,
            zorder=6,
            label='Sgr B2(N2)',
        )

        # ====================================================
        # Modelos químicos
        # ====================================================

        xmin = -0.5
        xmax = x_sgr + 0.5

        ax.hlines(
            datos_lit['fast'],
            xmin=xmin,
            xmax=xmax,
            linestyles='--',
            linewidth=2.5,
            label='Fast model',
        )

        ax.hlines(
            datos_lit['medium'],
            xmin=xmin,
            xmax=xmax,
            linestyles='-.',
            linewidth=2.5,
            label='Medium model',
        )

        ax.hlines(
            datos_lit['slow'],
            xmin=xmin,
            xmax=xmax,
            linestyles=':',
            linewidth=3.0,
            label='Slow model',
        )

        # ====================================================
        # Formato
        # ====================================================

        ax.set_yscale('log')

        ax.set_xlim(
            xmin,
            xmax,
        )

        ax.set_xticks(
            np.arange(len(etiquetas_x))
        )

        ax.set_xticklabels(
            etiquetas_x,
            rotation=35,
            ha='right',
            fontsize=18,
        )

        ax.tick_params(
            axis='y',
            labelsize=18,
        )

        ax.set_ylabel(
            r'$N(X)/N(\mathrm{CH_3OH})$',
            fontsize=22,
        )

        ax.set_title(
            etiqueta_molecula(molecula),
            fontsize=26,
        )

        ax.grid(
            axis='y',
            which='both',
            alpha=0.25,
        )

        ax.legend(
            fontsize=15,
            frameon=True,
            ncol=2,
            loc = "best"
        )

        fig.tight_layout()

        # ====================================================
        # Guardar
        # ====================================================

        nombre_archivo = (
            f'comparacion_modelos_{molecula}.pdf'
        )

        fig.savefig(
            ruta_salida / nombre_archivo,
            bbox_inches='tight',
        )

        plt.show()
        plt.close(fig)

        print(
            f'[modelos] Guardada figura: '
            f'{nombre_archivo}'
        )

plot_comparacion_modelos_quimicos(
    dict_ratio=dict_ratio_CH3OH,
    datos_modelos=DATOS_GARROD,
    ruta_salida=(
        '/home/jorge/TFM/figures/'
        'comparacion_modelos_quimicos'
    ),
)



    
