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
                label=region,
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
                label=region,
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
                label=region,
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

for region, resultados_region in dict_Ncol.items():

    dict_ratio_CH3OH[region] = {}

    referencia = resultados_region.get('CH3OH_v0')

    if referencia is None:
        continue

    N_CH3OH = referencia['valor']
    error_CH3OH = referencia['error']

    for molecula, resultado in resultados_region.items():

        if molecula == 'CH3OH_v0':
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
                label=region,
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
    
