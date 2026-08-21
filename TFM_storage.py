#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:24:33 2026

@author: jorge
"""

from pathlib import Path
from astropy import units as u
from astropy.table import QTable
import numpy as np
import TFM_config as cfg

# =========================
# FUNCIONES GENERALES
# =========================

def ensure_dir(path):
    """
    Crea una carpeta si no existe.

    Parameters
    ----------
    path : str or pathlib.Path
        Ruta de la carpeta.

    Returns
    -------
    path : pathlib.Path
        Ruta como Path.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(table, path, overwrite=True):
    """
    Guarda una QTable/Table en formato ECSV.

    Parameters
    ----------
    table : astropy.table.Table or astropy.table.QTable
        Tabla a guardar.

    path : str or pathlib.Path
        Ruta de salida.

    overwrite : bool
        Si True, sobrescribe la tabla si ya existe.
    """

    path = Path(path)
    ensure_dir(path.parent)

    table.write(path, format="ascii.ecsv", overwrite=overwrite)


def load_table(path):
    """
    Carga una tabla ECSV como QTable.

    Parameters
    ----------
    path : str or pathlib.Path
        Ruta de la tabla.

    Returns
    -------
    table : astropy.table.QTable
        Tabla cargada.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe la tabla: {path}")

    return QTable.read(path, format="ascii.ecsv")


# =========================
# TABLAS FILTRADAS
# =========================

def save_filtered_table(table, molecule_name, base_dir):
    """
    Guarda una tabla filtrada de una molécula.

    Ejemplo de salida:
        tables/filtradas/C2H5OH_g.ecsv
    """

    path = Path(base_dir) / "filtradas" / f"{molecule_name}.ecsv"
    save_table(table, path)


def load_filtered_table(molecule_name, base_dir):
    """
    Carga una tabla filtrada de una molécula.

    Ejemplo:
        tables/filtradas/C2H5OH_g.ecsv
    """

    path = Path(base_dir) / "filtradas" / f"{molecule_name}.ecsv"
    return load_table(path)


def load_filtered_tables(molecule_names, base_dir):
    """
    Carga varias tablas filtradas y devuelve un diccionario.

    Returns
    -------
    tablas : dict
        Diccionario:
            tablas["C2H5OH_g"] = QTable(...)
    """

    tablas = {}

    for name in molecule_names:
        tablas[name] = load_filtered_table(name, base_dir)

    return tablas


# =========================
# CONTINUOS
# =========================

def save_continuos(dict_T_cont, dict_sigma, path):
    """
    Guarda los continuos por ventana en una tabla ECSV.

    Parameters
    ----------
    dict_T_cont : dict
        Diccionario:
            dict_T_cont["B6-SPW7"] = valor con unidad K

    dict_sigma : dict
        Diccionario:
            dict_sigma["B6-SPW7"] = valor con unidad K

    path : str or pathlib.Path
        Ruta de salida. Ejemplo:
            tables/continuos.ecsv
    """

    tab = QTable(
        names=["ventana", "T_cont", "sigma_cont"],
        dtype=["U30", "f8", "f8"],
        units=[None, u.K, u.K],
    )

    for ventana in dict_T_cont:
        tab.add_row((
            ventana,
            dict_T_cont[ventana],
            dict_sigma[ventana],
        ))

    save_table(tab, path)


def load_continuos(path):
    """
    Carga continuos desde ECSV y reconstruye los diccionarios.

    Returns
    -------
    dict_T_cont : dict
    dict_sigma : dict
    """

    tab = load_table(path)

    dict_T_cont = {}
    dict_sigma = {}

    for row in tab:
        ventana = str(row["ventana"])
        dict_T_cont[ventana] = row["T_cont"]
        dict_sigma[ventana] = row["sigma_cont"]

    return dict_T_cont, dict_sigma


# =========================
# FWHM
# =========================

def save_fwhm_table(fwhm_dict, path):
    """
    Guarda FWHM por molécula en una tabla ECSV.

    Parameters
    ----------
    fwhm_dict : dict
        Diccionario:
            fwhm_dict["C2H5OH_g"] = 5.2 * u.km/u.s
    """

    tab = QTable(
        names=["molecula", "FWHM"],
        dtype=["U40", "f8"],
        units=[None, u.km / u.s],
    )

    for molecula, fwhm in fwhm_dict.items():
        tab.add_row((molecula, fwhm))

    save_table(tab, path)


def load_fwhm_table(path):
    """
    Carga FWHM desde una tabla ECSV y reconstruye el diccionario.

    Returns
    -------
    fwhm_dict : dict
        Diccionario:
            fwhm_dict["C2H5OH_g"] = valor con unidad km/s
    """

    tab = load_table(path)

    fwhm_dict = {}

    for row in tab:
        fwhm_dict[str(row["molecula"])] = row["FWHM"]

    return fwhm_dict


# =========================
# RESULTADOS DIAGRAMA ROTACIONAL
# =========================

def save_diagrot_results(resultados, path):
    """
    Guarda resultados de diagramas rotacionales.

    Parameters
    ----------
    resultados : dict
        Diccionario con estructura:

        resultados["C2H5CN"] = {
            "T_ex": 150*u.K,
            "Delta_Tex": 20*u.K,
            "N_col": 1e16/u.cm**2,
            "Delta_Ncol": 1e15/u.cm**2,
            "pendiente": -0.005,
            "ordenada": 35.2,
            "QTex": 12345.0,
        }
    """

    tab = QTable(
        names=[
            "molecula",
            "T_ex",
            "Delta_Tex",
            "N_col",
            "Delta_Ncol",
            "pendiente",
            "ordenada",
            "QTex",
        ],
        dtype=["U40", "f8", "f8", "f8", "f8", "f8", "f8", "f8"],
        units=[
            None,
            u.K,
            u.K,
            1 / u.cm**2,
            1 / u.cm**2,
            None,
            None,
            None,
        ],
    )

    for molecula, res in resultados.items():
        tab.add_row((
            molecula,
            res["T_ex"],
            res.get("Delta_Tex", float("nan") * u.K),
            res["N_col"],
            res.get("Delta_Ncol", float("nan") / u.cm**2),
            res.get("pendiente", float("nan")),
            res.get("ordenada", float("nan")),
            res.get("QTex", float("nan")),
        ))

    save_table(tab, path)


def load_diagrot_results(path):
    """
    Carga resultados de diagramas rotacionales desde ECSV.

    Returns
    -------
    resultados : dict
    """

    tab = load_table(path)

    resultados = {}

    for row in tab:
        molecula = str(row["molecula"])

        resultados[molecula] = {
            "T_ex": row["T_ex"],
            "Delta_Tex": row["Delta_Tex"],
            "N_col": row["N_col"],
            "Delta_Ncol": row["Delta_Ncol"],
            "pendiente": row["pendiente"],
            "ordenada": row["ordenada"],
            "QTex": row["QTex"],
        }

    return resultados

# =========================
# PARÁMETROS MODELO SINTÉTICO
# =========================

def save_spec_sint_params(params, path):
    """
    Guarda parámetros numéricos del modelo sintético.

    Nota:
    No guarda filtros regex, modelos de astropy ni diccionarios de espectros.
    Solo valores físicos/númericos persistibles.
    """

    tab = QTable(
        names=[
            "molecula",
            "T_ex",
            "N_col",
            "FWHM",
            "f0",
            "v_pik",
            "cat_mol",
            "id_cat",
        ],
        dtype=["U40", "f8", "f8", "f8", "f8", "f8", "U20", "i8"],
        units=[
            None,
            u.K,
            1 / u.cm**2,
            u.km / u.s,
            u.MHz,
            u.km / u.s,
            None,
            None,
        ],
    )

    for molecula, p in params.items():
        tab.add_row((
            molecula,
            p["T_ex"],
            p["N_col"],
            p["FWHM"],
            p["f0"],
            p["v_pik"],
            p["cat_mol"],
            p["id_cat"],
        ))

    save_table(tab, path)


def load_spec_sint_params(path):
    """
    Carga parámetros del modelo sintético.

    Returns
    -------
    params : dict
    """

    tab = load_table(path)

    params = {}

    for row in tab:
        molecula = str(row["molecula"])

        params[molecula] = {
            "T_ex": row["T_ex"],
            "N_col": row["N_col"],
            "FWHM": row["FWHM"],
            "f0": row["f0"],
            "v_pik": row["v_pik"],
            "cat_mol": str(row["cat_mol"]),
            "id_cat": int(row["id_cat"]),
        }

    return params


# =========================
# RESULTADOS CHI2
# =========================

def save_chi2_results(resultados, path):
    """
    Guarda resultados del ajuste chi2.

    Parameters
    ----------
    resultados : dict
        Ejemplo:

        resultados["C2H5OH_g"] = {
            "T_fit": 150*u.K,
            "N_fit": 1e16/u.cm**2,
            "deltaT": 20*u.K,
            "deltaN": 1e15/u.cm**2,
            "chi2_min": 123.4,
        }
    """

    tab = QTable(
        names=[
            "molecula",
            "T_fit",
            "N_fit",
            "deltaT",
            "deltaN",
            "chi2_min",
        ],
        dtype=["U40", "f8", "f8", "f8", "f8", "f8"],
        units=[None, u.K, 1 / u.cm**2, u.K, 1 / u.cm**2, None],
    )

    for molecula, res in resultados.items():
        tab.add_row((
            molecula,
            res["T_fit"],
            res["N_fit"],
            res["deltaT"],
            res["deltaN"],
            res["chi2_min"],
        ))

    save_table(tab, path)


def load_chi2_results(path):
    """
    Carga resultados chi2 desde ECSV.

    Returns
    -------
    resultados : dict
    """

    tab = load_table(path)

    resultados = {}

    for row in tab:
        molecula = str(row["molecula"])

        resultados[molecula] = {
            "T_fit": row["T_fit"],
            "N_fit": row["N_fit"],
            "deltaT": row["deltaT"],
            "deltaN": row["deltaN"],
            "chi2_min": row["chi2_min"],
        }

    return resultados

# =========================
# CONFIGURACIÓN MAESTRA DE MOLÉCULAS
# =========================

def create_molecule_config_table(rows=None):
    """
    Crea la tabla maestra de configuración de moléculas.

    Esta tabla NO guarda directamente objetos Python como regex, listas o ventanas.
    Guarda nombres simbólicos que luego TFM_runtime.py traducirá a objetos reales.

    Columns
    -------
    nombre : str
        Nombre interno usado en el pipeline, por ejemplo 'C2H5OH_g'.

    mol : str
        Nombre de la molécula para Splatalogue.

    intervalo_name : str
        Nombre simbólico del intervalo: 'Banda6', 'Banda3', 'SPW7', etc.

    catalogo : str
        'CDMS' o 'JPL'.

    id_splat_name : str
        Nombre de la variable en TFM_config.py, por ejemplo 'id_cdmsC2H5OH'.

    id_cat_name : str
        Nombre de la variable del catálogo para función de partición.

    B0_name : str
        Nombre de la constante rotacional en TFM_config.py.

    filtro_name : str
        Nombre del filtro regex en TFM_config.py, o 'None'.

    noconsid_name : str
        Nombre de la lista de frecuencias excluidas, o 'None'.

    f0 : Quantity
        Frecuencia de referencia.

    v_pik : Quantity
        Velocidad sistémica/pico.

    E_max : Quantity
        Energía máxima para Splatalogue.

    aij_min : Quantity
        Filtro mínimo de Aij.

    sijmu2_min : Quantity
        Filtro mínimo de Sijmu2.

    filt_inter : float
        Tolerancia de filtrado por diferencia de intensidad.
    """

    tab = QTable(
        names=[
            "nombre",
            "mol",
            "intervalo_name",
            "catalogo",
            "id_splat_name",
            "id_cat_name",
            "B0_name",
            "filtro_name",
            "noconsid_name",
            "f0",
            "v_pik",
            "E_max",
            "aij_min",
            "sijmu2_min",
            "filt_inter",
        ],
        dtype=[
            "U40",
            "U40",
            "U30",
            "U20",
            "U40",
            "U40",
            "U40",
            "U80",
            "U80",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
        ],
        units=[
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            u.MHz,
            u.km / u.s,
            u.K,
            1 / u.s,
            u.D**2,
            None,
        ],
    )

    if rows is not None:
        for row in rows:
            tab.add_row(row)

    return tab


def save_molecule_config(table, path):
    """
    Guarda la tabla maestra de configuración de moléculas.
    """

    save_table(table, path)


def load_molecule_config(path):
    """
    Carga la tabla maestra de configuración de moléculas.
    """

    return load_table(path)


def create_default_molecule_config_table():
    """
    Crea una primera versión de la tabla de configuración de moléculas.

    Puedes guardarla y luego editar el .ecsv a mano si quieres.
    """

    rows = [
    (
        "C2H5OH_g",
        "C2H5OH",
        "Banda6",
        "CDMS",
        "id_splatC2H5OH",
        "id_cdmsC2H5OH",
        "B0C2H5OH",
        "filtro_C2H5OH_quitar_anti",
        "None",
        230000 * u.MHz,
        63 * u.km / u.s,
        600 * u.K,
        1e-6 / u.s,
        10 * u.D**2,
        0.5,
    ),
    (
        "C2H5OH_anti",
        "C2H5OH",
        "Banda6",
        "CDMS",
        "id_splatC2H5OH",
        "id_cdmsC2H5OH",
        "B0C2H5OH",
        "filtro_C2H5OH_quitar_g",
        "None",
        230000 * u.MHz,
        63 * u.km / u.s,
        300 * u.K,
        1e-6 / u.s,
        10 * u.D**2,
        0.5,
    ),
    (
        "CH3OH_v0",
        "CH3OH",
        "Banda6",
        "CDMS",
        "id_splatCH3OH",
        "id_cdmsCH3OH",
        "B0CH3OH",
        "filtro_CH3OH_quitar_v1",
        "None",
        230000 * u.MHz,
        63 * u.km / u.s,
        820 * u.K,
        1e-6 / u.s,
        10 * u.D**2,
        0.5,
    ),
    (
        "CH3OH_v1",
        "CH3OH",
        "Banda6",
        "CDMS",
        "id_splatCH3OH",
        "id_cdmsCH3OH",
        "B0CH3OH",
        "filtro_CH3OH_quitar_v0",
        "None",
        230000 * u.MHz,
        63 * u.km / u.s,
        1000 * u.K,
        1e-6 / u.s,
        0 * u.D**2,
        0.5,
    ),
    (
        "C2H5CN",
        "CH3CH2CN",
        "Banda6",
        "CDMS",
        "id_splatC2H5CN",
        "id_cdmsC2H5CN",
        "B0C2H5CN",
        "None",
        "None",
        230000 * u.MHz,
        63 * u.km / u.s,
        400 * u.K,
        1e-6 / u.s,
        200 * u.D**2,
        0.5,
    ),
    (
        "CH3OCHO_v0E",
        "CH3OCHO",
        "Banda6",
        "CDMS,JPL",
        "id_splatCH3OCHO_v0",
        "id_JPLCH3OCHO",
        "B0CH3OCHO",
        "filtro_CH3OCHO_quitar_A",
        "None",
        230000 * u.MHz,
        63 * u.km / u.s,
        1000 * u.K,
        1e-6 / u.s,
        9 * u.D**2,
        0.5,
    ),
    (
        "CH3OCHO_v0A",
        "CH3OCHO",
        "Banda6",
        "CDMS,JPL",
        "id_splatCH3OCHO_v0",
        "id_JPLCH3OCHO",
        "B0CH3OCHO",
        "filtro_CH3OCHO_quitar_E",
        "None",
        230000 * u.MHz,
        63 * u.km / u.s,
        1000 * u.K,
        1e-6 / u.s,
        10 * u.D**2,
        0.5,
    ),
    (
        "CH3CHO_v0",
        "CH3CHO",
        "Banda6",
        "CDMS,JPL",
        "id_splatCH3CHO",
        "id_JPLCH3CHO",
        "B0CH3CHO",
        "filtro_CH3CHO_quedarse_v0",
        "None",
        230000 * u.MHz,
        63 * u.km / u.s,
        200 * u.K,
        1e-6 / u.s,
        100 * u.D**2,
        0.5,
    ),
    (
        "Acetona",
        "CH3",
        "Banda6",
        "CDMS,JPL",
        "id_splatAcetona",
        "id_JPLacetone",
        "B0acetone",
        "None",
        "list_nofiltAcetona",
        230000 * u.MHz,
        63 * u.km / u.s,
        116 * u.K,
        1e-6 / u.s,
        1000 * u.D**2,
        0.5,
    ),
    (
        "CH3OCH3",
        "CH3OCH3",
        "Banda6",
        "CDMS",
        "id_splatCH3OCH3",
        "id_cdmsCH3OCH3",
        "B0CH3OCH3",
        "filtro_estructuras_acetona_conEE",
        "None",
        230000 * u.MHz,
        63 * u.km / u.s,
        500 * u.K,
        1e-6 / u.s,
        90 * u.D**2,
        0.5,
    ),
    (
        "CH3CN",
        "CH3CN",
        "Banda3",
        "CDMS,JPL",
        "id_splatCH3CN",
        "id_JPLCH3CN",
        "B0CH3CN",
        "filtro_quitar_F",
        "None",
        93000 * u.MHz,
        63 * u.km / u.s,
        1000 * u.K,
        1e-6 / u.s,
        0 * u.D**2,
        0.5,
    ),
]

    return create_molecule_config_table(rows)

# =========================
# CONFIGURACIÓN DELTAS CHI2
# =========================

def create_chi2_deltas_table(rows=None):
    """
    Crea tabla con deltas para el grid de chi2.
    """

    tab = QTable(
        names=["nombre", "deltaT", "deltaN"],
        dtype=["U40", "f8", "f8"],
        units=[None, u.K, 1 / u.cm**2],
    )

    if rows is not None:
        for row in rows:
            tab.add_row(row)

    return tab


def save_chi2_deltas(table, path):
    """
    Guarda tabla de deltas para chi2.
    """

    save_table(table, path)


def load_chi2_deltas(path):
    """
    Carga tabla de deltas para chi2 y devuelve diccionario.
    """

    tab = load_table(path)

    deltas = {}

    for row in tab:
        nombre = str(row["nombre"])
        deltas[nombre] = {
            "deltaT": row["deltaT"],
            "deltaN": row["deltaN"],
        }

    return deltas


def create_default_chi2_deltas_table():
    """
    Crea una tabla inicial de deltas para chi2.
    Ajusta estos valores cuando tengas rangos mejores.
    """

    rows = [
        ("C2H5OH_g", 50 * u.K, 1e16 / u.cm**2),
        ("C2H5OH_anti", 50 * u.K, 1e16 / u.cm**2),
        ("CH3OH_v0", 75 * u.K, 5e16 / u.cm**2),
        ("CH3OH_v1", 75 * u.K, 5e16 / u.cm**2),
        ("C2H5CN", 50 * u.K, 1e15 / u.cm**2),
        ("CH3OCHO_v0E", 50 * u.K, 1e16 / u.cm**2),
        ("CH3OCHO_v0A", 50 * u.K, 1e16 / u.cm**2),
    ]

    return create_chi2_deltas_table(rows)


def path_chi2_results(region_name, base_dir=None):
    """
    Devuelve la ruta de la tabla de resultados chi2 para una región.
    """

    if base_dir is None:
        base_dir = cfg.rutachi2

    path_dir = Path(base_dir) / region_name
    path_dir.mkdir(parents=True, exist_ok=True)

    return path_dir / "chi2_resultados.ecsv"


def create_empty_chi2_results_table():
    """
    Crea una tabla vacía para guardar resultados de ajustes chi2.
    """

    tab = QTable(
        names=[
            "molecula",
            "region",
            "T_init",
            "N_init",
            "T_fit",
            "N_fit",
            "deltaT",
            "deltaN",
            "chi2_min",
            "n_grid",
            "n_iter",
            "converged",
            "fuente_init",
        ],
        dtype=[
            "U100",
            "U100",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "i4",
            "i4",
            "bool",
            "U50",
        ],
        units=[
            None,
            None,
            u.K,
            1 / u.cm**2,
            u.K,
            1 / u.cm**2,
            u.K,
            1 / u.cm**2,
            None,
            None,
            None,
            None,
            None,
        ],
    )

    return tab


def load_chi2_results(region_name, base_dir=None):
    """
    Carga la tabla de resultados chi2 de una región.

    Si no existe, devuelve una tabla vacía.
    """

    path = path_chi2_results(region_name, base_dir=base_dir)

    if path.exists():
        return QTable.read(path, format="ascii.ecsv")

    return create_empty_chi2_results_table()


def save_chi2_results(tab_chi2, region_name, base_dir=None):
    """
    Guarda la tabla de resultados chi2 de una región.
    """

    path = path_chi2_results(region_name, base_dir=base_dir)

    tab_chi2.write(
        path,
        format="ascii.ecsv",
        overwrite=True,
    )

    print(f"[chi2] Tabla de resultados guardada en: {path}")

    return path


def chi2_result_exists(tab_chi2, molecula):
    """
    Devuelve True si la molécula ya tiene resultado chi2 en la tabla.
    """

    if len(tab_chi2) == 0:
        return False

    return np.any(tab_chi2["molecula"] == molecula)


def get_chi2_result(tab_chi2, molecula):
    """
    Devuelve la fila de resultados chi2 de una molécula.
    """

    mask = tab_chi2["molecula"] == molecula

    if not np.any(mask):
        raise KeyError(f"No hay resultado chi2 guardado para {molecula}")

    return tab_chi2[mask][0]


def update_chi2_result(
        tab_chi2,
        molecula,
        region_name,
        T_init,
        N_init,
        T_fit,
        N_fit,
        deltaT,
        deltaN,
        chi2_min,
        n_grid,
        n_iter,
        converged,
        fuente_init="diagrot"):
    """
    Añade o actualiza el resultado chi2 de una molécula.
    """

    T_init = T_init.to(u.K)
    N_init = N_init.to(1 / u.cm**2)
    T_fit = T_fit.to(u.K)
    N_fit = N_fit.to(1 / u.cm**2)
    deltaT = deltaT.to(u.K)
    deltaN = deltaN.to(1 / u.cm**2)

    row_values = [
        molecula,
        region_name,
        T_init,
        N_init,
        T_fit,
        N_fit,
        deltaT,
        deltaN,
        float(chi2_min),
        int(n_grid),
        int(n_iter),
        bool(converged),
        str(fuente_init),
    ]

    mask = tab_chi2["molecula"] == molecula

    if np.any(mask):
        idx = np.where(mask)[0][0]

        for colname, value in zip(tab_chi2.colnames, row_values):
            tab_chi2[colname][idx] = value

        print(f"[chi2] Resultado actualizado para {molecula}")

    else:
        tab_chi2.add_row(row_values)
        print(f"[chi2] Resultado añadido para {molecula}")

    return tab_chi2