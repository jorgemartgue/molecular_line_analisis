#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:40:46 2026

@author: jorge
"""

from astroquery.splatalogue import Splatalogue
from astropy.table import Table, vstack, unique
import re
from datetime import datetime, timezone
from pathlib import Path
import TFM_config as cfg
from TFM_runtime import (
    _parse_linelist,
    _resolve_name,
    resolve_intervalo_region,
)
import astropy.units as u

_pat_html = re.compile(r"<[^>]+>")

def aplica_filtros(qn, filtros):
    s = _pat_html.sub('', str(qn))  # limpia html

    for f in filtros:
        if isinstance(f, str):
            if f in s:
                return False
        else:
            # regex compilada (re.Pattern)
            if f.search(s):
                return False
    return True

def buscador_splatalogue_cdms(elemento, intervalo, E_max, id_splat=None,
                              columnas=None, filtro_estructuras=None,
                              linelist=['CDMS']):
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

id_splat : int, optional
    Identifier used by Splatalogue to label the molecule (column `species_id`).
    If None (default), no filtering by species_id is applied.

columnas : array-like of str, optional
    Columns to retrieve from Splatalogue. If None (default), the following
    columns are used:
        ('species_id', 'name', 'resolved_QNs', 'orderedfreq',
         'aij', 'sijmu2', 'upper_state_energy_K', 'upperStateDegen',
         'ventana_obs', 'linelist').

filtro_estructuras : list of re.Pattern, optional
    List of compiled regular expressions used to filter out specific
    structures in the quantum numbers.

    Example:
        filtro_CH3OCHO_quitar_A = [re.compile(r'\\sA$')]

    This would remove all transitions labeled as 'A' for CH3OCHO.
    Default is None.

linelist : array-like of str, optional
    Spectroscopic catalogs to query (e.g., CDMS, JPL). Default is ['CDMS'].

    To use both CDMS and JPL:
        ['CDMS', 'JPL']

Returns
-------
tab : QTable
    Table containing all the transitions that satisfy the applied filters,
    including the selected columns.
    """

    tab = Table()

    if columnas is None:

        columnas = ('species_id', 'name', 'resolved_QNs', 'orderedfreq',
                    'aij', 'sijmu2', 'upper_state_energy_K', 'upperStateDegen',
                    'ventana_obs', 'linelist')

    for name, nu_min, nu_max, name_window in intervalo:

        t = Splatalogue.query_lines(nu_min, nu_max, chemical_name=elemento,
                                    energy_max=E_max.value, energy_type='eu_k',
                                    line_strengths=['Aij'],
                                    line_lists=linelist)

        # si la tabla está vacía, salta
        if len(t) == 0:
            continue

        if id_splat is not None:
            if 'species_id' in t.colnames:

                t = t[t['species_id'] == id_splat]
            elif 'moleculeTag' in t.colnames:

                t = t[t['moleculeTag'] == id_splat]
            else:
                continue

        if len(t) == 0:
            continue

        t['ventana_obs'] = [name] * len(t)

    # selecciona solo columnas disponibles
        cols_ok = [c for c in columnas if c in t.colnames]
        if len(cols_ok) == 0:
            continue
        t = t[cols_ok]

        if len(tab) == 0:

            tab = t

        else:

            tab = vstack([tab, t])

    if filtro_estructuras is not None and 'resolved_QNs' in tab.colnames:

        mask = [aplica_filtros(q,
                               filtro_estructuras) for q in tab['resolved_QNs']]
        tab = tab[mask]
        
    return tab

def path_catalogo_splatalogue(
        molecula,
        intervalo_name,
        base_dir=None):
    """
    Devuelve la ruta del catálogo local de una molécula y banda.
    """

    if base_dir is None:
        base_dir = cfg.rutatablas

    path_dir = Path(base_dir) / "catalogos_splatalogue"
    path_dir.mkdir(parents=True, exist_ok=True)

    return path_dir / f"{molecula}_{intervalo_name}.ecsv"

def _consultar_catalogo_completo(
        elemento,
        intervalos,
        id_splat=None,
        linelist=("CDMS",)):
    """
    Descarga todas las transiciones de una molécula situadas
    dentro de las ventanas espectrales observadas.

    No aplica filtros de energía, Aij, Sijmu2 ni estructura.
    Esos filtros se aplicarán posteriormente sobre la tabla local.
    """

    tablas = []

    columnas = (
        "species_id",
        "name",
        "resolved_QNs",
        "orderedfreq",
        "aij",
        "sijmu2",
        "upper_state_energy_K",
        "upperStateDegen",
        "ventana_obs",
        "linelist",
    )

    for nombre_cubo, nu_min, nu_max, nombre_ventana in intervalos:

        tab = Splatalogue.query_lines(
            nu_min,
            nu_max,
            chemical_name=elemento,
            line_strengths=["Aij"],
            line_lists=list(linelist),
        )

        if len(tab) == 0:
            continue

        # Seleccionar únicamente la entrada concreta del catálogo
        if id_splat is not None:

            if "species_id" in tab.colnames:
                tab = tab[tab["species_id"] == id_splat]

            elif "moleculeTag" in tab.colnames:
                tab = tab[tab["moleculeTag"] == id_splat]

            else:
                continue

        if len(tab) == 0:
            continue

        # Guardar a qué cubo/SPW pertenece cada transición
        tab["ventana_obs"] = [nombre_cubo] * len(tab)

        # Splatalogue puede no devolver siempre todas las columnas
        columnas_disponibles = [
            columna
            for columna in columnas
            if columna in tab.colnames
        ]

        tab = tab[columnas_disponibles]
        tablas.append(tab)

    if not tablas:
        return Table()

    catalogo = vstack(
        tablas,
        metadata_conflicts="silent",
    )

    # Eliminar posibles líneas duplicadas
    claves_unicas = [
        columna
        for columna in (
            "species_id",
            "orderedfreq",
            "resolved_QNs",
        )
        if columna in catalogo.colnames
    ]

    if claves_unicas:
        catalogo = unique(
            catalogo,
            keys=claves_unicas,
            keep="first",
        )

    return catalogo

def cargar_o_descargar_catalogo_molecula(
        molecula,
        tab_mol_config,
        intervalos_mol_region,
        recalcular=False,
        base_dir=None):
    """
    Carga el catálogo local de una molécula y banda.

    Si no existe, o recalcular=True, consulta Splatalogue,
    guarda el resultado en ECSV y devuelve la tabla.
    """

    # Seleccionar la configuración de la molécula
    tab_mol = tab_mol_config[
        tab_mol_config["nombre"] == molecula
    ]

    if len(tab_mol) == 0:
        raise KeyError(
            f"{molecula} no está en moleculas_config.ecsv"
        )

    if len(tab_mol) > 1:
        raise ValueError(
            f"{molecula} aparece más de una vez "
            "en moleculas_config.ecsv"
        )

    row = tab_mol[0]

    # Banda o intervalo lógico: Banda3, Banda6...
    intervalo_name = str(row["intervalo"]).strip()

    path = path_catalogo_splatalogue(
        molecula=molecula,
        intervalo_name=intervalo_name,
        base_dir=base_dir,
    )

    # Cargar el archivo existente
    if path.exists() and not recalcular:

        print(
            f"[catálogo] Cargando catálogo local: {path}"
        )

        return Table.read(
            path,
            format="ascii.ecsv",
        )

    # Resolver las ventanas reales de la región activa
    intervalos = resolve_intervalo_region(
        intervalo_name,
        intervalos_mol_region=intervalos_mol_region,
    )

    # Resolver los nombres almacenados en moleculas_config.ecsv
    id_splat = _resolve_name(row["id_splat"])
    linelist = _parse_linelist(row["catalogo"])
    filtro_estructuras = _resolve_name(row["filtro_estructuras"])

    print(
        f"[catálogo] Consultando Splatalogue para "
        f"{molecula}_{intervalo_name}"
    )

    catalogo = _consultar_catalogo_completo(
        elemento=str(row["mol"]),
        intervalos=intervalos,
        id_splat=id_splat,
        linelist=linelist,
    )

    if len(catalogo) == 0:
        raise ValueError(
            f"Splatalogue no devolvió líneas para {molecula} "
            f"en {intervalo_name}. Revisa mol, id_splat "
            "y catálogo."
        )

    # Aplicar el filtro estructural correspondiente a esta entrada
    # molecular antes de guardar el catálogo local.
    n_lineas_sin_filtrar = len(catalogo)

    if filtro_estructuras is not None:

        if "resolved_QNs" not in catalogo.colnames:
            raise KeyError(
                "El catálogo descargado no contiene la columna "
                "'resolved_QNs', necesaria para aplicar el filtro "
                f"de estructuras de {molecula}."
            )

        mask_estructura = [
            aplica_filtros(qn, filtro_estructuras)
            for qn in catalogo["resolved_QNs"]
        ]

        catalogo = catalogo[mask_estructura]

    if len(catalogo) == 0:
        raise ValueError(
            f"El filtro de estructuras ha eliminado todas las "
            f"transiciones de {molecula} en {intervalo_name}."
        )

    print(
        f"[catálogo] Filtro de estructuras: "
        f"{n_lineas_sin_filtrar} → {len(catalogo)} líneas"
    )

    # Metadatos para poder identificar la descarga
    catalogo.meta["molecula_pipeline"] = molecula
    catalogo.meta["molecula_splatalogue"] = str(row["mol"])
    catalogo.meta["intervalo"] = intervalo_name
    catalogo.meta["catalogos"] = ",".join(linelist)
    catalogo.meta["id_splat"] = str(id_splat)
    catalogo.meta["fecha_descarga_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    catalogo.meta["n_lineas"] = len(catalogo)

    catalogo.meta["rangos_MHz"] = "; ".join(
        (
            f"{nu_min.to_value(u.MHz):.6f}-"
            f"{nu_max.to_value(u.MHz):.6f}"
        )
        for _, nu_min, nu_max, _ in intervalos
    )

    catalogo.write(
        path,
        format="ascii.ecsv",
        overwrite=True,
    )

    print(
        f"[catálogo] Catálogo local guardado en: {path}"
    )

    return catalogo