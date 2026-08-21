#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import TFM_config as cfg


def _resolve_name(name):
    """
    Traduce un nombre simbólico de la tabla a un objeto real de TFM_config.
    """

    if name is None:
        return None

    name = str(name).strip()

    if name in ("None", "", "nan"):
        return None

    if not hasattr(cfg, name):
        raise AttributeError(f"TFM_config no tiene definido: {name}")

    return getattr(cfg, name)

def resolve_intervalo_region(nombre_intervalo, intervalos_mol_region):
    """
    Resuelve nombres lógicos de intervalos según la región activa.

    Ejemplo:
        'Banda6' -> config_region["intervalos_mol"]["Banda6"]

    No usa variables globales tipo cfg.Banda6.
    """

    if nombre_intervalo is None:
        return None

    nombre_intervalo = str(nombre_intervalo).strip()

    if nombre_intervalo in ("None", "", "nan"):
        return None

    if intervalos_mol_region is None:
        raise ValueError(
            "No se ha proporcionado intervalos_mol_region. "
            "No puedo resolver el intervalo lógico "
            f"'{nombre_intervalo}' sin la región activa."
        )

    if nombre_intervalo not in intervalos_mol_region:
        raise KeyError(
            f"El intervalo lógico '{nombre_intervalo}' no está definido "
            "para la región activa.\n"
            f"Intervalos disponibles: {list(intervalos_mol_region.keys())}"
        )

    return intervalos_mol_region[nombre_intervalo]

def _parse_linelist(catalogo):
    """
    Convierte:
        'CDMS'     -> ('CDMS',)
        'JPL'      -> ('JPL',)
        'CDMS,JPL' -> ('CDMS', 'JPL')
    """

    catalogo = str(catalogo).strip()

    if "," in catalogo:
        return tuple(c.strip() for c in catalogo.split(","))

    return (catalogo,)


def build_config_filtrador_from_table(
        tab_mol_config,
        region_name,
        dict_cubos_med,
        dict_T_contv2,
        fwhm,
        intervalos_mol_region=None):
    """
    Construye CONFIG_FILTRADOR combinando:

    - tab_mol_config:
        identidad espectroscópica de la molécula.

    - REGION_LINE_CONFIG[region_name]["filtrado"]:
        filtros específicos de cada región y molécula.
    """

    if region_name not in cfg.REGION_LINE_CONFIG:
        raise KeyError(
            f"La región {region_name} no está en REGION_LINE_CONFIG. "
            f"Regiones disponibles: {list(cfg.REGION_LINE_CONFIG.keys())}"
        )

    config_region = cfg.REGION_LINE_CONFIG[region_name]

    if "filtrado" not in config_region:
        raise KeyError(
            f"La región {region_name} no tiene definido el bloque 'filtrado'."
        )

    config_filtrado_region = config_region["filtrado"]

    config = {}

    for row in tab_mol_config:

        nombre = str(row["nombre"])

        if nombre not in fwhm:
            raise KeyError(
                f"No hay FWHM para {nombre}. "
                "Calcula primero la calibración de FWHM."
            )

        if nombre not in config_filtrado_region:
            raise KeyError(
                f"{nombre} no tiene configuración de filtrado "
                f"para la región {region_name}."
            )

        filtros = config_filtrado_region[nombre]

        parametros_obligatorios = [
            "E_max",
            "aij_min",
            "sijmu2_min",
            "filt_inter",
            "frec_nofiltrar",
        ]

        for parametro in parametros_obligatorios:
            if parametro not in filtros:
                raise KeyError(
                    f"Falta '{parametro}' en la configuración de filtrado "
                    f"de {nombre} para la región {region_name}."
                )

        config[nombre] = {
            # Identidad molecular: procede de tab_mol_config
            "mol": str(row["mol"]),

            "intervalo": resolve_intervalo_region(
                row["intervalo"],
                intervalos_mol_region=intervalos_mol_region,
            ),

            "id_splat1": _resolve_name(row["id_splat"]),

            "filtro_estructurasf": _resolve_name(
                row["filtro_estructuras"]
            ),

            "linelist": _parse_linelist(row["catalogo"]),

            "f0": row["f0"],

            "id_cat": _resolve_name(row["id_cat"]),

            "B0": _resolve_name(row["B0"]),

            # Filtros específicos de región y molécula
            "E_max": filtros["E_max"],

            "aij_min": filtros["aij_min"],

            "sijmu2_min": filtros["sijmu2_min"],

            "filt_inter": filtros["filt_inter"],

            "list_freq_nofilt": filtros["frec_nofiltrar"],

            # Datos de ejecución
            "Tcont": dict_T_contv2,

            "anch_fix": fwhm[nombre],

            "dict_espec": dict_cubos_med,
        }

    return config