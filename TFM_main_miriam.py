#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:42:19 2026

@author: jorge
"""

from astropy import units as u
import sys
import TFM_config as cfg
import numpy as np

from TFM_load_region import seleccionar_region, cargar_datos_region

from TFM_calibration import (
    cargar_o_calcular_calibracion_molecula,
    cargar_o_calcular_calibracion_pixeles_molecula,
    dicts_calibracion_region,
)

from TFM_continuum import (
    cargar_o_calcular_continuo_medio,
    cargar_o_calcular_continuo_pixeles,
    cargar_o_calcular_continuo_medio_percent,
    cargar_o_calcular_continuo_pixeles_percent,
)

from TFM_filtered_lines import (
    cargar_o_calcular_tabla_filtrada_molecula,
    cargar_o_calcular_tablas_filtradas_pixeles,
)

from TFM_diagrot_pipeline import (
    cargar_o_calcular_diagrot_molecula_medio,
    cargar_o_calcular_diagrot_pixeles,
    cargar_resultados_diagrot_region,
    corregir_mapas_T_N,
    save_diagrot_maps_fits,
)

from TFM_synthetic_pipeline import calcular_modelo_sintetico

from TFM_chi2_pipeline import (
    cargar_o_calcular_chi2_molecula,
    construir_residuo_base_chi2,
    obtener_orden_ajuste_chi2,
    obtener_dict_noconsid_chi2,
)
from TFM_storage import (
    load_molecule_config,
    load_diagrot_results,
    load_chi2_results,
    save_chi2_results,
    update_chi2_result,
    chi2_result_exists,
    get_chi2_result
)

import shutil
from astropy.table import QTable
from TFM_chi2_maps import (
    cargar_o_calcular_chi2_pixeles,
    save_chi2_maps_fits,
)

from TFM_splatalogue_tools import (
    cargar_o_descargar_catalogo_molecula,
)

import warnings

#Para ignorar todos los warnings
warnings.filterwarnings("ignore")

# ============================================================
# CREAR TABLA DE CONFIGURACIÓN DE MOLÉCULAS
# ============================================================

MOSTRAR_PLOTS = True

tabla_config_mol = input(
    "¿Quieres generar la tabla config mol desde MOLECULE_CONFIG? S/N: "
).strip().upper()

if tabla_config_mol == "S":

    PATH_CONFIG_MOLECULAS = (
        cfg.rutatablas / "config" / "moleculas_config.ecsv"
    )

    # --------------------------------------------------------
    # Validaciones
    # --------------------------------------------------------

    def validar_nombre_cfg(nombre):
        """
        Comprueba que un nombre simbólico existe en TFM_config.py.

        Permite None, 'None', una cadena vacía y 'nan'.
        """

        if nombre is None:
            return

        nombre = str(nombre).strip()

        if nombre in ("None", "", "nan"):
            return

        if not hasattr(cfg, nombre):
            raise AttributeError(
                "TFM_config.py no tiene definido el nombre "
                f"simbólico: {nombre}"
            )


    def validar_intervalo_logico(nombre):
        """
        Comprueba que el intervalo es una etiqueta lógica válida.
        """

        if nombre is None:
            return

        nombre = str(nombre).strip()

        if nombre in ("None", "", "nan"):
            return

        intervalos_validos = {
            "Banda3",
            "Banda6",
            "SPW7",
            "SPW6y7",
        }

        if nombre not in intervalos_validos:
            raise ValueError(
                f"Intervalo lógico no válido: {nombre}. "
                f"Opciones válidas: {sorted(intervalos_validos)}"
            )


    if not hasattr(cfg, "MOLECULE_CONFIG"):
        raise AttributeError(
            "TFM_config.py no contiene MOLECULE_CONFIG."
        )

    if len(cfg.MOLECULE_CONFIG) == 0:
        raise ValueError(
            "MOLECULE_CONFIG está vacío."
        )

    parametros_obligatorios = {
        "mol",
        "intervalo",
        "catalogo",
        "id_splat",
        "id_cat",
        "B0",
        "filtro_estructuras",
        "f0",
    }

    # --------------------------------------------------------
    # Crear la tabla vacía
    # --------------------------------------------------------

    tab_nueva = QTable(
        names=[
            "nombre",
            "mol",
            "intervalo",
            "catalogo",
            "id_splat",
            "id_cat",
            "B0",
            "filtro_estructuras",
            "f0",
        ],

        dtype=[
            "U100",
            "U100",
            "U100",
            "U100",
            "U100",
            "U100",
            "U100",
            "U100",
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
            u.MHz,
        ],
    )

    # --------------------------------------------------------
    # Construir las filas desde MOLECULE_CONFIG
    # --------------------------------------------------------

    for nombre, config_mol in cfg.MOLECULE_CONFIG.items():

        parametros_faltantes = (
            parametros_obligatorios - set(config_mol.keys())
        )

        if parametros_faltantes:
            raise KeyError(
                f"Faltan parámetros en MOLECULE_CONFIG['{nombre}']: "
                f"{sorted(parametros_faltantes)}"
            )

        validar_intervalo_logico(
            config_mol["intervalo"]
        )

        validar_nombre_cfg(
            config_mol["id_splat"]
        )

        validar_nombre_cfg(
            config_mol["id_cat"]
        )

        validar_nombre_cfg(
            config_mol["B0"]
        )

        validar_nombre_cfg(
            config_mol["filtro_estructuras"]
        )

        f0 = config_mol["f0"]

        if not isinstance(f0, u.Quantity):
            raise TypeError(
                f"MOLECULE_CONFIG['{nombre}']['f0'] debe tener "
                "unidades de frecuencia."
            )

        if not f0.unit.is_equivalent(u.MHz):
            raise u.UnitConversionError(
                f"MOLECULE_CONFIG['{nombre}']['f0'] tiene unidades "
                f"incorrectas: {f0.unit}"
            )

        tab_nueva.add_row(
            (
                nombre,
                config_mol["mol"],
                config_mol["intervalo"],
                config_mol["catalogo"],
                config_mol["id_splat"],
                config_mol["id_cat"],
                config_mol["B0"],
                config_mol["filtro_estructuras"],
                f0.to(u.MHz),
            )
        )

    print(
        "[config] Validación correcta: "
        "MOLECULE_CONFIG contiene todos los parámetros necesarios."
    )

    # --------------------------------------------------------
    # Backup y guardado
    # --------------------------------------------------------

    PATH_CONFIG_MOLECULAS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if PATH_CONFIG_MOLECULAS.exists():

        backup_path = PATH_CONFIG_MOLECULAS.with_suffix(
            ".backup.ecsv"
        )

        shutil.copy(
            PATH_CONFIG_MOLECULAS,
            backup_path,
        )

        print(
            f"[config] Backup guardado en: {backup_path}"
        )

    tab_nueva.write(
        PATH_CONFIG_MOLECULAS,
        format="ascii.ecsv",
        overwrite=True,
    )

    print(
        f"[config] Nueva tabla guardada en: "
        f"{PATH_CONFIG_MOLECULAS}"
    )

    print("\n[config] Tabla de configuración generada:")

    tab_nueva.pprint(
        max_width=-1,
        max_lines=-1,
    )

# ============================================================
# SELECCIÓN DE REGIÓN
# ============================================================

regiones_disponibles = list(cfg.regiones.keys())

print("\nRegiones disponibles:")
for i, reg in enumerate(regiones_disponibles, start=1):
    print(f"  {i}. {reg}")

REGION = input(
    "\nSelecciona la región que quieras "
    f"({', '.join(regiones_disponibles)}): "
).strip()

if REGION not in cfg.regiones:
    raise ValueError(
        f"Región '{REGION}' no reconocida. "
        f"Regiones disponibles: {regiones_disponibles}"
    )

config_region = seleccionar_region(REGION)

modo_diagrot = config_region["modo_diagrot"]

if "rutacarp" not in config_region:
    raise KeyError(
        f"La región {REGION} no tiene definida 'rutacarp' en cfg.regiones."
    )

RUTACARP_REGION = config_region["rutacarp"]
RUTAREGION_REGION = config_region["ruta"]
VENTANAS_REGION = config_region["ventanas_obs"]
INTERVALOS_MOL_REGION = config_region["intervalos_mol"]

print("[main] Carpeta activa de cubos:")
print(f"       {RUTACARP_REGION}")

print("[main] Región DS9 activa:")
print(f"       {RUTAREGION_REGION}")

(dict_cubos_med,
 dict_resol_esp,
 dict_cubos_comp) = cargar_datos_region(config_region)

continuar = input('¿Quieres continuar? Siguiente paso, calibración del continuo (S/N)')

if continuar == 'N':
    sys.exit(0)

# ============================================================
# CALIBRACIÓN DEL CONTINUO
# ============================================================

respuesta = input(
    "¿Quieres recalcular el continuo para la "
    "región seleccionada? [s/N]: "
).strip().lower()

RECALCULAR_CONTINUO = respuesta in {
    "s",
    "si",
    "sí",
}

# True  -> usa continuo por percentiles
# False -> usa continuo clásico por intervalos_Tcont
USAR_CONTINUO_PERCENT = True

#Usar este para d2, MM31, MM14, MM24
# PERCENTILES_CONT = {
#     "B3-SPW0": (5, 60),
#     "B3-SPW1": (5, 60),
#     "B3-SPW2": (5, 60),
#     "B3-SPW3": (5, 60),

#     "B6-SPW0": (5, 40),
#     "B6-SPW1": (5, 40),
#     "B6-SPW2": (5, 40),
#     "B6-SPW3": (5, 60),
#     "B6-SPW4": (2, 20),
#     "B6-SPW5": (5, 60),
#     "B6-SPW6": (5, 60),
#     "B6-SPW7": (5, 50),
# }

#Usar este para NORTH Y MM35
# PERCENTILES_CONT = {
#     "B3-SPW0": (5, 60),
#     "B3-SPW1": (5, 60),
#     "B3-SPW2": (5, 60),
#     "B3-SPW3": (5, 60),

#     "B6-SPW0": (2, 20),
#     "B6-SPW1": (2, 20),
#     "B6-SPW2": (5, 40),
#     "B6-SPW3": (5, 40),
#     "B6-SPW4": (2, 20),
#     "B6-SPW5": (5, 60),
#     "B6-SPW6": (5, 40),
#     "B6-SPW7": (5, 30),
# }

#USAR ESTA PARA e8mm

PERCENTILES_CONT = {
    "B3-SPW0": (5, 60),
    "B3-SPW1": (5, 60),
    "B3-SPW2": (5, 60),
    "B3-SPW3": (5, 60),

    "B6-SPW0": (5, 40),
    "B6-SPW1": (5, 20),
    "B6-SPW2": (5, 20),
    "B6-SPW3": (5, 40),
    "B6-SPW4": (5, 20),
    "B6-SPW5": (5, 40),
    "B6-SPW6": (5, 40),
    "B6-SPW7": (5, 30),
}


# None calcula todas las SPWs disponibles.
# Para limitarlo, por ejemplo:
# SPWS_CONTINUO = ["B6-SPW0", "B6-SPW1", "B6-SPW2", "B6-SPW3",
#                  "B6-SPW4", "B6-SPW5", "B6-SPW6", "B6-SPW7"]

SPWS_CONTINUO = None

VF_SYS = config_region.get("v_sys", 63 * u.km / u.s)
print(f"[region] Velocidad sistémica usada: {VF_SYS}")

dict_T_contv2 = None
dict_sigma_vent = None
dict_T_cont_pix = None
dict_sigma_pix = None

if config_region.get("calibrar_continuo_medio", False):

    if USAR_CONTINUO_PERCENT:

        dict_T_contv2, dict_sigma_vent = cargar_o_calcular_continuo_medio_percent(
            region_name=REGION,
            percentiles=PERCENTILES_CONT,
            ventanas_obs=VENTANAS_REGION,
            dict_cubos_med=dict_cubos_med,
            base_dir=cfg.rutatablas,
            SPWs=SPWS_CONTINUO,
            recalcular=RECALCULAR_CONTINUO,
            plots=MOSTRAR_PLOTS,
            rutacarp_region=RUTACARP_REGION,
            rutaregion_region=RUTAREGION_REGION,
        )

    else:

        dict_T_contv2, dict_sigma_vent = cargar_o_calcular_continuo_medio(
            region_name=REGION,
            intervalos_Tcont=cfg.intervalos_Tcont,
            ventanas_obs=VENTANAS_REGION,
            vfuent=VF_SYS,
            dict_cubos_med=dict_cubos_med,
            base_dir=cfg.rutatablas,
            recalcular=RECALCULAR_CONTINUO,
            plots=MOSTRAR_PLOTS,
            rutacarp_region=RUTACARP_REGION,
            rutaregion_region=RUTAREGION_REGION,
        )

if config_region.get("calibrar_continuo_pixeles", False):

    if USAR_CONTINUO_PERCENT:

        dict_T_cont_pix, dict_sigma_pix = cargar_o_calcular_continuo_pixeles_percent(
            region_name=REGION,
            percentiles=PERCENTILES_CONT,
            ventanas_obs=VENTANAS_REGION,
            dict_cubos_comp=dict_cubos_comp,
            base_dir=cfg.rutatablas,
            SPWs=SPWS_CONTINUO,
            recalcular=RECALCULAR_CONTINUO,
            rutacarp_region=RUTACARP_REGION,
            rutaregion_region=RUTAREGION_REGION,
        )

    else:

        dict_T_cont_pix, dict_sigma_pix = cargar_o_calcular_continuo_pixeles(
            region_name=REGION,
            intervalos_Tcont=cfg.intervalos_Tcont,
            ventanas_obs=VENTANAS_REGION,
            vfuent=VF_SYS,
            dict_cubos_comp=dict_cubos_comp,
            base_dir=cfg.rutatablas,
            recalcular=RECALCULAR_CONTINUO,
            plots=MOSTRAR_PLOTS,
            rutacarp_region=RUTACARP_REGION,
            rutaregion_region=RUTAREGION_REGION,
        )

# ============================================================
# CARGAR CONFIGURACIÓN DE MOLÉCULAS
# ============================================================

PATH_CONFIG_MOLECULAS = cfg.rutatablas / "config" / "moleculas_config.ecsv"

tab_mol_config = load_molecule_config(PATH_CONFIG_MOLECULAS)

fwhm_dict, v_pik_dict = dicts_calibracion_region(REGION)

print("[calibración] Moléculas ya calibradas en esta región:")
print(list(fwhm_dict.keys()))

moleculas_disponibles = [
    str(mol).strip()
    for mol in tab_mol_config["nombre"]
]

def seleccionar_molecula():

    print("\nMoléculas disponibles:")

    for i, mol in enumerate(moleculas_disponibles, start=1):
        print(f"  {i}. {mol}")

    while True:

        seleccion = input(
            "\nSelecciona la molécula por número o nombre "
            "('q' para terminar): "
        ).strip()

        # Salir del programa
        if seleccion.lower() in {"q", "salir"}:
            return None

        # Selección por número
        if seleccion.isdigit():

            indice = int(seleccion) - 1

            if 0 <= indice < len(moleculas_disponibles):
                return moleculas_disponibles[indice]

            print("[main] Número de molécula no válido.")
            continue

        # Selección por nombre
        if seleccion in moleculas_disponibles:
            return seleccion

        print(
            f"[main] Molécula '{seleccion}' no reconocida. "
            "Vuelve a intentarlo."
        )

def procesar_molecula(MOLECULA):
    
    # ============================================================
    # Catálogo molecular de Splatalogue
    # ============================================================

    respuesta = input(
        "¿Quieres recalcular el catalogo de splatalogue para la "
        "molécula seleccionada? [s/N]: "
    ).strip().lower()

    RECALCULAR_CATALOGO_SPLATALOGUE = respuesta in {
        "s",
        "si",
        "sí",
    }

    tab_catalogo_mol = cargar_o_descargar_catalogo_molecula(
        molecula=MOLECULA,
        tab_mol_config=tab_mol_config,
        intervalos_mol_region=INTERVALOS_MOL_REGION,
        recalcular=RECALCULAR_CATALOGO_SPLATALOGUE,
    )

    # ============================================================
    # CALIBRACIÓN DE FWHM DE LA MOLÉCULA
    # ============================================================



    tab_cali_mol, fwhm_mol, v_pik_model, fwhm_dict, v_pik_dict = cargar_o_calcular_calibracion_molecula(
        molecula=MOLECULA,
        region_name=REGION,
        dict_cubos_med=dict_cubos_med,
        recalcular=True,
        plots=MOSTRAR_PLOTS,
        v_sys=VF_SYS,
        ventanas_obs=VENTANAS_REGION,
        rutacarp_region=RUTACARP_REGION,
        rutaregion_region=RUTAREGION_REGION,
    )

    print(f"[calibración] FWHM usado para {MOLECULA}: {fwhm_mol}")
    print(f"[calibración] v_pik usado para {MOLECULA}: {v_pik_model}")
    print("[calibración] Diccionario v_pik disponible:")
    print(v_pik_dict)

    print(f"[calibración] v_pik_model usado para {MOLECULA}: {v_pik_model}")

    # ============================================================
    # CALIBRACIÓN PÍXEL A PÍXEL
    # ============================================================

    respuesta = input(
        "¿Quieres recalcular la calibración píxel a píxel para la "
        "molécula seleccionada? [s/N]: "
    ).strip().lower()

    RECALCULAR_CALIBRACION_PIXELES = respuesta in {
        "s",
        "si",
        "sí",
    }

    v_pik_map = None
    fwhm_map = None

    if modo_diagrot == "pixeles":

        v_pik_map, fwhm_map = (
            cargar_o_calcular_calibracion_pixeles_molecula(
                molecula=MOLECULA,
                region_name=REGION,
                dict_cubos_comp=dict_cubos_comp,
                ventanas_obs=VENTANAS_REGION,
                fwhm_global=fwhm_dict[MOLECULA],
                v_pik_global=v_pik_dict[MOLECULA],
                recalcular=RECALCULAR_CALIBRACION_PIXELES,
            )
        )

        print(
            f"[calibración pix] Mapas disponibles para {MOLECULA}: "
            f"shape={v_pik_map.shape}"
        )

    continuar = input(
        '¿Quieres continuar? SiguienteS paso, filtrado y diagrama rotacional (S/N)')

    if continuar == 'N':
        return
    # ============================================================
    # DIAGRAMA ROTACIONAL
    # ============================================================

    respuesta = input(
        "¿Quieres recalcular el filtrado y el diagrama "
        "rotacional para la molécula seleccionada? [s/N]: "
    ).strip().lower()

    recalcular = respuesta in {
        "s",
        "si",
        "sí",
    }

    RECALCULAR_FILTRADO = recalcular
    RECALCULAR_FILTRADO_PIXELES = recalcular
    RECALCULAR_DIAGROT = recalcular
    RECALCULAR_DIAGROT_PIXELES = recalcular

    guardar_plots=True

    resultados_diagrot = cargar_resultados_diagrot_region(REGION)

    print("[diagrot] Moléculas disponibles en resultados_diagrot:")
    print(list(resultados_diagrot.keys()))


    # ============================================================
    # DIAGRAMA ROTACIONAL: MODO MEDIO O PIXELES
    # ============================================================

    if modo_diagrot == "medio":
    
        # ============================================================
        # CARGA/CÁLCULO SEGURO DE TABLA FILTRADA MEDIA
        # ============================================================

        tab_filtrada = None
        tabla_filtrada_ok = False
    
        try:
            tab_filtrada = cargar_o_calcular_tabla_filtrada_molecula(
                molecula=MOLECULA,
                region_name=REGION,
                tab_mol_config=tab_mol_config,
                dict_cubos_med=dict_cubos_med,
                dict_T_cont=dict_T_contv2,
                fwhm_dict=fwhm_dict,
                recalcular=RECALCULAR_FILTRADO,
                base_dir=cfg.rutatablas,
                plots=MOSTRAR_PLOTS,
                v_filt_override=VF_SYS, 
                rutacarp_region=RUTACARP_REGION,
                rutaregion_region=RUTAREGION_REGION,
                intervalos_mol_region=INTERVALOS_MOL_REGION,
                tab_catalogo=tab_catalogo_mol,
            )

            if len(tab_filtrada) > 0:
                tabla_filtrada_ok = True
                print(f"[filtrado] Tabla filtrada media válida para {MOLECULA}")
            else:
                print(f"[filtrado] Tabla filtrada media vacía para {MOLECULA}")

        except Exception as e:
            print(f"[filtrado] No hay tabla filtrada media válida para {MOLECULA}")
            print(f"[filtrado] Motivo: {e}")
            print("[filtrado] Continuo si existe entrada en resultados_diagrot.")
            tab_filtrada = None
            tabla_filtrada_ok = False

        # ============================================================
        # DIAGROT MEDIO O VALORES MANUALES
        # ============================================================

        if tabla_filtrada_ok:

            print(f"[diagrot] Hay tabla filtrada media para {MOLECULA}")

            if MOLECULA in resultados_diagrot and not RECALCULAR_DIAGROT:
                print(
                    f"[diagrot] Ya existe resultado para {MOLECULA}. "
                    "Uso el resultado guardado."
                )
                resultado_diagrot = resultados_diagrot[MOLECULA]

            else:
                print(f"[diagrot] Calculando diagrama rotacional medio para {MOLECULA}")
    
                resultado_diagrot = cargar_o_calcular_diagrot_molecula_medio(
                    molecula=MOLECULA,
                    region_name=REGION,
                    tab_mol_config=tab_mol_config,
                    tab_filtrada=tab_filtrada,
                    recalcular=RECALCULAR_DIAGROT,
                    guardar_pdf=True,
                    plot_Q=True,
                    plots=MOSTRAR_PLOTS,
                )

                resultados_diagrot = cargar_resultados_diagrot_region(REGION)
    
        else:

            print(f"[diagrot] No hay tabla filtrada media válida para {MOLECULA}")

            if MOLECULA not in resultados_diagrot:
                raise KeyError(
                    f"No hay tabla filtrada para {MOLECULA} y tampoco hay entrada "
                    f"manual en resultados_diagrot para la región {REGION}.\n"
                    "Solución: añade manualmente T_ex, N_col, Delta_Tex y Delta_Ncol "
                    "en tables/diagrot/<REGION>/diagrot_resultados.ecsv."
                )

            print(
                f"[diagrot] Uso valores manuales/guardados de resultados_diagrot "
                f"para {MOLECULA}"
            )

            resultado_diagrot = resultados_diagrot[MOLECULA]


    elif modo_diagrot == "pixeles":

        print("[diagrot] Modo píxel a píxel")

        # ============================================================
        # TABLAS FILTRADAS POR PÍXEL
        # ============================================================
        # En modo píxeles sí necesitamos tabla de líneas base.
        # Si la tabla filtrada media está vacía, no se pueden medir mapas
        # de diagrama rotacional píxel a píxel.

        tab_pixeles = cargar_o_calcular_tablas_filtradas_pixeles(
            molecula=MOLECULA,
            region_name=REGION,
            tab_mol_config=tab_mol_config,
            dict_cubos_med=dict_cubos_med,
            dict_T_cont_med=dict_T_contv2,
            dict_cubos_comp=dict_cubos_comp,
            dict_T_cont_pix=dict_T_cont_pix,
            fwhm_dict=fwhm_dict,
            v_pik_map=v_pik_map,
            fwhm_map=fwhm_map,
            recalcular=RECALCULAR_FILTRADO_PIXELES,
            base_dir=cfg.rutatablas,
            plots=MOSTRAR_PLOTS,
            v_filt_override=VF_SYS,
            rutacarp_region=RUTACARP_REGION,
            rutaregion_region=RUTAREGION_REGION,
            intervalos_mol_region=INTERVALOS_MOL_REGION,
            tab_catalogo=tab_catalogo_mol,
        )

        mapas_diagrot = cargar_o_calcular_diagrot_pixeles(
            molecula=MOLECULA,
            region_name=REGION,
            tab_mol_config=tab_mol_config,
            tab_pixeles=tab_pixeles,
            dict_cubos_comp=dict_cubos_comp,
            recalcular=RECALCULAR_DIAGROT_PIXELES,
            guardar_fits=True,
            guardar_plots=True,
            plots=MOSTRAR_PLOTS,
        )
        # ============================================================
        # CORRECCIÓN DE MAPAS DEL DIAGRAMA ROTACIONAL
        # ============================================================

        CORREGIR_MAPAS_DIAGROT = True
    
        if CORREGIR_MAPAS_DIAGROT:

            print(
                "[diagrot_pix] Corrigiendo píxeles anómalos "
                "del diagrama rotacional"
            )

            (
                T_ex_corregido,
                N_col_corregido,
                Delta_Tex_corregido,
                Delta_Ncol_corregido,
                mask_corregidos_diagrot,
            ) = corregir_mapas_T_N(
                T_ex_map=mapas_diagrot["T_ex_map"],
                N_col_map=mapas_diagrot["N_col_map"],
                Delta_Tex_map=mapas_diagrot["Delta_Tex_map"],
                Delta_Ncol_map=mapas_diagrot["Delta_Ncol_map"],
                radio_deteccion=1,
                radio_busqueda=3,
                n_sigma=5.0,
                umbral_relativo=1.0,
                min_vecinos=3,
                limitar_incertidumbres=True,
            )

            mapas_diagrot["T_ex_map"] = (
                T_ex_corregido
            )

            mapas_diagrot["N_col_map"] = (
                N_col_corregido
            )

            mapas_diagrot["Delta_Tex_map"] = (
                Delta_Tex_corregido
            )

            mapas_diagrot["Delta_Ncol_map"] = (
                Delta_Ncol_corregido
            )

            mapas_diagrot["mask_corregidos"] = (
                mask_corregidos_diagrot
            )

            # Sobrescribir los FITS con los mapas corregidos.
            save_diagrot_maps_fits(
                region_name=REGION,
                molecula=MOLECULA,
                T_ex_map=mapas_diagrot["T_ex_map"],
                N_col_map=mapas_diagrot["N_col_map"],
                Delta_Tex_map=mapas_diagrot["Delta_Tex_map"],
                Delta_Ncol_map=mapas_diagrot["Delta_Ncol_map"],
                header_2d=mapas_diagrot["header"],
            )

            print(
                "[diagrot_pix] Número de píxeles corregidos: "
                f"{np.count_nonzero(mask_corregidos_diagrot)}"
            )
        
        # ------------------------------------------------------------
        # Para que el modelo sintético/chi2 sigan funcionando después,
        # cargamos también el resultado medio si existe.
        # En modo píxeles, mapas_diagrot no tiene una única T_ex/N_col media.
        # ------------------------------------------------------------

        resultados_diagrot = cargar_resultados_diagrot_region(REGION)

        if MOLECULA in resultados_diagrot:
            resultado_diagrot = resultados_diagrot[MOLECULA]
            print(
                f"[diagrot] Además existe resultado medio/manual para {MOLECULA}; "
                "se usará para modelo sintético y chi2."
            )
        else:
            print(
                f"[diagrot] Aviso: se han calculado mapas para {MOLECULA}, "
                "pero no hay entrada media/manual en resultados_diagrot."
            )
            print(
                "[diagrot] Si quieres hacer modelo sintético o chi2 medio, "
                "añade T_ex/N_col manuales o calcula un diagrot medio."
            )


    else:
        raise ValueError(
            f"modo_diagrot='{modo_diagrot}' no reconocido. "
            "Usa 'medio' o 'pixeles'."
        )

    if modo_diagrot == "pixeles":

        continuar = input(
            "¿Quieres continuar? Siguiente paso, ajuste chi2 por píxel (S/N)"
        ).strip().upper()
    
        if continuar == "N":
            return
        
        RECALCULAR_CHI2_PIX = False

        fila_mol = tab_mol_config[tab_mol_config["nombre"] == MOLECULA]

        if len(fila_mol) == 0:
            raise KeyError(f"{MOLECULA} no está en moleculas_config.ecsv.")

        intervalo_mol = str(fila_mol["intervalo"][0]).strip()

        dict_lin_noconsid_chi2 = obtener_dict_noconsid_chi2(
            region_name=REGION,
        )

        resultado_chi2_pix = cargar_o_calcular_chi2_pixeles(
            molecula=MOLECULA,
            region_name=REGION,
            tab_mol_config=tab_mol_config,
            mapas_diagrot=mapas_diagrot,
            tab_pixeles=tab_pixeles,
            dict_cubos_comp=dict_cubos_comp,
            dict_T_cont_pix=dict_T_cont_pix,
            dict_sigma_pix=dict_sigma_pix,
            fwhm_dict=fwhm_dict,
            v_pik_dict=v_pik_dict,
            fwhm_map=fwhm_map,
            v_pik_map=v_pik_map,
            intervalos_mol_region=INTERVALOS_MOL_REGION,
            header_2d=mapas_diagrot["header"],
            recalcular=RECALCULAR_CHI2_PIX,
            n_grid=10,
            max_iter=2,
            tol=1,
            dict_lin_noconsid=dict_lin_noconsid_chi2,
            debug=False,
            show_modelo= False
        )

        # ============================================================
        # CORRECCIÓN DE MAPAS CHI2
        # ============================================================

        CORREGIR_MAPAS_CHI2 = True

        if CORREGIR_MAPAS_CHI2:

            claves_necesarias = {
                "T_fit_map",
                "N_fit_map",
                "deltaT_map",
                "deltaN_map",
            }

            claves_faltantes = (
                claves_necesarias
                - set(resultado_chi2_pix.keys())
            )

            if claves_faltantes:
                raise KeyError(
                    "No se pueden corregir los mapas chi2. "
                    f"Faltan las claves: {sorted(claves_faltantes)}"
                )

            print(
                "[chi2_pix] Corrigiendo píxeles anómalos "
                "de los mapas chi2"
            )

            (
                T_fit_corregido,
                N_fit_corregido,
                deltaT_corregido,
                deltaN_corregido,
                mask_corregidos_chi2,
            ) = corregir_mapas_T_N(
                T_ex_map=resultado_chi2_pix["T_fit_map"],
                N_col_map=resultado_chi2_pix["N_fit_map"],
                Delta_Tex_map=resultado_chi2_pix["deltaT_map"],
                Delta_Ncol_map=resultado_chi2_pix["deltaN_map"],
                radio_deteccion=1,
                radio_busqueda=3,
                n_sigma=5.0,
                umbral_relativo=1.0,
                    min_vecinos=3,
                limitar_incertidumbres=False,
            )

            resultado_chi2_pix["T_fit_map"] = (
                T_fit_corregido
            )

            resultado_chi2_pix["N_fit_map"] = (
                N_fit_corregido
            )

            resultado_chi2_pix["deltaT_map"] = (
                deltaT_corregido
            )

            resultado_chi2_pix["deltaN_map"] = (
                deltaN_corregido
            )

            resultado_chi2_pix["mask_corregidos"] = (
                mask_corregidos_chi2
            )

            # Sobrescribir los FITS de χ² con los mapas corregidos.
            save_chi2_maps_fits(
                region_name=REGION,
                molecula=MOLECULA,
                    mapas=resultado_chi2_pix,
                header_2d=resultado_chi2_pix["header"],
            )

            print(
                "[chi2_pix] Número de píxeles corregidos: "
                f"{np.count_nonzero(mask_corregidos_chi2)}"
            )

        print("[chi2_pix] Mapas chi2 terminados.")
        return

    continuar = input(
        '¿Quieres continuar? Siguiente paso, modelo sintético (diag_rot) (S/N)')
    
    if continuar == 'N':
        return tab_filtrada
    
    # ============================================================
    # MODELO SINTÉTICO DESDE DIAGRAMA ROTACIONAL
    # ============================================================
    # UNA ÚNICA MOLÉCULA
    # ============================================================

    PATH_DIAGROT_RESULTADOS = (
        cfg.rutatablas / "diagrot" / REGION / "diagrot_resultados.ecsv"
    )

    resultados_diagrot = load_diagrot_results(PATH_DIAGROT_RESULTADOS)

    # modelo_sintetico_diagrot_delgado = calcular_modelo_sintetico(
    #     moleculas=MOLECULA,
    #     region_name=REGION,
    #     tab_mol_config=tab_mol_config,
    #     resultados_parametros=resultados_diagrot,
    #     fwhm_dict=fwhm_dict,
    #     dict_T_cont=dict_T_contv2,
    #     dict_cubos_med=dict_cubos_med,
    #     fuente_parametros="diagrot",
    #     modelo_radiativo="delgado",
    #     plot_lineas=True,
    #     show_plots=MOSTRAR_PLOTS,
    #     save_plots=True,
    #     v_pik_dict=v_pik_dict,
    # )

    modelo_sintetico_diagrot_opacidad = calcular_modelo_sintetico(
        moleculas=MOLECULA,
        region_name=REGION,
        tab_mol_config=tab_mol_config,
        resultados_parametros=resultados_diagrot,
        fwhm_dict=fwhm_dict,
        dict_T_cont=dict_T_contv2,
        dict_cubos_med=dict_cubos_med,
        fuente_parametros="diagrot",
        modelo_radiativo="opacidad",
        plot_lineas=True,
        show_plots=MOSTRAR_PLOTS,
        save_plots=True,
        v_pik_dict=v_pik_dict,
        dict_sigma=dict_sigma_vent,
        nsigma_lineas=3.0,
        intervalos_mol_region=INTERVALOS_MOL_REGION,
    )

    # ============================================================
    # MODELO COMPLETO
    # ============================================================

    PATH_DIAGROT_RESULTADOS = (
        cfg.rutatablas / "diagrot" / REGION / "diagrot_resultados.ecsv"
    )

    resultados_diagrot = load_diagrot_results(PATH_DIAGROT_RESULTADOS)

    MOLECULAS_MODELO = list(resultados_diagrot.keys())

    print("[spec_sint] Moléculas disponibles en diagrot:")
    print(MOLECULAS_MODELO)

    fwhm_dict, v_pik_dict = dicts_calibracion_region(REGION)

    faltan_vpik = [m for m in MOLECULAS_MODELO if m not in v_pik_dict]
    faltan_fwhm = [m for m in MOLECULAS_MODELO if m not in fwhm_dict]

    if len(faltan_vpik) > 0 or len(faltan_fwhm) > 0:
        raise KeyError(
            "Faltan moléculas calibradas antes de construir el modelo completo.\n"
            f"Faltan v_pik: {faltan_vpik}\n"
            f"Faltan FWHM: {faltan_fwhm}\n"
            "Ejecuta primero la calibración de esas moléculas en esta región."
        )

    print("[spec_sint] v_pik_dict usado:")
    print(v_pik_dict)

    # modelo_completo_diagrot_delgado = calcular_modelo_sintetico(
    #     moleculas=MOLECULAS_MODELO,
    #     region_name=REGION,
    #     tab_mol_config=tab_mol_config,
    #     resultados_parametros=resultados_diagrot,
    #     fwhm_dict=fwhm_dict,
    #     dict_T_cont=dict_T_contv2,
    #     dict_cubos_med=dict_cubos_med,
    #     fuente_parametros="diagrot",
    #     modelo_radiativo="delgado",
    #     plot_lineas=False,
    #     show_plots=MOSTRAR_PLOTS,
    #     save_plots=True,
    #     guardar_solo_final=True,
    #     v_pik_dict=v_pik_dict,
    # )

    # modelo_completo_diagrot_opacidad = calcular_modelo_sintetico(
    #     moleculas=MOLECULAS_MODELO,
    #     region_name=REGION,
    #     tab_mol_config=tab_mol_config,
    #     resultados_parametros=resultados_diagrot,
    #     fwhm_dict=fwhm_dict,
    #     dict_T_cont=dict_T_contv2,
    #     dict_cubos_med=dict_cubos_med,
    #     fuente_parametros="diagrot",
    #     modelo_radiativo="opacidad",
    #     plot_lineas=False,
    #     show_plots=MOSTRAR_PLOTS,
    #     save_plots=True,
    #     guardar_solo_final=True,
    #     v_pik_dict=v_pik_dict,
    #     dict_sigma=dict_sigma_vent,
    #     nsigma_lineas=3.0,
    # )

    continuar = input('¿Quieres continuar? Siguiente paso, ajuste de chi2 (S/N)')
    
    if continuar == 'N':
        return tab_filtrada
    
    # ============================================================
    # AJUSTE CHI2 - UNA MOLÉCULA
    # ============================================================

    RECALCULAR_CHI2 = False

    print("\n" + "=" * 70)
    print(f"[chi2] Ajuste de molécula: {MOLECULA}")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Detectar intervalo/banda
    # ------------------------------------------------------------

    fila_mol = tab_mol_config[tab_mol_config["nombre"] == MOLECULA]

    if len(fila_mol) == 0:
        raise KeyError(
            f"{MOLECULA} no está en moleculas_config.ecsv. "
            f"Moléculas disponibles: {list(tab_mol_config['nombre'])}"
        )

    intervalo_mol = str(fila_mol["intervalo"][0]).strip()

    print(f"[chi2] Intervalo/banda: {intervalo_mol}")

    orden_chi2_mol = obtener_orden_ajuste_chi2(
        tab_mol_config=tab_mol_config,
        molecula=MOLECULA,
    )

    print(
        f"[chi2] Orden de ajuste para {intervalo_mol}: "
        f"{orden_chi2_mol}"
    )

    # ------------------------------------------------------------
    # 2. Cargar resultados chi2 ya existentes
    # ------------------------------------------------------------

    tab_chi2_actual = load_chi2_results(REGION)


    # ------------------------------------------------------------
    # 3. Reconstruir residuo base correcto para esta molécula
    # ------------------------------------------------------------
    # Esto resta SOLO las moléculas anteriores ya ajustadas.
    # Si MOLECULA es la primera del orden, devuelve dict_cubos_med.

    base_chi2 = construir_residuo_base_chi2(
        molecula=MOLECULA,
        region_name=REGION,
        tab_mol_config=tab_mol_config,
        resultados_chi2=tab_chi2_actual,
        fwhm_dict=fwhm_dict,
        dict_T_cont=dict_T_contv2,
        dict_cubos_med=dict_cubos_med,
        show_plots=True,
        save_plots=False,
        v_pik_dict=v_pik_dict,
        intervalos_mol_region=INTERVALOS_MOL_REGION,
        rutacarp_region=RUTACARP_REGION,
        rutaregion_region=RUTAREGION_REGION,
    )

    dict_espec_base_chi2 = base_chi2["dict_espec_base"]
    
    print("[chi2] Moléculas restadas para construir el residuo base:")
    print(f"       {base_chi2['moleculas_restadas']}")
    
    
    # ------------------------------------------------------------
    # 4. Generar modelo base de la molécula actual
    # ------------------------------------------------------------
    # Genero un modelo inicial de la molécula actual solo para obtener
    # las líneas que se usarán en el cálculo de chi2.
    # IMPORTANTE:
    # Los residuos de este modelo NO se usan como espectro de entrada.
    # El ajuste se realiza sobre dict_espec_base_chi2, que contiene:
    # espectro original - moléculas anteriores ajustadas.

    modelo_base_mol = calcular_modelo_sintetico(
        moleculas=MOLECULA,
        region_name=REGION,
        tab_mol_config=tab_mol_config,
        resultados_parametros=resultados_diagrot,
        fwhm_dict=fwhm_dict,
        dict_T_cont=dict_T_contv2,
        dict_cubos_med=dict_espec_base_chi2,
        fuente_parametros="diagrot",
        modelo_radiativo="opacidad",
        plot_lineas=False,
        show_plots=True,
        save_plots=False,
        v_pik_dict=v_pik_dict,
        dict_sigma=dict_sigma_vent,
        nsigma_lineas=1.0,
        intervalos_mol_region=INTERVALOS_MOL_REGION,
    )

    tab_lineas_chi2 = modelo_base_mol["tab_lineas_plot"]

    print("[chi2] Líneas usadas en el ajuste:")

    for mol in tab_lineas_chi2:
        print(f"       {mol}: {len(tab_lineas_chi2[mol]['freq'])} líneas")
    

    # ------------------------------------------------------------
    # 5. Líneas a no considerar en chi2
    # ------------------------------------------------------------

    dict_lin_noconsid_chi2 = obtener_dict_noconsid_chi2(
            region_name=REGION,
        )
    
    print(f'[chi2] Las frecuencias excluidas para {MOLECULA} son: {dict_lin_noconsid_chi2[MOLECULA]}')
    # ------------------------------------------------------------
    # 6. Ejecutar ajuste chi2
    # ------------------------------------------------------------

    resultado_chi2 = cargar_o_calcular_chi2_molecula(
        molecula=MOLECULA,
        region_name=REGION,
        tab_mol_config=tab_mol_config,
        resultados_diagrot=resultados_diagrot,
        fwhm_dict=fwhm_dict,
        dict_T_cont=dict_T_contv2,
        dict_sigma_vent=dict_sigma_vent,
        dict_cubos_med=dict_espec_base_chi2,
        dict_resol_espec=dict_resol_esp,
        tab_lineas=tab_lineas_chi2,
        recalcular=RECALCULAR_CHI2,
        n_grid=10,
        tol=0.5,
        max_iter=20,
        dict_lin_noconsid=dict_lin_noconsid_chi2,
        model_sint=None,
        residuos=False,
        show_model=MOSTRAR_PLOTS,
        save_model=True, 
        debug=False,
        v_pik_dict=v_pik_dict,
        dict_cubos_plot=dict_espec_base_chi2,
        intervalos_mol_region=INTERVALOS_MOL_REGION,
        rutacarp_region=RUTACARP_REGION,
        rutaregion_region=RUTAREGION_REGION,
    )

    print("[chi2] Ajuste terminado.")
    print(f"[chi2] T_fit = {resultado_chi2.get('T_fit')}")
    print(f"[chi2] N_fit = {resultado_chi2.get('N_fit')}")
    print(f"[chi2] converged = {resultado_chi2.get('converged')}")
    tau_linea_menor_Eu = resultado_chi2.get(
        "tau_linea_menor_Eu"
        )

    linea_menor_Eu = resultado_chi2.get(
        "linea_menor_Eu"
        )

    path_tabla_opacidad = resultado_chi2.get(
        "path_tabla_opacidad"
        )

    if tau_linea_menor_Eu is not None:
        
        print("[chi2] Opacidad de la transición seleccionada:")
        print(
            f"       Eu = "
            f"{linea_menor_Eu['upper_state_energy_K']:.2f} K"
            )
        print(
            f"       tau = {tau_linea_menor_Eu:.4f}"
            )

    # ============================================================
    # MODELO SINTÉTICO COMPLETO CON RESULTADOS CHI2
    # ============================================================

    continuar = input(
        "¿Quieres calcular el modelo completo usando los resultados chi2? (S/N): "
    ).strip().upper()

    if continuar == "S":

        print("\n[spec_sint_chi2] Cargando resultados chi2 de la región...")
    
        resultados_chi2 = load_chi2_results(REGION)
    
        # ------------------------------------------------------------
        # CASO 1: resultados_chi2 es una tabla
        # ------------------------------------------------------------
        if hasattr(resultados_chi2, "colnames"):

            tab_chi2_region = resultados_chi2

            if "region" in tab_chi2_region.colnames:
                tab_chi2_region = tab_chi2_region[
                    tab_chi2_region["region"] == REGION
                ]

            if len(tab_chi2_region) == 0:
                raise ValueError(
                    f"No hay resultados chi2 guardados para la región {REGION}. "
                    "Calcula primero al menos un ajuste chi2."
                )

            if "molecula" not in tab_chi2_region.colnames:
                raise KeyError(
                    "La tabla de chi2 no tiene columna 'molecula'. "
                    f"Columnas disponibles: {tab_chi2_region.colnames}"
                )

            moleculas_disponibles = [
                str(mol).strip()
                for mol in tab_chi2_region["molecula"]
            ]

            MOLECULAS_MODELO_CHI2 = [
                mol for mol in orden_chi2_mol
                if mol in moleculas_disponibles
            ]

            resultados_chi2_modelo = tab_chi2_region

        # ------------------------------------------------------------
        # CASO 2: resultados_chi2 es un diccionario
        # ------------------------------------------------------------
        elif isinstance(resultados_chi2, dict):
    
            if len(resultados_chi2) == 0:
                raise ValueError(
                    f"No hay resultados chi2 guardados para la región {REGION}. "
                    "Calcula primero al menos un ajuste chi2."
                )
    
            MOLECULAS_MODELO_CHI2 = [
                mol for mol in orden_chi2_mol
                if mol in resultados_chi2
            ]
            resultados_chi2_modelo = resultados_chi2
    
        else:
            raise TypeError(
                "No reconozco el formato de resultados_chi2. "
                f"Tipo recibido: {type(resultados_chi2)}"
            )

        print("[spec_sint_chi2] Moléculas disponibles con chi2:")
        print(MOLECULAS_MODELO_CHI2)
    
        # ------------------------------------------------------------
        # Recargar calibraciones de la región
        # ------------------------------------------------------------
        fwhm_dict, v_pik_dict = dicts_calibracion_region(REGION)

        faltan_fwhm = [
            m for m in MOLECULAS_MODELO_CHI2
            if m not in fwhm_dict
        ]

        faltan_vpik = [
            m for m in MOLECULAS_MODELO_CHI2
            if m not in v_pik_dict
        ]

        if len(faltan_fwhm) > 0 or len(faltan_vpik) > 0:
            raise KeyError(
                "Faltan moléculas calibradas antes de construir el modelo completo chi2.\n"
                f"Faltan FWHM: {faltan_fwhm}\n"
                f"Faltan v_pik: {faltan_vpik}\n"
                "Ejecuta primero la calibración de esas moléculas en esta región."
            )

        modelo_completo_chi2_opacidad = calcular_modelo_sintetico(
            moleculas=MOLECULAS_MODELO_CHI2,
            region_name=REGION,
            tab_mol_config=tab_mol_config,
            resultados_parametros=resultados_chi2_modelo,
            fwhm_dict=fwhm_dict,
            dict_T_cont=dict_T_contv2,
            dict_cubos_med=dict_cubos_med,
            fuente_parametros="chi2",
            modelo_radiativo="opacidad",
            plot_lineas=True,
            show_plots=MOSTRAR_PLOTS,
            save_plots=True,
            guardar_solo_final=True,
            v_pik_dict=v_pik_dict,
            dict_sigma=dict_sigma_vent,
            nsigma_lineas=3.0,
            intervalos_mol_region=INTERVALOS_MOL_REGION,
        )

        print(
            "[spec_sint_chi2] Modelo completo chi2 con opacidad "
            "calculado correctamente."
        )
        
        return modelo_completo_chi2_opacidad, tab_filtrada 

# ============================================================
# BUCLE PRINCIPAL DE MOLÉCULAS
# ============================================================

while True:

    MOLECULA = seleccionar_molecula()

    if MOLECULA is None:
        print("[main] Programa terminado.")
        break

    print(
        "\n" + "=" * 70
    )
    print(
        f"[main] Molécula seleccionada: {MOLECULA}"
    )
    print(
        "=" * 70
    )

    resultados = procesar_molecula(MOLECULA)

    respuesta = input(
        "\n¿Quieres procesar otra molécula "
        "en esta misma región? [S/n]: "
    ).strip().lower()

    if respuesta in {"n", "no"}:
        print("[main] Programa terminado.")
        break

# ARREGLAR LO DE CHI² (INCLUIR MAPAS) 

# Hacer que puedas saltar de secciones sin tener que hacer todo el proceso, y 
# y si da error a elegir alguna fuente o molécula te pregunte de nuevo, no que 
# falle

# mover lo de crear la tabla de config mol a config
# Incluir que puedas editar lo de recalcular las cosas desde la terminal

# Crear un readme (ir haciendolo poco a poco)

# Hacer lo de que aparezca una barra de progreso

# Arreglar lo de los plots!!!
#Añadir modo grifo que sea que no muestra ninguna gráfica pero guarda las cosas

#2 Añadir HC(O)NH2 v12=0 y v12=1 C-13-S ISOTOPOLOGOS CH3C-13-N CH3CN-15
#H2CS C2H3CN
#OC-13-S C-13-H3OH E ISOTOPOLOGOS CH3CCH JUNTO LA ACETONA ethylene glycol


#OCS(Puede que solo tenga una línea) HCCCN (Lo mismo que OCS) CO18 t-HCOOH H2CO

#3
#Intentar añadir CH3CCH v=1 Banda 3
#C2H3CN v = 1 en la banda 3
#CH3C-13-N Banda 3
#4
#Intentar hacer los esquemas de energía para el CH3CN o el CH3OH 
#(codigo de miriam)
