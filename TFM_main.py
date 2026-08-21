#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:42:19 2026

@author: jorge
"""

from astropy import units as u

import TFM_config as cfg

from TFM_load_region import seleccionar_region, cargar_datos_region

from TFM_calibration import cargar_o_calcular_calibracion_molecula

from TFM_continuum import (
    cargar_o_calcular_continuo_medio,
    cargar_o_calcular_continuo_pixeles,
)

from TFM_filtered_lines import (
    cargar_o_calcular_tabla_filtrada_molecula,
    cargar_o_calcular_tablas_filtradas_pixeles,
)

from TFM_diagrot_pipeline import (
    cargar_o_calcular_diagrot_molecula_medio,
    cargar_o_calcular_diagrot_pixeles,
)

from TFM_synthetic_pipeline import calcular_modelo_sintetico
from TFM_chi2_pipeline import (
    cargar_o_calcular_chi2_molecula,
    construir_residuo_base_chi2,
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

# %%
# ============================================================
# CREAR TABLA DE CONFIGURACIÓN DE MOLÉCULAS
# ============================================================

# import shutil
# from astropy import units as u
# from astropy.table import QTable

# import TFM_config as cfg


# PATH_CONFIG_MOLECULAS = cfg.rutatablas / "config" / "moleculas_config.ecsv"


# def validar_nombre_cfg(nombre):
#     """
#     Comprueba que un nombre simbólico existe en TFM_config.py.
#     Permite None.
#     """
#     nombre = str(nombre).strip()

#     if nombre in ("None", "", "nan"):
#         return

#     if not hasattr(cfg, nombre):
#         raise AttributeError(
#             f"TFM_config.py no tiene definido el nombre simbólico: {nombre}"
#         )


# rows = [
#     # nombre, mol, intervalo, catalogo,
#     # id_splat, id_cat, B0,
#     # filtro_estructuras,
#     # list_frec_noconsid, frec_nofiltrar,
#     # f0, v_filt, v_pik,
#     # E_max, aij_min, sijmu2_min, filt_inter

#     (
#         "C2H5OH_g",
#         "C2H5OH",
#         "Banda6",
#         "CDMS",
#         "id_splatC2H5OH",
#         "id_cdmsC2H5OH",
#         "B0C2H5OH",
#         "filtro_C2H5OH_quitar_anti",
#         "list_noconsidC2H5OH",
#         "None",
#         232491 * u.MHz,
#         63 * u.km / u.s,
#         -61 * u.km / u.s,
#         600 * u.K,
#         1e-6 / u.s,
#         10 * u.D**2,
#         0.5,
#     ),

#     (
#         "C2H5OH_anti",
#         "C2H5OH",
#         "Banda6",
#         "CDMS",
#         "id_splatC2H5OH",
#         "id_cdmsC2H5OH",
#         "B0C2H5OH",
#         "filtro_C2H5OH_quitar_g",
#         "list_noconsidC2H5OH",
#         "None",
#         232491 * u.MHz,
#         63 * u.km / u.s,
#         -61 * u.km / u.s,
#         300 * u.K,
#         1e-6 / u.s,
#         10 * u.D**2,
#         0.5,
#     ),

#     (
#         "C2H5CN",
#         "CH3CH2CN",
#         "Banda6",
#         "CDMS",
#         "id_splatC2H5CN",
#         "id_cdmsC2H5CN",
#         "B0C2H5CN",
#         "None",
#         "list_noconsidC2H5CN",
#         "None",
#         233069.375 * u.MHz,
#         63 * u.km / u.s,
#         -61 * u.km / u.s,
#         400 * u.K,
#         1e-6 / u.s,
#         200 * u.D**2,
#         0.5,
#     ),

#     (
#         "CH3OH_v0",
#         "CH3OH",
#         "Banda6",
#         "CDMS",
#         "id_splatCH3OH",
#         "id_cdmsCH3OH",
#         "B0CH3OH",
#         "filtro_CH3OH_quitar_v1",
#         "None",
#         "None",
#         231281 * u.MHz,
#         63 * u.km / u.s,
#         -61 * u.km / u.s,
#         820 * u.K,
#         1e-6 / u.s,
#         10 * u.D**2,
#         0.5,
#     ),

#     (
#         "CH3OH_v1",
#         "CH3OH",
#         "Banda6",
#         "CDMS",
#         "id_splatCH3OH",
#         "id_cdmsCH3OH",
#         "B0CH3OH",
#         "filtro_CH3OH_quitar_v0",
#         "list_noconsidCH3OHv1",
#         "None",
#         231281 * u.MHz,
#         63 * u.km / u.s,
#         -61 * u.km / u.s,
#         1000 * u.K,
#         1e-6 / u.s,
#         0 * u.D**2,
#         0.5,
#     ),

#     (
#         "CH3OCHO_v0",
#         "CH3OCHO",
#         "Banda6",
#         "CDMS,JPL",
#         "id_splatCH3OCHO_v0",
#         "id_JPLCH3OCHO",
#         "B0CH3OCHO",
#         "None",
#         "list_noconsidCH3OCHO_v0",
#         "None",
#         218297.89 * u.MHz,
#         63 * u.km / u.s,
#         -61 * u.km / u.s,
#         550 * u.K,
#         1e-6 / u.s,
#         20 * u.D**2,
#         0.5,
#     ),

#     (
#         "CH3OCHO_v1",
#         "CH3OCHO",
#         "Banda6",
#         "CDMS,JPL",
#         "id_splatCH3OCHO_v1",
#         "id_JPLCH3OCHO",
#         "B0CH3OCHO",
#         "None",
#         "list_noconsidCH3OCHO_v1",
#         "None",
#         231724 * u.MHz,
#         63 * u.km / u.s,
#         -61 * u.km / u.s,
#         450 * u.K,
#         1e-6 / u.s,
#         30 * u.D**2,
#         0.5,
#     ),

#     (
#         "CH3CHO_v0",
#         "CH3CHO",
#         "Banda6",
#         "CDMS,JPL",
#         "id_splatCH3CHO",
#         "id_JPLCH3CHO",
#         "B0CH3CHO",
#         "filtro_CH3CHO_quedarse_v0",
#         "list_noconsidCH3CHO_v0",
#         "None",
#         230315.7923 * u.MHz,
#         63 * u.km / u.s,
#         -60.5 * u.km / u.s,
#         200 * u.K,
#         1e-6 / u.s,
#         100 * u.D**2,
#         0.5,
#     ),

#     (
#         "CH3OCH3",
#         "CH3OCH3",
#         "Banda6",
#         "CDMS",
#         "id_splatCH3OCH3",
#         "id_cdmsCH3OCH3",
#         "B0CH3OCH3",
#         "filtro_estructuras_acetona_conEE",
#         "list_noconsidCH3OCH3",
#         "None",
#         231669.354 * u.MHz,
#         63 * u.km / u.s,
#         -61 * u.km / u.s,
#         500 * u.K,
#         1e-6 / u.s,
#         90 * u.D**2,
#         0.5,
#     ),

#     (
#         "Acetona",
#         "CH3",
#         "Banda6",
#         "CDMS,JPL",
#         "id_splatAcetona",
#         "id_JPLacetone",
#         "B0acetone",
#         "None",
#         "list_noconsidAcetona",
#         "freq_nofiltrarAcetona",
#         218127.2074 * u.MHz,
#         63 * u.km / u.s,
#         -61 * u.km / u.s,
#         116 * u.K,
#         1e-6 / u.s,
#         1000 * u.D**2,
#         0.5,
#     ),

#     (
#         "CH3CN",
#         "CH3CN",
#         "Banda3",
#         "CDMS,JPL",
#         "id_splatCH3CN",
#         "id_JPLCH3CN",
#         "B0CH3CN",
#         "filtro_quitar_F",
#         "list_noconsidCH3CN",
#         "None",
#         91958.726 * u.MHz,
#         63 * u.km / u.s,
#         -61 * u.km / u.s,
#         1000 * u.K,
#         1e-6 / u.s,
#         0 * u.D**2,
#         0.5,
#     ),
# ]


# tab_nueva = QTable(
#     names=[
#         "nombre",
#         "mol",
#         "intervalo",
#         "catalogo",
#         "id_splat",
#         "id_cat",
#         "B0",
#         "filtro_estructuras",
#         "list_frec_noconsid",
#         "frec_nofiltrar",
#         "f0",
#         "v_filt",
#         "v_pik",
#         "E_max",
#         "aij_min",
#         "sijmu2_min",
#         "filt_inter",
#     ],
#     dtype=[
#         "U100",
#         "U100",
#         "U100",
#         "U100",
#         "U100",
#         "U100",
#         "U100",
#         "U100",
#         "U100",
#         "U100",
#         "f8",
#         "f8",
#         "f8",
#         "f8",
#         "f8",
#         "f8",
#         "f8",
#     ],
#     units=[
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#         None,
#         u.MHz,
#         u.km / u.s,
#         u.km / u.s,
#         u.K,
#         1 / u.s,
#         u.D**2,
#         None,
#     ],
# )

# for row in rows:
#     tab_nueva.add_row(row)


# # ============================================================
# # VALIDACIÓN DE NOMBRES SIMBÓLICOS
# # ============================================================

# for row in tab_nueva:
#     validar_nombre_cfg(row["intervalo"])
#     validar_nombre_cfg(row["id_splat"])
#     validar_nombre_cfg(row["id_cat"])
#     validar_nombre_cfg(row["B0"])
#     validar_nombre_cfg(row["filtro_estructuras"])
#     validar_nombre_cfg(row["list_frec_noconsid"])
#     validar_nombre_cfg(row["frec_nofiltrar"])

# print("[config] Validación correcta: todos los nombres simbólicos existen.")


# # ============================================================
# # BACKUP Y GUARDADO
# # ============================================================

# PATH_CONFIG_MOLECULAS.parent.mkdir(parents=True, exist_ok=True)

# if PATH_CONFIG_MOLECULAS.exists():
#     backup_path = PATH_CONFIG_MOLECULAS.with_suffix(".backup.ecsv")
#     shutil.copy(PATH_CONFIG_MOLECULAS, backup_path)
#     print(f"[config] Backup guardado en: {backup_path}")

# tab_nueva.write(
#     PATH_CONFIG_MOLECULAS,
#     format="ascii.ecsv",
#     overwrite=True,
# )

# print(f"[config] Nueva tabla guardada en: {PATH_CONFIG_MOLECULAS}")

# print(
#     tab_nueva[
#         "nombre",
#         "mol",
#         "intervalo",
#         "catalogo",
#         "list_frec_noconsid",
#         "frec_nofiltrar",
#         "f0",
#         "v_filt",
#         "v_pik",
#     ]
# )

# %%

# ============================================================
# SELECCIÓN DE REGIÓN
# ============================================================

REGION = "MF2"
# REGION = "mm31_d2"

config_region = seleccionar_region(REGION)

(dict_cubos_med,
 dict_resol_esp, dict_cubos_comp) = cargar_datos_region(config_region)

# %%

# ============================================================
# CALIBRACIÓN DEL CONTINUO
# ============================================================

RECALCULAR_CONTINUO = False
VF_SYS = config_region.get("v_sys", 63 * u.km / u.s)
print(f"[region] Velocidad sistémica usada: {VF_SYS}")

dict_T_contv2 = None
dict_sigma_vent = None
dict_T_cont_pix = None
dict_sigma_pix = None

if config_region.get("calibrar_continuo_medio", False):

    dict_T_contv2, dict_sigma_vent = cargar_o_calcular_continuo_medio(
        region_name=REGION,
        intervalos_Tcont=cfg.intervalos_Tcont,
        ventanas_obs=cfg.ventanas_obs,
        vfuent=VF_SYS,
        dict_cubos_med=dict_cubos_med,
        base_dir=cfg.rutatablas,
        recalcular=RECALCULAR_CONTINUO,
    )

if config_region.get("calibrar_continuo_pixeles", False):

    dict_T_cont_pix, dict_sigma_pix = cargar_o_calcular_continuo_pixeles(
        region_name=REGION,
        intervalos_Tcont=cfg.intervalos_Tcont,
        ventanas_obs=cfg.ventanas_obs,
        vfuent=VF_SYS,
        dict_cubos_comp=dict_cubos_comp,
        base_dir=cfg.rutatablas,
        recalcular=RECALCULAR_CONTINUO,
    )

# %%
# ============================================================
# CARGAR CONFIGURACIÓN DE MOLÉCULAS
# ============================================================

PATH_CONFIG_MOLECULAS = cfg.rutatablas / "config" / "moleculas_config.ecsv"

tab_mol_config = load_molecule_config(PATH_CONFIG_MOLECULAS)

# %%

MOLECULA = "C2H5OH_anti"

# ============================================================
# CALIBRACIÓN DE FWHM DE LA MOLÉCULA
# ============================================================

tab_cali_mol, fwhm_mol, fwhm_dict = cargar_o_calcular_calibracion_molecula(
    MOLECULA,
    dict_cubos_med,
    recalcular=True, v_sys= VF_SYS
)

# %%
# ============================================================
# DIAGRAMA ROTACIONAL
# ============================================================

RECALCULAR_FILTRADO = True
RECALCULAR_FILTRADO_PIXELES = True
RECALCULAR_DIAGROT = True
RECALCULAR_DIAGROT_PIXELES = True
guardar_plots=True

modo_diagrot = config_region["modo_diagrot"]

if modo_diagrot == "medio":

    print("[diagrot] Modo espectro promedio")

    tab_filtrada = cargar_o_calcular_tabla_filtrada_molecula(
        molecula=MOLECULA,
        region_name=REGION,
        tab_mol_config=tab_mol_config,
        dict_cubos_med=dict_cubos_med,
        dict_T_cont=dict_T_contv2,
        fwhm_dict=fwhm_dict,
        recalcular=RECALCULAR_FILTRADO, v_filt_override= VF_SYS
    )

    resultado_diagrot = cargar_o_calcular_diagrot_molecula_medio(
        molecula=MOLECULA,
        region_name=REGION,
        tab_mol_config=tab_mol_config,
        tab_filtrada=tab_filtrada,
        recalcular=RECALCULAR_DIAGROT,
        guardar_pdf=True,
        plot_Q=False,
    )

elif modo_diagrot == "pixeles":

    print("[diagrot] Modo píxel a píxel")

    tab_pixeles = cargar_o_calcular_tablas_filtradas_pixeles(
        molecula=MOLECULA,
        region_name=REGION,
        tab_mol_config=tab_mol_config,
        dict_cubos_med=dict_cubos_med,
        dict_T_cont_med=dict_T_contv2,
        dict_cubos_comp=dict_cubos_comp,
        dict_T_cont_pix=dict_T_cont_pix,
        fwhm_dict=fwhm_dict,
        recalcular=RECALCULAR_FILTRADO_PIXELES, v_filt_override= VF_SYS
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
    )

else:
    raise ValueError(
        f"modo_diagrot='{modo_diagrot}' no reconocido. "
        "Usa 'medio' o 'pixeles'."
    )
    
# %%
# ============================================================
# MODELO SINTÉTICO DESDE DIAGRAMA ROTACIONAL
# ============================================================
# UNA ÚNICA MOLÉCULA
# ============================================================

PATH_DIAGROT_RESULTADOS = (
    cfg.rutatablas / "diagrot" / REGION / "diagrot_resultados.ecsv"
)

resultados_diagrot = load_diagrot_results(PATH_DIAGROT_RESULTADOS)

modelo_sintetico_diagrot = calcular_modelo_sintetico(
    moleculas=MOLECULA,
    region_name=REGION,
    tab_mol_config=tab_mol_config,
    resultados_parametros=resultados_diagrot,
    fwhm_dict=fwhm_dict,
    dict_T_cont=dict_T_contv2,
    dict_cubos_med=dict_cubos_med,
    fuente_parametros="diagrot",
    plot_lineas=True,
    show_plots=True,
    save_plots=True, 
)

# %%


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

modelo_completo_diagrot = calcular_modelo_sintetico(
    moleculas=MOLECULAS_MODELO,
    region_name=REGION,
    tab_mol_config=tab_mol_config,
    resultados_parametros=resultados_diagrot,
    fwhm_dict=fwhm_dict,
    dict_T_cont=dict_T_contv2,
    dict_cubos_med=dict_cubos_med,
    fuente_parametros="diagrot",
    plot_lineas=False,
    show_plots=True,
    save_plots=True,
    guardar_solo_final=True,
)



# %%
# ============================================================
# AJUSTE CHI2 - UNA MOLÉCULA
# ============================================================

RECALCULAR_CHI2 = True

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

intervalo_mol = str(fila_mol["intervalo"][0])

print(f"[chi2] Intervalo/banda: {intervalo_mol}")


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
    orden_moleculas=cfg.CHI2_FIT_ORDER,
    show_plots=False,
    save_plots=False,
)

dict_espec_base_chi2 = base_chi2["dict_espec_base"]

print("[chi2] Moléculas restadas para construir el residuo base:")
print(f"       {base_chi2['moleculas_restadas']}")


# ------------------------------------------------------------
# 4. Generar modelo base de la molécula actual
# ------------------------------------------------------------
# Igual que en el enfoque antiguo:
# primero genero un modelo inicial de la molécula actual,
# y uso los residuos de ese modelo como entrada al minimchi2 con residuos=True.

modelo_base_mol = calcular_modelo_sintetico(
    moleculas=MOLECULA,
    region_name=REGION,
    tab_mol_config=tab_mol_config,
    resultados_parametros=resultados_diagrot,
    fwhm_dict=fwhm_dict,
    dict_T_cont=dict_T_contv2,
    dict_cubos_med=dict_espec_base_chi2,
    fuente_parametros="diagrot",
    plot_lineas=False,
    show_plots=False,
    save_plots=False,
)

tab_lineas_chi2 = modelo_base_mol["tab_lineas_plot"]
residuos_mol_chi2 = modelo_base_mol["residuos"]

print("[chi2] Líneas usadas en el ajuste:")

for mol in tab_lineas_chi2:
    print(f"       {mol}: {len(tab_lineas_chi2[mol]['freq'])} líneas")


# ------------------------------------------------------------
# 5. Líneas a no considerar en chi2
# ------------------------------------------------------------

if intervalo_mol == "Banda6":
    list_lin_noconsid_chi2 = cfg.list_freq_no_consid_B6

elif intervalo_mol == "Banda3":
    list_lin_noconsid_chi2 = getattr(cfg, "list_freq_no_consid_B3", None)

else:
    list_lin_noconsid_chi2 = None


# ------------------------------------------------------------
# 6. Ejecutar ajuste chi2
# ------------------------------------------------------------
# De momento recomiendo max_iter=1 hasta verificar que la primera malla va bien.
# Cuando esté validado, cambia a max_iter=10.

resultado_chi2 = cargar_o_calcular_chi2_molecula(
    molecula=MOLECULA,
    region_name=REGION,
    tab_mol_config=tab_mol_config,
    resultados_diagrot=resultados_diagrot,
    fwhm_dict=fwhm_dict,
    dict_T_cont=dict_T_contv2,
    dict_sigma_vent=dict_sigma_vent,
    dict_cubos_med=residuos_mol_chi2,
    dict_resol_espec=dict_resol_esp,
    tab_lineas=tab_lineas_chi2,
    recalcular=RECALCULAR_CHI2,
    n_grid=10,
    tol=1.0,
    max_iter=10,
    list_lin_noconsid=list_lin_noconsid_chi2,
    model_sint=None,
    residuos=True,
    show_model=True,
    save_model=True,
    debug=False,
)

print("[chi2] Ajuste terminado.")
print(f"[chi2] T_fit = {resultado_chi2.get('T_fit')}")
print(f"[chi2] N_fit = {resultado_chi2.get('N_fit')}")
print(f"[chi2] converged = {resultado_chi2.get('converged')}")