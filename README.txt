README DEL TFM:

COMO AÑADIR UNA NUEVA MOLÉCULA:

1: PONER TODAS LAS CONSTANTES B0, id_cdms/jpl, id_splat, list_noconsid, list_cali en config
2: AÑADIR FILA DE LA MOLÉCULA EN LA TABLA DE CONFIG:

Ejemplo de tabla:
    rows = [
        # nombre, mol, intervalo, catalogo,
        # id_splat, id_cat, B0,
        # filtro_estructuras,
        # list_frec_noconsid, frec_nofiltrar,
        # f0, 
        # E_max, aij_min, sijmu2_min, filt_inter

        (
            "C2H5OH_g",
            "C2H5OH",
            "Banda6",
            "CDMS",
            "id_splatC2H5OH",
            "id_cdmsC2H5OH",
            "B0C2H5OH",
            "filtro_C2H5OH_quitar_anti",
            "list_noconsidC2H5OH",
            "None",
            232491 * u.MHz,
            600 * u.K,
            1e-6 / u.s,
            10 * u.D**2,
            0.5,
        ),
        
3: CREAR PLAN DE CALIBRACIÓN: En calibration.py ==> get_calibration_plan(): añadir el plan de calibración de esta molécula:
Ejemplo de plan de calibración: 
{
        "C2H5OH_g": {
            "freqs": cfg.freqs_caliC2H5OH,
            "longs": [20] * len(cfg.freqs_caliC2H5OH) * u.km / u.s,
        },

4: AÑADIR MOLÉCULA EN LA LISTA DE ORDEN PARA AJUSTAR EL CHI² EN CONFIG.PY
5: AÑADIR FRECUENCIAS DE LA MOLECULA A NO CONSIDERAR POR EL CHI² EN EL DICCIONARIO DE CONFIG.PY
6: AÑADIR GRID_CONFIG PARA EL CHI²: (Esto es opcional)
Ejemplo:
    "C2H5OH_g": {
        "deltaT": 4 * u.K,
        "deltaN": 2.53e16 / u.cm**2,
        "n_grid": 10,
    },

7: AÑADIR MOLÉCULA AL DICCIONARIO DE CONFIG DE CADA REGIÓN


SOLUCIÓN DE ALGUNOS ERRORES QUE ME VOY ENCONTRANDO:

1- A la hora de ajustar el chi² hay veces que falla porque la incertidumbre en T_ex es demasiada alta y se rompe el código, no sé en qué depende para que se rompa, porque hay veces que sí que funciona y otras rompe porque sí. Para las veces que se rompe: Simplemente reduce la deltaT a un valor por debajo del valor obtenido del diagrama rotacional, eso debería funcionar si no es así, reduce también el deltaN.

2- Si al cambiar de región, el ajuste chi² se comporta de forma muy extraña para una molécula en concreto, prueba a revisar a mano las frecuencias que no hay que considerar en ese ajuste chi², probablemente haya alguna que no estés considerando que sí que haya que hacerlo y viceversa, alguna que la estás considerando y no haya que hacerlo.
