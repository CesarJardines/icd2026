"""Carga y limpieza básica del dataset PAMAP2 (sesión Protocol).

Este módulo solo lee los archivos crudos y les pone nombres legibles.
No interpola ni descarta filas: esas decisiones se toman en el notebook
después del análisis exploratorio, para que cada una quede justificada.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Frecuencia de muestreo de las IMU (Hz); cada fila equivale a 0.01 s
FS = 100

# Mapeo oficial de activityID (readme.pdf, sección II.2) a nombres en español.
# Solo se listan las 12 actividades del protocolo; 0 = transitoria.
ACTIVIDADES = {
    1: "acostado",
    2: "sentado",
    3: "de pie",
    4: "caminar",
    5: "correr",
    6: "bicicleta",
    7: "caminata nórdica",
    12: "subir escaleras",
    13: "bajar escaleras",
    16: "aspirar",
    17: "planchar",
    24: "saltar cuerda",
}

# Ubicación de cada IMU, en el orden en que aparecen en el archivo
UBICACIONES = ["muneca", "pecho", "tobillo"]

# Nombre con acento para títulos y leyendas (los identificadores van sin ñ)
NOMBRE_UBICACION = {"muneca": "muñeca", "pecho": "pecho", "tobillo": "tobillo"}

# Las 17 columnas de cada IMU (readme.pdf, sección II.1)
_COLUMNAS_IMU = (
    ["temp"]
    + [f"acc16_{e}" for e in "xyz"]
    + [f"acc6_{e}" for e in "xyz"]
    + [f"gyro_{e}" for e in "xyz"]
    + [f"mag_{e}" for e in "xyz"]
    + [f"orient_{i}" for i in range(1, 5)]
)


def nombres_columnas():
    """Devuelve los 54 nombres de columna legibles de un archivo .dat.

    Returns
    -------
    list of str
        ``tiempo``, ``actividad_id``, ``fc`` y 17 columnas por IMU con el
        prefijo de su ubicación (por ejemplo ``tobillo_acc16_x``).
    """
    nombres = ["tiempo", "actividad_id", "fc"]
    for ubicacion in UBICACIONES:
        nombres += [f"{ubicacion}_{c}" for c in _COLUMNAS_IMU]
    return nombres


def ruta_protocol(raiz):
    """Devuelve la carpeta Protocol/ y falla con un mensaje claro si no existe.

    Parameters
    ----------
    raiz : pathlib.Path
        Raíz del repositorio (la carpeta que contiene ``data/``).

    Returns
    -------
    pathlib.Path
        Ruta a ``data/raw/PAMAP2_Dataset/Protocol``.
    """
    ruta = Path(raiz) / "data" / "raw" / "PAMAP2_Dataset" / "Protocol"
    if not ruta.is_dir():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Descarga PAMAP2 desde UCI y coloca la "
            "carpeta Protocol/ en data/raw/PAMAP2_Dataset/ (ver README.md)."
        )
    return ruta


def cargar_sujeto(archivo):
    """Lee un archivo subjectXXX.dat y le agrega la columna ``sujeto``.

    Las señales se guardan en float32 para reducir a la mitad la memoria
    (~2.9 millones de filas); el tiempo se deja en float64 porque necesita
    resolución de centésimas en valores de miles de segundos.

    Parameters
    ----------
    archivo : pathlib.Path
        Ruta al archivo .dat.

    Returns
    -------
    pandas.DataFrame
        Una fila por muestra (0.01 s) con 55 columnas.
    """
    nombres = nombres_columnas()
    tipos = {c: np.float32 for c in nombres[2:]}
    tipos["tiempo"] = np.float64
    df = pd.read_csv(archivo, sep=" ", header=None, names=nombres, dtype=tipos)
    df["actividad_id"] = df["actividad_id"].astype(np.int16)
    # El número de sujeto viene en el nombre del archivo (subject101.dat → 101)
    df.insert(0, "sujeto", int(Path(archivo).stem.replace("subject", "")))
    return df


def cargar_protocolo(raiz):
    """Carga los 9 archivos de Protocol/ en un solo DataFrame.

    Parameters
    ----------
    raiz : pathlib.Path
        Raíz del repositorio.

    Returns
    -------
    pandas.DataFrame
        Todas las muestras de los 9 sujetos, en orden de sujeto y tiempo,
        con la columna ``actividad`` (nombre en español; NaN si es transitoria).
    """
    archivos = sorted(ruta_protocol(raiz).glob("subject*.dat"))
    df = pd.concat([cargar_sujeto(a) for a in archivos], ignore_index=True)
    df.insert(2, "actividad", df["actividad_id"].map(ACTIVIDADES))
    return df


def columnas_sensor(df, tipo):
    """Devuelve las columnas de un tipo de sensor en las tres IMU.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con los nombres de ``nombres_columnas``.
    tipo : str
        Prefijo del sensor: ``temp``, ``acc16``, ``acc6``, ``gyro``, ``mag``
        u ``orient``.

    Returns
    -------
    list of str
    """
    return [c for c in df.columns if c.split("_", 1)[-1].startswith(tipo)
            and c.split("_", 1)[0] in UBICACIONES]


def minutos_por_actividad_sujeto(df):
    """Tabla de minutos etiquetados por actividad (filas) y sujeto (columnas).

    Parameters
    ----------
    df : pandas.DataFrame
        Salida de ``cargar_protocolo``.

    Returns
    -------
    pandas.DataFrame
        Minutos por celda; las filas siguen el orden de ``ACTIVIDADES``.
    """
    etiquetadas = df[df["actividad_id"] != 0]
    tabla = (
        etiquetadas.groupby(["actividad", "sujeto"], observed=True)
        .size()
        .unstack(fill_value=0)
        / FS
        / 60
    )
    orden = [a for a in ACTIVIDADES.values() if a in tabla.index]
    return tabla.loc[orden]


def magnitud(df, ubicacion, sensor="acc16"):
    """Norma euclidiana de las tres componentes de un sensor.

    La magnitud no depende de cómo quedó orientada la IMU sobre el cuerpo,
    por eso es más comparable entre sujetos que cada eje por separado.

    Parameters
    ----------
    df : pandas.DataFrame
    ubicacion : str
        ``muneca``, ``pecho`` o ``tobillo``.
    sensor : str
        ``acc16``, ``acc6``, ``gyro`` o ``mag``.

    Returns
    -------
    pandas.Series
    """
    ejes = [f"{ubicacion}_{sensor}_{e}" for e in "xyz"]
    return np.sqrt((df[ejes].astype(np.float64) ** 2).sum(axis=1, skipna=False))


def huecos_nan(df, columna):
    """Longitud (en muestras) de cada racha consecutiva de NaN en una columna.

    Las rachas se calculan por sujeto para no unir el final de un archivo
    con el inicio del siguiente.

    Parameters
    ----------
    df : pandas.DataFrame
    columna : str

    Returns
    -------
    numpy.ndarray
        Una entrada por hueco.
    """
    longitudes = []
    for _, d in df.groupby("sujeto"):
        es_nan = d[columna].isna().to_numpy().astype(np.int8)
        bordes = np.diff(np.r_[0, es_nan, 0])
        longitudes.append(np.flatnonzero(bordes == -1) - np.flatnonzero(bordes == 1))
    return np.concatenate(longitudes)


# Límite de interpolación de las IMU: 1 s. El hueco más largo observado en el
# EDA dura 0.77 s, así que con este límite se rellenan todos.
LIMITE_INTERPOLACION_IMU = 100


def columnas_modelo(df):
    """Columnas de señal que se conservan para el modelado.

    Se descartan la orientación (inválida según la documentación) y el
    acelerómetro de ±6 g (se satura; ver §2.5). Quedan la frecuencia
    cardiaca y, por IMU, temperatura, acelerómetro de ±16 g, giroscopio y
    magnetómetro.

    Returns
    -------
    list of str
    """
    return ["fc"] + [
        c for c in df.columns
        if c.split("_", 1)[0] in UBICACIONES
        and not any(t in c for t in ("orient", "acc6"))
    ]


def interpolar_por_sujeto(df, limite_imu=LIMITE_INTERPOLACION_IMU):
    """Rellena los NaN de cada sujeto usando solo su propia señal en el tiempo.

    No usa etiquetas ni datos de otros sujetos, así que no puede filtrar
    información entre entrenamiento y prueba. Se aplica sobre la secuencia
    completa (incluidas las muestras transitorias) para no unir tramos que
    en realidad están separados por una transición.

    - Frecuencia cardiaca: interpolación lineal y luego ``ffill``/``bfill``
      para los extremos (se mide a ~9 Hz, no son pérdidas reales).
    - IMU: interpolación lineal solo dentro de huecos de hasta
      ``limite_imu`` muestras.

    Parameters
    ----------
    df : pandas.DataFrame
        Salida de ``cargar_protocolo``.
    limite_imu : int
        Máximo de muestras consecutivas que se rellenan en las IMU.

    Returns
    -------
    pandas.DataFrame
        Copia con ``sujeto``, ``tiempo``, ``actividad_id``, ``actividad`` y
        las columnas de ``columnas_modelo``.
    """
    señales = columnas_modelo(df)
    imu = [c for c in señales if c != "fc"]
    salida = df[["sujeto", "tiempo", "actividad_id", "actividad"] + señales].copy()
    for _, idx in salida.groupby("sujeto").groups.items():
        bloque = salida.loc[idx]
        salida.loc[idx, "fc"] = bloque["fc"].interpolate().ffill().bfill()
        salida.loc[idx, imu] = bloque[imu].interpolate(
            limit=limite_imu, limit_area="inside"
        )
    return salida
