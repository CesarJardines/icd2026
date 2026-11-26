"""Extracción de características por ventana.

Cada característica se calcula solo con las muestras de su propia ventana:
no aprende nada de otras ventanas, de otros sujetos ni de las etiquetas.
Por eso puede calcularse antes de partir los datos sin introducir fuga.
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

from src.datos import FS, UBICACIONES

# Sensores que se usan (nombre en las características → prefijo en los datos)
SENSORES = {"acc": "acc16", "gyro": "gyro"}
_EPS = 1e-12


def senales_base(df):
    """Construye las 24 señales que se resumen en cada ventana.

    Por cada IMU (3) y sensor (acelerómetro ±16 g y giroscopio): los ejes
    x, y, z y la magnitud, que no depende de la orientación del sensor.

    Parameters
    ----------
    df : pandas.DataFrame
        Datos ya interpolados.

    Returns
    -------
    dict of str → numpy.ndarray
        Nombre de la señal (por ejemplo ``tobillo_acc_mag``) → serie completa.
    """
    señales = {}
    for u in UBICACIONES:
        for corto, prefijo in SENSORES.items():
            ejes = [df[f"{u}_{prefijo}_{e}"].to_numpy(np.float64) for e in "xyz"]
            for e, valores in zip("xyz", ejes):
                señales[f"{u}_{corto}_{e}"] = valores
            señales[f"{u}_{corto}_mag"] = np.sqrt(sum(v ** 2 for v in ejes))
    return señales


def recortar(serie, inicios, largo):
    """Apila las ventanas de una serie en una matriz (ventanas × muestras)."""
    return serie[inicios[:, None] + np.arange(largo)]


def estadisticas_tiempo(W):
    """Media, desviación estándar, mínimo, máximo y RMS de cada ventana.

    Parameters
    ----------
    W : numpy.ndarray
        Matriz ventanas × muestras.

    Returns
    -------
    dict of str → numpy.ndarray
    """
    return {
        "media": W.mean(axis=1),
        "std": W.std(axis=1),
        "min": W.min(axis=1),
        "max": W.max(axis=1),
        "rms": np.sqrt((W ** 2).mean(axis=1)),
    }


def estadisticas_espectrales(W, fs=FS):
    """Energía, frecuencia dominante y entropía espectral de cada ventana.

    Se resta la media de la ventana y se ignora la componente de 0 Hz, para
    que estas características describan el movimiento y no la gravedad ni
    el nivel medio de la señal.

    - Energía: suma de la potencia espectral dividida entre el largo.
    - Frecuencia dominante: frecuencia (Hz) con mayor potencia.
    - Entropía espectral: entropía de Shannon de la potencia normalizada,
      dividida entre su máximo posible (0 = un solo tono, 1 = ruido plano).

    Parameters
    ----------
    W : numpy.ndarray
        Matriz ventanas × muestras.
    fs : float
        Frecuencia de muestreo (Hz).

    Returns
    -------
    dict of str → numpy.ndarray
    """
    n = W.shape[1]
    potencia = np.abs(rfft(W - W.mean(axis=1, keepdims=True), axis=1)) ** 2
    potencia, frecuencias = potencia[:, 1:], rfftfreq(n, 1 / fs)[1:]
    total = potencia.sum(axis=1, keepdims=True) + _EPS
    p = potencia / total
    entropia = -(p * np.log2(p + _EPS)).sum(axis=1) / np.log2(p.shape[1])
    return {
        "energia": potencia.sum(axis=1) / n,
        "frec_dom": frecuencias[potencia.argmax(axis=1)],
        "entropia": entropia,
    }


def extraer_caracteristicas(df, meta, fs=FS):
    """Calcula la matriz de características de todas las ventanas.

    Parameters
    ----------
    df : pandas.DataFrame
        Datos ya interpolados (mismo índice posicional que usó ``meta``).
    meta : pandas.DataFrame
        Salida de ``ventanas.indices_ventanas``.
    fs : float
        Frecuencia de muestreo (Hz).

    Returns
    -------
    pandas.DataFrame
        Una fila por ventana: 24 señales × 8 estadísticas, más la frecuencia
        cardiaca media y la temperatura media de cada IMU.
    """
    inicios = meta["inicio"].to_numpy()
    largo = int((meta["fin"] - meta["inicio"]).iloc[0])
    columnas = {}
    for nombre, serie in senales_base(df).items():
        W = recortar(serie, inicios, largo)
        for stat, valores in {**estadisticas_tiempo(W),
                              **estadisticas_espectrales(W, fs)}.items():
            columnas[f"{nombre}_{stat}"] = valores
    columnas["fc_media"] = recortar(df["fc"].to_numpy(np.float64), inicios, largo).mean(axis=1)
    for u in UBICACIONES:
        temp = df[f"{u}_temp"].to_numpy(np.float64)
        columnas[f"{u}_temp_media"] = recortar(temp, inicios, largo).mean(axis=1)
    return pd.DataFrame(columnas, index=meta.index)


def pares_para_similitud(meta, seed):
    """Forma tres grupos de pares de ventanas para medir su parecido.

    (a) consecutivas: mismo segmento, posiciones contiguas (comparten 50 %);
    (b) mismo sujeto y actividad, sin compartir ninguna muestra;
    (c) misma actividad, sujetos distintos.
    Para (b) y (c) se sortea una pareja por ventana con semilla fija.

    Parameters
    ----------
    meta : pandas.DataFrame
        Salida de ``ventanas.indices_ventanas``.
    seed : int
        Semilla para el sorteo.

    Returns
    -------
    dict of str → numpy.ndarray
        Grupo → arreglo (n_pares × 2) de posiciones de fila en ``meta``.
    """
    rng = np.random.default_rng(seed)
    m = meta.reset_index(drop=True)
    siguiente = (m["segmento"].shift(-1) == m["segmento"]).to_numpy()
    consecutivas = np.column_stack([np.flatnonzero(siguiente), np.flatnonzero(siguiente) + 1])

    mismo_sujeto, otro_sujeto = [], []
    for _, g in m.groupby("actividad"):
        idx = g.index.to_numpy()
        suj = g["sujeto"].to_numpy()
        ini = g["inicio"].to_numpy()
        for k, i in enumerate(idx):
            # mismo sujeto y sin muestras compartidas (separadas ≥ 1 ventana completa)
            cand = idx[(suj == suj[k]) & (np.abs(ini - ini[k]) >= 512)]
            if len(cand):
                mismo_sujeto.append((i, rng.choice(cand)))
            cand = idx[suj != suj[k]]
            if len(cand):
                otro_sujeto.append((i, rng.choice(cand)))
    return {
        "consecutivas (mismo sujeto)": consecutivas,
        "no vecinas (mismo sujeto)": np.array(mismo_sujeto),
        "otro sujeto": np.array(otro_sujeto),
    }


def construir_dataset(raiz):
    """Ejecuta carga, interpolación, ventanas y características de una vez.

    Es el mismo camino que sigue el notebook paso a paso (§1–§3); lo usa el
    script de experimentos para no depender del notebook.

    Parameters
    ----------
    raiz : pathlib.Path
        Raíz del repositorio.

    Returns
    -------
    X : pandas.DataFrame
    y : numpy.ndarray
    grupos : numpy.ndarray
    meta : pandas.DataFrame
    """
    from src import datos, ventanas  # importación local para evitar ciclos

    datos_limpios = datos.interpolar_por_sujeto(datos.cargar_protocolo(raiz))
    meta = ventanas.indices_ventanas(datos_limpios)
    X = extraer_caracteristicas(datos_limpios, meta)
    return X, meta["actividad"].to_numpy(), meta["sujeto"].to_numpy(), meta
