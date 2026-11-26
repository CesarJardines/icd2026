"""Segmentación de las señales en ventanas deslizantes.

Una ventana nunca cruza un cambio de actividad, de sujeto ni un salto de
tiempo: se generan solo dentro de segmentos contiguos. Para cada ventana se
guardan su sujeto, su actividad, su segmento y su posición dentro del
segmento, porque el esquema A′ (bloques temporales) los necesita.
"""

import numpy as np
import pandas as pd

from src.datos import ACTIVIDADES, FS

# 5.12 s a 100 Hz, con 50 % de traslape (Reiss y Stricker, 2012)
LARGO_VENTANA = 512
PASO_VENTANA = 256


def segmentos_contiguos(df, tolerancia=1.5 / FS):
    """Asigna un identificador de segmento contiguo a cada muestra.

    Un segmento nuevo empieza cuando cambia el sujeto, cambia la actividad
    o hay un salto de tiempo mayor que ``tolerancia``. El cálculo se hace
    sobre la secuencia completa (incluida la etiqueta 0), así dos tramos de
    la misma actividad separados por una transición quedan en segmentos
    distintos.

    Parameters
    ----------
    df : pandas.DataFrame
        Muestras ordenadas por sujeto y tiempo, con ``sujeto``,
        ``actividad_id`` y ``tiempo``.
    tolerancia : float
        Separación máxima (s) entre muestras consecutivas del mismo segmento.

    Returns
    -------
    numpy.ndarray
        Identificador entero de segmento por fila.
    """
    sujeto = df["sujeto"].to_numpy()
    actividad = df["actividad_id"].to_numpy()
    tiempo = df["tiempo"].to_numpy()
    cambio = np.r_[
        True,
        (sujeto[1:] != sujeto[:-1])
        | (actividad[1:] != actividad[:-1])
        | (np.diff(tiempo) > tolerancia),
    ]
    return np.cumsum(cambio)


def indices_ventanas(df, largo=LARGO_VENTANA, paso=PASO_VENTANA):
    """Calcula las ventanas de cada segmento etiquetado (sin la etiqueta 0).

    Parameters
    ----------
    df : pandas.DataFrame
        Muestras ordenadas por sujeto y tiempo (índice 0..n-1).
    largo : int
        Muestras por ventana.
    paso : int
        Desplazamiento entre ventanas consecutivas.

    Returns
    -------
    pandas.DataFrame
        Una fila por ventana con ``sujeto``, ``actividad_id``, ``actividad``,
        ``segmento``, ``posicion`` (orden dentro del segmento) e ``inicio`` /
        ``fin`` (posiciones de fila; ``fin`` es exclusivo).
    """
    segmento = segmentos_contiguos(df)
    # Inicio, largo, sujeto y actividad de cada segmento
    limites = np.flatnonzero(np.r_[True, segmento[1:] != segmento[:-1]])
    largos = np.diff(np.r_[limites, len(df)])
    sujetos = df["sujeto"].to_numpy()[limites]
    actividades = df["actividad_id"].to_numpy()[limites]

    filas = []
    for ini_seg, n, suj, act, seg in zip(
        limites, largos, sujetos, actividades, segmento[limites]
    ):
        if act == 0 or n < largo:
            continue
        for pos, ini in enumerate(range(ini_seg, ini_seg + n - largo + 1, paso)):
            filas.append((suj, act, seg, pos, ini, ini + largo))

    meta = pd.DataFrame(
        filas,
        columns=["sujeto", "actividad_id", "segmento", "posicion", "inicio", "fin"],
    )
    meta.insert(2, "actividad", meta["actividad_id"].map(ACTIVIDADES))
    return meta
