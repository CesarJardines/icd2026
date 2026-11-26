"""Carga del segundo dataset: UCI HAR (Human Activity Recognition Using Smartphones).

Anguita et al. (2013). 30 sujetos con un teléfono en la cintura, 6
actividades, ventanas de 2.56 s (128 muestras a 50 Hz) con 50 % de
traslape y 561 características ya calculadas por los autores.

Los autores separaron los sujetos en ``train`` (21) y ``test`` (9). Aquí
se juntan las dos partes en un solo conjunto para aplicar nuestros propios
esquemas de partición (A, A′ y B), igual que en PAMAP2.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ACTIVIDADES_HAR = {
    1: "caminar", 2: "subir escaleras", 3: "bajar escaleras",
    4: "sentado", 5: "de pie", 6: "acostado",
}
LARGO_VENTANA_HAR = 128  # 2.56 s a 50 Hz
PASO_VENTANA_HAR = 64    # 50 % de traslape


def ruta_har(raiz):
    """Devuelve la carpeta del dataset y falla con un mensaje claro si no existe."""
    ruta = Path(raiz) / "data" / "raw" / "UCI HAR Dataset"
    if not ruta.is_dir():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Descarga «Human Activity Recognition Using "
            "Smartphones» desde UCI y descomprímelo en data/raw/ (ver README.md).")
    return ruta


def _continua_ventana_anterior(senal):
    """True si la ventana i comparte su primera mitad con la segunda mitad de i-1.

    Con 50 % de traslape, dos ventanas contiguas en el tiempo tienen
    exactamente las mismas 64 lecturas en común. Si no coinciden, la
    ventana inicia un tramo nuevo (otra actividad, otra repetición u otro
    sujeto).
    """
    mitad = senal.shape[1] // 2
    continua = np.isclose(senal[:-1, mitad:], senal[1:, :mitad], atol=1e-6).all(axis=1)
    return np.r_[False, continua]


def _cargar_parte(ruta, parte):
    """Lee una de las dos partes (``train`` o ``test``) del dataset."""
    X = pd.read_csv(ruta / parte / f"X_{parte}.txt", sep=r"\s+", header=None)
    y = np.loadtxt(ruta / parte / f"y_{parte}.txt", dtype=int)
    sujeto = np.loadtxt(ruta / parte / f"subject_{parte}.txt", dtype=int)
    # La señal cruda de una sola componente basta para reconocer el traslape
    senal = np.loadtxt(ruta / parte / "Inertial Signals" / f"body_acc_x_{parte}.txt")
    continua = _continua_ventana_anterior(senal) & np.r_[False, sujeto[1:] == sujeto[:-1]]
    return X, y, sujeto, continua


def cargar_har(raiz):
    """Carga UCI HAR completo con metadatos de ventana compatibles con ``particiones``.

    Returns
    -------
    X : pandas.DataFrame
        10 299 ventanas × 561 características.
    y : numpy.ndarray
        Actividad (nombre en español).
    grupos : numpy.ndarray
        Sujeto de cada ventana.
    meta : pandas.DataFrame
        ``sujeto``, ``actividad``, ``parte_original``, ``segmento``,
        ``posicion``, ``inicio`` y ``fin``. ``inicio`` y ``fin`` son índices
        de muestra reconstruidos: dentro de un segmento avanzan 64 muestras
        por ventana; entre segmentos hay un salto grande para que nunca se
        consideren traslapados.
    """
    ruta = ruta_har(raiz)
    nombres = pd.read_csv(ruta / "features.txt", sep=r"\s+", header=None)[1]
    # features.txt repite algunos nombres (bandas de energía); se numeran para hacerlos únicos
    nombres = [f"{i:03d}_{n}" for i, n in enumerate(nombres, start=1)]

    partes = [_cargar_parte(ruta, p) for p in ["train", "test"]]
    X = pd.concat([p[0] for p in partes], ignore_index=True)
    X.columns = nombres
    y_id = np.concatenate([p[1] for p in partes])
    sujeto = np.concatenate([p[2] for p in partes])
    continua = np.concatenate([p[3] for p in partes])
    parte = np.repeat(["train", "test"], [len(p[1]) for p in partes])

    # Un segmento nuevo empieza donde la ventana no continúa a la anterior
    segmento = np.cumsum(~continua) - 1
    posicion = pd.Series(segmento).groupby(segmento).cumcount().to_numpy()
    inicio = segmento * 1_000_000 + posicion * PASO_VENTANA_HAR
    meta = pd.DataFrame({
        "sujeto": sujeto, "actividad_id": y_id,
        "actividad": pd.Series(y_id).map(ACTIVIDADES_HAR).to_numpy(),
        "parte_original": parte, "segmento": segmento, "posicion": posicion,
        "inicio": inicio, "fin": inicio + LARGO_VENTANA_HAR,
    })
    # Un segmento no puede mezclar actividades ni sujetos
    assert (meta.groupby("segmento")[["actividad", "sujeto"]].nunique() == 1).all().all()
    return X, meta["actividad"].to_numpy(), sujeto, meta
