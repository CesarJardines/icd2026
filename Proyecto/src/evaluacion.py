"""Métricas y utilidades de gráficas compartidas por todo el notebook.

Las métricas trabajan sobre las predicciones guardadas por
``src/experimentos.py`` (una fila por ventana de prueba, con columnas
``esquema``, ``modelo``, ``balanceo``, ``fold``, ``sujeto``, ``y_real`` y
``y_pred``).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Resolución pedida para reutilizar las figuras en el reporte y las diapositivas
DPI_FIGURAS = 200


def configurar_estilo():
    """Aplica un estilo de gráficas uniforme para todo el notebook."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold"})


def guardar_figura(fig, nombre, carpeta):
    """Guarda una figura como PNG a 200 dpi en la carpeta de figuras.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura a guardar.
    nombre : str
        Nombre del archivo sin extensión (por ejemplo ``eda_01_minutos``).
    carpeta : pathlib.Path
        Carpeta destino (normalmente ``figures/``).

    Returns
    -------
    pathlib.Path
        Ruta del archivo creado.
    """
    ruta = Path(carpeta) / f"{nombre}.png"
    fig.savefig(ruta, dpi=DPI_FIGURAS, bbox_inches="tight")
    return ruta


# ---------------------------------------------------------------- resultados
def cargar_predicciones(carpeta):
    """Lee todas las predicciones de la Fase 3 y marca las del análisis principal.

    ``principal`` es True para las combinaciones que se usan en el resto del
    proyecto: RF y SVM con el balanceo elegido en B; los demás modelos sin
    balanceo.

    Parameters
    ----------
    carpeta : pathlib.Path
        ``results/fase3``.

    Returns
    -------
    pandas.DataFrame
    """
    carpeta = Path(carpeta)
    pred = pd.concat([pd.read_csv(a) for a in sorted(carpeta.glob("pred_*.csv"))],
                     ignore_index=True)
    eleccion = json.loads((carpeta / "balanceo_elegido.json").read_text())
    elegido = pred["modelo"].map({m: e["elegido"] for m, e in eleccion.items()}).fillna("ninguno")
    pred["principal"] = pred["balanceo"] == elegido
    return pred


def metricas_por_fold(pred):
    """F1 macro (solo clases presentes), F1 ponderado y exactitud de cada fold.

    Returns
    -------
    pandas.DataFrame
        Una fila por (esquema, modelo, balanceo, fold).
    """
    filas = []
    for (esq, mod, bal, fold), g in pred.groupby(["esquema", "modelo", "balanceo", "fold"]):
        presentes = np.unique(g["y_real"])
        filas.append({
            "esquema": esq, "modelo": mod, "balanceo": bal, "fold": fold,
            "sujeto": g["sujeto"].iloc[0] if g["sujeto"].nunique() == 1 else np.nan,
            "n": len(g), "clases": len(presentes),
            "f1_macro": f1_score(g["y_real"], g["y_pred"], labels=presentes, average="macro"),
            "f1_ponderado": f1_score(g["y_real"], g["y_pred"], average="weighted"),
            "exactitud": accuracy_score(g["y_real"], g["y_pred"]),
        })
    return pd.DataFrame(filas)


def resumen_esquemas(pred, clases):
    """Tabla modelo × esquema con las distintas formas de medir el desempeño.

    - ``f1_fold_media`` / ``f1_fold_std``: F1 macro por fold, solo con las
      clases presentes en ese fold.
    - ``f1_agrupado``: F1 macro sobre todas las predicciones juntas (12 clases).
    - ``f1_ponderado`` y ``exactitud``: sobre todas las predicciones juntas.

    Parameters
    ----------
    pred : pandas.DataFrame
    clases : list of str
        Las 12 actividades.

    Returns
    -------
    pandas.DataFrame
    """
    por_fold = metricas_por_fold(pred)
    filas = []
    for (esq, mod, bal), g in pred.groupby(["esquema", "modelo", "balanceo"]):
        f = por_fold[(por_fold["esquema"] == esq) & (por_fold["modelo"] == mod)
                     & (por_fold["balanceo"] == bal)]
        filas.append({
            "modelo": mod, "esquema": esq, "balanceo": bal, "folds": len(f),
            "f1_fold_media": f["f1_macro"].mean(),
            "f1_fold_std": f["f1_macro"].std(ddof=1) if len(f) > 1 else np.nan,
            "f1_agrupado": f1_score(g["y_real"], g["y_pred"], labels=clases, average="macro"),
            "f1_ponderado": f1_score(g["y_real"], g["y_pred"], average="weighted"),
            "exactitud": accuracy_score(g["y_real"], g["y_pred"]),
        })
    return pd.DataFrame(filas)


def f1_por_clase(pred, clases):
    """F1 de cada actividad sobre las predicciones agrupadas.

    Returns
    -------
    pandas.Series
        Índice: actividad.
    """
    valores = f1_score(pred["y_real"], pred["y_pred"], labels=clases, average=None)
    return pd.Series(valores, index=clases)


def matriz_confusion(pred, clases):
    """Matriz de confusión normalizada por fila (cada fila suma 1).

    Returns
    -------
    pandas.DataFrame
        Filas: actividad real; columnas: actividad predicha.
    """
    m = confusion_matrix(pred["y_real"], pred["y_pred"], labels=clases, normalize="true")
    return pd.DataFrame(m, index=clases, columns=clases)


def pares_mas_confundidos(matriz, n=5):
    """Los ``n`` pares (real → predicha) con mayor proporción de error."""
    valores = matriz.to_numpy().copy()
    np.fill_diagonal(valores, 0)
    m = pd.DataFrame(valores, index=matriz.index, columns=matriz.columns)
    pares = m.stack().sort_values(ascending=False).head(n)
    return pares.rename("proporción").reset_index().rename(
        columns={"level_0": "real", "level_1": "predicha"})
