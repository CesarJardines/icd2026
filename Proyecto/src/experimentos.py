"""Experimentos de la Fase 3: modelos × esquemas de partición.

Se ejecuta aparte del notebook porque tarda alrededor de media hora:

    conda activate pamap2-fuga
    python -m src.experimentos            # desde la raíz del repositorio

Cada combinación (esquema, modelo, balanceo) se guarda en su propio CSV en
``results/fase3/``; si el proceso se interrumpe, al volver a correrlo
continúa con lo que falta. ``--forzar`` borra todo y empieza de cero.

Orden de ejecución:
1. Experimento de balanceo: RF y SVM con ``class_weight=None`` y
   ``"balanced"`` en los esquemas A y B.
2. Se elige, por modelo, la opción con mayor F1 macro medio en B.
3. Todos los modelos en A, A′, B y C (RF y SVM con la opción elegida).
"""

import argparse
import json
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import caracteristicas, particiones

SEED = 42
ESQUEMAS = ["A", "A′", "B", "C"]
MODELOS = ["dummy_frecuente", "dummy_estratificado", "rf", "svm", "knn"]
CON_BALANCEO = ["rf", "svm"]  # kNN no admite class_weight


def crear_modelo(nombre, balanceo, seed=SEED):
    """Devuelve el ``Pipeline`` de un modelo y su rejilla de hiperparámetros.

    El escalado va dentro del ``Pipeline`` para que se ajuste solo con el
    entrenamiento de cada fold.

    Parameters
    ----------
    nombre : str
        Uno de ``MODELOS``.
    balanceo : str
        ``"ninguno"`` o ``"balanced"`` (pesos inversos a la frecuencia).
    seed : int

    Returns
    -------
    (sklearn.pipeline.Pipeline, dict or None)
    """
    peso = None if balanceo == "ninguno" else "balanced"
    if nombre == "dummy_frecuente":
        return Pipeline([("modelo", DummyClassifier(strategy="most_frequent"))]), None
    if nombre == "dummy_estratificado":
        return Pipeline([("modelo", DummyClassifier(strategy="stratified", random_state=seed))]), None
    if nombre == "rf":
        modelo = RandomForestClassifier(n_estimators=200, class_weight=peso,
                                        random_state=seed, n_jobs=1)
        rejilla = {"modelo__max_depth": [None, 20], "modelo__min_samples_leaf": [1, 5]}
        return Pipeline([("modelo", modelo)]), rejilla
    if nombre == "svm":
        modelo = SVC(kernel="rbf", class_weight=peso)
        rejilla = {"modelo__C": [1, 10, 100], "modelo__gamma": ["scale", 0.01]}
        return Pipeline([("escalado", StandardScaler()), ("modelo", modelo)]), rejilla
    if nombre == "knn":
        rejilla = {"modelo__n_neighbors": [1, 5, 15], "modelo__weights": ["uniform", "distance"]}
        return Pipeline([("escalado", StandardScaler()), ("modelo", KNeighborsClassifier())]), rejilla
    raise ValueError(f"Modelo desconocido: {nombre}")


def preparar_esquemas(y, grupos, meta, seed=SEED):
    """Construye los folds externos y la función de folds internos de cada esquema.

    Verifica con ``assert`` que B y C no compartan sujetos y que A′ no
    comparta muestras entre entrenamiento y prueba.

    Returns
    -------
    dict
        Esquema → {"folds": lista de folds, "internos": función(tr) → folds,
        "sujetos_c": tupla (solo en C)}.
    """
    sujetos_c = particiones.elegir_sujetos_c(meta, seed)
    esquemas = {
        "A": {"folds": particiones.esquema_a(y, seed),
              "internos": lambda tr: particiones.internos_a(y[tr], seed)},
        "A′": {"folds": particiones.esquema_a_prima(meta),
               "internos": lambda tr: particiones.internos_a_prima(meta, tr)},
        "B": {"folds": particiones.esquema_b(grupos),
              "internos": lambda tr: particiones.internos_por_sujeto(grupos[tr])},
        "C": {"folds": particiones.esquema_c(grupos, sujetos_c),
              "internos": lambda tr: particiones.internos_por_sujeto(grupos[tr]),
              "sujetos_c": sujetos_c},
    }
    particiones.verificar_sin_sujetos_compartidos(esquemas["B"]["folds"], grupos)
    particiones.verificar_sin_sujetos_compartidos(esquemas["C"]["folds"], grupos)
    particiones.verificar_sin_traslape(esquemas["A′"]["folds"], meta)
    return esquemas


def evaluar(nombre, balanceo, esquema, X, y, grupos, esquemas, seed=SEED):
    """Entrena y evalúa un modelo en todos los folds externos de un esquema.

    Si el modelo tiene rejilla, en cada fold se hace una búsqueda interna
    (``GridSearchCV`` con ``f1_macro``) que usa solo el entrenamiento de ese
    fold, con la misma separación que el esquema externo.

    Returns
    -------
    predicciones : pandas.DataFrame
        Una fila por ventana de prueba.
    parametros : list of dict
        Mejores hiperparámetros por fold.
    """
    X = np.asarray(X)
    filas, parametros = [], []
    for k, (tr, te) in enumerate(esquemas[esquema]["folds"]):
        modelo, rejilla = crear_modelo(nombre, balanceo, seed)
        if rejilla:
            busqueda = GridSearchCV(modelo, rejilla, scoring="f1_macro",
                                    cv=esquemas[esquema]["internos"](tr), n_jobs=-1)
            busqueda.fit(X[tr], y[tr])
            pred = busqueda.predict(X[te])
            parametros.append({"fold": k, **busqueda.best_params_,
                               "f1_interno": round(busqueda.best_score_, 4)})
        else:
            pred = modelo.fit(X[tr], y[tr]).predict(X[te])
        filas.append(pd.DataFrame({
            "esquema": esquema, "modelo": nombre, "balanceo": balanceo, "fold": k,
            "indice": te, "sujeto": grupos[te], "y_real": y[te], "y_pred": pred,
        }))
    return pd.concat(filas, ignore_index=True), parametros


def f1_medio_por_fold(pred):
    """F1 macro medio por fold, calculado solo con las clases presentes en cada fold."""
    return np.mean([
        f1_score(g["y_real"], g["y_pred"], labels=np.unique(g["y_real"]), average="macro")
        for _, g in pred.groupby("fold")
    ])


def _correr(nombre, balanceo, esquema, X, y, grupos, esquemas, carpeta):
    """Ejecuta una combinación si su CSV no existe y la guarda."""
    archivo = carpeta / f"pred_{esquema}_{nombre}_{balanceo}.csv"
    if archivo.exists():
        print(f"  ya existe: {archivo.name}")
        return pd.read_csv(archivo)
    t0 = time.time()
    pred, params = evaluar(nombre, balanceo, esquema, X, y, grupos, esquemas)
    pred.to_csv(archivo, index=False)
    (carpeta / f"params_{esquema}_{nombre}_{balanceo}.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=1, default=str))
    print(f"  {esquema:3} {nombre:20} {balanceo:9} F1 fold = {f1_medio_por_fold(pred):.3f}"
          f"   ({time.time() - t0:.0f} s)", flush=True)
    return pred


def ejecutar(raiz, forzar=False, modelos=MODELOS, esquemas_a_correr=ESQUEMAS):
    """Corre todos los experimentos de la Fase 3 y guarda sus resultados.

    Parameters
    ----------
    raiz : pathlib.Path
        Raíz del repositorio.
    forzar : bool
        Si es True, borra ``results/fase3/`` y recalcula todo.

    Returns
    -------
    pathlib.Path
        Carpeta con los resultados.
    """
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    carpeta = Path(raiz) / "results" / "fase3"
    if forzar and carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    print("Construyendo el dataset…", flush=True)
    X, y, grupos, meta = caracteristicas.construir_dataset(raiz)
    esquemas = preparar_esquemas(y, grupos, meta)
    (carpeta / "esquema_c.json").write_text(json.dumps(
        {"sujetos_prueba": esquemas["C"]["sujetos_c"]}))
    print(f"X = {X.shape}; sujetos de prueba en C: {esquemas['C']['sujetos_c']}")

    # 1. Experimento de balanceo (A y B)
    print("\n1. Experimento de balanceo", flush=True)
    for esquema in ["A", "B"]:
        for nombre in [m for m in CON_BALANCEO if m in modelos]:
            for balanceo in ["ninguno", "balanced"]:
                _correr(nombre, balanceo, esquema, X, y, grupos, esquemas, carpeta)

    # 2. Opción ganadora en B, por modelo
    eleccion = {}
    for nombre in [m for m in CON_BALANCEO if m in modelos]:
        f1 = {b: f1_medio_por_fold(pd.read_csv(carpeta / f"pred_B_{nombre}_{b}.csv"))
              for b in ["ninguno", "balanced"]}
        eleccion[nombre] = {"elegido": max(f1, key=f1.get),
                            "f1_B": {b: round(v, 4) for b, v in f1.items()}}
    (carpeta / "balanceo_elegido.json").write_text(json.dumps(eleccion, indent=1))
    print(f"\n2. Balanceo elegido: {eleccion}")

    # 3. Todos los modelos en todos los esquemas
    print("\n3. Modelos × esquemas", flush=True)
    for esquema in esquemas_a_correr:
        for nombre in modelos:
            balanceo = eleccion.get(nombre, {}).get("elegido", "ninguno")
            _correr(nombre, balanceo, esquema, X, y, grupos, esquemas, carpeta)
    print("\nListo.")
    return carpeta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--forzar", action="store_true", help="borra resultados previos")
    args = parser.parse_args()
    ejecutar(Path(__file__).resolve().parent.parent, forzar=args.forzar)
