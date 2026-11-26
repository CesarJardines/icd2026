"""Experimentos de la Fase 4 (P3): ¿qué características siguen siendo útiles?

Se ejecuta aparte del notebook, igual que ``src/experimentos.py``:

    conda activate pamap2-fuga
    python -m src.experimentos_p3          # desde la raíz del repositorio

Cada análisis se guarda en ``results/fase4/``; si ya existe, se omite.
``--forzar`` borra todo y empieza de cero.

Todos los análisis usan Random Forest con los hiperparámetros que eligió la
búsqueda anidada de la Fase 3 **en cada fold** (``results/fase3/params_*``),
así que no hay una búsqueda nueva que pueda ver los datos de prueba.

1. Importancia por permutación en A y en B, calculada sobre la prueba de
   cada fold: atributo por atributo y por grupo de sensores.
2. Información mutua de cada atributo con la actividad (y con el sujeto),
   calculada solo con el entrenamiento de cada fold LOSO.
3. RFE dentro de cada fold LOSO (de 10 en 10 atributos) y F1 en el sujeto de
   prueba para cada tamaño del subconjunto.
4. PCA (95 % de la varianza) dentro del pipeline, en LOSO, con RF y kNN.
5. Comprobación: RF, SVM y kNN en LOSO sin temperatura ni frecuencia
   cardiaca, los atributos más ligados a la identidad del sujeto.
"""

import argparse
import json
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import caracteristicas, particiones

SEED = 42
N_REPETICIONES = 5  # permutaciones por atributo
PASO_RFE = 10
MINIMO_RFE = 6  # 196 → 186 → … → 16 → 6


def f1_presentes(y_real, y_pred):
    """F1 macro solo sobre las clases presentes en la prueba (convención de la Fase 3)."""
    return f1_score(y_real, y_pred, labels=np.unique(y_real), average="macro")


def grupo_de(nombre):
    """Grupo de sensores al que pertenece un atributo (p. ej. ``tobillo_acc``)."""
    partes = nombre.split("_")
    if partes[0] == "fc":
        return "frecuencia cardiaca"
    if partes[1] == "temp":
        return "temperatura"
    return f"{partes[0]}_{partes[1]}"


def rf_del_fold(params, seed=SEED, n_jobs=-1):
    """Random Forest con los hiperparámetros elegidos en un fold de la Fase 3."""
    return RandomForestClassifier(
        n_estimators=200, max_depth=params["modelo__max_depth"],
        min_samples_leaf=params["modelo__min_samples_leaf"],
        random_state=seed, n_jobs=n_jobs)


def _leer_params(raiz, esquema):
    archivo = Path(raiz) / "results" / "fase3" / f"params_{esquema}_rf_ninguno.json"
    return json.loads(archivo.read_text())


def permutacion_por_grupos(modelo, X_te, y_te, grupos_cols, rng, n_rep=N_REPETICIONES):
    """Caída del F1 al permutar juntos todos los atributos de un grupo.

    Los atributos de un mismo sensor están muy correlacionados: si se
    permuta solo uno, el modelo lo sustituye con sus vecinos y la
    importancia individual se subestima. Permutar el grupo completo (con la
    misma permutación de filas) mide cuánto aporta el sensor como un todo.
    """
    base = f1_presentes(y_te, modelo.predict(X_te))
    filas = []
    for grupo, cols in grupos_cols.items():
        caidas = []
        for _ in range(n_rep):
            Xp = X_te.copy()
            orden = rng.permutation(len(Xp))
            Xp[:, cols] = Xp[orden][:, cols]
            caidas.append(base - f1_presentes(y_te, modelo.predict(Xp)))
        filas.append({"grupo": grupo, "importancia": np.mean(caidas),
                      "desv_repeticiones": np.std(caidas)})
    return pd.DataFrame(filas)


def importancia_permutacion(X, y, grupos, folds, params, nombre_esquema):
    """Importancia por permutación sobre la prueba de cada fold.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Importancia por atributo y por grupo de sensores, una fila por fold.
    """
    nombres = list(X.columns)
    Xv = X.to_numpy()
    grupos_cols = {}
    for j, n in enumerate(nombres):
        grupos_cols.setdefault(grupo_de(n), []).append(j)
    rng = np.random.default_rng(SEED)
    por_atributo, por_grupo = [], []
    for k, (tr, te) in enumerate(folds):
        t0 = time.time()
        modelo = rf_del_fold(params[k]).fit(Xv[tr], y[tr])
        modelo.set_params(n_jobs=1)  # la paralelización va por atributo
        r = permutation_importance(
            modelo, Xv[te], y[te], scoring=lambda m, Xs, ys: f1_presentes(ys, m.predict(Xs)),
            n_repeats=N_REPETICIONES, random_state=SEED, n_jobs=-1)
        sujeto = int(grupos[te][0]) if len(np.unique(grupos[te])) == 1 else np.nan
        por_atributo.append(pd.DataFrame({
            "esquema": nombre_esquema, "fold": k, "sujeto": sujeto, "atributo": nombres,
            "importancia": r.importances_mean, "desv_repeticiones": r.importances_std}))
        g = permutacion_por_grupos(modelo, Xv[te], y[te], grupos_cols, rng)
        g.insert(0, "sujeto", sujeto)
        g.insert(0, "fold", k)
        g.insert(0, "esquema", nombre_esquema)
        por_grupo.append(g)
        print(f"    {nombre_esquema} fold {k}: {time.time() - t0:.0f} s", flush=True)
    return pd.concat(por_atributo, ignore_index=True), pd.concat(por_grupo, ignore_index=True)


def informacion_mutua(X, y, grupos, folds):
    """Información mutua de cada atributo con la actividad y con el sujeto.

    Se calcula solo con el entrenamiento de cada fold LOSO. La información
    con el sujeto mide cuánto «delata» un atributo a la persona: un atributo
    muy ligado al sujeto y poco a la actividad es candidato a no generalizar.
    """
    filas = []
    for k, (tr, te) in enumerate(folds):
        Xtr = X.iloc[tr]
        mi_act = mutual_info_classif(Xtr, y[tr], random_state=SEED, n_jobs=-1)
        mi_suj = mutual_info_classif(Xtr, grupos[tr], random_state=SEED, n_jobs=-1)
        filas.append(pd.DataFrame({
            "fold": k, "sujeto": int(grupos[te][0]), "atributo": X.columns,
            "mi_actividad": mi_act, "mi_sujeto": mi_suj}))
        print(f"    MI fold {k}", flush=True)
    return pd.concat(filas, ignore_index=True)


def curva_rfe(X, y, grupos, folds, params):
    """RFE con Random Forest dentro de cada fold LOSO y F1 por tamaño.

    El ranking se aprende solo con el entrenamiento del fold; después, para
    cada tamaño k se entrena un RF con los k mejores atributos y se evalúa
    en el sujeto de prueba.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Curva (fold, k, F1) y ranking de RFE de cada fold.
    """
    Xv = X.to_numpy()
    nombres = np.array(X.columns)
    curva, rankings = [], []
    for k, (tr, te) in enumerate(folds):
        t0 = time.time()
        rfe = RFE(rf_del_fold(params[k]), n_features_to_select=MINIMO_RFE, step=PASO_RFE)
        rfe.fit(Xv[tr], y[tr])
        rankings.append(pd.DataFrame({"fold": k, "sujeto": int(grupos[te][0]),
                                      "atributo": nombres, "ranking": rfe.ranking_}))
        # ranking 1 = los últimos 6 que sobreviven; ranking r = eliminados en el paso r-1
        for r in range(1, rfe.ranking_.max() + 1):
            cols = rfe.ranking_ <= r
            modelo = rf_del_fold(params[k]).fit(Xv[tr][:, cols], y[tr])
            curva.append({"fold": k, "sujeto": int(grupos[te][0]), "n_atributos": int(cols.sum()),
                          "f1": f1_presentes(y[te], modelo.predict(Xv[te][:, cols]))})
        print(f"    RFE fold {k}: {time.time() - t0:.0f} s", flush=True)
    return pd.DataFrame(curva), pd.concat(rankings, ignore_index=True)


def con_pca(X, y, grupos, folds, params):
    """F1 LOSO con PCA (95 % de varianza) dentro del pipeline, con RF y kNN.

    El escalado y el PCA se ajustan solo con el entrenamiento del fold. kNN
    usa k = 5 con pesos por distancia, que fue lo que eligió la búsqueda
    anidada en los 9 folds de B.
    """
    Xv = X.to_numpy()
    filas = []
    for k, (tr, te) in enumerate(folds):
        for nombre, modelo in [("rf", rf_del_fold(params[k])),
                               ("knn", KNeighborsClassifier(5, weights="distance"))]:
            pipe = Pipeline([("escalado", StandardScaler()),
                             ("pca", PCA(n_components=0.95, svd_solver="full")),
                             ("modelo", modelo)]).fit(Xv[tr], y[tr])
            filas.append({"fold": k, "sujeto": int(grupos[te][0]), "modelo": nombre,
                          "componentes": int(pipe["pca"].n_components_),
                          "f1": f1_presentes(y[te], pipe.predict(Xv[te]))})
    return pd.DataFrame(filas)


def sin_atributos_de_persona(X, y, grupos, folds, params):
    """F1 LOSO de RF, SVM y kNN con y sin temperatura y frecuencia cardiaca.

    Estos 4 atributos son los más ligados a la identidad del sujeto
    (información mutua con el sujeto). Si la caída por personas de kNN y SVM
    se debe a ellos, quitarlos debería mejorar a esos modelos en LOSO.
    Se usan los hiperparámetros que eligió la búsqueda anidada en B: SVM con
    C = 1 y gamma = "scale", kNN con k = 5 por distancia (iguales en los 9
    folds) y RF con los de cada fold.
    """
    from sklearn.svm import SVC

    de_persona = [c for c in X.columns if grupo_de(c) in ("temperatura", "frecuencia cardiaca")]
    variantes = {"todos": list(X.columns),
                 "sin temperatura ni FC": [c for c in X.columns if c not in de_persona]}
    filas = []
    for k, (tr, te) in enumerate(folds):
        modelos = {
            "rf": rf_del_fold(params[k]),
            "svm": Pipeline([("escalado", StandardScaler()), ("modelo", SVC(C=1, gamma="scale"))]),
            "knn": Pipeline([("escalado", StandardScaler()),
                             ("modelo", KNeighborsClassifier(5, weights="distance"))]),
        }
        for variante, cols in variantes.items():
            Xv = X[cols].to_numpy()
            for nombre, modelo in modelos.items():
                pred = modelo.fit(Xv[tr], y[tr]).predict(Xv[te])
                filas.append({"fold": k, "sujeto": int(grupos[te][0]), "modelo": nombre,
                              "variante": variante, "f1": f1_presentes(y[te], pred)})
    return pd.DataFrame(filas)


def _guardar(carpeta, nombre, funcion):
    """Ejecuta ``funcion`` si sus CSV no existen; devuelve los DataFrames guardados."""
    archivos = [carpeta / f"{n}.csv" for n in nombre]
    if all(a.exists() for a in archivos):
        print(f"  ya existe: {', '.join(a.name for a in archivos)}")
        return
    t0 = time.time()
    salida = funcion()
    salida = salida if isinstance(salida, tuple) else (salida,)
    for a, df in zip(archivos, salida):
        df.to_csv(a, index=False)
    print(f"  listo: {', '.join(a.name for a in archivos)} ({time.time() - t0:.0f} s)", flush=True)


def ejecutar(raiz, forzar=False):
    """Corre los análisis de P3 y guarda sus resultados en ``results/fase4/``.

    Returns
    -------
    pathlib.Path
    """
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    carpeta = Path(raiz) / "results" / "fase4"
    if forzar and carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    print("Construyendo el dataset…", flush=True)
    X, y, grupos, meta = caracteristicas.construir_dataset(raiz)
    folds_a = particiones.esquema_a(y, SEED)
    folds_b = particiones.esquema_b(grupos)
    particiones.verificar_sin_sujetos_compartidos(folds_b, grupos)
    params_a, params_b = _leer_params(raiz, "A"), _leer_params(raiz, "B")

    print("\n1. Importancia por permutación", flush=True)
    _guardar(carpeta, ["perm_atributos_B", "perm_grupos_B"],
             lambda: importancia_permutacion(X, y, grupos, folds_b, params_b, "B"))
    _guardar(carpeta, ["perm_atributos_A", "perm_grupos_A"],
             lambda: importancia_permutacion(X, y, grupos, folds_a, params_a, "A"))

    print("\n2. Información mutua", flush=True)
    _guardar(carpeta, ["info_mutua_B"], lambda: informacion_mutua(X, y, grupos, folds_b))

    print("\n3. RFE en LOSO", flush=True)
    _guardar(carpeta, ["rfe_curva_B", "rfe_ranking_B"],
             lambda: curva_rfe(X, y, grupos, folds_b, params_b))

    print("\n4. PCA en LOSO", flush=True)
    _guardar(carpeta, ["pca_B"], lambda: con_pca(X, y, grupos, folds_b, params_b))

    print("\n5. Sin atributos ligados a la persona (temperatura y FC)", flush=True)
    _guardar(carpeta, ["sin_persona_B"],
             lambda: sin_atributos_de_persona(X, y, grupos, folds_b, params_b))
    print("\nListo.")
    return carpeta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--forzar", action="store_true", help="borra resultados previos")
    args = parser.parse_args()
    ejecutar(Path(__file__).resolve().parent.parent, forzar=args.forzar)
