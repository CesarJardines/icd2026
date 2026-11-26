"""Esquemas de partición A, A′, B y C.

Todos los modelos usan exactamente estas particiones, definidas una sola vez.
Cada esquema tiene un ciclo externo (evaluación) y un ciclo interno
(búsqueda de hiperparámetros) que imita la misma separación, para no
reintroducir fuga al elegir hiperparámetros.

Las particiones se representan como listas de pares ``(entrenamiento,
prueba)`` con posiciones de fila (0..n-1), que es el formato que aceptan
``GridSearchCV(cv=...)`` y los ciclos manuales.
"""

from itertools import combinations

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold

N_FOLDS_A = 5
N_BLOQUES = 5
N_FOLDS_INTERNOS = 4


# ---------------------------------------------------------------- A: aleatoria
def esquema_a(y, seed):
    """Partición aleatoria estratificada de ventanas (5 folds).

    Parameters
    ----------
    y : array-like
        Actividad de cada ventana.
    seed : int

    Returns
    -------
    list of (numpy.ndarray, numpy.ndarray)
    """
    skf = StratifiedKFold(N_FOLDS_A, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def internos_a(y_entrenamiento, seed):
    """Ciclo interno de A: estratificado y aleatorio, igual que el externo."""
    skf = StratifiedKFold(N_FOLDS_INTERNOS, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y_entrenamiento)), y_entrenamiento))


# ------------------------------------------------------- A′: bloques temporales
def asignar_bloques(meta, n_bloques=N_BLOQUES):
    """Divide cada segmento contiguo en ``n_bloques`` tramos temporales.

    Parameters
    ----------
    meta : pandas.DataFrame
        Metadatos de ventana con ``segmento`` y ``posicion``.
    n_bloques : int

    Returns
    -------
    numpy.ndarray
        Número de bloque (0..n_bloques-1) de cada ventana.
    """
    tamaño = meta.groupby("segmento")["posicion"].transform("size").to_numpy()
    # El bloque crece con la posición: las primeras ventanas del segmento van al
    # bloque 0, las últimas al bloque n_bloques-1
    return np.minimum(meta["posicion"].to_numpy() * n_bloques // tamaño, n_bloques - 1)


def _purgar_traslape(meta, entrenamiento, prueba):
    """Quita del entrenamiento las ventanas que comparten muestras con la prueba.

    Dos ventanas comparten muestras si están en el mismo segmento y sus
    intervalos ``[inicio, fin)`` se cruzan; con 50 % de traslape eso solo
    pasa con la ventana inmediatamente anterior o posterior.
    """
    inicio = meta["inicio"].to_numpy()
    fin = meta["fin"].to_numpy()
    segmento = meta["segmento"].to_numpy()
    prohibidas = np.zeros(len(meta), dtype=bool)
    for seg in np.unique(segmento[prueba]):
        en_prueba = prueba[segmento[prueba] == seg]
        candidatas = entrenamiento[segmento[entrenamiento] == seg]
        ini_p, fin_p = inicio[en_prueba], fin[en_prueba]
        for c in candidatas:
            if np.any((inicio[c] < fin_p) & (ini_p < fin[c])):
                prohibidas[c] = True
    return entrenamiento[~prohibidas[entrenamiento]]


def particion_por_bloques(meta, bloques):
    """Un fold por bloque: prueba = ese bloque de todos los segmentos.

    Parameters
    ----------
    meta : pandas.DataFrame
        Metadatos de las ventanas consideradas (posiciones 0..n-1).
    bloques : numpy.ndarray
        Bloque de cada ventana.

    Returns
    -------
    list of (numpy.ndarray, numpy.ndarray)
    """
    todas = np.arange(len(meta))
    folds = []
    for b in np.unique(bloques):
        prueba = todas[bloques == b]
        entrenamiento = _purgar_traslape(meta, todas[bloques != b], prueba)
        folds.append((entrenamiento, prueba))
    return folds


def esquema_a_prima(meta):
    """Esquema A′: 5 bloques temporales por segmento, sin ventanas vecinas."""
    return particion_por_bloques(meta.reset_index(drop=True), asignar_bloques(meta))


def internos_a_prima(meta, entrenamiento):
    """Ciclo interno de A′: un fold por cada bloque restante, también purgado."""
    sub = meta.iloc[entrenamiento].reset_index(drop=True)
    return particion_por_bloques(sub, asignar_bloques(meta)[entrenamiento])


# ---------------------------------------------------------------- B: LOSO
def esquema_b(grupos):
    """Leave-One-Subject-Out: un fold por sujeto (9 folds)."""
    logo = LeaveOneGroupOut()
    return list(logo.split(np.zeros(len(grupos)), groups=grupos))


def internos_por_sujeto(grupos_entrenamiento, n_folds=N_FOLDS_INTERNOS):
    """Ciclo interno de B y C: GroupKFold por sujeto."""
    n_folds = min(n_folds, len(np.unique(grupos_entrenamiento)))
    gkf = GroupKFold(n_folds)
    return list(gkf.split(np.zeros(len(grupos_entrenamiento)), groups=grupos_entrenamiento))


# ------------------------------------------------------- C: hold-out de 3 sujetos
def elegir_sujetos_c(meta, seed, n=3, excluir=(109,)):
    """Elige, con semilla fija, los sujetos de prueba del esquema C.

    Se consideran todas las combinaciones de ``n`` sujetos (sin los de
    ``excluir``) que incluyan al menos un sujeto con saltar cuerda y uno con
    correr, y se sortea una con ``seed``. La elección se hace antes de
    entrenar cualquier modelo y no se cambia después.

    Returns
    -------
    tuple of int
    """
    con = lambda act: set(meta.loc[meta["actividad"] == act, "sujeto"])  # noqa: E731
    saltar, correr = con("saltar cuerda"), con("correr")
    candidatos = sorted(set(meta["sujeto"]) - set(excluir))
    validas = [c for c in combinations(candidatos, n)
               if set(c) & saltar and set(c) & correr]
    rng = np.random.default_rng(seed)
    return tuple(int(s) for s in validas[rng.integers(len(validas))])


def esquema_c(grupos, sujetos_prueba):
    """Hold-out fijo: entrena con 6 sujetos y prueba con los 3 apartados."""
    es_prueba = np.isin(grupos, sujetos_prueba)
    return [(np.flatnonzero(~es_prueba), np.flatnonzero(es_prueba))]


# ---------------------------------------------------------------- verificaciones
def verificar_sin_sujetos_compartidos(folds, grupos):
    """Falla si algún sujeto aparece en entrenamiento y prueba del mismo fold."""
    for k, (tr, te) in enumerate(folds):
        comunes = set(grupos[tr]) & set(grupos[te])
        assert not comunes, f"Fold {k}: sujetos {comunes} en ambos lados"


def verificar_sin_traslape(folds, meta):
    """Falla si alguna ventana de prueba comparte muestras con una de entrenamiento."""
    m = meta.reset_index(drop=True)
    for k, (tr, te) in enumerate(folds):
        for seg in np.intersect1d(m["segmento"].to_numpy()[tr], m["segmento"].to_numpy()[te]):
            a = m.iloc[tr][m["segmento"].to_numpy()[tr] == seg]
            b = m.iloc[te][m["segmento"].to_numpy()[te] == seg]
            cruce = (a["inicio"].to_numpy()[:, None] < b["fin"].to_numpy()[None, :]) & \
                    (b["inicio"].to_numpy()[None, :] < a["fin"].to_numpy()[:, None])
            assert not cruce.any(), f"Fold {k}: traslape en el segmento {seg}"
