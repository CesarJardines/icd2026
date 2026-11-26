"""Fase 5: réplica de P1 en UCI HAR (esquemas A, A′ y B).

    conda activate pamap2-fuga
    python -m src.experimentos_har         # desde la raíz del repositorio

Usa exactamente los mismos modelos, rejillas, búsqueda anidada y funciones
de partición que la Fase 3 (``src/experimentos.py`` y
``src/particiones.py``), para que la comparación entre datasets sea justa.
Modelos: las dos líneas base, Random Forest y kNN (sin balanceo: en PAMAP2
los pesos por clase no ayudaron, y UCI HAR está casi balanceado).
Resultados en ``results/fase5/``; si un CSV ya existe, se omite.
"""

import argparse
import json
import shutil
import time
import warnings
from pathlib import Path

from sklearn.exceptions import UndefinedMetricWarning

from src import datos_har, particiones
from src.experimentos import SEED, evaluar, f1_medio_por_fold

MODELOS_HAR = ["dummy_frecuente", "dummy_estratificado", "rf", "knn"]


def preparar_esquemas_har(y, grupos, meta, seed=SEED):
    """Folds externos e internos de A, A′ y B, con las mismas verificaciones."""
    esquemas = {
        "A": {"folds": particiones.esquema_a(y, seed),
              "internos": lambda tr: particiones.internos_a(y[tr], seed)},
        "A′": {"folds": particiones.esquema_a_prima(meta),
               "internos": lambda tr: particiones.internos_a_prima(meta, tr)},
        "B": {"folds": particiones.esquema_b(grupos),
              "internos": lambda tr: particiones.internos_por_sujeto(grupos[tr])},
    }
    particiones.verificar_sin_sujetos_compartidos(esquemas["B"]["folds"], grupos)
    particiones.verificar_sin_traslape(esquemas["A′"]["folds"], meta)
    return esquemas


def ejecutar(raiz, forzar=False):
    """Corre los modelos en A, A′ y B sobre UCI HAR y guarda las predicciones."""
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    carpeta = Path(raiz) / "results" / "fase5"
    if forzar and carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    X, y, grupos, meta = datos_har.cargar_har(raiz)
    esquemas = preparar_esquemas_har(y, grupos, meta)
    print(f"UCI HAR: X = {X.shape}, {len(set(grupos))} sujetos, "
          f"{meta['segmento'].nunique()} segmentos", flush=True)

    for esquema in ["A", "A′", "B"]:
        for nombre in MODELOS_HAR:
            archivo = carpeta / f"pred_{esquema}_{nombre}_ninguno.csv"
            if archivo.exists():
                print(f"  ya existe: {archivo.name}")
                continue
            t0 = time.time()
            pred, params = evaluar(nombre, "ninguno", esquema, X, y, grupos, esquemas)
            pred.to_csv(archivo, index=False)
            (carpeta / f"params_{esquema}_{nombre}_ninguno.json").write_text(
                json.dumps(params, ensure_ascii=False, indent=1, default=str))
            print(f"  {esquema:3} {nombre:20} F1 fold = {f1_medio_por_fold(pred):.3f}"
                  f"   ({time.time() - t0:.0f} s)", flush=True)
    print("\nListo.")
    return carpeta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--forzar", action="store_true", help="borra resultados previos")
    args = parser.parse_args()
    ejecutar(Path(__file__).resolve().parent.parent, forzar=args.forzar)
