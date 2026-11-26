# Fuga de información en el reconocimiento de actividades humanas (PAMAP2)

Proyecto final del curso **Introducción a la Ciencia de Datos 2026** (CICESE).
Autor: César Eduardo Jardines Mendoza.

## Descripción

Los modelos de reconocimiento de actividades humanas con sensores inerciales suelen evaluarse con una **partición aleatoria de ventanas**. Como las ventanas se traslapan y cada persona aporta cientos de ellas, ventanas casi idénticas del mismo sujeto quedan en entrenamiento y en prueba, y el desempeño se infla. Este proyecto mide esa inflación en PAMAP2 comparando cuatro esquemas de partición:

| Esquema | Qué hace |
|---|---|
| **A** | Ventanas al azar (`StratifiedKFold`, 5 folds) |
| **A′** | Bloques temporales dentro de cada sujeto, quitando la ventana vecina de cada bloque de prueba |
| **B** | Un sujeto de prueba por fold (LOSO, 9 folds) |
| **C** | Hold-out fijo de 3 sujetos (101, 102 y 107) |

Responde tres preguntas:

- **P1.** ¿Cuánto cae el F1 macro de A a B, y qué parte de esa caída es fuga por ventanas vecinas y qué parte es diferencia entre personas?
- **P2.** ¿Qué actividades son las más afectadas?
- **P3.** ¿Qué características siguen siendo útiles con personas nuevas?

Los modelos son dos líneas base (`DummyClassifier`), Random Forest, SVM (RBF) y kNN, con búsqueda anidada de hiperparámetros. Además, P1 se replica en un segundo dataset (UCI HAR).

El entregable principal es [`notebooks/proyecto_pamap2.ipynb`](notebooks/proyecto_pamap2.ipynb), que se entrega ejecutado con todas sus salidas.

## Datos

Los datos **no están en el repositorio**: PAMAP2 pesa unos 1.6 GB y UCI HAR unos 270 MB. Hay que descargarlos y descomprimirlos en `data/raw/`.

### PAMAP2 (dataset principal)

- **Fuente:** [PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring), UCI Machine Learning Repository. DOI: [10.24432/C5NW2H](https://doi.org/10.24432/C5NW2H).
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Cita:** Reiss, A. y Stricker, D. (2012). PAMAP2 Physical Activity Monitoring [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NW2H
- **Contenido:** 9 sujetos con tres sensores inerciales (muñeca, pecho y tobillo) a 100 Hz y un monitor de frecuencia cardiaca, haciendo 12 actividades.
- **Instalación:** descarga el zip, descomprímelo y deja la carpeta `Protocol/` en

  ```
  data/raw/PAMAP2_Dataset/Protocol/subject101.dat … subject109.dat
  ```

  Solo se usa `Protocol/`; la carpeta `Optional/` no hace falta.

### UCI HAR (segundo dataset, §6b del notebook)

- **Fuente:** [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), UCI Machine Learning Repository. DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K).
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Cita:**
  - Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L. y Parra, X. (2013). Human Activity Recognition Using Smartphones [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K
  - Anguita, D., Ghio, A., Oneto, L., Parra, X. y Reyes-Ortiz, J. L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. *European Symposium on Artificial Neural Networks (ESANN)*.
- **Contenido:** 30 sujetos con un teléfono en la cintura, 6 actividades, 10 299 ventanas de 2.56 s y 561 características.
- **Instalación:** descarga el zip (que contiene otro zip adentro), descomprímelo y deja la carpeta en

  ```
  data/raw/UCI HAR Dataset/train/…  y  data/raw/UCI HAR Dataset/test/…
  ```

Si falta alguna de las carpetas, el notebook se detiene con un mensaje que remite a este README.

## Requisitos e instalación

Se usa Python 3.11 con las versiones fijadas en [`environment.yml`](environment.yml): numpy, pandas, scipy, scikit-learn, matplotlib, seaborn y jupyter. No se usa `imbalanced-learn`.

```bash
conda env create -f environment.yml
conda activate pamap2-fuga
python -m ipykernel install --user --name pamap2-fuga --display-name "Python (pamap2-fuga)"
```

## Cómo ejecutar

Los experimentos pesados (entrenar todos los modelos en todos los folds, con búsqueda anidada) corren en scripts de `src/` y guardan sus resultados en `results/`. El notebook carga esos resultados y, si no existen, llama al script correspondiente. Así el notebook completo corre en menos de un minuto.

Desde la raíz del repositorio:

```bash
conda activate pamap2-fuga
python -m src.experimentos        # Fase 3: esquemas A, A′, B y C, y balanceo (≈ 10 min)
python -m src.experimentos_p3     # Fase 4: importancia, información mutua, RFE y PCA (≈ 5 min)
python -m src.experimentos_har    # Fase 5: réplica en UCI HAR (≈ 25 min)
```

Los tiempos corresponden a un Apple M4 con 10 núcleos. Cada script omite los resultados que ya existen; para repetir todo desde cero, usa `--forzar`.

Después abre `notebooks/proyecto_pamap2.ipynb` con el kernel `Python (pamap2-fuga)` y ejecuta **Kernel → Restart & Run All**. Para recalcular desde el notebook en lugar de usar los scripts, cambia a `True` las banderas `RECALCULAR` (Fase 3), `RECALCULAR_P3` o `RECALCULAR_HAR` de las celdas correspondientes.

La semilla es `SEED = 42` en todo lo aleatorio, y todas las rutas son relativas a la raíz del repositorio.

## Estructura

```
pamap2-fuga/
├── README.md
├── environment.yml              # entorno conda con versiones fijadas
├── notebooks/
│   └── proyecto_pamap2.ipynb    # entregable principal (ejecutado)
├── src/
│   ├── datos.py                 # carga y limpieza de PAMAP2
│   ├── ventanas.py              # segmentación en ventanas de 5.12 s
│   ├── caracteristicas.py       # 196 características por ventana
│   ├── particiones.py           # esquemas A, A′, B y C, y sus verificaciones contra la fuga
│   ├── evaluacion.py            # métricas, resúmenes y utilidades de gráficas
│   ├── experimentos.py          # Fase 3: modelos × esquemas, balanceo
│   ├── experimentos_p3.py       # Fase 4: experimentos de P3
│   ├── datos_har.py             # carga de UCI HAR y reconstrucción del orden temporal
│   └── experimentos_har.py      # Fase 5: réplica en UCI HAR
├── data/                        # no se sube (ver «Datos»)
│   └── raw/
├── figures/                     # figuras PNG (200 dpi) para el reporte y la presentación
└── results/                     # tablas CSV; fase3/, fase4/ y fase5/ guardan las predicciones por fold
```

## Uso de IA generativa

Se usaron Claude y Claude Code (Anthropic) para planear, escribir y revisar código y redactar explicaciones. El detalle está en la última celda del notebook.
