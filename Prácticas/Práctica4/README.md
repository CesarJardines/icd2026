# Práctica 4: Ensambles de clasificadores

César Eduardo Jardines Mendoza

Cuatrimestre III-2026

CICESE

## Descripción

En esta práctica revisamos cuánto ganan los ensambles (*bagging*, *boosting*, votación y *stacking*) sobre los cinco clasificadores base de la Práctica 3 (árbol de decisión, Naive Bayes, kNN, SVM y regresión logística) cuando estos ya están bien ajustados. Después buscamos el mejor modelo para el conjunto *Bank Marketing* de la Tarea 4.

Todo el trabajo está en el notebook `Práctica_4_Ensambles.ipynb`, que tiene dos partes:

- **Parte 1. Deserción estudiantil.** Clasificadores base, *bagging*, *AdaBoost*, votación dura y suave, *stacking*, *Random Forest*, *Extra-Trees*, XGBoost y LightGBM, evaluados con los mismos cinco pliegues de la Práctica 3.
- **Parte 2. *Bank Marketing*.** Búsqueda del mejor modelo, comparación con lo que funcionó en la deserción y conclusiones.

## Conjuntos de datos

### Deserción estudiantil

Es el conjunto *Predict Students' Dropout and Academic Success* de la UCI, el mismo de la Práctica 3. Cada renglón es un estudiante de licenciatura y el objetivo es predecir si va a desertar, seguir inscrito o graduarse.

| Archivo | Instancias | Contenido |
|---|---|---|
| `desercion-entrenamiento.csv` | 3539 | `id`, 36 atributos y la clase `Target` |
| `desercion-prueba.csv` | 885 | `id` y los 30 atributos que se conocen al cierre del primer semestre |

- **Clase:** `Target`, con tres valores: *Dropout* (1137 estudiantes), *Enrolled* (635) y *Graduate* (1767). *Enrolled* es la clase minoritaria y la más difícil de predecir.
- **Atributos utilizados:** los 30 que se conocen al cierre del primer semestre (9 nominales, 8 binarios y 13 numéricos), igual que en la Práctica 3. El archivo de prueba solo se usa para obtener esta lista de atributos.
- **Evaluación:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, los mismos pliegues de la Práctica 3.

### *Bank Marketing*

Es el conjunto *Bank Marketing* de la UCI, el mismo de la Tarea 4. Cada renglón es una llamada de una campaña de telemarketing de un banco portugués (de mayo de 2008 a noviembre de 2010), y el objetivo es predecir si el cliente acepta un depósito a plazo.

| Archivo | Instancias | Contenido |
|---|---|---|
| `bank-additional-full.csv` | 41,188 | 20 atributos y la clase `y`, separados por `;` |

- **Clase:** `y`, que se convierte a 1 (*yes*, 4640 clientes, 11.3 %) y 0 (*no*, 36,548 clientes).
- **Atributos utilizados:** los 19 restantes al quitar `duration`, ya que la duración de la llamada solo se conoce después de hacerla. De estos, 10 son categóricos (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`) y 9 son numéricos (datos del cliente, de la campaña y cinco indicadores económicos).
- **Evaluación:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)`, como pide la consigna.
- **Nota:** el archivo está ordenado por fecha, por lo que cualquier validación interna (por ejemplo en el *stacking* o en el ajuste del umbral) debe revolver los datos. Esto se explica en la fase 9 del notebook.

El archivo `bank-additional-names.txt` tiene la descripción original de cada atributo.

## Requisitos

El notebook se ejecutó con las siguientes versiones:

| Librería | Versión |
|---|---|
| Python | 3.13.5 |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| lightgbm | 4.7.0 |
| joblib | 1.6.0 |

Para instalarlas:

```bash
pip install numpy==2.5.4 pandas==3.1.1 scipy==1.18.1 matplotlib==3.11.2 scikit-learn==1.9.1 xgboost==3.4.1 lightgbm==4.7.0 joblib==1.6.0
```

La primera celda de código del notebook imprime las versiones con las que se está ejecutando. Con otras versiones los resultados pueden variar un poco (por ejemplo, Naive Bayes depende de cómo `KBinsDiscretizer` calcula los cuantiles), y los tiempos de entrenamiento dependen de la computadora.

## Uso

1. Colocar en la misma carpeta el notebook y los tres archivos de datos:

   ```
   Práctica_4_Ensambles.ipynb
   desercion-entrenamiento.csv
   desercion-prueba.csv
   bank-additional-full.csv
   ```

   El notebook usa rutas relativas, por lo que no hay que cambiar ninguna ruta. En Google Colab, los archivos se suben al panel de archivos (carpeta `/content`).

2. Ejecutar el notebook completo con **Restart & Run All** (en Colab, *Entorno de ejecución → Reiniciar y ejecutar todo*). Las semillas están fijas, entonces el F1 macro de cada modelo debe salir igual en cada ejecución. La ejecución completa tarda aproximadamente entre 30 y 45 minutos, sobre todo por las búsquedas de hiperparámetros.

3. Al terminar, el notebook guarda el mejor modelo de *Bank Marketing* en `cesar-jardines_modelo.joblib`.

### Uso del modelo guardado

El modelo final es LightGBM con pesos balanceados y el umbral ajustado. Como el preprocesamiento y el umbral están dentro del modelo, basta con pasarle las columnas originales del archivo sin `duration` ni `y`:

```python
import joblib
import pandas as pd

modelo = joblib.load("cesar-jardines_modelo.joblib")

datos = pd.read_csv("bank-additional-full.csv", sep=";")
X = datos.drop(columns=["duration", "y"])

predicciones = modelo.predict(X)  # 1 = yes, 0 = no
```

## Resultados principales

| Conjunto | Mejor modelo | F1 macro |
|---|---|---|
| Deserción estudiantil | XGBoost | 0.6934 ± 0.0155 |
| *Bank Marketing* | LightGBM con umbral ajustado | 0.7192 ± 0.0076 |

En la deserción los ensambles ganan muy poco sobre los clasificadores base ya ajustados (XGBoost supera al SVM por 0.011, dentro de la desviación entre pliegues). En *Bank Marketing* los ensambles de árboles sí superan claramente a la regresión logística. La discusión completa está en las conclusiones del notebook.
