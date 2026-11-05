# Práctica 3: Tarea de clasificación

**César Eduardo Jardines Mendoza**
Introducción a la Ciencia de Datos · Cuatrimestre III-2026 · CICESE

En esta práctica se construye un clasificador que predice, al cierre del primer semestre, la situación que tendrá un estudiante al final de la duración normal de su carrera: graduado (*graduate*), desertor (*dropout*) o todavía inscrito (*enrolled*). Se comparan cinco clasificadores (árbol de decisión, Naive Bayes, kNN, SVM y regresión logística), cada uno optimizado con validación cruzada estratificada, y con el mejor de ellos se generan las predicciones para el conjunto de prueba.

---

## Descripción del dataset

Para esta práctica se utilizó el dataset *Predict Students' Dropout and Academic Success*, disponible en el repositorio de aprendizaje automático de la UCI (https://archive.ics.uci.edu/dataset/697). Fue donado en diciembre del 2021 por investigadores del Instituto Politécnico de Portalegre, en Portugal, y se distribuye bajo la licencia CC BY 4.0.

Los datos corresponden a 4424 estudiantes de licenciatura que ingresaron entre los ciclos 2008/09 y 2018/19. Cada estudiante está descrito por 36 atributos que podemos agrupar en tres tipos de información: la que se conoce al momento de la inscripción (trayectoria académica previa, datos demográficos y factores socioeconómicos), el desempeño académico al cierre del primer y segundo semestre, y tres indicadores macroeconómicos (tasa de desempleo, inflación y PIB). Varios atributos nominales vienen codificados como números enteros, y el significado de cada código se encuentra en la ficha del dataset en la UCI.

Para la práctica el dataset viene dividido en dos archivos, ubicados en la raíz del repositorio junto al notebook:

| Archivo | Estudiantes | Contenido |
|---|---|---|
| `desercion-entrenamiento.csv` | 3539 | `id`, los 36 atributos y la variable objetivo `Target` |
| `desercion-prueba.csv` | 885 | `id` y solo los 30 atributos que se conocen al cierre del primer semestre, sin `Target` |

Las clases del conjunto de entrenamiento no están balanceadas: *Graduate* representa el 49.9 %, *Dropout* el 32.1 % y *Enrolled* el 17.9 %. No hay valores faltantes en ninguno de los dos archivos.

**Referencia**

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

---

## Estructura del repositorio

```
.
├── README.md
├── Práctica_3_Clasificación.ipynb   # Notebook con todo el análisis
├── desercion-entrenamiento.csv       # Conjunto de entrenamiento
├── desercion-prueba.csv              # Conjunto de prueba
└── predicciones.csv                  # Predicciones del clasificador final (generado por el notebook)
```

---

## Requisitos de ejecución

El notebook se desarrolló y ejecutó en Google Colab con las siguientes versiones:

| Librería | Versión |
|---|---|
| Python | 3.13.18 |
| numpy | 2.4.6 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |
| scikit-learn | 1.6.1 |
| scipy | incluida con scikit-learn |

Es importante considerar que con otras versiones de scikit-learn los resultados pueden variar ligeramente, y algunas funciones pueden mostrar advertencias de desuso. Para reproducir exactamente los resultados, se recomienda instalar las versiones indicadas:

```bash
pip install numpy==2.4.6 pandas==2.2.3 matplotlib==3.10.0 scikit-learn==1.6.1
```

La ejecución completa del notebook tarda aproximadamente entre 15 y 20 minutos en Google Colab, siendo las búsquedas de hiperparámetros del SVM y de la regresión logística las más lentas.

---

## Instrucciones de uso

### Opción 1: Google Colab

1. Abrir `Práctica_3_Clasificación.ipynb` en Google Colab.
2. En el panel de archivos (ícono de carpeta a la izquierda), subir los dos archivos CSV directamente, sin crear carpetas. Los archivos deben quedar en `/content/`.
3. Ejecutar todo el notebook desde el menú **Entorno de ejecución → Reiniciar y ejecutar todo**.
4. Al terminar, el archivo `predicciones.csv` aparece en el panel de archivos, en `/content/`. Para descargarlo, colocar el cursor sobre el archivo, hacer clic en los tres puntos y elegir **Descargar**.

Los archivos subidos a Colab se borran cuando la sesión se cierra, por lo que es necesario volver a subirlos en cada sesión nueva.

### Opción 2: Ejecución local

1. Clonar el repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>
   ```
2. Instalar las librerías indicadas en los requisitos.
3. Abrir el notebook con Jupyter desde la raíz del repositorio, para que las rutas relativas a los archivos CSV funcionen:
   ```bash
   jupyter notebook Práctica_3_Clasificación.ipynb
   ```
4. Ejecutar todo el notebook con **Kernel → Restart & Run All**. El archivo `predicciones.csv` se genera en la raíz del repositorio.

### Notas sobre la reproducibilidad

- Todas las rutas del notebook son relativas, por lo que debe ejecutarse desde la misma carpeta donde se encuentran los archivos CSV.
- La semilla de aleatoriedad está fijada en `SEED = 42` para los pliegues de la validación cruzada, las búsquedas aleatorias de hiperparámetros y los clasificadores que la utilizan.
- Los cinco clasificadores se evalúan con los mismos 5 pliegues estratificados, generados una sola vez al inicio.

---

## Contenido del notebook

El notebook está organizado en las siguientes fases:

1. **Carga y descripción de los datos:** distribución de las clases, tipo de cada atributo y momento en que se conoce.
2. **Validación cruzada y preprocesamiento común:** pliegues estratificados, preprocesamiento base y línea base.
3. **Árbol de decisión.**
4. **Naive Bayes.**
5. **k vecinos más cercanos (kNN).**
6. **Máquinas de soporte vectorial (SVM).**
7. **Regresión logística.**
8. **Comparación de los clasificadores:** F1 macro, sensibilidad y especificidad por clase, y tiempo de búsqueda.
9. **Matriz de confusión del mejor clasificador** y análisis de la clase *Dropout*.
10. **Estimación de la mejora al cierre del segundo semestre** y discusión sobre si conviene esperar.
11. **Clasificador final y predicciones** sobre el conjunto de prueba.
**Conclusiones.**

En cada clasificador el preprocesamiento se realiza dentro de un Pipeline, y los hiperparámetros se optimizan con búsqueda aleatoria usando el F1 macro como métrica.

---

## Resultados principales

| Clasificador | F1 macro (validación cruzada) |
|---|---|
| **SVM (kernel RBF)** | **0.6828 ± 0.0052** |
| Regresión logística | 0.6824 ± 0.0084 |
| Árbol de decisión | 0.6457 ± 0.0102 |
| Naive Bayes (Bernoulli) | 0.6202 ± 0.0099 |
| kNN | 0.5967 ± 0.0177 |

El clasificador final es el SVM con kernel RBF (C ≈ 25, gamma ≈ 0.0021 y pesos balanceados por clase). El archivo `predicciones.csv` contiene sus predicciones para los 885 estudiantes del conjunto de prueba, con exactamente dos columnas, `id` y `Target`, cuyos valores son `dropout`, `enrolled` o `graduate`.

---

## Uso de inteligencia artificial

En la elaboración de esta práctica se utilizó Claude (Anthropic) como herramienta de apoyo. La descripción de su uso se encuentra en la última sección del notebook.
