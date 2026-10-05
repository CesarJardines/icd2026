
# Procesamiento de datos: Acceso y permanencia en la educación en México (ENAPE 2021)


## Descripción del dataset

Este dataset proviene de la **Encuesta Nacional sobre Acceso y Permanencia en la
Educación (ENAPE) 2021**, levantada por el Instituto Nacional de Estadística y
Geografía (**INEGI**) en coordinación con la Secretaría de Educación Pública
(**SEP**). Su objetivo es generar información estadística sobre las
características educativas de la población de **0 a 29 años** en México, con
especial atención en el grupo de 3 a 29 años.

La encuesta capta, entre otros temas, la inscripción en los ciclos escolares
2020-2021 y 2021-2022, las razones de no inscripción y de abandono escolar, las
condiciones de estudio durante la pandemia, el equipamiento tecnológico de la
vivienda y la valoración de la educación en el hogar.

**Periodo:** el levantamiento se realizó en **noviembre de 2021**, por vía
telefónica, debido a la contingencia por COVID-19. Los ciclos escolares de
referencia son **2020-2021** y **2021-2022**.

**Unidad de observación:** los microdatos se distribuyen en dos tablas
relacionadas de uno a muchos:

| Tabla | Unidad de observación | Llave | Núm. de variables |
|-------|-----------------------|-------|-------------------|
| `tvivienda` | Vivienda | `FOLIO` | 17 |
| `tmodulo` | Persona de 0 a 29 años | `FOLIO` + `N_REN` | 101 |

**Fuente de acceso:** los microdatos se descargaron directamente del sitio del
INEGI: [https://www.inegi.org.mx/programas/enape/2021/](https://www.inegi.org.mx/programas/enape/2021/).
La documentación (diccionario de datos, cuestionario y diseño muestral) está
disponible en la [Red Nacional de Metadatos del INEGI](https://www.inegi.org.mx/rnm/index.php/catalog/832).

### Pregunta de análisis y población de estudio

La variable objetivo es la inscripción en el ciclo escolar 2021-2022 (`PB3_1`),
recodificada como `NO_INSCRITO` (1 = no inscrito, 0 = inscrito). La pregunta
que guía el procesamiento es:

> ¿Qué características de la persona, de su hogar y de su trayectoria en el ciclo
> escolar anterior se asocian con no estar inscrita(o) en el ciclo 2021-2022?

Tras el análisis exploratorio, la población se acotó a personas de **6 a 17
años** (14,350 registros), rango que corresponde a la edad típica de cursar
primaria, secundaria y media superior. En este grupo, el **6.6%** no estaba
inscrito, lo que representa un desbalance marcado entre clases.

### Estructura del dataset (variables principales utilizadas)

| Atributo | Descripción | Tipo |
|----------|-------------|------|
| `FOLIO` / `N_REN` | Identificadores de la vivienda y de la persona dentro de la vivienda. | Identificador |
| `SEXO` | Sexo de la persona. | Categórica (binaria) |
| `EDAD` | Edad en años cumplidos. | Numérica |
| `ENT` | Entidad federativa. | Categórica |
| `PA3_*` | Apartado A: inscripción, tipo de escuela, conclusión del grado, regularización y formas de evaluación en el ciclo 2020-2021. | Categórica |
| `NIVEL_A` / `GRADO_A` | Nivel y grado escolar del ciclo 2020-2021. | Categórica |
| `PB3_1` | ¿Está inscrita(o) en el ciclo 2021-2022? **Variable objetivo** del análisis. | Categórica (binaria) |
| `P1_1`, `P1_2_1`, `P1_2_2` | Número de personas, hombres y mujeres en la vivienda. | Numérica |
| `P1_4_1` a `P1_4_6` | Disponibilidad de computadora, laptop, TV digital, tablet, smartphone e internet fijo. | Categórica (binaria) |
| `P1_5` | Razón por la que no disponen de internet fijo. | Categórica |
| `P4_1_1` a `P4_1_3` | Grado de acuerdo con frases sobre el valor de la educación. | Categórica (ordinal) |
| `FACTOR` | Factor de expansión de la muestra. | Numérica |

La descripción completa de las variables se encuentra en los archivos
`diccionario_datos_tmodulo_enape_2021.csv` y
`diccionario_datos_tvivienda_enape_2021.csv`, incluidos en la descarga del INEGI.

**Nota sobre licencia:** la información del INEGI se rige por los
[Términos de Libre Uso de la Información del INEGI](https://www.inegi.org.mx/inegi/terminos.html),
que permiten su uso y redistribución siempre que se otorguen los créditos
correspondientes al INEGI como autor y no se aparente que el uso representa una
postura oficial del Instituto.

**Fuente: INEGI. Encuesta Nacional sobre Acceso y Permanencia en la Educación
(ENAPE) 2021. Microdatos.**

## Requisitos de ejecución

El notebook fue desarrollado y ejecutado en **Google Colab**, que ya incluye
preinstaladas las librerías necesarias (Python 3, pandas, numpy, matplotlib,
scikit-learn, scipy e imbalanced-learn). Si `imblearn` no estuviera disponible,
puede instalarse desde una celda con:

```bash
!pip install imbalanced-learn
```

Para ejecutar el notebook en un entorno local, instala:

```bash
pip install pandas numpy matplotlib scikit-learn scipy imbalanced-learn jupyter
```

## Instrucciones de uso

1. **Obtener los datos.** Descarga los microdatos en formato CSV desde la
   sección de *Datos abiertos* de la ENAPE 2021:
   [https://www.inegi.org.mx/programas/enape/2021/](https://www.inegi.org.mx/programas/enape/2021/).
   La descarga contiene dos carpetas: `conjunto_de_datos_tmodulo_enape_2021` y
   `conjunto_de_datos_tvivienda_enape_2021`.
2. **Colocar los archivos.** Sube a la misma carpeta donde se ejecuta el
   notebook (en Colab, súbelos a la sesión o móntalos desde Google Drive)
   los siguientes archivos:
   - **Datos:** `conjunto_de_datos_tmodulo_enape_2021.csv` y
     `conjunto_de_datos_tvivienda_enape_2021.csv`, ubicados en la subcarpeta
     `conjunto_de_datos` de cada tabla.
   - **Catálogos:** `pa3_7_1.csv` y `grado_a.csv` (de la carpeta `catalogos`
     de `tmodulo`), y `p4_1_1.csv` y `p1_5.csv` (de la carpeta `catalogos` de
     `tvivienda`). Se utilizan para identificar los códigos especiales
     ("No sabe", "No responde", etc.).
3. **Ajustar las rutas de lectura.** Confirma que las celdas con
   `pd.read_csv(...)` apunten a la ubicación de tus archivos. Los datos se leen
   como texto (`dtype=str`) para conservar los ceros a la izquierda de `FOLIO`.
4. **Ejecutar el notebook completo.** En Colab: `Entorno de ejecución >
   Ejecutar todas`. Las celdas deben ejecutarse **en orden**, ya que cada etapa
   del procesamiento modifica los datos de la anterior.
5. **Revisar el resultado.** Las gráficas y tablas quedan guardadas dentro del
   notebook al exportarlo; no es necesario volver a ejecutarlo para verlas en
   GitHub.

## Técnicas aplicadas y resultados principales

| Técnica | Método | Antes → Después | Hallazgo principal |
|---------|--------|-----------------|--------------------|
| Limpieza de datos | Eliminación de variables (sin información, redundantes y con fuga de información), reglas de flujo del cuestionario e imputación por moda | 116 → 40 variables; 470,403 → 0 celdas vacías | El 100% de los vacíos restantes se explicó por 5 reglas del cuestionario; solo 475 valores se imputaron |
| Extracción de características | Índices y razones construidos a partir de variables del hogar | +4 variables | El índice de equipamiento muestra un gradiente claro: de 18.5% de no inscripción (0 equipos) a 1% (6 equipos) |
| Selección de características | Información mutua con prueba de permutación, V de Cramér y relevancia propia | 39 → 25 variables | Varias variables debían su relevancia solo a la categoría "No aplica" |
| Reducción de dimensionalidad | PCA (90% de varianza explicada) | 90 → 60 dimensiones | PC1 refleja la trayectoria escolar del ciclo anterior y PC2 el nivel escolar |
| Aumento de datos | SMOTE, aplicado solo al conjunto de entrenamiento | 763 → 10,717 no inscritos | Clases balanceadas, con algunos puntos sintéticos fuera de las regiones con datos reales |

## Referencias

Ver la sección de referencias dentro del notebook.

## Declaración de uso de IA

Ver la sección correspondiente dentro del notebook.
