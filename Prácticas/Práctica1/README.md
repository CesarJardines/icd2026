# Análisis exploratorio: Matrícula de educación superior en México (ANUIES)

Práctica de la Maestría en Ciencia de Datos (ICD) — Análisis exploratorio 
de datos reales sobre matrícula universitaria en México.

## Descripción del dataset

Este dataset mide la matrícula de estudiantes inscritos en instituciones 
de educación superior en México, desagregada por carrera, campus, 
entidad federativa, sexo y grado académico.

Fue recolectado por la Asociación Nacional de Universidades e 
Instituciones de Educación Superior (ANUIES), a partir de la información 
que reportan todas las instituciones de educación superior del país 
—públicas y privadas— mediante el **Formato 911**, un instrumento 
estadístico promovido por la Secretaría de Educación Pública (SEP) al 
inicio de cada ciclo escolar.

Los datos utilizados en este análisis corresponden a los ciclos 
[2018–2022, según `df["Year"].min()` / 
`df["Year"].max()`].

**Unidad de observación:** cada fila representa un conteo agregado de 
estudiantes para una combinación única de carrera, campus, año, sexo, 
estado y grado académico. **No** es el registro de un estudiante 
individual.

**Fuente de acceso:** los datos se obtuvieron a través de la plataforma 
[DataMéxico](https://www.economia.gob.mx/datamexico) (Secretaría de 
Economía e INEGI), que funciona como distribuidor secundario de la 
información recolectada originalmente por ANUIES.
Enlace a la consulta específica: [COMPLETAR con tu enlace de DataMéxico]

### Estructura del dataset

| Atributo            | Descripción                                                                                     | Tipo         |
|---------------------|--------------------------------------------------------------------------------------------------|--------------|
| Career ID           | Identificador único (hash) asignado por DataMéxico a cada carrera. No es estable entre ciclos escolares. | Categórica |
| Career              | Nombre del programa académico o carrera.                                                          | Categórica |
| Campus ID           | Identificador numérico único del campus o plantel.                                                | Categórica |
| Campus              | Nombre del campus o plantel de la institución educativa.                                          | Categórica |
| Year                | Ciclo escolar de referencia del registro.                                                          | Numérica |
| Sex ID / Sex        | Código y descripción del sexo reportado.                                                           | Categórica (binaria) |
| State ID / State    | Clave y nombre de la entidad federativa del campus.                                                | Categórica |
| Academic Degree ID / Academic Degree | Código y descripción del grado académico del programa.                        | Categórica (ordinal) |
| Students            | Número total de estudiantes matriculados para esa combinación. **Variable objetivo** del análisis. | Numérica |

**Nota sobre licencia:** los Términos y Condiciones de uso de la 
plataforma restringen el uso comercial y la redistribución pública de 
los datos crudos. Por esta razón, **el archivo CSV original no se 
incluye en este repositorio**; ver instrucciones abajo para obtenerlo.

## Requisitos de ejecución

El notebook fue desarrollado y ejecutado en **Google Colab**, que ya 
incluye preinstaladas las librerías necesarias (Python 3, pandas, numpy, 
matplotlib, seaborn). No se requiere instalación adicional si se ejecuta 
ahí.

Para ejecutar el notebook en un entorno local en su lugar, instala:

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

Ver [`requirements.md`](./requirements.md) para el detalle de versiones.

## Instrucciones de uso

1. **Obtener los datos.** Descarga el CSV desde la consulta de 
   DataMéxico: [https://www.economia.gob.mx/datamexico/es/vizbuilder?cube=anuies_status&drilldowns%5B0%5D=Career+Fields.Career&drilldowns%5B1%5D=Campus&drilldowns%5B2%5D=Year&drilldowns%5B3%5D=Sex&drilldowns%5B4%5D=Geography+Municipality.Geography.State&drilldowns%5B5%5D=Academic+Degree&measures%5B0%5D=Students]. Exporta en formato CSV desde 
   la pestaña "Tabla de Datos" con "Resultado completo" activado.

   O bien ve a Datos/ y ahí encontraras `datos.csv
2. **Colocar el archivo.** Sube el CSV descargado a la misma carpeta 
   donde se ejecuta el notebook (en Colab, súbelo a la sesión, o 
   móntalo desde Google Drive).
3. **Ajustar la ruta de lectura.** Confirma que la celda `df = 
   pd.read_csv(...)` apunte al nombre exacto de tu archivo descargado.
4. **Ejecutar el notebook completo.** En Colab: `Entorno de ejecución > 
   Ejecutar todas`. Esto corre las celdas en orden, desde la carga de 
   librerías hasta el cierre con los hallazgos.
5. **Revisar el resultado.** Las gráficas y tablas quedan guardadas 
   dentro del notebook al exportarlo; no es necesario volver a correrlo 
   para verlas en GitHub.

## Referencias

Ver la sección de referencias dentro del notebook.

## Declaración de uso de IA

Ver la sección correspondiente dentro del notebook.