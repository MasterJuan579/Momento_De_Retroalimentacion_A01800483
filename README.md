# Momento de Retroalimentación — Módulo 2

**Curso:** Inteligencia artificial avanzada para la ciencia de datos (Gpo 601)
**Profesor:** Jorge Adolfo Ramírez Uresti
**Autor:** Juan Pablo Pérez Gutiérrez — **A01800483**

Este repositorio contiene las dos entregas del módulo, ambas sobre el mismo problema
(**Spaceship Titanic**: predecir si un pasajero fue transportado a otra dimensión), lo que permite
compararlas de forma directa.

| Entrega | Enfoque | Código | Reporte |
|---|---|---|---|
| **Parte I** — sin framework | Árbol CART y Random Forest programados a mano | [`main.py`](main.py), [`src/`](src) | [REPORTE.md](REPORTE.md) |
| **Parte II** — con framework | scikit-learn: Pipeline, ColumnTransformer y GridSearchCV | [`main_framework.py`](main_framework.py) | [Reporte_M2_Parte2_Framework.pdf](Reporte_M2_Parte2_Framework.pdf) |

> **Hallazgo de la comparación:** entrenadas con la misma configuración y los mismos datos, ambas
> implementaciones alcanzan **exactamente la misma exactitud (0.7914)**, pero scikit-learn entrena
> 39 veces más rápido. El código propio explica *por qué* funciona el algoritmo; el framework es lo
> que permite iterar.

---

# Parte I — Árbol de Decisión y Random Forest sin framework

Indicador **SMA0401A**: implementación de una técnica de aprendizaje máquina sin uso de framework.

## Qué es esto

Una implementación **manual** de un árbol de decisión CART y de un Random Forest construido sobre él,
aplicada al dataset **Spaceship Titanic** para predecir si un pasajero fue transportado a otra dimensión.

Ninguna parte del aprendizaje viene de una librería: el índice de Gini, la búsqueda del mejor punto de
corte, la construcción recursiva del árbol, la poda, el *bagging*, el subespacio aleatorio de
características, la votación del ensamble, la matriz de confusión y todas las métricas están escritas
línea por línea en este repositorio.

## Resultados sobre el conjunto de prueba

| Modelo | Exactitud | F1 | AUC |
|---|---|---|---|
| Línea base (clase mayoritaria) | 0.5031 | — | — |
| Árbol de decisión (profundidad 6) | 0.7738 | 0.7820 | 0.8573 |
| **Random Forest (100 árboles)** | **0.8029** | **0.8043** | **0.8886** |

El análisis completo está en [REPORTE.md](REPORTE.md).

---

## Cómo ejecutarlo

Requiere Python 3.9 o superior.

```bash
pip install -r requirements.txt
```

### Entrenar y evaluar (genera todas las gráficas y métricas)

```bash
python main.py --entrenar
```

Tarda alrededor de 40 segundos. Deja en `resultados/` las gráficas, la bitácora completa de la
corrida y el modelo entrenado.

### Predecir un pasajero desde la consola

```bash
python main.py --predecir
```

Pregunta los datos del pasajero uno por uno y responde con la clase predicha y su probabilidad.
Cualquier campo que se deje vacío se imputa con el valor típico del entrenamiento.

### Predecir un archivo completo

```bash
python main.py --predecir-csv data/test.csv
```

Escribe `resultados/predicciones.csv` con una predicción y una probabilidad por fila.

### Ver las reglas que aprendió el árbol

```bash
python main.py --arbol --niveles 3
```

### Verificar que la implementación es correcta

```bash
python verificar.py
```

Corre 44 pruebas que comparan la salida del código contra resultados conocidos de antemano: valores
exactos del índice de Gini y de la entropía, recuperación de una regla sintética que se conoce por
construcción, respeto de los criterios de paro, validez de las probabilidades, métricas calculadas a
mano sobre una matriz de confusión de ejemplo, y reproducibilidad con semilla fija.

### Opciones adicionales

```bash
python main.py --entrenar --arboles 200 --criterio entropia --semilla 7
```

`--help` lista todas las opciones disponibles.

---

## Estructura del proyecto

```
├── main.py               Punto de entrada: entrenar, predecir, inspeccionar
├── verificar.py          Pruebas de correccion del algoritmo
├── src/
│   ├── data_loader.py    Carga, imputacion, ingenieria de caracteristicas, particion
│   ├── decision_tree.py  Arbol CART desde cero  <- el algoritmo principal
│   ├── random_forest.py  Bagging, subespacio aleatorio, votacion, out-of-bag
│   ├── metrics.py        Matriz de confusion, exactitud, precision, recall, F1, AUC
│   └── plots.py          Graficas (solo visualizacion)
├── data/                 Dataset Spaceship Titanic
├── resultados/           Salida generada por --entrenar
└── REPORTE.md            Reporte de resultados
```

El proyecto corre con un intérprete de Python normal. No depende de un IDE ni de un notebook.

---

## Sobre las dependencias

La actividad prohíbe usar librerías que ya traigan el algoritmo implementado. Este proyecto **no usa
ninguna**. Las tres dependencias que sí utiliza cumplen funciones que no forman parte del aprendizaje:

| Librería | Para qué se usa | Para qué **no** se usa |
|---|---|---|
| `pandas` | Leer los CSV y construir las columnas derivadas | Nada relacionado con el modelo |
| `numpy` | Guardar los datos en arreglos y hacer aritmética (`sum`, `argsort`, `cumsum`) | No se usa ninguna función que resuelva parte del algoritmo |
| `matplotlib` | Dibujar las gráficas del reporte | No participa en el entrenamiento ni en el cálculo de métricas |

No aparecen en el proyecto `scikit-learn`, `scipy`, `statsmodels`, `xgboost` ni ninguna librería
equivalente. Se puede comprobar con:

```bash
grep -rn "sklearn\|scipy\|statsmodels\|xgboost\|tensorflow\|torch" main.py verificar.py src/
```

El comando no devuelve ninguna coincidencia.

---

## El dataset

**Spaceship Titanic** (Kaggle): 8,693 pasajeros con 13 características. La variable objetivo,
`Transported`, indica si el pasajero fue transportado a otra dimensión tras la colisión de la nave
con una anomalía espaciotemporal.

Un detalle importante: el archivo `data/test.csv` es el conjunto de evaluación del *leaderboard* de
Kaggle y **no contiene la columna `Transported`**, así que no sirve para medir desempeño. Por eso los
tres conjuntos se obtienen particionando `data/train.csv` de forma estratificada:

| Conjunto | Muestras | Uso |
|---|---|---|
| Entrenamiento | 6,085 (70%) | Construir los árboles |
| Validación | 1,304 (15%) | Elegir la profundidad y demás hiperparámetros |
| Prueba | 1,304 (15%) | Evaluación final, una sola vez |

---

# Parte II — La misma solución con scikit-learn

Segunda entrega del módulo: resolver el mismo problema apoyándose por completo en un framework,
demostrando dominio sobre su uso y sobre la configuración del algoritmo.

## Resultados sobre el conjunto de prueba

| Modelo | Exactitud | Precisión | Sensibilidad | F1 | AUC |
|---|---|---|---|---|---|
| Línea base (clase mayoritaria) | 0.5038 | — | — | — | — |
| Regresión Logística | 0.7814 | 0.7793 | 0.7900 | 0.7846 | 0.8776 |
| Random Forest | 0.7929 | 0.8136 | 0.7641 | 0.7881 | 0.8907 |
| **Gradient Boosting** (seleccionado) | **0.8029** | 0.8003 | 0.8113 | **0.8057** | **0.9022** |

El análisis completo está en [Reporte_M2_Parte2_Framework.pdf](Reporte_M2_Parte2_Framework.pdf).

## Qué se configuró

- **Pipeline + ColumnTransformer**: todo el preprocesamiento vive dentro del modelo, así que la
  imputación y el escalado se reajustan dentro de cada pliegue de la validación cruzada y no hay
  fuga de datos. El objeto guardado acepta una fila cruda del CSV original.
- **FunctionTransformer**: la ingeniería de características (`Deck`, `CabinNum`, `Side`,
  `GroupSize`, `TotalSpend`, `NoSpend`) también forma parte del pipeline.
- **GridSearchCV** con validación cruzada estratificada de 5 pliegues: 30 combinaciones de
  hiperparámetros, equivalentes a 150 ajustes de modelo.
- **Importancia por permutación** en lugar de la importancia por impureza, porque se calcula sobre
  datos no vistos y no favorece a las variables de alta cardinalidad.

## Cómo ejecutarlo

```bash
python main_framework.py --entrenar
```

Tarda alrededor de 2 minutos (incluye la búsqueda de hiperparámetros). Deja las gráficas, la
bitácora y el modelo entrenado en `resultados_framework/`.

```bash
python main_framework.py --predecir
```

```bash
python main_framework.py --predecir-csv data/test.csv
```
