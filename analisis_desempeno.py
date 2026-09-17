"""
Analisis y reporte sobre el desempeno del modelo.

Modelo analizado: Random Forest de scikit-learn (implementacion de main_framework.py)
Dataset:          Spaceship Titanic (data/train.csv)
Autor:            Juan Pablo Perez Gutierrez  --  A01800483

El analisis sigue cuatro pasos:

    1. Separacion Train / Validation / Test (70 / 15 / 15, estratificada).
    2. Diagnostico del modelo base (hiperparametros por defecto):
         - sesgo     (error en entrenamiento)
         - varianza  (brecha entrenamiento-validacion, dispersion entre pliegues
                      y descomposicion sesgo-varianza por bootstrap)
         - ajuste    (underfit / fit / overfit)
    3. Regularizacion: curvas de validacion para entender cada hiperparametro y
       busqueda aleatoria con validacion cruzada para elegir la configuracion.
    4. Diagnostico del modelo regularizado y comparacion antes / despues.

El conjunto de prueba solo se usa al final, una vez por modelo.

Uso:
    python analisis_desempeno.py
"""

import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
    validation_curve,
)

from main_framework import COLUMNA_OBJETIVO, construir_pipeline

CARPETA = "resultados_analisis"
SEMILLA = 42

# Umbrales usados para clasificar sesgo y varianza. Se justifican en el reporte:
#   - El sesgo se mide con el error de entrenamiento RELATIVO al techo del
#     problema. Los tres modelos de la entrega anterior (lineal, bagging y
#     boosting) se estancan alrededor de 80 % de exactitud, lo que sugiere un
#     error irreducible cercano a 20 %. Un modelo cuyo error de entrenamiento
#     esta muy por debajo de ese techo no tiene problema de sesgo.
#   - La varianza se mide con la brecha entre entrenamiento y validacion.
UMBRAL_SESGO = {"bajo": 0.10, "medio": 0.20}      # error de entrenamiento
UMBRAL_VARIANZA = {"bajo": 0.03, "medio": 0.08}   # brecha train - validacion


def titulo(texto):
    """Imprime un encabezado de seccion."""
    print()
    print("=" * 76)
    print(texto)
    print("=" * 76)


def nivel(valor, umbrales):
    """Traduce un valor numerico a bajo / medio / alto segun los umbrales."""
    if valor < umbrales["bajo"]:
        return "bajo"
    if valor < umbrales["medio"]:
        return "medio"
    return "alto"


def diagnostico_ajuste(sesgo, varianza):
    """Deduce el nivel de ajuste a partir del sesgo y la varianza.

    - Varianza alta con sesgo bajo: el modelo memoriza -> overfit.
    - Sesgo alto con varianza baja: el modelo es demasiado simple -> underfit.
    - En cualquier otro caso el modelo esta razonablemente ajustado -> fit.
    """
    if varianza == "alto":
        return "overfit"
    if sesgo == "alto" and varianza == "bajo":
        return "underfit"
    return "fit"


def guardar(figura, nombre):
    """Guarda una figura en la carpeta de resultados."""
    os.makedirs(CARPETA, exist_ok=True)
    ruta = os.path.join(CARPETA, nombre)
    figura.tight_layout()
    figura.savefig(ruta, dpi=150)
    plt.close(figura)
    print(f"  grafica: {ruta}")
    return ruta


# --- Evaluacion -------------------------------------------------------------

def metricas(modelo, X, y):
    """Calcula exactitud, precision, recall, F1 y AUC de un modelo ajustado."""
    prediccion = modelo.predict(X)
    proba = modelo.predict_proba(X)[:, 1]
    return {
        "exactitud": accuracy_score(y, prediccion),
        "precision": precision_score(y, prediccion),
        "sensibilidad": recall_score(y, prediccion),
        "f1": f1_score(y, prediccion),
        "auc": roc_auc_score(y, proba),
    }


def evaluar_tres_conjuntos(modelo, datos):
    """Evalua un modelo ya entrenado en entrenamiento, validacion y prueba."""
    return {
        "entrenamiento": metricas(modelo, datos["X_entrena"], datos["y_entrena"]),
        "validacion": metricas(modelo, datos["X_valida"], datos["y_valida"]),
        "prueba": metricas(modelo, datos["X_prueba"], datos["y_prueba"]),
    }


def descomposicion_sesgo_varianza(pipeline, datos, n_modelos=20):
    """Estima sesgo y varianza con la descomposicion para perdida 0-1.

    Se entrenan n_modelos copias del modelo, cada una sobre una muestra
    bootstrap distinta del conjunto de entrenamiento, y se predice el conjunto
    de validacion con todas. Para cada pasajero de validacion:

        prediccion principal = la clase que predice la mayoria de los modelos
        sesgo     = 1 si la prediccion principal es incorrecta
        varianza  = fraccion de modelos que NO coinciden con la principal

    Promediando sobre todos los pasajeros se obtiene una medida directa de
    cuanto se equivoca el modelo "en promedio" (sesgo) y de cuanto cambian sus
    predicciones cuando cambian los datos de entrenamiento (varianza). Es la
    descomposicion de Domingos (2000) para clasificacion.

    Returns:
        Diccionario con el sesgo, la varianza y el error promedio.
    """
    generador = np.random.default_rng(SEMILLA)
    X, y = datos["X_entrena"], datos["y_entrena"]
    n = len(y)
    predicciones = []

    for i in range(n_modelos):
        indices = generador.integers(0, n, size=n)
        modelo = clone(pipeline)
        modelo.set_params(clasificador__random_state=int(generador.integers(0, 10**6)))
        modelo.fit(X.iloc[indices], y.iloc[indices])
        predicciones.append(modelo.predict(datos["X_valida"]))

    predicciones = np.array(predicciones)                 # (modelos, muestras)
    y_valida = datos["y_valida"].to_numpy()

    voto = predicciones.mean(axis=0)
    principal = (voto >= 0.5).astype(int)
    sesgo = float((principal != y_valida).mean())
    varianza = float((predicciones != principal).mean())
    error = float((predicciones != y_valida).mean())

    return {"sesgo": sesgo, "varianza": varianza, "error_promedio": error}


def diagnosticar(nombre, pipeline, modelo_ajustado, datos, validacion_cruzada):
    """Diagnostico completo de sesgo, varianza y ajuste para un modelo.

    Args:
        nombre: etiqueta del modelo para la salida.
        pipeline: pipeline sin ajustar (se usa para validacion cruzada y bootstrap).
        modelo_ajustado: el mismo pipeline ya entrenado con todo el entrenamiento.
        datos: diccionario con los tres subconjuntos.
        validacion_cruzada: objeto StratifiedKFold.

    Returns:
        Diccionario con todos los indicadores y el diagnostico.
    """
    titulo(f"DIAGNOSTICO: {nombre}")

    resultados = evaluar_tres_conjuntos(modelo_ajustado, datos)
    ac_tr = resultados["entrenamiento"]["exactitud"]
    ac_va = resultados["validacion"]["exactitud"]
    ac_te = resultados["prueba"]["exactitud"]

    print(f"  Exactitud  entrenamiento {ac_tr:.4f} | validacion {ac_va:.4f} | prueba {ac_te:.4f}")

    # Validacion cruzada: la dispersion entre pliegues es otra senal de varianza.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = cross_validate(pipeline, datos["X_entrena"], datos["y_entrena"],
                            cv=validacion_cruzada, scoring="accuracy",
                            return_train_score=True, n_jobs=-1)
    print(f"  Validacion cruzada (5 pliegues): entrenamiento {cv['train_score'].mean():.4f}, "
          f"validacion {cv['test_score'].mean():.4f} +/- {cv['test_score'].std():.4f}")

    print("  Descomposicion sesgo-varianza (20 modelos bootstrap)...")
    inicio = time.time()
    descomposicion = descomposicion_sesgo_varianza(pipeline, datos)
    print(f"    sesgo {descomposicion['sesgo']:.4f} | varianza {descomposicion['varianza']:.4f} "
          f"| error promedio {descomposicion['error_promedio']:.4f}  ({time.time() - inicio:.0f} s)")

    error_entrena = 1 - ac_tr
    brecha = ac_tr - ac_va
    sesgo = nivel(error_entrena, UMBRAL_SESGO)
    varianza = nivel(brecha, UMBRAL_VARIANZA)
    ajuste = diagnostico_ajuste(sesgo, varianza)

    print(f"\n  Error de entrenamiento: {error_entrena:.4f}  ->  SESGO {sesgo.upper()}")
    print(f"  Brecha train - val:     {brecha:.4f}  ->  VARIANZA {varianza.upper()}")
    print(f"  Nivel de ajuste:        {ajuste.upper()}")

    return {
        "nombre": nombre,
        "conjuntos": resultados,
        "cv_entrena": float(cv["train_score"].mean()),
        "cv_valida": float(cv["test_score"].mean()),
        "cv_valida_std": float(cv["test_score"].std()),
        "cv_pliegues": [float(v) for v in cv["test_score"]],
        "descomposicion": descomposicion,
        "error_entrenamiento": float(error_entrena),
        "brecha": float(brecha),
        "sesgo": sesgo,
        "varianza": varianza,
        "ajuste": ajuste,
    }


# --- Graficas ---------------------------------------------------------------

def grafica_curva_aprendizaje(pipeline, datos, validacion_cruzada, titulo_grafica,
                              nombre, eje=None):
    """Curva de aprendizaje: exactitud contra tamano del entrenamiento.

    Es la grafica clave para el diagnostico:
      - Curvas separadas con entrenamiento cerca de 1.0 -> varianza alta.
      - Ambas curvas juntas pero bajas -> sesgo alto.
      - Ambas curvas juntas y altas -> buen ajuste.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tamanos, entrena, valida = learning_curve(
            pipeline, datos["X_entrena"], datos["y_entrena"],
            train_sizes=np.linspace(0.1, 1.0, 8), cv=validacion_cruzada,
            scoring="accuracy", n_jobs=-1, shuffle=True, random_state=SEMILLA)

    propio = eje is None
    if propio:
        figura, eje = plt.subplots(figsize=(7.5, 4.8))

    for puntajes, etiqueta, color, marca in ((entrena, "Entrenamiento", "#1f77b4", "o"),
                                             (valida, "Validacion cruzada", "#d62728", "s")):
        media, desv = puntajes.mean(axis=1), puntajes.std(axis=1)
        eje.plot(tamanos, media, marker=marca, color=color, label=etiqueta)
        eje.fill_between(tamanos, media - desv, media + desv, color=color, alpha=0.15)

    brecha = entrena.mean(axis=1)[-1] - valida.mean(axis=1)[-1]
    eje.annotate(f"brecha final = {brecha:.3f}",
                 xy=(tamanos[-1], (entrena.mean(axis=1)[-1] + valida.mean(axis=1)[-1]) / 2),
                 xytext=(-130, 0), textcoords="offset points", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="gray"))

    eje.set_ylim(0.70, 1.01)
    eje.set_xlabel("Muestras de entrenamiento")
    eje.set_ylabel("Exactitud")
    eje.set_title(titulo_grafica)
    eje.legend(loc="lower right")
    eje.grid(alpha=0.3)

    datos_curva = {
        "tamanos": [int(t) for t in tamanos],
        "entrena": [float(v) for v in entrena.mean(axis=1)],
        "valida": [float(v) for v in valida.mean(axis=1)],
    }

    if propio:
        guardar(figura, nombre)
    return datos_curva


def grafica_curva_validacion(pipeline, datos, validacion_cruzada, parametro, valores,
                             etiquetas_x, titulo_grafica, nombre, xlabel):
    """Curva de validacion: exactitud contra el valor de un hiperparametro.

    Muestra la transicion de underfit (izquierda o derecha, segun el parametro)
    a overfit y permite ubicar la zona de buen ajuste.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entrena, valida = validation_curve(
            pipeline, datos["X_entrena"], datos["y_entrena"],
            param_name=parametro, param_range=valores, cv=validacion_cruzada,
            scoring="accuracy", n_jobs=-1)

    posiciones = np.arange(len(valores))
    figura, eje = plt.subplots(figsize=(7.5, 4.6))
    for puntajes, etiqueta, color, marca in ((entrena, "Entrenamiento", "#1f77b4", "o"),
                                             (valida, "Validacion cruzada", "#d62728", "s")):
        media, desv = puntajes.mean(axis=1), puntajes.std(axis=1)
        eje.plot(posiciones, media, marker=marca, color=color, label=etiqueta)
        eje.fill_between(posiciones, media - desv, media + desv, color=color, alpha=0.15)

    mejor = int(np.argmax(valida.mean(axis=1)))
    eje.axvline(mejor, color="gray", linestyle="--", linewidth=1)
    eje.annotate(f"mejor validacion: {etiquetas_x[mejor]}\n({valida.mean(axis=1)[mejor]:.4f})",
                 xy=(mejor, valida.mean(axis=1)[mejor]), xytext=(12, -40),
                 textcoords="offset points", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="gray"))

    eje.set_xticks(posiciones, labels=etiquetas_x)
    eje.set_xlabel(xlabel)
    eje.set_ylabel("Exactitud")
    eje.set_title(titulo_grafica)
    eje.legend(loc="best")
    eje.grid(alpha=0.3)
    guardar(figura, nombre)

    return {
        "valores": [str(e) for e in etiquetas_x],
        "entrena": [float(v) for v in entrena.mean(axis=1)],
        "valida": [float(v) for v in valida.mean(axis=1)],
        "mejor": str(etiquetas_x[mejor]),
    }


def grafica_tres_conjuntos(antes, despues, nombre="comparacion_train_val_test.png"):
    """Barras de exactitud en entrenamiento, validacion y prueba, antes y despues."""
    conjuntos = ["entrenamiento", "validacion", "prueba"]
    etiquetas = ["Entrenamiento", "Validacion", "Prueba"]
    x = np.arange(len(conjuntos))
    ancho = 0.36

    valores_antes = [antes["conjuntos"][c]["exactitud"] for c in conjuntos]
    valores_despues = [despues["conjuntos"][c]["exactitud"] for c in conjuntos]

    figura, eje = plt.subplots(figsize=(8, 4.8))
    b1 = eje.bar(x - ancho / 2, valores_antes, ancho, label="Antes (por defecto)", color="#c44e52")
    b2 = eje.bar(x + ancho / 2, valores_despues, ancho, label="Despues (regularizado)",
                 color="#55a868")
    for barras in (b1, b2):
        for barra in barras:
            eje.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.004,
                     f"{barra.get_height():.3f}", ha="center", fontsize=9)

    eje.set_xticks(x, labels=etiquetas)
    eje.set_ylim(0.70, 1.04)
    eje.set_ylabel("Exactitud")
    eje.set_title("Exactitud por conjunto: antes y despues de regularizar")
    eje.legend(loc="upper right")
    eje.grid(axis="y", alpha=0.3)
    return guardar(figura, nombre)


def grafica_indicadores(antes, despues, nombre="indicadores_sesgo_varianza.png"):
    """Barras de los indicadores de sesgo y varianza, antes y despues."""
    indicadores = [
        ("Error de\nentrenamiento", "error_entrenamiento"),
        ("Brecha\ntrain - val", "brecha"),
        ("Sesgo\n(bootstrap)", ("descomposicion", "sesgo")),
        ("Varianza\n(bootstrap)", ("descomposicion", "varianza")),
    ]

    def leer(resultado, clave):
        if isinstance(clave, tuple):
            return resultado[clave[0]][clave[1]]
        return resultado[clave]

    x = np.arange(len(indicadores))
    ancho = 0.36
    valores_antes = [leer(antes, c) for _, c in indicadores]
    valores_despues = [leer(despues, c) for _, c in indicadores]

    figura, eje = plt.subplots(figsize=(8.5, 4.8))
    b1 = eje.bar(x - ancho / 2, valores_antes, ancho, label="Antes (por defecto)", color="#c44e52")
    b2 = eje.bar(x + ancho / 2, valores_despues, ancho, label="Despues (regularizado)",
                 color="#55a868")
    for barras in (b1, b2):
        for barra in barras:
            eje.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.003,
                     f"{barra.get_height():.3f}", ha="center", fontsize=9)

    eje.set_xticks(x, labels=[e for e, _ in indicadores])
    eje.set_ylim(0, max(valores_antes + valores_despues) * 1.25)
    eje.set_ylabel("Valor (menor es mejor)")
    eje.set_title("Indicadores de sesgo y varianza")
    eje.legend(loc="upper center", ncol=2)
    eje.grid(axis="y", alpha=0.3)
    return guardar(figura, nombre)


def grafica_matrices(modelo_antes, modelo_despues, datos, nombre="matrices_confusion.png"):
    """Matrices de confusion en prueba, lado a lado."""
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4.6))
    for eje, modelo, etiqueta in ((ejes[0], modelo_antes, "Antes (por defecto)"),
                                  (ejes[1], modelo_despues, "Despues (regularizado)")):
        ConfusionMatrixDisplay.from_estimator(
            modelo, datos["X_prueba"], datos["y_prueba"], ax=eje, cmap="Blues",
            colorbar=False, display_labels=["No transp.", "Si transp."])
        eje.set_title(etiqueta)
        eje.set_xlabel("Prediccion")
        eje.set_ylabel("Real")
    return guardar(figura, nombre)


def grafica_pliegues(antes, despues, nombre="dispersion_pliegues.png"):
    """Exactitud de cada pliegue de validacion cruzada, antes y despues."""
    figura, eje = plt.subplots(figsize=(7, 4.4))
    pliegues = np.arange(1, 6)
    eje.plot(pliegues, antes["cv_pliegues"], "o-", color="#c44e52",
             label=f"Antes  (desv. {antes['cv_valida_std']:.4f})")
    eje.plot(pliegues, despues["cv_pliegues"], "s-", color="#55a868",
             label=f"Despues (desv. {despues['cv_valida_std']:.4f})")
    eje.set_xticks(pliegues)
    eje.set_xlabel("Pliegue de validacion cruzada")
    eje.set_ylabel("Exactitud en el pliegue")
    eje.set_title("Estabilidad entre pliegues")
    eje.legend()
    eje.grid(alpha=0.3)
    return guardar(figura, nombre)


# --- Programa principal -----------------------------------------------------

def main():
    os.makedirs(CARPETA, exist_ok=True)
    inicio_total = time.time()

    titulo("1. SEPARACION TRAIN / VALIDATION / TEST")
    df = pd.read_csv("data/train.csv")
    y = df[COLUMNA_OBJETIVO].astype(int)
    X = df.drop(columns=[COLUMNA_OBJETIVO])

    X_entrena, X_resto, y_entrena, y_resto = train_test_split(
        X, y, test_size=0.30, random_state=SEMILLA, stratify=y)
    X_valida, X_prueba, y_valida, y_prueba = train_test_split(
        X_resto, y_resto, test_size=0.50, random_state=SEMILLA, stratify=y_resto)

    datos = {"X_entrena": X_entrena, "y_entrena": y_entrena,
             "X_valida": X_valida, "y_valida": y_valida,
             "X_prueba": X_prueba, "y_prueba": y_prueba}

    particion = {}
    for etiqueta, yy in (("entrenamiento", y_entrena), ("validacion", y_valida),
                         ("prueba", y_prueba)):
        particion[etiqueta] = {"muestras": int(len(yy)), "positivos": float(yy.mean())}
        print(f"  {etiqueta:<14} {len(yy):>6} muestras   {100 * yy.mean():.2f} % transportados")

    validacion_cruzada = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    # ------------------------------------------------------------------
    titulo("2. MODELO BASE: Random Forest con hiperparametros por defecto")
    print("  n_estimators=100, max_depth=None, min_samples_leaf=1, max_features='sqrt'")
    base = construir_pipeline(RandomForestClassifier(random_state=SEMILLA, n_jobs=-1),
                              escalar=False)
    modelo_base = clone(base).fit(X_entrena, y_entrena)
    arboles = modelo_base.named_steps["clasificador"].estimators_
    print(f"  profundidad promedio de los arboles: "
          f"{np.mean([a.get_depth() for a in arboles]):.1f}")
    print(f"  hojas promedio por arbol: {np.mean([a.get_n_leaves() for a in arboles]):.0f}")

    diag_antes = diagnosticar("Modelo base (por defecto)", base, modelo_base, datos,
                              validacion_cruzada)
    diag_antes["profundidad_promedio"] = float(np.mean([a.get_depth() for a in arboles]))
    diag_antes["hojas_promedio"] = float(np.mean([a.get_n_leaves() for a in arboles]))

    print()
    curva_antes = grafica_curva_aprendizaje(base, datos, validacion_cruzada,
                                            "Curva de aprendizaje: modelo base",
                                            "curva_aprendizaje_base.png")

    # ------------------------------------------------------------------
    titulo("3. CURVAS DE VALIDACION: efecto de cada hiperparametro")
    curvas = {}

    valores_profundidad = [2, 4, 6, 8, 10, 12, 15, 20, 30, None]
    curvas["max_depth"] = grafica_curva_validacion(
        base, datos, validacion_cruzada, "clasificador__max_depth", valores_profundidad,
        [str(v) if v is not None else "sin\nlimite" for v in valores_profundidad],
        "Curva de validacion: profundidad maxima", "curva_validacion_max_depth.png",
        "max_depth")

    valores_hoja = [1, 2, 4, 8, 16, 32, 64, 128]
    curvas["min_samples_leaf"] = grafica_curva_validacion(
        base, datos, validacion_cruzada, "clasificador__min_samples_leaf", valores_hoja,
        [str(v) for v in valores_hoja],
        "Curva de validacion: minimo de muestras por hoja",
        "curva_validacion_min_samples_leaf.png", "min_samples_leaf")

    valores_alfa = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    curvas["ccp_alpha"] = grafica_curva_validacion(
        base, datos, validacion_cruzada, "clasificador__ccp_alpha", valores_alfa,
        [str(v) for v in valores_alfa],
        "Curva de validacion: poda por costo-complejidad (ccp_alpha)",
        "curva_validacion_ccp_alpha.png", "ccp_alpha")

    for parametro, curva in curvas.items():
        print(f"  {parametro:<18} mejor valor en validacion cruzada: {curva['mejor']}")

    # ------------------------------------------------------------------
    titulo("4. REGULARIZACION: busqueda aleatoria con validacion cruzada")
    espacio = {
        "clasificador__n_estimators": [200, 300, 500],
        "clasificador__max_depth": [8, 10, 12, 15, 20, None],
        "clasificador__min_samples_leaf": [1, 2, 4, 8, 16],
        "clasificador__min_samples_split": [2, 5, 10, 20],
        "clasificador__max_features": ["sqrt", 0.3, 0.5],
        "clasificador__max_samples": [None, 0.5, 0.7],
        "clasificador__ccp_alpha": [0.0, 0.0001, 0.0002, 0.0005],
    }
    combinaciones = int(np.prod([len(v) for v in espacio.values()]))
    print(f"  Espacio de busqueda: {combinaciones} combinaciones; se prueban 60 al azar "
          f"(300 ajustes con 5 pliegues).")

    inicio = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        busqueda = RandomizedSearchCV(
            base, espacio, n_iter=60, cv=validacion_cruzada, scoring="accuracy",
            n_jobs=-1, random_state=SEMILLA, return_train_score=True)
        busqueda.fit(X_entrena, y_entrena)
    tiempo_busqueda = time.time() - inicio

    mejores = {k.replace("clasificador__", ""): v for k, v in busqueda.best_params_.items()}
    print(f"  Tiempo: {tiempo_busqueda:.0f} s")
    print(f"  Mejores hiperparametros: {mejores}")
    print(f"  Exactitud en validacion cruzada: {busqueda.best_score_:.4f}")

    # Tabla de las 10 mejores configuraciones, util para el reporte.
    tabla_busqueda = pd.DataFrame(busqueda.cv_results_)
    tabla_busqueda["brecha"] = tabla_busqueda["mean_train_score"] - tabla_busqueda["mean_test_score"]
    top = tabla_busqueda.sort_values("rank_test_score").head(10)
    top_registros = []
    for _, fila in top.iterrows():
        top_registros.append({
            "params": {k.replace("clasificador__", ""): (None if v is None else
                                                          (v if isinstance(v, str) else float(v)))
                       for k, v in fila["params"].items()},
            "cv_entrena": float(fila["mean_train_score"]),
            "cv_valida": float(fila["mean_test_score"]),
            "cv_std": float(fila["std_test_score"]),
            "brecha": float(fila["brecha"]),
        })

    regularizado = busqueda.best_estimator_
    pipeline_regularizado = clone(base).set_params(**busqueda.best_params_)
    arboles = regularizado.named_steps["clasificador"].estimators_

    diag_despues = diagnosticar("Modelo regularizado", pipeline_regularizado, regularizado,
                                datos, validacion_cruzada)
    diag_despues["profundidad_promedio"] = float(np.mean([a.get_depth() for a in arboles]))
    diag_despues["hojas_promedio"] = float(np.mean([a.get_n_leaves() for a in arboles]))
    diag_despues["hiperparametros"] = {k: (v if isinstance(v, (str, type(None))) else float(v))
                                       for k, v in mejores.items()}
    print(f"  profundidad promedio de los arboles: {diag_despues['profundidad_promedio']:.1f}")
    print(f"  hojas promedio por arbol: {diag_despues['hojas_promedio']:.0f}")

    print()
    curva_despues = grafica_curva_aprendizaje(pipeline_regularizado, datos, validacion_cruzada,
                                              "Curva de aprendizaje: modelo regularizado",
                                              "curva_aprendizaje_regularizado.png")

    # ------------------------------------------------------------------
    titulo("5. COMPARACION ANTES / DESPUES")

    # Curvas de aprendizaje lado a lado (se recalculan sobre ejes compartidos).
    figura, ejes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    grafica_curva_aprendizaje(base, datos, validacion_cruzada,
                              "Antes: hiperparametros por defecto", None, eje=ejes[0])
    grafica_curva_aprendizaje(pipeline_regularizado, datos, validacion_cruzada,
                              "Despues: modelo regularizado", None, eje=ejes[1])
    guardar(figura, "curvas_aprendizaje_comparacion.png")

    grafica_tres_conjuntos(diag_antes, diag_despues)
    grafica_indicadores(diag_antes, diag_despues)
    grafica_matrices(modelo_base, regularizado, datos)
    grafica_pliegues(diag_antes, diag_despues)

    matrices = {}
    for etiqueta, modelo in (("antes", modelo_base), ("despues", regularizado)):
        matriz = confusion_matrix(y_prueba, modelo.predict(X_prueba))
        matrices[etiqueta] = [[int(v) for v in fila] for fila in matriz]

    print()
    print(f"  {'Indicador':<34}{'Antes':>10}{'Despues':>10}{'Cambio':>10}")
    filas = [
        ("Exactitud entrenamiento", diag_antes["conjuntos"]["entrenamiento"]["exactitud"],
         diag_despues["conjuntos"]["entrenamiento"]["exactitud"]),
        ("Exactitud validacion", diag_antes["conjuntos"]["validacion"]["exactitud"],
         diag_despues["conjuntos"]["validacion"]["exactitud"]),
        ("Exactitud prueba", diag_antes["conjuntos"]["prueba"]["exactitud"],
         diag_despues["conjuntos"]["prueba"]["exactitud"]),
        ("AUC prueba", diag_antes["conjuntos"]["prueba"]["auc"],
         diag_despues["conjuntos"]["prueba"]["auc"]),
        ("F1 prueba", diag_antes["conjuntos"]["prueba"]["f1"],
         diag_despues["conjuntos"]["prueba"]["f1"]),
        ("Brecha train - val", diag_antes["brecha"], diag_despues["brecha"]),
        ("Validacion cruzada (media)", diag_antes["cv_valida"], diag_despues["cv_valida"]),
        ("Validacion cruzada (desv.)", diag_antes["cv_valida_std"], diag_despues["cv_valida_std"]),
        ("Sesgo bootstrap", diag_antes["descomposicion"]["sesgo"],
         diag_despues["descomposicion"]["sesgo"]),
        ("Varianza bootstrap", diag_antes["descomposicion"]["varianza"],
         diag_despues["descomposicion"]["varianza"]),
    ]
    for etiqueta, a, d in filas:
        print(f"  {etiqueta:<34}{a:>10.4f}{d:>10.4f}{d - a:>+10.4f}")

    print()
    print(f"  Diagnostico antes:   sesgo {diag_antes['sesgo']}, varianza "
          f"{diag_antes['varianza']}, {diag_antes['ajuste']}")
    print(f"  Diagnostico despues: sesgo {diag_despues['sesgo']}, varianza "
          f"{diag_despues['varianza']}, {diag_despues['ajuste']}")

    resumen = {
        "particion": particion,
        "umbrales": {"sesgo": UMBRAL_SESGO, "varianza": UMBRAL_VARIANZA},
        "antes": diag_antes,
        "despues": diag_despues,
        "curva_aprendizaje_antes": curva_antes,
        "curva_aprendizaje_despues": curva_despues,
        "curvas_validacion": curvas,
        "busqueda": {"combinaciones": combinaciones, "iteraciones": 60,
                     "tiempo_s": tiempo_busqueda, "mejor_cv": float(busqueda.best_score_),
                     "top10": top_registros},
        "matrices_prueba": matrices,
    }
    ruta = os.path.join(CARPETA, "resultados.json")
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Resultados guardados en {ruta}")
    print(f"  Tiempo total: {time.time() - inicio_total:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
