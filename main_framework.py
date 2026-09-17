"""
Momento de Retroalimentacion: uso de un FRAMEWORK de aprendizaje maquina.

Framework:  scikit-learn
Algoritmo:  Random Forest (comparado contra Regresion Logistica y
            Histogram-based Gradient Boosting)
Dataset:    Spaceship Titanic (data/train.csv)
Autor:      Juan Pablo Perez Gutierrez  --  A01800483

A diferencia de la entrega anterior (main.py), donde el algoritmo se programo a
mano, aqui todo el aprendizaje lo hace scikit-learn. El trabajo consiste en
configurarlo correctamente:

    - Pipeline y ColumnTransformer para que el preprocesamiento viva DENTRO del
      modelo y no pueda haber fuga de datos entre entrenamiento y prueba.
    - FunctionTransformer para que la ingenieria de caracteristicas tambien
      forme parte del pipeline y este acepte el CSV crudo.
    - GridSearchCV con validacion cruzada estratificada para elegir los
      hiperparametros de cada modelo.
    - Comparacion de tres familias de algoritmos con las mismas particiones.

Modos de uso desde la terminal:

    python main_framework.py --entrenar
        Ajusta los tres modelos con busqueda de hiperparametros, los compara,
        evalua el mejor sobre el conjunto de prueba y guarda graficas y modelo.

    python main_framework.py --predecir
        Modo interactivo: captura un pasajero por consola y lo clasifica.

    python main_framework.py --predecir-csv data/test.csv
        Predice en lote un CSV sin etiqueta.

Este archivo se ejecuta con un interprete de Python normal; no requiere IDE ni
notebook.
"""

import argparse
import os
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CARPETA_RESULTADOS = "resultados_framework"
RUTA_MODELO = os.path.join(CARPETA_RESULTADOS, "modelo_sklearn.joblib")
SEMILLA = 42

# --- Definicion de columnas -------------------------------------------------

COLUMNA_OBJETIVO = "Transported"

COLUMNAS_GASTO = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Columnas que el ColumnTransformer trata como categoricas (One-Hot).
CATEGORICAS = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]

# Columnas que el ColumnTransformer trata como numericas (imputar + escalar).
NUMERICAS = ["Age", "CabinNum", "GroupSize", "TotalSpend", "NoSpend"] + COLUMNAS_GASTO


def titulo(texto):
    """Imprime un encabezado de seccion para que la salida sea legible."""
    print()
    print("=" * 74)
    print(texto)
    print("=" * 74)


# --- Ingenieria de caracteristicas dentro del pipeline ----------------------

def derivar_caracteristicas(df):
    """Crea las columnas derivadas a partir de las columnas crudas del CSV.

    Esta funcion se envuelve en un FunctionTransformer para que forme parte del
    Pipeline. La ventaja de hacerlo asi, en vez de transformar el DataFrame por
    fuera, es que el pipeline queda completo: recibe una fila tal como viene del
    CSV original y se encarga de todo. Eso hace imposible olvidar un paso al
    predecir datos nuevos.

    Args:
        df: DataFrame con las columnas crudas del dataset.

    Returns:
        Una copia del DataFrame con las columnas derivadas agregadas.
    """
    df = df.copy()

    # Cabin viene como "cubierta/numero/lado": cada parte es una senal distinta.
    partes = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = partes[0]
    df["CabinNum"] = pd.to_numeric(partes[1], errors="coerce")
    df["Side"] = partes[2]

    # PassengerId tiene el formato gggg_pp; los primeros digitos son el grupo
    # de viaje, y su tamano distingue a quien viajaba solo de quien iba en grupo.
    grupo = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = grupo.map(grupo.value_counts()).astype(float)

    # El gasto total resume las cinco amenidades; NoSpend marca a quien no
    # consumio nada (tipicamente los pasajeros en criosueno).
    df["TotalSpend"] = df[COLUMNAS_GASTO].sum(axis=1, skipna=True)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(float)

    # CryoSleep y VIP son booleanas con nulos, lo que las vuelve de tipo objeto.
    # Se pasan a texto para que el OneHotEncoder las trate como categoricas.
    for columna in ("CryoSleep", "VIP"):
        df[columna] = df[columna].astype("object").where(df[columna].notna(), np.nan)
        df[columna] = df[columna].map({True: "True", False: "False"}).astype("object")

    return df[CATEGORICAS + NUMERICAS]


def construir_preprocesador(escalar=True):
    """Arma el ColumnTransformer con el tratamiento de cada grupo de columnas.

    Args:
        escalar: si True, estandariza las numericas. Los modelos de arboles no
            lo necesitan (son invariantes a transformaciones monotonas), pero la
            regresion logistica si, porque su regularizacion penaliza los
            coeficientes y estos dependen de la escala de cada variable.

    Returns:
        Un ColumnTransformer listo para usarse dentro de un Pipeline.
    """
    pasos_numericos = [("imputador", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_numericos.append(("escalador", StandardScaler()))

    transformador_numerico = Pipeline(pasos_numericos)

    transformador_categorico = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore" evita que una categoria nunca vista en
        # entrenamiento haga fallar la prediccion en produccion.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", transformador_numerico, NUMERICAS),
        ("cat", transformador_categorico, CATEGORICAS),
    ])


def construir_pipeline(clasificador, escalar=True):
    """Encadena ingenieria de caracteristicas, preprocesamiento y clasificador.

    Args:
        clasificador: cualquier estimador de scikit-learn.
        escalar: si se estandarizan las variables numericas.

    Returns:
        Un Pipeline que acepta el DataFrame crudo del CSV.
    """
    return Pipeline([
        ("caracteristicas", FunctionTransformer(derivar_caracteristicas)),
        ("preprocesamiento", construir_preprocesador(escalar=escalar)),
        ("clasificador", clasificador),
    ])


# --- Configuracion de los modelos a comparar --------------------------------

def definir_modelos():
    """Define los tres modelos y la rejilla de hiperparametros de cada uno.

    Se comparan tres familias distintas a proposito:
        - Regresion logistica: modelo lineal, sirve de referencia.
        - Random Forest: ensamble por bagging, el algoritmo principal.
        - Histogram Gradient Boosting: ensamble por boosting, el competidor.

    Returns:
        Diccionario nombre -> (pipeline, rejilla de hiperparametros).
    """
    modelos = {}

    # --- Regresion logistica ---
    # C es el inverso de la fuerza de regularizacion: valores chicos regularizan
    # mas. Necesita escalado porque la penalizacion depende de la escala.
    modelos["Regresion Logistica"] = (
        construir_pipeline(
            LogisticRegression(max_iter=2000, random_state=SEMILLA),
            escalar=True,
        ),
        {
            "clasificador__C": [0.01, 0.1, 1.0, 10.0],
            "clasificador__penalty": ["l2"],
        },
    )

    # --- Random Forest ---
    # max_depth y min_samples_leaf controlan el sobreajuste de cada arbol;
    # max_features controla que tan decorrelacionados quedan entre si, que es
    # el mecanismo del que depende el ensamble para reducir varianza.
    modelos["Random Forest"] = (
        construir_pipeline(
            RandomForestClassifier(n_estimators=300, random_state=SEMILLA, n_jobs=-1),
            escalar=False,
        ),
        {
            "clasificador__max_depth": [None, 12, 20],
            "clasificador__min_samples_leaf": [1, 2, 5],
            "clasificador__max_features": ["sqrt", 0.5],
        },
    )

    # --- Gradient Boosting ---
    # learning_rate y max_iter se compensan entre si: una tasa baja necesita
    # mas iteraciones. max_leaf_nodes limita la complejidad de cada arbol.
    modelos["Gradient Boosting"] = (
        construir_pipeline(
            HistGradientBoostingClassifier(random_state=SEMILLA),
            escalar=False,
        ),
        {
            "clasificador__learning_rate": [0.05, 0.1],
            "clasificador__max_iter": [200, 400],
            "clasificador__max_leaf_nodes": [15, 31],
        },
    )

    return modelos


# --- Metricas ---------------------------------------------------------------

def evaluar(modelo, X, y, etiqueta):
    """Calcula el conjunto de metricas de clasificacion binaria.

    Args:
        modelo: pipeline ya entrenado.
        X: DataFrame de entrada.
        y: etiquetas verdaderas.
        etiqueta: nombre del modelo, para el reporte.

    Returns:
        Diccionario con las metricas y la matriz de confusion.
    """
    prediccion = modelo.predict(X)
    proba = modelo.predict_proba(X)[:, 1]

    matriz = confusion_matrix(y, prediccion)
    vn, fp, fn, vp = matriz.ravel()

    return {
        "modelo": etiqueta,
        "exactitud": accuracy_score(y, prediccion),
        "precision": precision_score(y, prediccion),
        "sensibilidad": recall_score(y, prediccion),
        # La especificidad no viene en sklearn: es el recall de la clase 0.
        "especificidad": vn / (vn + fp) if (vn + fp) else 0.0,
        "f1": f1_score(y, prediccion),
        "auc": roc_auc_score(y, proba),
        "vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp),
        "matriz": matriz,
        "prediccion": prediccion,
        "proba": proba,
    }


def formato_metricas(m):
    """Devuelve las metricas como un bloque de texto alineado."""
    return "\n".join([
        f"  Exactitud (accuracy)      {m['exactitud']:.4f}",
        f"  Precision                 {m['precision']:.4f}",
        f"  Sensibilidad (recall)     {m['sensibilidad']:.4f}",
        f"  Especificidad             {m['especificidad']:.4f}",
        f"  F1                        {m['f1']:.4f}",
        f"  AUC (ROC)                 {m['auc']:.4f}",
    ])


def formato_matriz(m):
    """Devuelve la matriz de confusion como una tabla de texto legible."""
    return (
        f"                       PREDICCION\n"
        f"                  No transp.   Si transp.\n"
        f"  REAL No transp.  {m['vn']:9d}    {m['fp']:9d}   <- {m['fp']} falsos positivos\n"
        f"       Si transp.  {m['fn']:9d}    {m['vp']:9d}\n"
        f"                        ^\n"
        f"                        {m['fn']} falsos negativos"
    )


# --- Graficas ---------------------------------------------------------------

def guardar(figura, nombre):
    """Guarda la figura en la carpeta de resultados y libera la memoria."""
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(CARPETA_RESULTADOS, nombre)
    figura.tight_layout()
    figura.savefig(ruta, dpi=150)
    plt.close(figura)
    return ruta


def graficar_comparacion(resultados_val, nombre="comparacion_modelos.png"):
    """Grafica en barras la exactitud y el AUC de los tres modelos."""
    nombres = [r["modelo"] for r in resultados_val]
    exactitudes = [r["exactitud"] for r in resultados_val]
    aucs = [r["auc"] for r in resultados_val]

    x = np.arange(len(nombres))
    ancho = 0.35

    figura, eje = plt.subplots(figsize=(8, 4.8))
    eje.bar(x - ancho / 2, exactitudes, ancho, label="Exactitud", color="#1f77b4")
    eje.bar(x + ancho / 2, aucs, ancho, label="AUC", color="#ff7f0e")

    for i, (a, u) in enumerate(zip(exactitudes, aucs)):
        eje.text(i - ancho / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)
        eje.text(i + ancho / 2, u + 0.005, f"{u:.3f}", ha="center", fontsize=9)

    eje.set_xticks(x, labels=nombres)
    eje.set_ylim(0.7, 1.0)
    eje.set_ylabel("Valor de la metrica")
    eje.set_title("Comparacion de modelos sobre el conjunto de validacion")
    eje.legend()
    eje.grid(axis="y", alpha=0.3)
    return guardar(figura, nombre)


def graficar_matriz(modelo, X, y, etiqueta, nombre):
    """Dibuja la matriz de confusion usando el display de scikit-learn."""
    figura, eje = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay.from_estimator(
        modelo, X, y, ax=eje, cmap="Blues", colorbar=True,
        display_labels=["No transportado", "Si transportado"],
    )
    eje.set_title(etiqueta)
    eje.set_xlabel("Prediccion del modelo")
    eje.set_ylabel("Valor real")
    return guardar(figura, nombre)


def graficar_roc(modelos_entrenados, X, y, nombre="curva_roc.png"):
    """Dibuja las curvas ROC de todos los modelos en los mismos ejes."""
    figura, eje = plt.subplots(figsize=(6, 5.4))
    for etiqueta, modelo in modelos_entrenados.items():
        RocCurveDisplay.from_estimator(modelo, X, y, ax=eje, name=etiqueta)
    eje.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar (AUC = 0.5)")
    eje.set_xlabel("Tasa de falsos positivos")
    eje.set_ylabel("Tasa de verdaderos positivos")
    eje.set_title("Curvas ROC sobre el conjunto de prueba")
    eje.legend(loc="lower right", fontsize=9)
    eje.grid(alpha=0.3)
    return guardar(figura, nombre)


def graficar_importancias(importancias, nombres, nombre="importancias.png"):
    """Grafica la importancia por permutacion en barras horizontales."""
    orden = np.argsort(importancias.importances_mean)
    valores = importancias.importances_mean[orden]
    errores = importancias.importances_std[orden]
    etiquetas = [nombres[i] for i in orden]

    figura, eje = plt.subplots(figsize=(8, 7))
    eje.barh(range(len(valores)), valores, xerr=errores, color="#1f77b4")
    eje.set_yticks(range(len(valores)), labels=etiquetas)
    eje.set_xlabel("Caida de exactitud al permutar la variable")
    eje.set_title("Importancia por permutacion (Random Forest)")
    eje.grid(axis="x", alpha=0.3)
    return guardar(figura, nombre)


# --- Modo de entrenamiento --------------------------------------------------

def modo_entrenar(args):
    """Ajusta, compara y evalua los modelos; guarda graficas y modelo final."""
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    registro = []

    def reportar(texto=""):
        print(texto)
        registro.append(str(texto))

    titulo("1. CARGA Y PARTICION DE LOS DATOS")
    df = pd.read_csv(args.datos)
    y = df[COLUMNA_OBJETIVO].astype(int)
    X = df.drop(columns=[COLUMNA_OBJETIVO])

    reportar(f"Archivo: {args.datos}   ({X.shape[0]} filas, {X.shape[1]} columnas)")
    reportar("Nota: data/test.csv es el conjunto ciego de Kaggle y no trae la")
    reportar("      etiqueta Transported, por eso el conjunto de prueba se")
    reportar("      obtiene particionando el archivo de entrenamiento.")

    # Dos particiones sucesivas producen el reparto 70 / 15 / 15. stratify
    # conserva la proporcion de clases en cada parte.
    X_entrena, X_resto, y_entrena, y_resto = train_test_split(
        X, y, test_size=0.30, random_state=SEMILLA, stratify=y)
    X_valida, X_prueba, y_valida, y_prueba = train_test_split(
        X_resto, y_resto, test_size=0.50, random_state=SEMILLA, stratify=y_resto)

    reportar("")
    reportar(f"{'Subconjunto':<16}{'Muestras':>10}{'% Transported':>16}")
    for etiqueta, yy in (("Entrenamiento", y_entrena), ("Validacion", y_valida),
                         ("Prueba", y_prueba)):
        reportar(f"{etiqueta:<16}{len(yy):>10}{100 * yy.mean():>15.2f}%")

    base = max(y_prueba.mean(), 1 - y_prueba.mean())
    reportar(f"\nLinea base (predecir siempre la clase mayoritaria): {base:.4f}")

    titulo("2. BUSQUEDA DE HIPERPARAMETROS CON VALIDACION CRUZADA")
    reportar("Cada modelo se ajusta con GridSearchCV sobre el conjunto de")
    reportar("entrenamiento, usando validacion cruzada estratificada de 5")
    reportar("pliegues. El pipeline completo entra a la validacion cruzada, asi")
    reportar("que la imputacion y el escalado se reajustan dentro de cada")
    reportar("pliegue y no hay fuga de informacion entre pliegues.\n")

    validacion_cruzada = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    modelos = definir_modelos()
    entrenados = {}
    resultados_validacion = []

    for etiqueta, (pipeline, rejilla) in modelos.items():
        n_combinaciones = int(np.prod([len(v) for v in rejilla.values()]))
        reportar(f"--- {etiqueta} ---")
        reportar(f"    Combinaciones probadas: {n_combinaciones}  "
                 f"({n_combinaciones * 5} ajustes con la validacion cruzada)")

        inicio = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            busqueda = GridSearchCV(
                pipeline, rejilla, cv=validacion_cruzada,
                scoring="accuracy", n_jobs=-1, refit=True,
            )
            busqueda.fit(X_entrena, y_entrena)

        mejor = {k.replace("clasificador__", ""): v
                 for k, v in busqueda.best_params_.items()}
        reportar(f"    Mejores hiperparametros: {mejor}")
        reportar(f"    Exactitud en validacion cruzada: {busqueda.best_score_:.4f}")
        reportar(f"    Tiempo: {time.time() - inicio:.1f} s")

        entrenados[etiqueta] = busqueda.best_estimator_
        metricas = evaluar(busqueda.best_estimator_, X_valida, y_valida, etiqueta)
        resultados_validacion.append(metricas)
        reportar(f"    Exactitud en el conjunto de validacion: "
                 f"{metricas['exactitud']:.4f}  (AUC {metricas['auc']:.4f})")
        reportar("")

    titulo("3. COMPARACION DE LOS TRES MODELOS (conjunto de validacion)")
    reportar(f"{'Modelo':<22}{'Exactitud':>11}{'F1':>9}{'AUC':>9}")
    for m in resultados_validacion:
        reportar(f"{m['modelo']:<22}{m['exactitud']:>11.4f}"
                 f"{m['f1']:>9.4f}{m['auc']:>9.4f}")

    ruta = graficar_comparacion(resultados_validacion)
    reportar(f"\nGrafica guardada: {ruta}")

    # La seleccion es por exactitud en validacion, con el AUC como criterio de
    # desempate. Sin el desempate, dos modelos con la misma exactitud se
    # resolverian por el orden en que aparecen en el diccionario, que es
    # arbitrario; el AUC es una medida mas fina porque evalua el ordenamiento
    # completo de probabilidades y no solo las decisiones al umbral 0.5.
    ganador = max(resultados_validacion, key=lambda m: (m["exactitud"], m["auc"]))
    nombre_ganador = ganador["modelo"]
    modelo_final = entrenados[nombre_ganador]

    empatados = [m["modelo"] for m in resultados_validacion
                 if m["exactitud"] == ganador["exactitud"]]
    if len(empatados) > 1:
        reportar(f"\nEmpate en exactitud ({ganador['exactitud']:.4f}) entre: "
                 f"{', '.join(empatados)}")
        reportar("El desempate se resuelve por AUC.")
    reportar(f"\nModelo seleccionado por validacion: {nombre_ganador}")

    titulo("4. EVALUACION FINAL SOBRE EL CONJUNTO DE PRUEBA")
    reportar("El conjunto de prueba no participo ni en la validacion cruzada ni")
    reportar("en la seleccion del modelo; se evalua una sola vez, aqui.\n")

    resultados_prueba = {}
    for etiqueta, modelo in entrenados.items():
        metricas = evaluar(modelo, X_prueba, y_prueba, etiqueta)
        resultados_prueba[etiqueta] = metricas
        reportar(f"--- {etiqueta} ---")
        reportar(formato_matriz(metricas))
        reportar(formato_metricas(metricas))
        reportar("")
        archivo = "matriz_" + etiqueta.lower().replace(" ", "_") + ".png"
        graficar_matriz(modelo, X_prueba, y_prueba, etiqueta, archivo)

    reportar("Reporte de clasificacion del modelo seleccionado "
             f"({nombre_ganador}):")
    reportar(classification_report(
        y_prueba, resultados_prueba[nombre_ganador]["prediccion"],
        target_names=["No transportado", "Si transportado"]))

    ruta = graficar_roc(entrenados, X_prueba, y_prueba)
    reportar(f"Grafica guardada: {ruta}")

    titulo("5. IMPORTANCIA DE LAS CARACTERISTICAS (por permutacion)")
    reportar("La importancia por permutacion mide cuanto cae la exactitud del")
    reportar("modelo al desordenar una variable. A diferencia de la importancia")
    reportar("por impureza, se calcula sobre datos que el modelo no vio y no")
    reportar("favorece a las variables de alta cardinalidad.\n")

    modelo_rf = entrenados["Random Forest"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        importancias = permutation_importance(
            modelo_rf, X_prueba, y_prueba, n_repeats=10,
            random_state=SEMILLA, n_jobs=-1, scoring="accuracy")

    nombres_columnas = list(X_prueba.columns)
    orden = np.argsort(-importancias.importances_mean)
    reportar(f"{'Caracteristica':>16}{'Caida de exactitud':>22}")
    for i in orden:
        reportar(f"{nombres_columnas[i]:>16}"
                 f"{importancias.importances_mean[i]:>16.4f}"
                 f" +/- {importancias.importances_std[i]:.4f}")

    ruta = graficar_importancias(importancias, nombres_columnas)
    reportar(f"Grafica guardada: {ruta}")

    titulo("6. RESUMEN COMPARATIVO (conjunto de prueba)")
    reportar(f"{'Modelo':<24}{'Exactitud':>11}{'Precision':>11}"
             f"{'Recall':>9}{'F1':>9}{'AUC':>9}")
    reportar(f"{'Linea base':<24}{base:>11.4f}{'-':>11}{'-':>9}{'-':>9}{'-':>9}")
    for etiqueta, m in resultados_prueba.items():
        reportar(f"{etiqueta:<24}{m['exactitud']:>11.4f}{m['precision']:>11.4f}"
                 f"{m['sensibilidad']:>9.4f}{m['f1']:>9.4f}{m['auc']:>9.4f}")

    # --- Comparacion opcional contra la implementacion manual ---
    comparar_con_implementacion_manual(
        X_entrena, y_entrena, X_prueba, y_prueba, resultados_prueba, reportar)

    joblib.dump({"modelo": modelo_final, "nombre": nombre_ganador}, RUTA_MODELO)

    ruta_registro = os.path.join(CARPETA_RESULTADOS, "resultados.txt")
    with open(ruta_registro, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(registro))

    print(f"\nModelo guardado en:    {RUTA_MODELO}")
    print(f"Bitacora guardada en:  {ruta_registro}")
    print("\nListo. Usa 'python main_framework.py --predecir' para probar.")


def comparar_con_implementacion_manual(X_entrena, y_entrena, X_prueba, y_prueba,
                                       resultados_prueba, reportar):
    """Compara el Random Forest de scikit-learn contra el implementado a mano.

    La implementacion manual vive en src/ (entrega anterior de este mismo
    modulo). Si no esta disponible, esta seccion simplemente se omite y el
    programa sigue corriendo: main_framework.py no depende de ella.

    Ambos modelos se entrenan con EXACTAMENTE las mismas filas y se evaluan con
    el mismo conjunto de prueba, asi que la comparacion es directa.
    """
    try:
        from src.data_loader import Preprocesador, derivar_caracteristicas as derivar_manual
        from src.random_forest import RandomForest as RandomForestManual
    except ImportError:
        return

    titulo("7. COMPARACION CONTRA LA IMPLEMENTACION MANUAL")
    reportar("Se entrena el Random Forest programado a mano (entrega anterior)")
    reportar("con las mismas filas de entrenamiento y se evalua con el mismo")
    reportar("conjunto de prueba, para que la comparacion sea directa.\n")

    preprocesador = Preprocesador().ajustar(derivar_manual(X_entrena))
    X_entrena_manual = preprocesador.transformar(derivar_manual(X_entrena))
    X_prueba_manual = preprocesador.transformar(derivar_manual(X_prueba))

    inicio = time.time()
    bosque_manual = RandomForestManual(
        n_arboles=300, max_profundidad=14, min_muestras_hoja=2, semilla=SEMILLA,
        calcular_oob=False,
    ).entrenar(X_entrena_manual, y_entrena.to_numpy())
    tiempo_manual = time.time() - inicio

    proba = bosque_manual.predecir_proba(X_prueba_manual)[:, 1]
    prediccion = (proba >= 0.5).astype(int)

    exactitud = accuracy_score(y_prueba, prediccion)
    auc = roc_auc_score(y_prueba, proba)
    f1 = f1_score(y_prueba, prediccion)
    sklearn_rf = resultados_prueba["Random Forest"]

    reportar(f"{'Implementacion':<30}{'Exactitud':>11}{'F1':>9}{'AUC':>9}{'Tiempo':>10}")
    reportar(f"{'Manual (main.py, 300 arboles)':<30}{exactitud:>11.4f}"
             f"{f1:>9.4f}{auc:>9.4f}{tiempo_manual:>9.1f}s")
    reportar(f"{'scikit-learn (300 arboles)':<30}{sklearn_rf['exactitud']:>11.4f}"
             f"{sklearn_rf['f1']:>9.4f}{sklearn_rf['auc']:>9.4f}"
             f"{'':>10}")
    reportar(f"\nDiferencia de exactitud: "
             f"{sklearn_rf['exactitud'] - exactitud:+.4f} a favor de scikit-learn")


# --- Modos de prediccion ----------------------------------------------------

def cargar_modelo():
    """Carga el modelo entrenado desde disco.

    Raises:
        SystemExit: si todavia no se ha entrenado ningun modelo.
    """
    if not os.path.exists(RUTA_MODELO):
        print("No hay un modelo entrenado todavia.")
        print("Ejecuta primero:  python main_framework.py --entrenar")
        raise SystemExit(1)
    return joblib.load(RUTA_MODELO)


def _preguntar(mensaje, convertir=str, opciones=None, por_defecto=None):
    """Pide un dato al usuario por consola, validando la respuesta."""
    sufijo = f" {opciones}" if opciones else ""
    if por_defecto is not None:
        sufijo += f" [{por_defecto}]"

    while True:
        respuesta = input(f"  {mensaje}{sufijo}: ").strip()
        if not respuesta:
            return por_defecto
        if opciones and respuesta not in opciones:
            print(f"    Valor no valido. Opciones: {', '.join(opciones)}")
            continue
        try:
            return convertir(respuesta)
        except ValueError:
            print("    No se pudo interpretar el valor, intenta de nuevo.")


def modo_predecir(args):
    """Pide los datos de un pasajero por consola y muestra la prediccion."""
    guardado = cargar_modelo()
    modelo = guardado["modelo"]

    titulo("PREDICCION INTERACTIVA")
    print(f"Modelo en uso: {guardado['nombre']}")
    print("Deja vacio cualquier campo que no conozcas: el imputador del")
    print("pipeline lo rellena con el valor tipico del entrenamiento.\n")

    fila = {
        "PassengerId": "9999_01",
        "HomePlanet": _preguntar("Planeta de origen", opciones=["Earth", "Europa", "Mars"]),
        "CryoSleep": _preguntar("En criosueno", opciones=["True", "False"]),
        "Cabin": _preguntar("Cabina (formato cubierta/numero/lado, ej. F/123/S)"),
        "Destination": _preguntar("Destino",
                                  opciones=["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]),
        "Age": _preguntar("Edad", convertir=float),
        "VIP": _preguntar("Pasajero VIP", opciones=["True", "False"]),
        "RoomService": _preguntar("Gasto en RoomService", convertir=float, por_defecto=0.0),
        "FoodCourt": _preguntar("Gasto en FoodCourt", convertir=float, por_defecto=0.0),
        "ShoppingMall": _preguntar("Gasto en ShoppingMall", convertir=float, por_defecto=0.0),
        "Spa": _preguntar("Gasto en Spa", convertir=float, por_defecto=0.0),
        "VRDeck": _preguntar("Gasto en VRDeck", convertir=float, por_defecto=0.0),
        "Name": None,
    }

    # Las respuestas de texto "True"/"False" se convierten al booleano que
    # tendria la columna si viniera del CSV original.
    for columna in ("CryoSleep", "VIP"):
        if fila[columna] is not None:
            fila[columna] = (fila[columna] == "True")

    # El pipeline recibe el DataFrame crudo: el mismo objeto se encarga de
    # derivar caracteristicas, imputar, codificar y clasificar.
    proba = float(modelo.predict_proba(pd.DataFrame([fila]))[0, 1])

    titulo("RESULTADO")
    print(f"  {'SI' if proba >= 0.5 else 'NO'} fue transportado "
          f"(probabilidad {proba:.1%})")

    if 0.4 <= proba <= 0.6:
        print("\n  Aviso: la probabilidad esta cerca del 50%, el modelo no")
        print("  distingue con claridad este caso.")


def modo_predecir_csv(args):
    """Predice en lote todas las filas de un CSV y guarda el resultado."""
    guardado = cargar_modelo()
    modelo = guardado["modelo"]

    titulo(f"PREDICCION EN LOTE: {args.predecir_csv}")
    df = pd.read_csv(args.predecir_csv)
    print(f"Modelo en uso: {guardado['nombre']}")
    print(f"Filas a predecir: {len(df)}")

    proba = modelo.predict_proba(df)[:, 1]
    prediccion = proba >= 0.5

    salida = pd.DataFrame({
        "PassengerId": df["PassengerId"],
        "Transported": prediccion,
        "Probabilidad": np.round(proba, 4),
    })

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(CARPETA_RESULTADOS, "predicciones.csv")
    salida.to_csv(ruta, index=False)

    print(f"\nPredichos como transportados:     {int(prediccion.sum())} "
          f"({100 * prediccion.mean():.1f}%)")
    print(f"Predichos como no transportados:  {int((~prediccion).sum())}")
    print(f"\nArchivo guardado en: {ruta}")
    print("\nPrimeras 10 predicciones:")
    print(salida.head(10).to_string(index=False))


# --- Interfaz de linea de comandos ------------------------------------------

def construir_analizador():
    """Define los argumentos que acepta el programa."""
    analizador = argparse.ArgumentParser(
        description="Random Forest con scikit-learn aplicado al dataset "
                    "Spaceship Titanic.")

    grupo = analizador.add_mutually_exclusive_group()
    grupo.add_argument("--entrenar", action="store_true",
                       help="ajusta, compara y evalua los modelos (modo por defecto)")
    grupo.add_argument("--predecir", action="store_true",
                       help="predice un pasajero capturado por consola")
    grupo.add_argument("--predecir-csv", metavar="RUTA",
                       help="predice todas las filas de un CSV sin etiqueta")

    analizador.add_argument("--datos", default="data/train.csv",
                            help="CSV etiquetado de entrada")

    return analizador


def main():
    """Punto de entrada: interpreta los argumentos y ejecuta el modo pedido."""
    args = construir_analizador().parse_args()

    if args.predecir:
        modo_predecir(args)
    elif args.predecir_csv:
        modo_predecir_csv(args)
    else:
        modo_entrenar(args)


if __name__ == "__main__":
    sys.exit(main())
