"""
Pruebas de correccion de la implementacion.

El objetivo de este archivo no es medir desempeno sobre el dataset real, sino
demostrar que el algoritmo esta bien programado. Cada prueba compara la salida
del codigo contra un resultado que se conoce de antemano por calculo manual o
por construccion del ejemplo.

Se ejecuta con:  python verificar.py
"""

import numpy as np

from src.decision_tree import ArbolDecision, impureza_entropia, impureza_gini
from src.metrics import calcular_auc, calcular_metricas, matriz_confusion
from src.random_forest import RandomForest

pruebas_ejecutadas = 0
pruebas_fallidas = 0


def verificar(condicion, descripcion, detalle=""):
    """Registra el resultado de una comprobacion y lo imprime."""
    global pruebas_ejecutadas, pruebas_fallidas
    pruebas_ejecutadas += 1
    if condicion:
        print(f"  [OK]    {descripcion}")
    else:
        pruebas_fallidas += 1
        print(f"  [FALLA] {descripcion}  {detalle}")


def seccion(nombre):
    """Imprime el encabezado de un grupo de pruebas."""
    print(f"\n{nombre}")
    print("-" * len(nombre))


# --- 1. Medidas de impureza -------------------------------------------------

def probar_impureza():
    """Los valores de Gini y entropia se conocen exactamente en casos limite."""
    seccion("1. Medidas de impureza")

    # Nodo puro: todas las muestras de la misma clase -> impureza 0.
    verificar(abs(impureza_gini(0.0)) < 1e-12, "Gini de un nodo puro (p=0) es 0")
    verificar(abs(impureza_gini(1.0)) < 1e-12, "Gini de un nodo puro (p=1) es 0")

    # Nodo mitad y mitad: Gini = 1 - 0.25 - 0.25 = 0.5, que es su maximo.
    verificar(abs(impureza_gini(0.5) - 0.5) < 1e-12, "Gini de un nodo 50/50 es 0.5")

    # Con p=0.25: 1 - 0.0625 - 0.5625 = 0.375
    verificar(abs(impureza_gini(0.25) - 0.375) < 1e-12, "Gini con p=0.25 es 0.375")

    # Entropia: 0 en nodo puro, 1 bit en nodo 50/50.
    verificar(abs(float(impureza_entropia(0.0))) < 1e-12, "Entropia de un nodo puro es 0")
    verificar(abs(float(impureza_entropia(0.5)) - 1.0) < 1e-12,
              "Entropia de un nodo 50/50 es 1 bit")


# --- 2. Aprendizaje de una regla conocida -----------------------------------

def probar_regla_conocida():
    """El arbol debe recuperar exactamente una regla que se conoce de antemano."""
    seccion("2. El arbol recupera una regla conocida")

    # Se construye un problema donde la etiqueta depende de una sola condicion:
    # y = 1 si y solo si la caracteristica 0 es mayor que 5. Las otras tres
    # caracteristicas son ruido puro y el arbol deberia ignorarlas.
    generador = np.random.default_rng(0)
    X = generador.uniform(0, 10, size=(500, 4))
    y = (X[:, 0] > 5).astype(int)

    arbol = ArbolDecision(max_profundidad=3).entrenar(X, y)
    acierto = float((arbol.predecir(X) == y).mean())

    verificar(acierto == 1.0, "Acierto perfecto sobre una regla separable",
              f"(obtenido {acierto:.4f})")
    verificar(arbol.raiz.caracteristica == 0,
              "La raiz divide por la caracteristica correcta",
              f"(uso la caracteristica {arbol.raiz.caracteristica})")
    verificar(abs(arbol.raiz.umbral - 5.0) < 0.1,
              "El umbral aprendido es aproximadamente 5",
              f"(obtenido {arbol.raiz.umbral:.3f})")
    verificar(arbol.importancias_[0] > 0.99,
              "Toda la importancia se concentra en la caracteristica util",
              f"(obtenido {arbol.importancias_[0]:.4f})")

    # Una regla con dos condiciones (AND) requiere dos niveles de profundidad.
    y2 = ((X[:, 0] > 5) & (X[:, 1] < 3)).astype(int)
    arbol2 = ArbolDecision(max_profundidad=4).entrenar(X, y2)
    verificar(float((arbol2.predecir(X) == y2).mean()) == 1.0,
              "Acierto perfecto sobre una regla con dos condiciones (AND)")


# --- 3. Criterios de paro ---------------------------------------------------

def probar_criterios_de_paro():
    """Los limites de profundidad y de tamano de hoja deben respetarse."""
    seccion("3. Criterios de paro y poda")

    generador = np.random.default_rng(1)
    X = generador.normal(size=(400, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    for limite in (1, 3, 5):
        arbol = ArbolDecision(max_profundidad=limite).entrenar(X, y)
        verificar(arbol.profundidad_alcanzada_ <= limite,
                  f"Con max_profundidad={limite} el arbol no lo excede",
                  f"(alcanzo {arbol.profundidad_alcanzada_})")

    # Un arbol de profundidad 0 no puede dividir: es una sola hoja.
    arbol_hoja = ArbolDecision(max_profundidad=0).entrenar(X, y)
    verificar(arbol_hoja.n_nodos_ == 1 and arbol_hoja.raiz.es_hoja,
              "Con max_profundidad=0 el arbol es una sola hoja")

    # Un nodo puro no debe dividirse aunque quede profundidad disponible.
    y_puro = np.ones(len(X), dtype=int)
    arbol_puro = ArbolDecision(max_profundidad=10).entrenar(X, y_puro)
    verificar(arbol_puro.n_nodos_ == 1, "Un conjunto de una sola clase no se divide")

    # Con hojas grandes el arbol debe quedar mas pequeno que sin restriccion.
    arbol_libre = ArbolDecision(min_muestras_hoja=1).entrenar(X, y)
    arbol_restringido = ArbolDecision(min_muestras_hoja=50).entrenar(X, y)
    verificar(arbol_restringido.n_nodos_ < arbol_libre.n_nodos_,
              "min_muestras_hoja mas grande produce un arbol mas pequeno",
              f"({arbol_restringido.n_nodos_} vs {arbol_libre.n_nodos_} nodos)")


# --- 4. Probabilidades y predicciones ---------------------------------------

def probar_probabilidades():
    """Las probabilidades deben ser validas y coherentes con las clases."""
    seccion("4. Probabilidades y predicciones")

    generador = np.random.default_rng(2)
    X = generador.normal(size=(300, 4))
    y = (X[:, 2] > 0).astype(int)

    arbol = ArbolDecision(max_profundidad=4).entrenar(X, y)
    proba = arbol.predecir_proba(X)

    verificar(np.allclose(proba.sum(axis=1), 1.0),
              "Las dos probabilidades de cada muestra suman 1")
    verificar(bool(((proba >= 0) & (proba <= 1)).all()),
              "Todas las probabilidades estan entre 0 y 1")
    verificar(bool((arbol.predecir(X) == (proba[:, 1] >= 0.5).astype(int)).all()),
              "predecir() coincide con aplicar el umbral 0.5 a predecir_proba()")

    # Con umbral 0 todo se clasifica como positivo; con umbral mayor a 1, nada.
    verificar(bool((arbol.predecir(X, umbral=0.0) == 1).all()),
              "Con umbral 0 todas las predicciones son de la clase 1")
    verificar(bool((arbol.predecir(X, umbral=1.01) == 0).all()),
              "Con umbral mayor a 1 todas las predicciones son de la clase 0")


# --- 5. Metricas ------------------------------------------------------------

def probar_metricas():
    """Las metricas se comparan contra conteos hechos a mano."""
    seccion("5. Matriz de confusion y metricas")

    # Caso construido a mano:
    #   reales:    1 1 1 1 0 0 0 0
    #   predichos: 1 1 1 0 1 0 0 0
    # -> VP=3, FN=1, FP=1, VN=3
    y_real = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])

    matriz = matriz_confusion(y_real, y_pred)
    verificar(matriz[1, 1] == 3, "Verdaderos positivos correctos", f"(obtenido {matriz[1, 1]})")
    verificar(matriz[1, 0] == 1, "Falsos negativos correctos", f"(obtenido {matriz[1, 0]})")
    verificar(matriz[0, 1] == 1, "Falsos positivos correctos", f"(obtenido {matriz[0, 1]})")
    verificar(matriz[0, 0] == 3, "Verdaderos negativos correctos", f"(obtenido {matriz[0, 0]})")
    verificar(matriz.sum() == len(y_real), "La matriz suma el total de muestras")

    m = calcular_metricas(y_real, y_pred)
    # Exactitud = 6/8 = 0.75; precision = 3/4 = 0.75; sensibilidad = 3/4 = 0.75
    verificar(abs(m["exactitud"] - 0.75) < 1e-12, "Exactitud = 0.75")
    verificar(abs(m["precision"] - 0.75) < 1e-12, "Precision = 0.75")
    verificar(abs(m["sensibilidad"] - 0.75) < 1e-12, "Sensibilidad = 0.75")
    verificar(abs(m["especificidad"] - 0.75) < 1e-12, "Especificidad = 0.75")
    verificar(abs(m["f1"] - 0.75) < 1e-12, "F1 = 0.75")

    # Prediccion perfecta y prediccion invertida.
    perfecta = calcular_metricas(y_real, y_real)
    verificar(perfecta["exactitud"] == 1.0, "Prediccion perfecta da exactitud 1")
    invertida = calcular_metricas(y_real, 1 - y_real)
    verificar(invertida["exactitud"] == 0.0, "Prediccion invertida da exactitud 0")

    # AUC: separacion perfecta = 1.0; puntuaciones sin informacion = 0.5.
    verificar(abs(calcular_auc(y_real, y_real.astype(float)) - 1.0) < 1e-12,
              "AUC de una separacion perfecta es 1.0")
    verificar(abs(calcular_auc(y_real, np.full(len(y_real), 0.5)) - 0.5) < 1e-12,
              "AUC de una puntuacion constante es 0.5")


# --- 6. Random Forest -------------------------------------------------------

def probar_bosque():
    """El ensamble debe entrenarse bien y mejorar a un arbol individual ruidoso."""
    seccion("6. Random Forest")

    generador = np.random.default_rng(3)
    X = generador.normal(size=(600, 8))
    # Problema con senal y ruido: la frontera depende de tres caracteristicas
    # y ademas se voltea el 10% de las etiquetas para dificultarlo.
    y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0).astype(int)
    ruido = generador.random(len(y)) < 0.10
    y[ruido] = 1 - y[ruido]

    bosque = RandomForest(n_arboles=25, max_profundidad=8, semilla=7).entrenar(X, y)

    verificar(len(bosque.arboles) == 25, "El bosque contiene el numero pedido de arboles")
    verificar(bosque.oob_score_ is not None and 0.5 < bosque.oob_score_ < 1.0,
              "La estimacion out-of-bag esta en un rango razonable",
              f"(obtenido {bosque.oob_score_})")
    verificar(abs(bosque.importancias_.sum() - 1.0) < 1e-9,
              "Las importancias del bosque suman 1",
              f"(suman {bosque.importancias_.sum():.6f})")

    proba = bosque.predecir_proba(X)
    verificar(np.allclose(proba.sum(axis=1), 1.0),
              "Las probabilidades del bosque suman 1 por muestra")

    # Los arboles deben ser distintos entre si: si el azar funcionara mal,
    # todos usarian la misma caracteristica en la raiz.
    raices = {a.raiz.caracteristica for a in bosque.arboles}
    verificar(len(raices) > 1,
              "Los arboles del bosque son diversos (distintas raices)",
              f"(raices distintas: {len(raices)})")

    # Un arbol profundo sin poda sobreajusta el ruido; el bosque debe superarlo
    # al evaluar en datos nuevos generados con la misma regla.
    X_nuevo = generador.normal(size=(400, 8))
    y_nuevo = ((X_nuevo[:, 0] + X_nuevo[:, 1] - X_nuevo[:, 2]) > 0).astype(int)

    arbol_solo = ArbolDecision(semilla=7).entrenar(X, y)
    acierto_arbol = float((arbol_solo.predecir(X_nuevo) == y_nuevo).mean())
    acierto_bosque = float((bosque.predecir(X_nuevo) == y_nuevo).mean())

    verificar(acierto_bosque > acierto_arbol,
              "El bosque generaliza mejor que un arbol sin poda",
              f"(bosque {acierto_bosque:.4f} vs arbol {acierto_arbol:.4f})")


# --- 7. Reproducibilidad ----------------------------------------------------

def probar_reproducibilidad():
    """Con la misma semilla, dos entrenamientos deben ser identicos."""
    seccion("7. Reproducibilidad")

    generador = np.random.default_rng(4)
    X = generador.normal(size=(300, 6))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)

    bosque_a = RandomForest(n_arboles=10, max_profundidad=6, semilla=99).entrenar(X, y)
    bosque_b = RandomForest(n_arboles=10, max_profundidad=6, semilla=99).entrenar(X, y)
    verificar(np.array_equal(bosque_a.predecir(X), bosque_b.predecir(X)),
              "Dos bosques con la misma semilla dan predicciones identicas")

    bosque_c = RandomForest(n_arboles=10, max_profundidad=6, semilla=100).entrenar(X, y)
    verificar(not np.array_equal(bosque_a.predecir_proba(X), bosque_c.predecir_proba(X)),
              "Dos bosques con semillas distintas dan bosques distintos")


def main():
    """Ejecuta todas las pruebas y devuelve un codigo de salida."""
    print("=" * 70)
    print("VERIFICACION DE LA IMPLEMENTACION")
    print("=" * 70)

    probar_impureza()
    probar_regla_conocida()
    probar_criterios_de_paro()
    probar_probabilidades()
    probar_metricas()
    probar_bosque()
    probar_reproducibilidad()

    print()
    print("=" * 70)
    if pruebas_fallidas == 0:
        print(f"RESULTADO: {pruebas_ejecutadas} pruebas ejecutadas, todas correctas.")
    else:
        print(f"RESULTADO: {pruebas_fallidas} de {pruebas_ejecutadas} pruebas fallaron.")
    print("=" * 70)

    return 1 if pruebas_fallidas else 0


if __name__ == "__main__":
    raise SystemExit(main())
