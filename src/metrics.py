"""
Matriz de confusion y metricas de clasificacion, calculadas a mano.

Convencion usada en todo el modulo:
    clase 1 = positiva = el pasajero SI fue transportado (Transported = True)
    clase 0 = negativa = el pasajero NO fue transportado

La matriz de confusion se organiza asi:

                        Prediccion
                     0 (No)   1 (Si)
    Real  0 (No)       VN       FP
          1 (Si)       FN       VP

    VP = verdaderos positivos   FP = falsos positivos
    VN = verdaderos negativos   FN = falsos negativos
"""

import numpy as np


def matriz_confusion(y_real, y_predicho):
    """Cuenta los cuatro tipos de acierto y error.

    Args:
        y_real: vector de etiquetas verdaderas (0/1).
        y_predicho: vector de etiquetas predichas (0/1).

    Returns:
        Matriz numpy 2x2 de enteros con la forma [[VN, FP], [FN, VP]].
    """
    y_real = np.asarray(y_real, dtype=int)
    y_predicho = np.asarray(y_predicho, dtype=int)

    vn = int(np.sum((y_real == 0) & (y_predicho == 0)))
    fp = int(np.sum((y_real == 0) & (y_predicho == 1)))
    fn = int(np.sum((y_real == 1) & (y_predicho == 0)))
    vp = int(np.sum((y_real == 1) & (y_predicho == 1)))

    return np.array([[vn, fp], [fn, vp]])


def _division_segura(numerador, denominador):
    """Divide evitando el error cuando el denominador es cero.

    Un denominador cero significa que la metrica no esta definida (por ejemplo,
    precision cuando el modelo nunca predice la clase positiva). Se devuelve 0.0
    en ese caso, que es la convencion habitual.
    """
    return float(numerador) / float(denominador) if denominador else 0.0


def calcular_metricas(y_real, y_predicho, y_proba=None):
    """Calcula el conjunto completo de metricas de clasificacion binaria.

    Args:
        y_real: vector de etiquetas verdaderas (0/1).
        y_predicho: vector de etiquetas predichas (0/1).
        y_proba: probabilidades de la clase positiva. Si se proporciona, se
            agrega el AUC de la curva ROC.

    Returns:
        Diccionario con las metricas y los cuatro conteos de la matriz.
    """
    matriz = matriz_confusion(y_real, y_predicho)
    vn, fp = int(matriz[0, 0]), int(matriz[0, 1])
    fn, vp = int(matriz[1, 0]), int(matriz[1, 1])
    total = vn + fp + fn + vp

    # Precision: de todo lo que el modelo marco como positivo, cuanto acerto.
    precision = _division_segura(vp, vp + fp)
    # Sensibilidad (recall): de todos los positivos reales, cuantos detecto.
    sensibilidad = _division_segura(vp, vp + fn)
    # Especificidad: de todos los negativos reales, cuantos identifico bien.
    especificidad = _division_segura(vn, vn + fp)
    # F1: media armonica entre precision y sensibilidad. Penaliza que una de
    # las dos sea baja, cosa que el promedio simple no haria.
    f1 = _division_segura(2 * precision * sensibilidad, precision + sensibilidad)

    resultado = {
        "exactitud": _division_segura(vp + vn, total),
        "precision": precision,
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "f1": f1,
        "vp": vp, "vn": vn, "fp": fp, "fn": fn,
        "n": total,
    }

    if y_proba is not None:
        resultado["auc"] = calcular_auc(y_real, y_proba)

    return resultado


def curva_roc(y_real, y_proba):
    """Construye la curva ROC recorriendo todos los umbrales posibles.

    La idea: se ordenan las muestras de mayor a menor probabilidad predicha y se
    va bajando el umbral de decision. Con cada paso se recalcula la tasa de
    verdaderos positivos (TPR) y la de falsos positivos (FPR). El resultado es
    la curva que traza el modelo al variar su umbral.

    Manejo de empates: un arbol asigna la misma probabilidad a todas las
    muestras que caen en la misma hoja, asi que los empates son muy frecuentes.
    Un umbral no puede separar dos muestras con identica probabilidad, asi que
    todas las muestras empatadas tienen que colapsarse en un solo punto de la
    curva. Si no se hiciera, el AUC dependeria del orden en que vinieron las
    filas y quedaria inflado.

    Args:
        y_real: vector de etiquetas verdaderas (0/1).
        y_proba: probabilidades de la clase positiva.

    Returns:
        Tupla (fpr, tpr) con dos arreglos numpy que empiezan en (0, 0).
    """
    y_real = np.asarray(y_real, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    orden = np.argsort(-y_proba, kind="mergesort")
    y_ordenado = y_real[orden]
    proba_ordenada = y_proba[orden]

    total_positivos = int(y_ordenado.sum())
    total_negativos = len(y_ordenado) - total_positivos
    if total_positivos == 0 or total_negativos == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    vp_acumulado = np.cumsum(y_ordenado)
    fp_acumulado = np.cumsum(1 - y_ordenado)

    # Solo se conservan las posiciones donde la probabilidad cambia (fin de cada
    # grupo de empates), mas la ultima posicion para cerrar la curva en (1, 1).
    cambios = np.flatnonzero(proba_ordenada[:-1] != proba_ordenada[1:])
    posiciones = np.append(cambios, len(y_ordenado) - 1)

    tpr = np.concatenate([[0.0], vp_acumulado[posiciones] / total_positivos])
    fpr = np.concatenate([[0.0], fp_acumulado[posiciones] / total_negativos])

    return fpr, tpr


def calcular_auc(y_real, y_proba):
    """Area bajo la curva ROC, calculada con la regla del trapecio.

    Interpretacion: es la probabilidad de que el modelo asigne una puntuacion
    mas alta a un positivo tomado al azar que a un negativo tomado al azar.
    0.5 equivale a adivinar y 1.0 es una separacion perfecta.

    Args:
        y_real: vector de etiquetas verdaderas (0/1).
        y_proba: probabilidades de la clase positiva.

    Returns:
        El AUC como float.
    """
    fpr, tpr = curva_roc(y_real, y_proba)
    # Regla del trapecio: se suma el area de cada trapecio entre puntos vecinos.
    anchos = np.diff(fpr)
    alturas_medias = (tpr[1:] + tpr[:-1]) / 2.0
    return float(np.sum(anchos * alturas_medias))


def formato_matriz(matriz, titulo="Matriz de confusion"):
    """Devuelve la matriz de confusion como una tabla de texto legible.

    Args:
        matriz: matriz 2x2 con la forma [[VN, FP], [FN, VP]].
        titulo: encabezado de la tabla.

    Returns:
        Una cadena lista para imprimir en consola.
    """
    vn, fp = int(matriz[0, 0]), int(matriz[0, 1])
    fn, vp = int(matriz[1, 0]), int(matriz[1, 1])

    return (
        f"{titulo}\n"
        f"                       PREDICCION\n"
        f"                  No transp.   Si transp.\n"
        f"  REAL No transp.  {vn:9d}    {fp:9d}   <- {fp} falsos positivos\n"
        f"       Si transp.  {fn:9d}    {vp:9d}\n"
        f"                        ^\n"
        f"                        {fn} falsos negativos"
    )


def formato_metricas(metricas, titulo="Metricas"):
    """Devuelve las metricas como un bloque de texto alineado.

    Args:
        metricas: diccionario producido por calcular_metricas.
        titulo: encabezado del bloque.

    Returns:
        Una cadena lista para imprimir en consola.
    """
    lineas = [titulo, "-" * len(titulo)]
    lineas.append(f"  Exactitud (accuracy)      {metricas['exactitud']:.4f}")
    lineas.append(f"  Precision                 {metricas['precision']:.4f}")
    lineas.append(f"  Sensibilidad (recall)     {metricas['sensibilidad']:.4f}")
    lineas.append(f"  Especificidad             {metricas['especificidad']:.4f}")
    lineas.append(f"  F1                        {metricas['f1']:.4f}")
    if "auc" in metricas:
        lineas.append(f"  AUC (ROC)                 {metricas['auc']:.4f}")
    lineas.append(f"  Muestras evaluadas        {metricas['n']}")
    return "\n".join(lineas)
