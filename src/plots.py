"""
Graficas de resultados para el reporte.

Este modulo es exclusivamente de visualizacion: matplotlib se usa para dibujar,
nunca para aprender ni para calcular metricas. Todos los numeros que se grafican
vienen ya calculados por los modulos propios (metrics, decision_tree,
random_forest).

Se usa el backend "Agg" para que las figuras se guarden como archivos PNG sin
necesidad de abrir ventanas, de modo que el programa corra igual en una terminal
sin entorno grafico.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import curva_roc

CARPETA_SALIDA = "resultados"


def _guardar(figura, nombre_archivo):
    """Guarda la figura en la carpeta de resultados y libera la memoria."""
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    ruta = os.path.join(CARPETA_SALIDA, nombre_archivo)
    figura.tight_layout()
    figura.savefig(ruta, dpi=150)
    plt.close(figura)
    return ruta


def graficar_matriz_confusion(matriz, titulo, nombre_archivo):
    """Dibuja la matriz de confusion como un mapa de calor anotado.

    Args:
        matriz: matriz 2x2 con la forma [[VN, FP], [FN, VP]].
        titulo: titulo de la figura.
        nombre_archivo: nombre del PNG de salida.

    Returns:
        La ruta del archivo generado.
    """
    figura, eje = plt.subplots(figsize=(5.5, 4.6))
    imagen = eje.imshow(matriz, cmap="Blues")

    etiquetas = ["No transportado", "Si transportado"]
    eje.set_xticks([0, 1], labels=etiquetas)
    eje.set_yticks([0, 1], labels=etiquetas)
    eje.set_xlabel("Prediccion del modelo")
    eje.set_ylabel("Valor real")
    eje.set_title(titulo)

    # Se anota el conteo y el porcentaje dentro de cada celda. El color del
    # texto se invierte en las celdas oscuras para que siga siendo legible.
    total = matriz.sum()
    umbral_color = matriz.max() / 2.0
    for fila in range(2):
        for columna in range(2):
            valor = int(matriz[fila, columna])
            color = "white" if valor > umbral_color else "black"
            eje.text(columna, fila, f"{valor}\n({100 * valor / total:.1f}%)",
                     ha="center", va="center", color=color, fontsize=12)

    figura.colorbar(imagen, ax=eje, label="Numero de pasajeros")
    return _guardar(figura, nombre_archivo)


def graficar_curva_profundidad(profundidades, acierto_entrena, acierto_valida,
                               nombre_archivo="curva_profundidad.png"):
    """Grafica el acierto en entrenamiento y validacion contra la profundidad.

    Es la grafica que muestra el sobreajuste: la curva de entrenamiento sigue
    subiendo mientras la de validacion se estanca y luego baja.

    Args:
        profundidades: lista de profundidades probadas.
        acierto_entrena: acierto en entrenamiento para cada profundidad.
        acierto_valida: acierto en validacion para cada profundidad.
        nombre_archivo: nombre del PNG de salida.

    Returns:
        La ruta del archivo generado.
    """
    figura, eje = plt.subplots(figsize=(7.5, 4.6))
    eje.plot(profundidades, acierto_entrena, "o-", label="Entrenamiento", color="#1f77b4")
    eje.plot(profundidades, acierto_valida, "s-", label="Validacion", color="#d62728")

    # Se marca la profundidad con mejor validacion, que es la que se elige.
    mejor = int(np.argmax(acierto_valida))
    eje.axvline(profundidades[mejor], color="gray", linestyle="--", linewidth=1)
    eje.annotate(f"mejor: profundidad {profundidades[mejor]}\n"
                 f"validacion {acierto_valida[mejor]:.4f}",
                 xy=(profundidades[mejor], acierto_valida[mejor]),
                 xytext=(10, -35), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

    eje.set_xlabel("Profundidad maxima del arbol")
    eje.set_ylabel("Exactitud")
    eje.set_title("Sobreajuste del arbol de decision al aumentar la profundidad")
    eje.legend()
    eje.grid(alpha=0.3)
    return _guardar(figura, nombre_archivo)


def graficar_curva_arboles(cantidades, acierto_valida, acierto_oob=None,
                           nombre_archivo="curva_arboles.png"):
    """Grafica el acierto del bosque conforme se agregan arboles.

    Args:
        cantidades: lista con el numero de arboles probado.
        acierto_valida: acierto en validacion para cada cantidad.
        acierto_oob: acierto out-of-bag para cada cantidad (opcional).
        nombre_archivo: nombre del PNG de salida.

    Returns:
        La ruta del archivo generado.
    """
    figura, eje = plt.subplots(figsize=(7.5, 4.6))
    eje.plot(cantidades, acierto_valida, "o-", label="Validacion", color="#d62728")
    if acierto_oob is not None:
        eje.plot(cantidades, acierto_oob, "^--", label="Out-of-bag", color="#2ca02c")

    eje.set_xlabel("Numero de arboles en el bosque")
    eje.set_ylabel("Exactitud")
    eje.set_title("Efecto del tamano del ensamble")
    eje.legend()
    eje.grid(alpha=0.3)
    return _guardar(figura, nombre_archivo)


def graficar_importancias(nombres, importancias, titulo,
                          nombre_archivo="importancias.png"):
    """Grafica la importancia de cada caracteristica en barras horizontales.

    Args:
        nombres: nombres de las caracteristicas.
        importancias: importancia normalizada de cada una.
        titulo: titulo de la figura.
        nombre_archivo: nombre del PNG de salida.

    Returns:
        La ruta del archivo generado.
    """
    orden = np.argsort(importancias)
    nombres_ordenados = [nombres[i] for i in orden]
    valores = np.asarray(importancias)[orden]

    figura, eje = plt.subplots(figsize=(7.5, 6))
    eje.barh(range(len(valores)), valores, color="#1f77b4")
    eje.set_yticks(range(len(valores)), labels=nombres_ordenados)
    eje.set_xlabel("Importancia (reduccion de impureza normalizada)")
    eje.set_title(titulo)
    eje.grid(axis="x", alpha=0.3)
    return _guardar(figura, nombre_archivo)


def graficar_roc(curvas, nombre_archivo="curva_roc.png"):
    """Dibuja una o varias curvas ROC en los mismos ejes.

    Args:
        curvas: lista de tuplas (etiqueta, y_real, y_proba, auc).
        nombre_archivo: nombre del PNG de salida.

    Returns:
        La ruta del archivo generado.
    """
    figura, eje = plt.subplots(figsize=(5.8, 5.4))

    for etiqueta, y_real, y_proba, auc in curvas:
        fpr, tpr = curva_roc(y_real, y_proba)
        eje.plot(fpr, tpr, label=f"{etiqueta} (AUC = {auc:.4f})")

    # La diagonal representa un clasificador que adivina al azar.
    eje.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar (AUC = 0.5)")

    eje.set_xlabel("Tasa de falsos positivos")
    eje.set_ylabel("Tasa de verdaderos positivos")
    eje.set_title("Curva ROC sobre el conjunto de prueba")
    eje.legend(loc="lower right")
    eje.grid(alpha=0.3)
    return _guardar(figura, nombre_archivo)
