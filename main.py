"""
Momento de Retroalimentacion: implementacion de una tecnica de aprendizaje
maquina SIN framework.

Algoritmo: arbol de decision (CART) y Random Forest, ambos programados a mano.
Dataset:   Spaceship Titanic (data/train.csv).
Autor:     Juan Pablo Perez  --  A01800483

Modos de uso desde la terminal:

    python main.py --entrenar
        Ejecuta el experimento completo: carga los datos, busca la mejor
        profundidad para un solo arbol, entrena el bosque, evalua ambos modelos
        sobre el conjunto de prueba, guarda las graficas en resultados/ y deja
        el modelo entrenado listo para predecir.

    python main.py --predecir
        Modo interactivo: pregunta los datos de un pasajero en la consola y
        responde si el modelo cree que fue transportado, con su probabilidad.

    python main.py --predecir-csv data/test.csv
        Predice en lote un CSV completo y escribe resultados/predicciones.csv.

    python main.py --arbol
        Imprime en texto el arbol de decision entrenado.

Este archivo se ejecuta con un interprete de Python normal; no requiere IDE ni
notebook.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from src.data_loader import cargar_datos, cargar_sin_etiqueta, derivar_caracteristicas
from src.decision_tree import ArbolDecision
from src.metrics import calcular_metricas, formato_matriz, formato_metricas, matriz_confusion
from src.random_forest import RandomForest
from src import plots

CARPETA_RESULTADOS = "resultados"
RUTA_MODELO = os.path.join(CARPETA_RESULTADOS, "modelo.pkl")

# Profundidades que se prueban para elegir la poda del arbol individual.
PROFUNDIDADES_A_PROBAR = list(range(1, 21))

# Tamanos de ensamble que se reportan en la curva de numero de arboles.
CANTIDADES_ARBOLES = [1, 5, 10, 25, 50, 75, 100]


def titulo(texto):
    """Imprime un encabezado de seccion para que la salida sea legible."""
    print()
    print("=" * 74)
    print(texto)
    print("=" * 74)


# --- Modo de entrenamiento --------------------------------------------------

def modo_entrenar(args):
    """Ejecuta el experimento completo y guarda modelo, metricas y graficas."""
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    registro = []   # se acumula todo lo impreso para guardarlo en un .txt

    def reportar(texto=""):
        print(texto)
        registro.append(str(texto))

    titulo("1. CARGA Y PREPARACION DE LOS DATOS")
    datos = cargar_datos(ruta=args.datos, semilla=args.semilla)
    reportar(f"Archivo: {args.datos}")
    reportar("Nota: data/test.csv es el set de Kaggle y no trae la etiqueta")
    reportar("      Transported, por eso el conjunto de prueba se obtiene")
    reportar("      particionando el archivo de entrenamiento.")
    reportar()
    reportar(datos.resumen())
    reportar()
    reportar(f"Caracteristicas usadas ({len(datos.nombres)}): {', '.join(datos.nombres)}")

    # Referencia minima: predecir siempre la clase mayoritaria. Cualquier
    # modelo util tiene que superar esto.
    clase_mayoritaria = int(round(datos.y_entrena.mean()))
    base = float((datos.y_prueba == clase_mayoritaria).mean())
    reportar(f"\nLinea base (predecir siempre la clase mayoritaria): {base:.4f}")

    titulo("2. ARBOL DE DECISION: EFECTO DE LA PROFUNDIDAD")
    reportar("Se entrena un arbol por cada profundidad y se compara su acierto")
    reportar("en entrenamiento contra el de validacion. La brecha entre ambas")
    reportar("curvas es la medida directa del sobreajuste.\n")
    reportar(f"{'Profundidad':>12} {'Entrenamiento':>15} {'Validacion':>12} {'Nodos':>8}")

    acierto_entrena, acierto_valida = [], []
    for profundidad in PROFUNDIDADES_A_PROBAR:
        arbol = ArbolDecision(max_profundidad=profundidad,
                              min_muestras_hoja=args.min_hoja,
                              criterio=args.criterio).entrenar(datos.X_entrena, datos.y_entrena)
        ac_tr = float((arbol.predecir(datos.X_entrena) == datos.y_entrena).mean())
        ac_va = float((arbol.predecir(datos.X_valida) == datos.y_valida).mean())
        acierto_entrena.append(ac_tr)
        acierto_valida.append(ac_va)
        reportar(f"{profundidad:>12} {ac_tr:>15.4f} {ac_va:>12.4f} {arbol.n_nodos_:>8}")

    mejor_profundidad = PROFUNDIDADES_A_PROBAR[int(np.argmax(acierto_valida))]
    reportar(f"\nMejor profundidad segun validacion: {mejor_profundidad}")

    # Un arbol sin poda, solo para dejar documentado el sobreajuste extremo.
    arbol_sin_podar = ArbolDecision(min_muestras_hoja=1).entrenar(datos.X_entrena, datos.y_entrena)
    reportar(f"Arbol SIN poda -> entrenamiento "
             f"{float((arbol_sin_podar.predecir(datos.X_entrena) == datos.y_entrena).mean()):.4f}, "
             f"validacion "
             f"{float((arbol_sin_podar.predecir(datos.X_valida) == datos.y_valida).mean()):.4f}, "
             f"{arbol_sin_podar.n_nodos_} nodos, profundidad {arbol_sin_podar.profundidad_alcanzada_}")

    ruta = plots.graficar_curva_profundidad(PROFUNDIDADES_A_PROBAR,
                                            acierto_entrena, acierto_valida)
    reportar(f"Grafica guardada: {ruta}")

    arbol_final = ArbolDecision(max_profundidad=mejor_profundidad,
                                min_muestras_hoja=args.min_hoja,
                                criterio=args.criterio).entrenar(datos.X_entrena, datos.y_entrena)

    titulo("3. RANDOM FOREST")
    reportar(f"Entrenando {args.arboles} arboles con bagging y subespacio")
    reportar(f"aleatorio de caracteristicas (sqrt de {len(datos.nombres)} = "
             f"{max(1, int(np.sqrt(len(datos.nombres))))} por nodo)...\n")

    inicio = time.time()
    bosque = RandomForest(n_arboles=args.arboles,
                          max_profundidad=args.profundidad_bosque,
                          min_muestras_hoja=args.min_hoja_bosque,
                          criterio=args.criterio,
                          semilla=args.semilla).entrenar(datos.X_entrena, datos.y_entrena,
                                                         verboso=True)
    reportar(f"\nEntrenamiento terminado en {time.time() - inicio:.1f} segundos.\n")
    reportar(bosque.resumen())

    cantidades = [k for k in CANTIDADES_ARBOLES if k <= args.arboles]
    curva_val = bosque.curva_por_numero_arboles(datos.X_valida, datos.y_valida, cantidades)
    curva_oob = bosque.curva_oob_por_numero_arboles(datos.X_entrena, datos.y_entrena, cantidades)

    reportar(f"\n{'Arboles':>9} {'Validacion':>12} {'Out-of-bag':>12}")
    for k, val, oob in zip(cantidades, curva_val, curva_oob):
        reportar(f"{k:>9} {val:>12.4f} {oob:>12.4f}")

    ruta = plots.graficar_curva_arboles(cantidades, curva_val, curva_oob)
    reportar(f"Grafica guardada: {ruta}")

    titulo("4. EVALUACION FINAL SOBRE EL CONJUNTO DE PRUEBA")
    reportar("El conjunto de prueba no se uso en ningun momento para entrenar")
    reportar("ni para elegir hiperparametros; se evalua una sola vez, aqui.\n")

    resultados_finales = {}
    curvas_roc = []

    for etiqueta, modelo, archivo in (
            (f"Arbol de decision (profundidad {mejor_profundidad})", arbol_final,
             "matriz_confusion_arbol.png"),
            (f"Random Forest ({args.arboles} arboles)", bosque,
             "matriz_confusion_bosque.png")):

        proba = modelo.predecir_proba(datos.X_prueba)[:, 1]
        prediccion = (proba >= 0.5).astype(int)
        metricas = calcular_metricas(datos.y_prueba, prediccion, proba)
        matriz = matriz_confusion(datos.y_prueba, prediccion)

        reportar()
        reportar(formato_matriz(matriz, f"{etiqueta} - matriz de confusion"))
        reportar()
        reportar(formato_metricas(metricas, f"{etiqueta} - metricas"))

        ruta = plots.graficar_matriz_confusion(matriz, etiqueta, archivo)
        reportar(f"Grafica guardada: {ruta}")

        resultados_finales[etiqueta] = metricas
        curvas_roc.append((etiqueta, datos.y_prueba, proba, metricas["auc"]))

    ruta = plots.graficar_roc(curvas_roc)
    reportar(f"\nGrafica guardada: {ruta}")

    titulo("5. IMPORTANCIA DE LAS CARACTERISTICAS (segun el bosque)")
    orden = np.argsort(-bosque.importancias_)
    reportar(f"{'Caracteristica':>16} {'Importancia':>13}")
    for i in orden:
        reportar(f"{datos.nombres[i]:>16} {bosque.importancias_[i]:>13.4f}")

    ruta = plots.graficar_importancias(datos.nombres, bosque.importancias_,
                                       f"Importancia de caracteristicas "
                                       f"(Random Forest, {args.arboles} arboles)")
    reportar(f"Grafica guardada: {ruta}")

    titulo("6. REGLAS APRENDIDAS POR EL ARBOL (primeros 3 niveles)")
    texto_arbol = arbol_final.a_texto(datos.nombres, max_profundidad=3)
    reportar(texto_arbol)

    # El arbol completo se guarda aparte porque puede ser muy largo.
    ruta_arbol = os.path.join(CARPETA_RESULTADOS, "arbol_completo.txt")
    with open(ruta_arbol, "w", encoding="utf-8") as archivo:
        archivo.write(arbol_final.a_texto(datos.nombres))
    reportar(f"\nArbol completo guardado en: {ruta_arbol}")

    titulo("7. RESUMEN COMPARATIVO (conjunto de prueba)")
    reportar(f"{'Modelo':<40} {'Exactitud':>10} {'F1':>8} {'AUC':>8}")
    reportar(f"{'Linea base (clase mayoritaria)':<40} {base:>10.4f} {'-':>8} {'-':>8}")
    for etiqueta, metricas in resultados_finales.items():
        reportar(f"{etiqueta:<40} {metricas['exactitud']:>10.4f} "
                 f"{metricas['f1']:>8.4f} {metricas['auc']:>8.4f}")

    # Se guarda el modelo para que los modos de prediccion no reentrenen.
    with open(RUTA_MODELO, "wb") as archivo:
        pickle.dump({
            "bosque": bosque,
            "arbol": arbol_final,
            "preprocesador": datos.preprocesador,
            "nombres": datos.nombres,
            "mejor_profundidad": mejor_profundidad,
        }, archivo)

    ruta_registro = os.path.join(CARPETA_RESULTADOS, "resultados.txt")
    with open(ruta_registro, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(registro))

    print(f"\nModelo guardado en:      {RUTA_MODELO}")
    print(f"Bitacora guardada en:    {ruta_registro}")
    print("\nListo. Usa 'python main.py --predecir' para probar predicciones.")


# --- Modos de prediccion ----------------------------------------------------

def cargar_modelo():
    """Carga el modelo entrenado desde disco.

    Returns:
        El diccionario guardado por el modo de entrenamiento.

    Raises:
        SystemExit: si todavia no se ha entrenado ningun modelo.
    """
    if not os.path.exists(RUTA_MODELO):
        print("No hay un modelo entrenado todavia.")
        print("Ejecuta primero:  python main.py --entrenar")
        raise SystemExit(1)

    with open(RUTA_MODELO, "rb") as archivo:
        return pickle.load(archivo)


def _preguntar(mensaje, convertir=str, opciones=None, por_defecto=None):
    """Pide un dato al usuario por consola, validando la respuesta.

    Args:
        mensaje: texto que se muestra.
        convertir: funcion que convierte el texto a su tipo final.
        opciones: lista de respuestas validas, o None si no se restringe.
        por_defecto: valor que se usa si el usuario deja el campo vacio.

    Returns:
        El valor convertido, o None si el usuario dejo el campo vacio y no hay
        valor por defecto (se tratara como dato faltante y se imputara).
    """
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
    modelo = cargar_modelo()

    titulo("PREDICCION INTERACTIVA")
    print("Escribe los datos del pasajero. Deja vacio cualquier campo que no")
    print("conozcas: se imputara con el valor tipico del entrenamiento.\n")

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

    # Se arma un DataFrame de una sola fila y se pasa por exactamente el mismo
    # preprocesamiento que se uso en el entrenamiento.
    df = pd.DataFrame([fila])
    df = derivar_caracteristicas(df)
    X = modelo["preprocesador"].transformar(df)

    proba_bosque = float(modelo["bosque"].predecir_proba(X)[0, 1])
    proba_arbol = float(modelo["arbol"].predecir_proba(X)[0, 1])

    titulo("RESULTADO")
    print(f"  Random Forest:      {'SI' if proba_bosque >= 0.5 else 'NO'} fue transportado "
          f"(probabilidad {proba_bosque:.1%})")
    print(f"  Arbol de decision:  {'SI' if proba_arbol >= 0.5 else 'NO'} fue transportado "
          f"(probabilidad {proba_arbol:.1%})")

    # Una probabilidad cercana a 0.5 significa que el modelo no tiene una
    # opinion clara; conviene decirselo al usuario en vez de dar un si o no seco.
    if 0.4 <= proba_bosque <= 0.6:
        print("\n  Aviso: la probabilidad esta cerca del 50%, el modelo no")
        print("  distingue con claridad este caso.")


def modo_predecir_csv(args):
    """Predice en lote todas las filas de un CSV y guarda el resultado."""
    modelo = cargar_modelo()

    titulo(f"PREDICCION EN LOTE: {args.predecir_csv}")
    X, identificadores = cargar_sin_etiqueta(args.predecir_csv, modelo["preprocesador"])
    print(f"Filas a predecir: {len(X)}")

    proba = modelo["bosque"].predecir_proba(X)[:, 1]
    prediccion = proba >= 0.5

    salida = pd.DataFrame({
        "PassengerId": identificadores,
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


def modo_arbol(args):
    """Imprime en texto el arbol de decision entrenado."""
    modelo = cargar_modelo()
    titulo(f"ARBOL DE DECISION ENTRENADO (profundidad {modelo['mejor_profundidad']})")
    print(modelo["arbol"].a_texto(modelo["nombres"], max_profundidad=args.niveles))
    print(f"\n(Mostrando {args.niveles} niveles. El arbol completo esta en "
          f"{CARPETA_RESULTADOS}/arbol_completo.txt)")


# --- Interfaz de linea de comandos ------------------------------------------

def construir_analizador():
    """Define los argumentos que acepta el programa."""
    analizador = argparse.ArgumentParser(
        description="Arbol de decision y Random Forest implementados sin framework, "
                    "aplicados al dataset Spaceship Titanic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    grupo = analizador.add_mutually_exclusive_group()
    grupo.add_argument("--entrenar", action="store_true",
                       help="entrena, evalua y genera todas las graficas (modo por defecto)")
    grupo.add_argument("--predecir", action="store_true",
                       help="predice un pasajero capturado por consola")
    grupo.add_argument("--predecir-csv", metavar="RUTA",
                       help="predice todas las filas de un CSV sin etiqueta")
    grupo.add_argument("--arbol", action="store_true",
                       help="imprime el arbol de decision entrenado")

    analizador.add_argument("--datos", default="data/train.csv",
                            help="CSV etiquetado de entrada (por defecto data/train.csv)")
    analizador.add_argument("--arboles", type=int, default=100,
                            help="numero de arboles del bosque (por defecto 100)")
    analizador.add_argument("--profundidad-bosque", type=int, default=14,
                            help="profundidad maxima de cada arbol del bosque")
    analizador.add_argument("--min-hoja", type=int, default=5,
                            help="minimo de muestras por hoja en el arbol individual")
    analizador.add_argument("--min-hoja-bosque", type=int, default=2,
                            help="minimo de muestras por hoja en los arboles del bosque")
    analizador.add_argument("--criterio", default="gini", choices=["gini", "entropia"],
                            help="medida de impureza a usar")
    analizador.add_argument("--semilla", type=int, default=42,
                            help="semilla aleatoria, para reproducir los resultados")
    analizador.add_argument("--niveles", type=int, default=3,
                            help="niveles a mostrar con --arbol")

    return analizador


def main():
    """Punto de entrada: interpreta los argumentos y ejecuta el modo pedido."""
    args = construir_analizador().parse_args()

    if args.predecir:
        modo_predecir(args)
    elif args.predecir_csv:
        modo_predecir_csv(args)
    elif args.arbol:
        modo_arbol(args)
    else:
        # Sin argumentos, el comportamiento por defecto es entrenar.
        modo_entrenar(args)


if __name__ == "__main__":
    sys.exit(main())
