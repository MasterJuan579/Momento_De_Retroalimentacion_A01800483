"""
Arbol de decision binario (CART) implementado desde cero.

Todo el aprendizaje esta escrito a mano en este archivo: el calculo de la
impureza, la busqueda exhaustiva del mejor punto de corte, la construccion
recursiva del arbol, los criterios de paro y la prediccion. numpy se usa
solamente para almacenar los datos en arreglos y para operaciones aritmeticas
basicas (ordenar, sumar acumulados); no se invoca ninguna funcion que resuelva
parte del algoritmo.

Como funciona el algoritmo, en resumen:

1. En cada nodo se busca la pregunta binaria de la forma
   "caracteristica_j <= umbral" que deja los dos grupos resultantes lo mas
   puros posible (lo mas cercano a contener una sola clase).
2. La pureza se mide con el indice de Gini. Un nodo con 50/50 de cada clase
   tiene Gini 0.5 (maxima impureza); un nodo con una sola clase tiene Gini 0.
3. El proceso se repite recursivamente en cada mitad hasta que se cumple algun
   criterio de paro (profundidad maxima, pocas muestras, o ninguna pregunta
   mejora la impureza lo suficiente).
4. Cada hoja guarda la proporcion de clases de las muestras que cayeron en
   ella; esa proporcion es la probabilidad que devuelve el modelo.
"""

import numpy as np


class Nodo:
    """Un nodo del arbol: o hace una pregunta, o es una hoja con una respuesta.

    Atributos:
        caracteristica: indice de la columna por la que se pregunta (None en hojas).
        umbral: valor de corte; van a la izquierda las muestras con valor <= umbral.
        izquierda / derecha: nodos hijos (None en hojas).
        proba: proporcion de cada clase en el nodo, por ejemplo [0.2, 0.8].
        n_muestras: cuantas muestras de entrenamiento llegaron a este nodo.
        impureza: impureza (Gini o entropia) del nodo.
    """

    __slots__ = ("caracteristica", "umbral", "izquierda", "derecha",
                 "proba", "n_muestras", "impureza")

    def __init__(self, proba, n_muestras, impureza):
        self.caracteristica = None
        self.umbral = None
        self.izquierda = None
        self.derecha = None
        self.proba = proba
        self.n_muestras = n_muestras
        self.impureza = impureza

    @property
    def es_hoja(self):
        """True si el nodo no hace ninguna pregunta y da una respuesta directa."""
        return self.caracteristica is None


def impureza_gini(prop_positivos):
    """Indice de Gini para un problema de dos clases.

    Gini = 1 - p0^2 - p1^2. Vale 0 cuando el nodo es puro y 0.5 cuando las dos
    clases estan perfectamente mezcladas. Se puede evaluar sobre un arreglo
    completo de proporciones a la vez.

    Args:
        prop_positivos: proporcion de la clase 1 (escalar o arreglo numpy).

    Returns:
        La impureza, del mismo tipo que la entrada.
    """
    p = prop_positivos
    return 1.0 - p * p - (1.0 - p) * (1.0 - p)


def impureza_entropia(prop_positivos):
    """Entropia de Shannon en bits para un problema de dos clases.

    Entropia = -p0*log2(p0) - p1*log2(p1). Vale 0 en un nodo puro y 1 cuando
    las clases estan al 50/50. Se define 0*log2(0) = 0.

    Args:
        prop_positivos: proporcion de la clase 1 (escalar o arreglo numpy).

    Returns:
        La impureza, del mismo tipo que la entrada.
    """
    p = np.asarray(prop_positivos, dtype=float)
    q = 1.0 - p
    # np.where evalua ambas ramas, asi que primero se sustituyen los ceros por
    # unos para que el logaritmo no genere avisos; el termino se anula despues.
    termino_p = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
    termino_q = np.where(q > 0, q * np.log2(np.where(q > 0, q, 1.0)), 0.0)
    resultado = -(termino_p + termino_q)
    return resultado if resultado.ndim else float(resultado)


CRITERIOS = {"gini": impureza_gini, "entropia": impureza_entropia}


class ArbolDecision:
    """Clasificador binario basado en un arbol de decision CART.

    Args:
        max_profundidad: profundidad maxima del arbol. None significa sin limite.
        min_muestras_division: minimo de muestras para intentar dividir un nodo.
        min_muestras_hoja: minimo de muestras que debe quedar en cada hijo.
        min_ganancia: reduccion minima de impureza para aceptar una division.
        criterio: "gini" o "entropia".
        max_caracteristicas: cuantas columnas se consideran al azar en cada nodo.
            None usa todas; "sqrt" usa la raiz cuadrada del total. Este parametro
            es el que convierte al arbol en el componente de un Random Forest.
        semilla: semilla para el muestreo aleatorio de caracteristicas.
    """

    def __init__(self, max_profundidad=None, min_muestras_division=2,
                 min_muestras_hoja=1, min_ganancia=0.0, criterio="gini",
                 max_caracteristicas=None, semilla=None):
        if criterio not in CRITERIOS:
            raise ValueError(f"Criterio no soportado: {criterio}")

        self.max_profundidad = max_profundidad
        self.min_muestras_division = min_muestras_division
        self.min_muestras_hoja = min_muestras_hoja
        self.min_ganancia = min_ganancia
        self.criterio = criterio
        self.max_caracteristicas = max_caracteristicas
        self.semilla = semilla

        self.raiz = None
        self.n_caracteristicas = 0
        self.importancias_ = None
        self.profundidad_alcanzada_ = 0
        self.n_nodos_ = 0
        self.n_hojas_ = 0

    # --- Entrenamiento ------------------------------------------------------

    def entrenar(self, X, y):
        """Construye el arbol a partir de los datos de entrenamiento.

        Args:
            X: matriz numpy (n_muestras, n_caracteristicas) de valores float.
            y: vector numpy de etiquetas 0/1 de longitud n_muestras.

        Returns:
            El propio arbol entrenado.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=np.int64)

        self.n_caracteristicas = X.shape[1]
        self._funcion_impureza = CRITERIOS[self.criterio]
        self._generador = np.random.default_rng(self.semilla)
        # Acumulador de la reduccion de impureza aportada por cada caracteristica.
        self._importancias_brutas = np.zeros(self.n_caracteristicas)
        self._n_total = len(y)
        self.n_nodos_ = 0
        self.n_hojas_ = 0
        self.profundidad_alcanzada_ = 0

        indices = np.arange(len(y))
        self.raiz = self._construir(X, y, indices, profundidad=0)

        # Las importancias se normalizan para que sumen 1 y sean comparables.
        total = self._importancias_brutas.sum()
        if total > 0:
            self.importancias_ = self._importancias_brutas / total
        else:
            self.importancias_ = self._importancias_brutas

        return self

    def _construir(self, X, y, indices, profundidad):
        """Construye recursivamente el subarbol para un subconjunto de muestras.

        Args:
            X, y: datos completos de entrenamiento.
            indices: indices de las muestras que llegaron a este nodo.
            profundidad: profundidad actual, empezando en 0 en la raiz.

        Returns:
            El Nodo raiz del subarbol construido.
        """
        y_nodo = y[indices]
        n = len(indices)
        n_positivos = int(y_nodo.sum())
        prop_positivos = n_positivos / n

        impureza = float(self._funcion_impureza(prop_positivos))
        nodo = Nodo(proba=np.array([1.0 - prop_positivos, prop_positivos]),
                    n_muestras=n, impureza=impureza)
        self.n_nodos_ += 1
        self.profundidad_alcanzada_ = max(self.profundidad_alcanzada_, profundidad)

        # Criterios de paro que se comprueban antes de buscar una division:
        # el nodo ya es puro, tiene muy pocas muestras, o se alcanzo el limite
        # de profundidad. En cualquiera de esos casos se convierte en hoja.
        if (n_positivos == 0 or n_positivos == n
                or n < self.min_muestras_division
                or (self.max_profundidad is not None and profundidad >= self.max_profundidad)):
            self.n_hojas_ += 1
            return nodo

        caracteristica, umbral, mascara_izq, ganancia = self._buscar_mejor_division(
            X, y_nodo, indices, impureza)

        # Si ninguna pregunta mejora la impureza lo suficiente, tambien se
        # convierte en hoja. Este es el mecanismo de poda previa (pre-pruning).
        if caracteristica is None or ganancia <= self.min_ganancia:
            self.n_hojas_ += 1
            return nodo

        # La reduccion de impureza se pondera por la fraccion de muestras que
        # pasa por el nodo: una mejora en la raiz vale mas que la misma mejora
        # en un nodo profundo con pocas muestras.
        self._importancias_brutas[caracteristica] += (n / self._n_total) * ganancia

        nodo.caracteristica = caracteristica
        nodo.umbral = umbral
        nodo.izquierda = self._construir(X, y, indices[mascara_izq], profundidad + 1)
        nodo.derecha = self._construir(X, y, indices[~mascara_izq], profundidad + 1)
        return nodo

    def _caracteristicas_candidatas(self):
        """Elige que columnas se evaluan en el nodo actual.

        Con max_caracteristicas=None se consideran todas (arbol clasico). Con
        "sqrt" se toma una muestra aleatoria sin reemplazo del tamano de la raiz
        cuadrada del total, que es lo que da diversidad a un Random Forest.

        Returns:
            Arreglo con los indices de las columnas a evaluar.
        """
        todas = np.arange(self.n_caracteristicas)
        if self.max_caracteristicas is None:
            return todas

        if self.max_caracteristicas == "sqrt":
            k = max(1, int(np.sqrt(self.n_caracteristicas)))
        elif self.max_caracteristicas == "log2":
            k = max(1, int(np.log2(self.n_caracteristicas)))
        else:
            k = max(1, min(int(self.max_caracteristicas), self.n_caracteristicas))

        return self._generador.choice(todas, size=k, replace=False)

    def _buscar_mejor_division(self, X, y_nodo, indices, impureza_padre):
        """Encuentra la pregunta que mas reduce la impureza en este nodo.

        Para cada caracteristica candidata se ordenan sus valores y se evaluan
        TODOS los puntos de corte posibles de una sola vez usando sumas
        acumuladas: la posicion i del arreglo ordenado representa el corte que
        manda las primeras i+1 muestras a la izquierda. Eso permite calcular la
        impureza de las dos mitades para todos los cortes con unas pocas
        operaciones vectoriales en lugar de un ciclo de Python.

        Args:
            X: matriz completa de caracteristicas.
            y_nodo: etiquetas de las muestras del nodo.
            indices: indices de las muestras del nodo.
            impureza_padre: impureza del nodo antes de dividir.

        Returns:
            Tupla (caracteristica, umbral, mascara_izquierda, ganancia). Si no
            existe ninguna division valida, devuelve (None, None, None, 0.0).
        """
        n = len(indices)
        total_positivos = int(y_nodo.sum())

        mejor_ganancia = 0.0
        mejor_caracteristica = None
        mejor_umbral = None

        # Tamanos de las mitades para cada posible punto de corte.
        n_izq = np.arange(1, n, dtype=float)
        n_der = n - n_izq

        for caracteristica in self._caracteristicas_candidatas():
            valores = X[indices, caracteristica]

            # mergesort es estable: garantiza el mismo resultado en cada corrida
            # aunque haya valores repetidos.
            orden = np.argsort(valores, kind="mergesort")
            valores_ord = valores[orden]
            y_ord = y_nodo[orden]

            # Positivos acumulados a la izquierda de cada punto de corte.
            pos_izq = np.cumsum(y_ord)[:-1].astype(float)
            pos_der = total_positivos - pos_izq

            # Un corte solo es valido si separa dos valores distintos (si no,
            # dejaria muestras identicas en lados opuestos) y si deja al menos
            # min_muestras_hoja muestras en cada hijo.
            validos = valores_ord[:-1] < valores_ord[1:]
            if self.min_muestras_hoja > 1:
                validos &= (n_izq >= self.min_muestras_hoja)
                validos &= (n_der >= self.min_muestras_hoja)
            if not validos.any():
                continue

            impureza_izq = self._funcion_impureza(pos_izq / n_izq)
            impureza_der = self._funcion_impureza(pos_der / n_der)

            # Impureza combinada, ponderada por el tamano de cada mitad.
            impureza_division = (n_izq * impureza_izq + n_der * impureza_der) / n

            # Los cortes invalidos se descartan poniendoles impureza infinita.
            impureza_division = np.where(validos, impureza_division, np.inf)

            posicion = int(np.argmin(impureza_division))
            ganancia = impureza_padre - impureza_division[posicion]

            if ganancia > mejor_ganancia:
                mejor_ganancia = float(ganancia)
                mejor_caracteristica = int(caracteristica)
                # El umbral se coloca a la mitad entre los dos valores vecinos.
                mejor_umbral = float((valores_ord[posicion] + valores_ord[posicion + 1]) / 2.0)

        if mejor_caracteristica is None:
            return None, None, None, 0.0

        mascara_izq = X[indices, mejor_caracteristica] <= mejor_umbral
        return mejor_caracteristica, mejor_umbral, mascara_izq, mejor_ganancia

    # --- Prediccion ---------------------------------------------------------

    def predecir_proba(self, X):
        """Devuelve la probabilidad de cada clase para cada muestra.

        El recorrido se hace por lotes: en cada nodo se parte el bloque de
        muestras en dos segun la pregunta, en vez de recorrer el arbol una vez
        por muestra. Es el mismo resultado, pero mucho mas rapido.

        Args:
            X: matriz numpy (n_muestras, n_caracteristicas).

        Returns:
            Matriz (n_muestras, 2) donde la columna 1 es P(clase = 1).

        Raises:
            RuntimeError: si el arbol no ha sido entrenado.
        """
        if self.raiz is None:
            raise RuntimeError("El arbol debe entrenarse antes de predecir.")

        X = np.asarray(X, dtype=float)
        salida = np.zeros((len(X), 2))
        self._predecir_lote(self.raiz, X, np.arange(len(X)), salida)
        return salida

    def _predecir_lote(self, nodo, X, indices, salida):
        """Reparte recursivamente un bloque de muestras por el arbol."""
        if len(indices) == 0:
            return
        if nodo.es_hoja:
            salida[indices] = nodo.proba
            return

        mascara = X[indices, nodo.caracteristica] <= nodo.umbral
        self._predecir_lote(nodo.izquierda, X, indices[mascara], salida)
        self._predecir_lote(nodo.derecha, X, indices[~mascara], salida)

    def predecir(self, X, umbral=0.5):
        """Devuelve la clase predicha (0 o 1) para cada muestra.

        Args:
            X: matriz de caracteristicas.
            umbral: probabilidad a partir de la cual se predice la clase 1.

        Returns:
            Vector de enteros 0/1.
        """
        return (self.predecir_proba(X)[:, 1] >= umbral).astype(int)

    # --- Inspeccion ---------------------------------------------------------

    def a_texto(self, nombres=None, max_profundidad=None):
        """Dibuja el arbol entrenado como texto, para poder revisarlo a mano.

        Args:
            nombres: lista con el nombre de cada caracteristica.
            max_profundidad: hasta que nivel imprimir. None imprime todo.

        Returns:
            Una cadena con el arbol en formato de arbol ASCII.
        """
        if self.raiz is None:
            return "(arbol sin entrenar)"

        lineas = []
        self._a_texto_recursivo(self.raiz, nombres, 0, max_profundidad, "", lineas)
        return "\n".join(lineas)

    def _a_texto_recursivo(self, nodo, nombres, profundidad, limite, prefijo, lineas):
        """Acumula en la lista una linea por nodo, con sangria por nivel."""
        if nodo.es_hoja:
            clase = int(np.argmax(nodo.proba))
            lineas.append(f"{prefijo}--> Transported={bool(clase)} "
                          f"(p={nodo.proba[1]:.3f}, n={nodo.n_muestras})")
            return

        if limite is not None and profundidad >= limite:
            lineas.append(f"{prefijo}... (podado en la impresion, n={nodo.n_muestras})")
            return

        nombre = (nombres[nodo.caracteristica] if nombres
                  else f"caracteristica[{nodo.caracteristica}]")
        lineas.append(f"{prefijo}[{nombre} <= {nodo.umbral:.3f}]  "
                      f"n={nodo.n_muestras}, {self.criterio}={nodo.impureza:.3f}")
        self._a_texto_recursivo(nodo.izquierda, nombres, profundidad + 1, limite,
                                prefijo + "   si  ", lineas)
        self._a_texto_recursivo(nodo.derecha, nombres, profundidad + 1, limite,
                                prefijo + "   no  ", lineas)
