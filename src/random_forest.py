"""
Random Forest implementado desde cero sobre el arbol de decision propio.

Un solo arbol profundo memoriza el entrenamiento: aprende reglas muy especificas
que no generalizan (varianza alta). El Random Forest ataca ese problema
promediando muchos arboles que se equivocan de formas distintas. Para que los
arboles sean distintos entre si se introducen dos fuentes de azar:

1. Bagging (bootstrap aggregating): cada arbol se entrena con una muestra del
   mismo tamano que el original pero tomada CON reemplazo. En promedio cada
   muestra bootstrap contiene solo el 63.2% de las filas originales; el 36.8%
   restante son las muestras "out-of-bag" (fuera de bolsa) de ese arbol.

2. Subespacio aleatorio: en cada nodo, cada arbol solo puede elegir entre un
   subconjunto aleatorio de caracteristicas (por defecto la raiz cuadrada del
   total). Esto evita que todos los arboles usen la misma caracteristica
   dominante en la raiz y queden correlacionados.

Al final se promedian las probabilidades de todos los arboles (voto suave). Los
errores individuales, al ser en buena medida independientes, se cancelan entre
si y el ensamble generaliza mejor que cualquiera de sus arboles.

Las muestras out-of-bag permiten ademas estimar el desempeno del bosque sin
gastar el conjunto de prueba: cada fila se evalua solo con los arboles que no
la vieron durante su entrenamiento.
"""

import numpy as np

from .decision_tree import ArbolDecision


class RandomForest:
    """Ensamble de arboles de decision entrenados con bagging.

    Args:
        n_arboles: cuantos arboles componen el bosque.
        max_profundidad: profundidad maxima de cada arbol. None = sin limite.
        min_muestras_division: minimo de muestras para intentar dividir un nodo.
        min_muestras_hoja: minimo de muestras que debe quedar en cada hijo.
        max_caracteristicas: cuantas columnas ve cada nodo ("sqrt", "log2",
            un entero, o None para todas).
        criterio: "gini" o "entropia".
        semilla: semilla maestra; de ella se derivan las semillas de cada arbol,
            asi que dos ejecuciones con la misma semilla dan el mismo bosque.
        calcular_oob: si es True, estima el acierto con las muestras out-of-bag.
    """

    def __init__(self, n_arboles=100, max_profundidad=None,
                 min_muestras_division=2, min_muestras_hoja=1,
                 max_caracteristicas="sqrt", criterio="gini",
                 semilla=42, calcular_oob=True):
        self.n_arboles = n_arboles
        self.max_profundidad = max_profundidad
        self.min_muestras_division = min_muestras_division
        self.min_muestras_hoja = min_muestras_hoja
        self.max_caracteristicas = max_caracteristicas
        self.criterio = criterio
        self.semilla = semilla
        self.calcular_oob = calcular_oob

        self.arboles = []
        self.bolsas_ = []          # indices bootstrap usados por cada arbol
        self.importancias_ = None
        self.oob_score_ = None
        self.n_caracteristicas = 0

    def entrenar(self, X, y, verboso=False):
        """Entrena los arboles del bosque.

        Args:
            X: matriz numpy (n_muestras, n_caracteristicas).
            y: vector numpy de etiquetas 0/1.
            verboso: si es True, imprime el avance cada 10 arboles.

        Returns:
            El propio bosque entrenado.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=np.int64)
        n_muestras = len(y)
        self.n_caracteristicas = X.shape[1]

        # Un generador maestro produce una semilla distinta por arbol. Guardar
        # las semillas hace que el bosque completo sea reproducible.
        generador = np.random.default_rng(self.semilla)
        semillas = generador.integers(0, 2**31 - 1, size=self.n_arboles)

        self.arboles = []
        self.bolsas_ = []
        importancias = np.zeros(self.n_caracteristicas)

        # Acumuladores para la estimacion out-of-bag: para cada muestra se suma
        # la probabilidad predicha por los arboles que no la usaron y se cuenta
        # cuantos arboles fueron.
        suma_oob = np.zeros(n_muestras)
        conteo_oob = np.zeros(n_muestras)

        for i, semilla_arbol in enumerate(semillas):
            gen_arbol = np.random.default_rng(int(semilla_arbol))

            # Bagging: muestreo con reemplazo del mismo tamano que el original.
            indices_bolsa = gen_arbol.integers(0, n_muestras, size=n_muestras)

            arbol = ArbolDecision(
                max_profundidad=self.max_profundidad,
                min_muestras_division=self.min_muestras_division,
                min_muestras_hoja=self.min_muestras_hoja,
                criterio=self.criterio,
                max_caracteristicas=self.max_caracteristicas,
                semilla=int(semilla_arbol),
            )
            arbol.entrenar(X[indices_bolsa], y[indices_bolsa])

            self.arboles.append(arbol)
            self.bolsas_.append(indices_bolsa)
            importancias += arbol.importancias_

            if self.calcular_oob:
                # Las muestras que no entraron en la bolsa de este arbol son su
                # conjunto out-of-bag: el arbol nunca las vio.
                en_bolsa = np.zeros(n_muestras, dtype=bool)
                en_bolsa[indices_bolsa] = True
                fuera = np.flatnonzero(~en_bolsa)
                if len(fuera) > 0:
                    suma_oob[fuera] += arbol.predecir_proba(X[fuera])[:, 1]
                    conteo_oob[fuera] += 1

            if verboso and (i + 1) % 10 == 0:
                print(f"    arboles entrenados: {i + 1}/{self.n_arboles}")

        # La importancia del bosque es el promedio de la de sus arboles.
        self.importancias_ = importancias / self.n_arboles

        if self.calcular_oob:
            evaluables = conteo_oob > 0
            if evaluables.any():
                proba_oob = suma_oob[evaluables] / conteo_oob[evaluables]
                prediccion_oob = (proba_oob >= 0.5).astype(int)
                self.oob_score_ = float((prediccion_oob == y[evaluables]).mean())

        return self

    def predecir_proba(self, X):
        """Promedia las probabilidades de todos los arboles (voto suave).

        Args:
            X: matriz numpy (n_muestras, n_caracteristicas).

        Returns:
            Matriz (n_muestras, 2) donde la columna 1 es P(clase = 1).

        Raises:
            RuntimeError: si el bosque no ha sido entrenado.
        """
        if not self.arboles:
            raise RuntimeError("El bosque debe entrenarse antes de predecir.")

        X = np.asarray(X, dtype=float)
        acumulado = np.zeros((len(X), 2))
        for arbol in self.arboles:
            acumulado += arbol.predecir_proba(X)
        return acumulado / len(self.arboles)

    def predecir(self, X, umbral=0.5):
        """Devuelve la clase predicha (0 o 1) para cada muestra.

        Args:
            X: matriz de caracteristicas.
            umbral: probabilidad a partir de la cual se predice la clase 1.

        Returns:
            Vector de enteros 0/1.
        """
        return (self.predecir_proba(X)[:, 1] >= umbral).astype(int)

    # --- Analisis del efecto del tamano del ensamble ------------------------

    def curva_por_numero_arboles(self, X, y, cantidades):
        """Mide el acierto del bosque usando solo los primeros k arboles.

        Permite dibujar la curva "acierto vs numero de arboles" sin reentrenar
        un bosque distinto por cada tamano: basta acumular las probabilidades
        de los arboles en orden y evaluar en los cortes pedidos.

        Args:
            X: matriz de caracteristicas a evaluar.
            y: etiquetas verdaderas correspondientes.
            cantidades: lista creciente de tamanos de ensamble a reportar.

        Returns:
            Lista con el acierto para cada cantidad solicitada.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        acumulado = np.zeros(len(X))
        aciertos = []
        objetivos = set(cantidades)

        for k, arbol in enumerate(self.arboles, start=1):
            acumulado += arbol.predecir_proba(X)[:, 1]
            if k in objetivos:
                prediccion = (acumulado / k >= 0.5).astype(int)
                aciertos.append(float((prediccion == y).mean()))

        return aciertos

    def curva_oob_por_numero_arboles(self, X, y, cantidades):
        """Igual que la curva anterior, pero con la estimacion out-of-bag.

        Cada muestra se evalua unicamente con los arboles (de los primeros k)
        que no la incluyeron en su bolsa bootstrap. Es una estimacion de
        generalizacion que no consume el conjunto de validacion ni el de prueba.

        Args:
            X: matriz de entrenamiento con la que se construyo el bosque.
            y: etiquetas de entrenamiento.
            cantidades: lista creciente de tamanos de ensamble a reportar.

        Returns:
            Lista con el acierto out-of-bag para cada cantidad solicitada.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n = len(y)

        suma = np.zeros(n)
        conteo = np.zeros(n)
        aciertos = []
        objetivos = set(cantidades)

        for k, (arbol, bolsa) in enumerate(zip(self.arboles, self.bolsas_), start=1):
            en_bolsa = np.zeros(n, dtype=bool)
            en_bolsa[bolsa] = True
            fuera = np.flatnonzero(~en_bolsa)
            suma[fuera] += arbol.predecir_proba(X[fuera])[:, 1]
            conteo[fuera] += 1

            if k in objetivos:
                evaluables = conteo > 0
                proba = suma[evaluables] / conteo[evaluables]
                prediccion = (proba >= 0.5).astype(int)
                aciertos.append(float((prediccion == y[evaluables]).mean()))

        return aciertos

    def resumen(self):
        """Devuelve un texto con las estadisticas del bosque entrenado."""
        if not self.arboles:
            return "(bosque sin entrenar)"

        nodos = [a.n_nodos_ for a in self.arboles]
        profundidades = [a.profundidad_alcanzada_ for a in self.arboles]
        lineas = [
            f"Arboles:                {len(self.arboles)}",
            f"Nodos por arbol:        {np.mean(nodos):.1f} en promedio",
            f"Profundidad por arbol:  {np.mean(profundidades):.1f} en promedio",
            f"Caracteristicas/nodo:   {self.max_caracteristicas}",
        ]
        if self.oob_score_ is not None:
            lineas.append(f"Acierto out-of-bag:     {self.oob_score_:.4f}")
        return "\n".join(lineas)
