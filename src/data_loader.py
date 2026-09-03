"""
Carga, limpieza e ingenieria de caracteristicas del dataset Spaceship Titanic.

Se usa pandas unicamente para leer los CSV y manipular tablas, y numpy para
producir las matrices numericas que consume el arbol. Ninguna funcion de
aprendizaje automatico proviene de una libreria: la imputacion, la codificacion
de categorias y la particion estratificada estan escritas a mano en este archivo.

Nota importante sobre los datos:
    data/test.csv es el conjunto de evaluacion de Kaggle y NO contiene la
    columna objetivo Transported. Por eso no sirve para medir desempeno.
    El conjunto de prueba se obtiene particionando data/train.csv.
"""

import numpy as np
import pandas as pd

# --- Definicion de columnas del dataset -------------------------------------

COLUMNA_OBJETIVO = "Transported"

# Las cinco amenidades de la nave en las que un pasajero pudo gastar dinero.
COLUMNAS_GASTO = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Caracteristicas categoricas (se imputan con la moda y se codifican a enteros).
COLUMNAS_CATEGORICAS = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]

# Caracteristicas numericas (se imputan con la mediana y se usan tal cual).
COLUMNAS_NUMERICAS = ["Age", "CabinNum", "GroupSize", "TotalSpend", "NoSpend"] + COLUMNAS_GASTO

# Orden final de las columnas de la matriz de caracteristicas.
NOMBRES_CARACTERISTICAS = COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS


# --- Ingenieria de caracteristicas ------------------------------------------

def derivar_caracteristicas(df):
    """Crea columnas nuevas a partir de las columnas crudas del CSV.

    Ninguna de estas transformaciones mira la variable objetivo, asi que puede
    aplicarse antes de partir los datos sin provocar fuga de informacion.

    Transformaciones:
        Cabin (B/0/P)   -> Deck (B), CabinNum (0), Side (P)
        PassengerId     -> GroupSize (cuantos pasajeros comparten el grupo)
        gastos          -> TotalSpend (suma) y NoSpend (1 si no gasto nada)

    Args:
        df: DataFrame con las columnas crudas del dataset.

    Returns:
        Una copia del DataFrame con las columnas derivadas agregadas.
    """
    df = df.copy()

    # La cabina viene como "cubierta/numero/lado"; cada parte aporta senal
    # distinta, asi que se separa en tres caracteristicas independientes.
    partes_cabina = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = partes_cabina[0]
    df["CabinNum"] = pd.to_numeric(partes_cabina[1], errors="coerce")
    df["Side"] = partes_cabina[2]

    # El identificador tiene el formato gggg_pp: los primeros cuatro digitos
    # son el grupo de viaje. El tamano del grupo indica si alguien viajo solo
    # o acompanado, lo cual influye en el desenlace.
    grupo = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = grupo.map(grupo.value_counts()).astype(float)

    # El gasto total resume las cinco amenidades en un solo numero. La bandera
    # NoSpend distingue a quien no consumio nada (tipicamente los pasajeros en
    # criosueno, que permanecieron dormidos todo el viaje).
    df["TotalSpend"] = df[COLUMNAS_GASTO].sum(axis=1, skipna=True)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(float)

    return df


# --- Preprocesador ----------------------------------------------------------

class Preprocesador:
    """Aprende los parametros de limpieza con el entrenamiento y los reaplica.

    Los valores de imputacion (moda y mediana) y el diccionario de codigos de
    cada categoria se calculan UNICAMENTE con el conjunto de entrenamiento. Si
    se calcularan con todos los datos, informacion del conjunto de prueba se
    filtraria al modelo y las metricas quedarian infladas.
    """

    def __init__(self):
        self.modas = {}        # columna categorica -> valor mas frecuente
        self.medianas = {}     # columna numerica   -> mediana
        self.codigos = {}      # columna categorica -> {categoria: entero}
        self.ajustado = False

    def ajustar(self, df):
        """Calcula los parametros de imputacion y codificacion.

        Args:
            df: DataFrame de entrenamiento, ya con las columnas derivadas.

        Returns:
            El propio preprocesador, para poder encadenar llamadas.
        """
        for columna in COLUMNAS_CATEGORICAS:
            valores = df[columna].dropna().astype(str)
            # La moda es la categoria con mayor frecuencia observada.
            conteos = valores.value_counts()
            self.modas[columna] = conteos.index[0]
            # Las categorias se ordenan alfabeticamente para que el codigo
            # asignado sea el mismo en cualquier ejecucion (reproducibilidad).
            categorias = sorted(valores.unique())
            self.codigos[columna] = {nombre: i for i, nombre in enumerate(categorias)}

        for columna in COLUMNAS_NUMERICAS:
            self.medianas[columna] = float(df[columna].median())

        self.ajustado = True
        return self

    def transformar(self, df):
        """Imputa, codifica y devuelve la matriz numerica de caracteristicas.

        Args:
            df: DataFrame ya con las columnas derivadas.

        Returns:
            Arreglo numpy de forma (n_muestras, n_caracteristicas) con dtype
            float, listo para el arbol.

        Raises:
            RuntimeError: si el preprocesador no fue ajustado antes.
        """
        if not self.ajustado:
            raise RuntimeError("El preprocesador debe ajustarse antes de transformar.")

        columnas = []

        for columna in COLUMNAS_CATEGORICAS:
            # Se rellenan los faltantes con la moda del entrenamiento y se
            # traduce cada categoria a su codigo entero. Una categoria nunca
            # vista en entrenamiento se trata como si fuera la moda.
            serie = df[columna].astype(str).where(df[columna].notna(), self.modas[columna])
            codigo_moda = self.codigos[columna][self.modas[columna]]
            codificada = serie.map(self.codigos[columna]).fillna(codigo_moda)
            columnas.append(codificada.to_numpy(dtype=float))

        for columna in COLUMNAS_NUMERICAS:
            serie = df[columna].fillna(self.medianas[columna])
            columnas.append(serie.to_numpy(dtype=float))

        return np.column_stack(columnas)


# --- Particion estratificada ------------------------------------------------

def particion_estratificada(y, proporciones=(0.70, 0.15, 0.15), semilla=42):
    """Reparte los indices en tres subconjuntos conservando el balance de clases.

    Se implementa a mano: se agrupan los indices por clase, se barajan con una
    semilla fija y se reparten respetando las proporciones pedidas. Asi los tres
    subconjuntos tienen aproximadamente el mismo porcentaje de cada clase que el
    dataset original.

    Args:
        y: vector de etiquetas (0/1) de longitud n.
        proporciones: tupla (entrenamiento, validacion, prueba) que suma 1.
        semilla: semilla del generador aleatorio, para que sea reproducible.

    Returns:
        Tupla de tres arreglos de indices: (entrenamiento, validacion, prueba).
    """
    generador = np.random.default_rng(semilla)
    prop_entrena, prop_valida, _ = proporciones

    idx_entrena, idx_valida, idx_prueba = [], [], []

    for clase in np.unique(y):
        indices_clase = np.flatnonzero(y == clase)
        generador.shuffle(indices_clase)

        n = len(indices_clase)
        corte_1 = int(round(n * prop_entrena))
        corte_2 = corte_1 + int(round(n * prop_valida))

        idx_entrena.append(indices_clase[:corte_1])
        idx_valida.append(indices_clase[corte_1:corte_2])
        idx_prueba.append(indices_clase[corte_2:])

    # Se vuelven a barajar los indices ya concatenados para que las clases no
    # queden agrupadas en bloques dentro de cada subconjunto.
    partes = []
    for lista in (idx_entrena, idx_valida, idx_prueba):
        parte = np.concatenate(lista)
        generador.shuffle(parte)
        partes.append(parte)

    return tuple(partes)


# --- Punto de entrada del modulo --------------------------------------------

class Datos:
    """Contenedor simple con los tres subconjuntos ya preprocesados."""

    def __init__(self, X_entrena, y_entrena, X_valida, y_valida,
                 X_prueba, y_prueba, preprocesador, nombres):
        self.X_entrena = X_entrena
        self.y_entrena = y_entrena
        self.X_valida = X_valida
        self.y_valida = y_valida
        self.X_prueba = X_prueba
        self.y_prueba = y_prueba
        self.preprocesador = preprocesador
        self.nombres = nombres

    def resumen(self):
        """Devuelve un texto con el tamano y balance de cada subconjunto."""
        lineas = ["Subconjunto      Muestras   % Transported"]
        for etiqueta, y in (("Entrenamiento", self.y_entrena),
                            ("Validacion   ", self.y_valida),
                            ("Prueba       ", self.y_prueba)):
            lineas.append(f"{etiqueta}    {len(y):8d}   {100 * y.mean():11.2f}%")
        return "\n".join(lineas)


def cargar_datos(ruta="data/train.csv", proporciones=(0.70, 0.15, 0.15), semilla=42):
    """Carga el CSV etiquetado y devuelve los tres subconjuntos listos para usar.

    Args:
        ruta: ruta al CSV que contiene la columna objetivo.
        proporciones: reparto (entrenamiento, validacion, prueba).
        semilla: semilla para la particion.

    Returns:
        Un objeto Datos con las matrices, las etiquetas y el preprocesador
        ya ajustado (necesario para transformar pasajeros nuevos despues).
    """
    df = pd.read_csv(ruta)
    df = derivar_caracteristicas(df)

    # La etiqueta viene como booleano True/False; se convierte a 1/0.
    y = df[COLUMNA_OBJETIVO].astype(int).to_numpy()

    idx_entrena, idx_valida, idx_prueba = particion_estratificada(y, proporciones, semilla)

    # El preprocesador se ajusta solo con las filas de entrenamiento.
    preprocesador = Preprocesador().ajustar(df.iloc[idx_entrena])

    X = preprocesador.transformar(df)

    return Datos(
        X_entrena=X[idx_entrena], y_entrena=y[idx_entrena],
        X_valida=X[idx_valida], y_valida=y[idx_valida],
        X_prueba=X[idx_prueba], y_prueba=y[idx_prueba],
        preprocesador=preprocesador,
        nombres=list(NOMBRES_CARACTERISTICAS),
    )


def cargar_sin_etiqueta(ruta, preprocesador):
    """Prepara un CSV sin columna objetivo (por ejemplo data/test.csv).

    Args:
        ruta: ruta al CSV a predecir.
        preprocesador: preprocesador ya ajustado con el entrenamiento.

    Returns:
        Tupla (matriz de caracteristicas, serie con los PassengerId).
    """
    df = pd.read_csv(ruta)
    identificadores = df["PassengerId"].copy()
    df = derivar_caracteristicas(df)
    return preprocesador.transformar(df), identificadores
