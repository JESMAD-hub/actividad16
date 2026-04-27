import re
import unicodedata
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Word2Vec
from gensim.models import Word2Vec


# =====================================================
# 1. CARGA DE DATOS
# =====================================================

def cargar_datos():
    datos = {
        'reseña': [
            "Excelente producto, cumple con todas las expectativas",
            "Mala calidad, no lo recomiendo",
            "Regular, podría ser mejor",
            "Increíble, superó mis expectativas",
            "Pésimo servicio al cliente",
            "Muy buen producto, funciona perfecto :)",
            "No me gustó, llegó tarde y defectuoso",
            "Está bien, aunque esperaba más",
            "Me encantó, excelente compra",
            "Horrible experiencia, jamás vuelvo a comprar :(",
            "Producto aceptable por el precio",
            "La calidad es buena y el envío fue rápido",
            "No es malo, pero tampoco excelente",
            "Muy malo, no sirve",
            "Funciona correctamente, cumple su función",
            "Terrible, llegó roto",
            "Buen precio y buena calidad",
            "No recomiendo este producto",
            "Está regular, nada sorprendente",
            "Me fascinó, lo volvería a comprar",
            "El producto no está mal",
            "No fue una mala compra",
            "Es demasiado caro para lo que ofrece",
            "La atención fue excelente",
            "El envío fue lento, pero el producto está bien",
            "Muy satisfecho con la compra",
            "Definitivamente no cumple con lo prometido",
            "Buen material y diseño bonito",
            "No me funcionó como esperaba",
            "Calidad promedio, precio justo"
        ],
        'sentimiento': [
            'positivo', 'negativo', 'neutral', 'positivo', 'negativo',
            'positivo', 'negativo', 'neutral', 'positivo', 'negativo',
            'neutral', 'positivo', 'neutral', 'negativo', 'positivo',
            'negativo', 'positivo', 'negativo', 'neutral', 'positivo',
            'neutral', 'neutral', 'negativo', 'positivo', 'neutral',
            'positivo', 'negativo', 'positivo', 'negativo', 'neutral'
        ]
    }
    return pd.DataFrame(datos)


# =====================================================
# 2. PREPROCESAMIENTO DE TEXTO
# =====================================================

emoticonos = {
    ":)": " positivo ",
    ":-)": " positivo ",
    ":D": " positivo ",
    ":(": " negativo ",
    ":-(": " negativo ",
    "😃": " positivo ",
    "😊": " positivo ",
    "😍": " positivo ",
    "😡": " negativo ",
    "😞": " negativo "
}

lenguaje_informal = {
    "super": "muy",
    "súper": "muy",
    "xq": "porque",
    "q": "que",
    "k": "que",
    "bn": "bien",
    "malisimo": "muy malo",
    "buenisimo": "muy bueno"
}

negaciones = {"no", "nunca", "jamás", "jamas", "tampoco", "ni"}


def quitar_acentos(texto):
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore")
    texto = texto.decode("utf-8")
    return texto


def reemplazar_emoticonos(texto):
    for emo, palabra in emoticonos.items():
        texto = texto.replace(emo, palabra)
    return texto


def normalizar_lenguaje(texto):
    palabras = texto.split()
    nuevas = []

    for palabra in palabras:
        nuevas.append(lenguaje_informal.get(palabra, palabra))

    return " ".join(nuevas)


def expandir_negaciones(texto):
    palabras = texto.split()
    resultado = []
    activar_negacion = False
    ventana = 0

    for palabra in palabras:
        if palabra in negaciones:
            resultado.append(palabra)
            activar_negacion = True
            ventana = 2
        elif activar_negacion and ventana > 0:
            resultado.append("no_" + palabra)
            ventana -= 1
            if ventana == 0:
                activar_negacion = False
        else:
            resultado.append(palabra)

    return " ".join(resultado)


def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = reemplazar_emoticonos(texto)
    texto = quitar_acentos(texto)
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-zA-ZñÑ\s_]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    texto = normalizar_lenguaje(texto)
    texto = expandir_negaciones(texto)
    return texto


# =====================================================
# 3. EXTRACCIÓN DE CARACTERÍSTICAS
# =====================================================

def vectorizar_tfidf(textos):
    vectorizador = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2)
    )
    X = vectorizador.fit_transform(textos)
    return X, vectorizador


def vectorizar_bow(textos):
    vectorizador = CountVectorizer(
        max_features=100,
        ngram_range=(1, 2)
    )
    X = vectorizador.fit_transform(textos)
    return X, vectorizador


def vectorizar_word2vec(textos):
    textos_tokenizados = [texto.split() for texto in textos]

    modelo_w2v = Word2Vec(
        sentences=textos_tokenizados,
        vector_size=50,
        window=3,
        min_count=1,
        workers=1,
        seed=42
    )

    vectores = []

    for tokens in textos_tokenizados:
        vectores_palabras = []

        for token in tokens:
            if token in modelo_w2v.wv:
                vectores_palabras.append(modelo_w2v.wv[token])

        if len(vectores_palabras) > 0:
            vectores.append(np.mean(vectores_palabras, axis=0))
        else:
            vectores.append(np.zeros(50))

    return np.array(vectores), modelo_w2v


def mostrar_caracteristicas_relevantes(vectorizador, modelo, top_n=10):
    if not hasattr(modelo, "coef_"):
        print("\nEste modelo no permite extraer coeficientes directamente.")
        return

    nombres = vectorizador.get_feature_names_out()
    coeficientes = modelo.coef_

    print("\nCaracterísticas más relevantes por clase:")

    for i, clase in enumerate(modelo.classes_):
        indices = coeficientes[i].argsort()[-top_n:][::-1]
        print(f"\nClase: {clase}")

        for idx in indices:
            print(f"{nombres[idx]} -> {coeficientes[i][idx]:.4f}")


# =====================================================
# 4. MODELADO Y EVALUACIÓN
# =====================================================

def entrenar_y_evaluar(X, y, nombre_vectorizacion):
    print("\n" + "=" * 70)
    print(f"VECTORIFICACIÓN: {nombre_vectorizacion}")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=1000),
        "SVM Lineal": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    if nombre_vectorizacion != "Word2Vec":
        modelos["Naive Bayes"] = MultinomialNB()

    resultados = {}

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X_test)

        accuracy = accuracy_score(y_test, predicciones)
        resultados[nombre] = accuracy

        print("\nModelo:", nombre)
        print("Accuracy:", round(accuracy, 4))
        print("Matriz de confusión:")
        print(confusion_matrix(y_test, predicciones))
        print("Reporte de clasificación:")
        print(classification_report(y_test, predicciones, zero_division=0))

        scores = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")
        print("Validación cruzada:", scores)
        print("Promedio CV:", round(scores.mean(), 4))

    return resultados, modelos


# =====================================================
# 5. EJECUCIÓN PRINCIPAL
# =====================================================

df = cargar_datos()

print("Datos cargados:")
print(df.head())

df["reseña_limpia"] = df["reseña"].apply(limpiar_texto)

print("\nDatos después del preprocesamiento:")
print(df[["reseña", "reseña_limpia", "sentimiento"]].head(10))

y = df["sentimiento"]


# TF-IDF
X_tfidf, vectorizador_tfidf = vectorizar_tfidf(df["reseña_limpia"])
resultados_tfidf, modelos_tfidf = entrenar_y_evaluar(X_tfidf, y, "TF-IDF")


# Bag of Words
X_bow, vectorizador_bow = vectorizar_bow(df["reseña_limpia"])
resultados_bow, modelos_bow = entrenar_y_evaluar(X_bow, y, "Bag of Words")


# Word2Vec
X_w2v, modelo_w2v = vectorizar_word2vec(df["reseña_limpia"])
resultados_w2v, modelos_w2v = entrenar_y_evaluar(X_w2v, y, "Word2Vec")


# Características relevantes usando Regresión Logística con TF-IDF
mostrar_caracteristicas_relevantes(
    vectorizador_tfidf,
    modelos_tfidf["Regresión Logística"],
    top_n=10
)


# =====================================================
# 6. PREDICCIÓN DE NUEVAS RESEÑAS
# =====================================================

nuevas_resenas = [
    "El producto es excelente y llegó rápido",
    "No lo recomiendo, es de mala calidad",
    "Está regular, cumple pero esperaba más",
    "Muy buena compra, me encantó :)"
]

nuevas_limpias = [limpiar_texto(texto) for texto in nuevas_resenas]
X_nuevas = vectorizador_tfidf.transform(nuevas_limpias)

modelo_final = modelos_tfidf["Regresión Logística"]
predicciones = modelo_final.predict(X_nuevas)

print("\n" + "=" * 70)
print("PREDICCIONES NUEVAS")
print("=" * 70)

for reseña, pred in zip(nuevas_resenas, predicciones):
    print(f"Reseña: {reseña}")
    print(f"Sentimiento predicho: {pred}")
    print("-" * 40)
