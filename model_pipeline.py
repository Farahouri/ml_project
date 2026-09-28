"""
model_pipeline.py
Fonctions modularisées pour le pipeline ML de prédiction du churn client.
"""

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare_data(filepath, test_size=0.2, random_state=1):
    """
    Charge et prétraite les données depuis un fichier CSV.

    Args:
        filepath (str): chemin vers le fichier CSV.
        test_size (float): proportion du jeu de test.
        random_state (int): graine aléatoire.

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    df = pd.read_csv(filepath)

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    columns_to_drop = ["Surname", "Geography"]
    df = df.drop(columns=columns_to_drop)

    x = df.drop(["Exited"], axis=1)
    y = df["Exited"]
    x = x.drop(columns=["RowNumber", "CustomerId"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """
    Entraîne un RandomForestClassifier.

    Returns:
        RandomForestClassifier: le modèle entraîné.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(x_train, y_train)
    return rf


def evaluate_model(model, x_test, y_test):
    """
    Évalue le modèle : accuracy, rapport de classification, matrice de confusion.

    Returns:
        dict: métriques d'évaluation.
    """
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy score: {accuracy * 100:.2f}%")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", matrix)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def save_model(model, filepath="classifier.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé dans : {filepath}")


def load_model(filepath="classifier.joblib"):
    """Charge un modèle sauvegardé."""
    model = joblib.load(filepath)
    print(f"Modèle chargé depuis : {filepath}")
    return model
