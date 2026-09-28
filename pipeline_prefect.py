"""
pipeline_prefect.py
Pipeline ML orchestré avec Prefect.
"""

import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


# ---------- TASKS ----------


@task(name="Installer les dépendances")
def install_dependencies():
    subprocess.run(  # nosec B603
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )
    print("Dépendances installées.")


@task(name="Formatage du code")
def format_code():
    subprocess.run(
        [sys.executable, "-m", "black"] + CODE_FILES, check=True
    )  # nosec B603
    print("Formatage terminé.")


@task(name="Qualité du code")
def quality_code():
    subprocess.run(  # nosec B603
        [sys.executable, "-m", "flake8", "--max-line-length=100"] + CODE_FILES,
        check=True,
    )
    print("Qualité du code OK.")


@task(name="Sécurité du code")
def security_code():
    subprocess.run(  # nosec B603
        [sys.executable, "-m", "bandit", "-q"] + CODE_FILES, check=True
    )
    print("Sécurité du code OK.")


@task(name="Tests unitaires")
def unit_tests():
    subprocess.run(  # nosec B603
        [sys.executable, "-m", "pytest", "test_model_pipeline.py", "-v"],
        check=True,
    )
    print("Tests unitaires OK.")


@task(name="Préparation des données")
def prepare_task(data_path=DATA_PATH):
    return prepare_data(data_path)


@task(name="Entraînement du modèle")
def train_task(x_train, y_train):
    return train_model(x_train, y_train)


@task(name="Sauvegarde du modèle")
def save_task(model, model_path=MODEL_PATH):
    save_model(model, model_path)


@task(name="Chargement du modèle")
def load_task(model_path=MODEL_PATH):
    return load_model(model_path)


@task(name="Évaluation du modèle")
def evaluate_task(model, x_test, y_test):
    return evaluate_model(model, x_test, y_test)


# ---------- FLOWS ----------


@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    quality_code()
    security_code()
    unit_tests()


@flow(name="all")
def all_flow():
    install_dependencies()
    format_code()
    quality_code()
    security_code()
    unit_tests()
    x_train, x_test, y_train, y_test = prepare_task()
    model = train_task(x_train, y_train)
    save_task(model)
    evaluate_task(model, x_test, y_test)


@flow(name="train")
def train_flow():
    x_train, x_test, y_train, y_test = prepare_task()
    model = train_task(x_train, y_train)
    save_task(model)


@flow(name="evaluate")
def evaluate_flow():
    x_train, x_test, y_train, y_test = prepare_task()
    model = load_task()
    evaluate_task(model, x_test, y_test)


# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect")
    parser.add_argument(
        "--flow",
        choices=["all", "train", "evaluate", "code"],
        default="all",
        help="Flow à exécuter",
    )
    args = parser.parse_args()

    if args.flow == "all":
        all_flow()
    elif args.flow == "train":
        train_flow()
    elif args.flow == "evaluate":
        evaluate_flow()
    elif args.flow == "code":
        code_flow()
