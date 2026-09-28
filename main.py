"""
main.py
Point d'entrée pour exécuter le pipeline ML via des arguments CLI.
"""

import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ML - Prédiction du churn client"
    )
    parser.add_argument(
        "--step",
        choices=["prepare", "train", "evaluate", "all"],
        default="all",
        help="Étape à exécuter",
    )
    parser.add_argument("--data", default=DATA_PATH, help="Chemin vers le CSV")
    parser.add_argument("--model", default=MODEL_PATH, help="Chemin du fichier modèle")
    args = parser.parse_args()

    x_train, x_test, y_train, y_test = prepare_data(args.data)
    print("Données préparées avec succès.")

    if args.step in ("train", "all"):
        model = train_model(x_train, y_train)
        save_model(model, args.model)
        print("Modèle entraîné et sauvegardé.")

    if args.step in ("evaluate", "all"):
        model = load_model(args.model)
        evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
