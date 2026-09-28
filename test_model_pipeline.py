"""
test_model_pipeline.py
Tests unitaires pour les fonctions de model_pipeline.py
"""

import os

from sklearn.ensemble import RandomForestClassifier

from model_pipeline import prepare_data, train_model, save_model, load_model

DATA_PATH = "Churn_Modelling.csv"


def test_prepare_data_shapes():
    x_train, x_test, y_train, y_test = prepare_data(DATA_PATH)
    assert x_train.shape[0] == 8000
    assert x_test.shape[0] == 2000
    assert "Surname" not in x_train.columns
    assert "Geography" not in x_train.columns


def test_train_model_returns_random_forest():
    x_train, _, y_train, _ = prepare_data(DATA_PATH)
    model = train_model(x_train, y_train, n_estimators=10)
    assert isinstance(model, RandomForestClassifier)


def test_save_and_load_model(tmp_path):
    x_train, _, y_train, _ = prepare_data(DATA_PATH)
    model = train_model(x_train, y_train, n_estimators=10)
    path = os.path.join(tmp_path, "model.joblib")
    save_model(model, path)
    loaded = load_model(path)
    assert isinstance(loaded, RandomForestClassifier)
