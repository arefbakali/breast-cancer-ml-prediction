"""
Tests unitaires exécutés par la CI (GitHub Actions) à chaque push.

Ils ne réentraînent pas les modèles (trop long pour une CI) mais vérifient :
  1. que models/pipeline.pkl (scaler + modèle réunis) se charge correctement,
  2. qu'il prédit sur le jeu de test avec une accuracy raisonnable
     (garde-fou contre une régression silencieuse du modèle),
  3. que le module de journalisation des prédictions fonctionne (base du
     monitoring de drift).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURES = ["Age", "BMI", "Glucose", "Insulin", "HOMA", "Leptin", "Adiponectin", "Resistin", "MCP.1"]
TARGET = "Classification"
RANDOM_STATE = 42
MIN_ACCEPTABLE_ACCURACY = 0.60  # garde-fou large : sur ce petit dataset, KNN ~0.75


@pytest.fixture(scope="module")
def pipeline():
    with open(PROJECT_DIR / "models" / "pipeline.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def test_split():
    df = pd.read_csv(PROJECT_DIR / "data" / "dataR2.csv")
    if set(df[TARGET].dropna().unique()) == {1, 2}:
        df[TARGET] = df[TARGET].map({1: 0, 2: 1})
    X, y = df[FEATURES], df[TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    return X_test, y_test


def test_pipeline_loads(pipeline):
    assert pipeline is not None
    assert hasattr(pipeline, "predict")


def test_pipeline_accuracy_above_threshold(pipeline, test_split):
    X_test, y_test = test_split
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc >= MIN_ACCEPTABLE_ACCURACY, (
        f"Accuracy {acc:.3f} en dessous du seuil minimal {MIN_ACCEPTABLE_ACCURACY} — "
        "possible régression du modèle."
    )


def test_pipeline_predicts_valid_classes(pipeline, test_split):
    X_test, _ = test_split
    y_pred = pipeline.predict(X_test)

    assert set(np.unique(y_pred)).issubset({0, 1})


def test_single_patient_prediction_shape(pipeline):
    """Vérifie qu'une prédiction unitaire (comme dans l'app Streamlit) fonctionne."""
    patient = np.array([[50.0, 24.0, 90.0, 10.0, 2.0, 20.0, 10.0, 10.0, 300.0]])
    prediction = pipeline.predict(patient)

    assert prediction.shape == (1,)
    assert prediction[0] in (0, 1)


def test_prediction_logger(tmp_path, monkeypatch):
    """Vérifie que log_prediction écrit bien une ligne exploitable par le monitoring."""
    import sys
    sys.path.insert(0, str(PROJECT_DIR))
    from monitoring import logger as logger_module

    fake_log_path = tmp_path / "predictions_log.csv"
    monkeypatch.setattr(logger_module, "LOG_PATH", fake_log_path)

    logger_module.log_prediction(
        feature_values={feat: 1.0 for feat in FEATURES},
        prediction=1,
        probability=0.87,
    )

    assert fake_log_path.exists()
    logged_df = pd.read_csv(fake_log_path)
    assert len(logged_df) == 1
    assert logged_df.loc[0, "prediction"] == 1
    assert abs(logged_df.loc[0, "probability"] - 0.87) < 1e-6
