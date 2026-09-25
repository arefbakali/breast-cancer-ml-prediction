"""
train_model.py
----------------
Ré-entraîne les 6 modèles ML du projet et logue chaque run dans MLflow :
hyperparamètres, métriques, matrice de confusion + courbe ROC (en image),
et le pipeline complet (scaler + modèle) avec sa signature d'entrée/sortie.

Pas de Model Registry ici : ce projet est mené seul, l'app Streamlit lit
directement models/pipeline.pkl, donc un registre de versions n'apporterait
rien de plus que le tracking déjà présent dans mlruns/ (voir README pour la
justification de ce choix).

Usage :
    python train/train_model.py

Pour visualiser les runs :
    mlflow ui --backend-store-uri ./mlruns
    (puis ouvrir http://localhost:5000)
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression as LassoLogReg  # Lasso = LogReg avec pénalité L1

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "dataR2.csv"
MODELS_DIR = PROJECT_DIR / "models"
FEATURES = ["Age", "BMI", "Glucose", "Insulin", "HOMA", "Leptin", "Adiponectin", "Resistin", "MCP.1"]
TARGET = "Classification"

EXPERIMENT_NAME = "breast-cancer-detection"


def load_data():
    df = pd.read_csv(DATA_PATH)
    if set(df[TARGET].dropna().unique()) == {1, 2}:
        df[TARGET] = df[TARGET].map({1: 0, 2: 1})
    return df


def eval_metrics(y_true, y_pred, y_proba=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }
    if y_proba is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            pass
    return metrics


def log_confusion_matrix(y_true, y_pred, run_name):
    """Sauvegarde la matrice de confusion en image et la logue comme artefact du run."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Sain", "Cancer"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Matrice de confusion — {run_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close(fig)
    mlflow.log_artifact("confusion_matrix.png")


def log_roc_curve(pipeline, X_test, y_test, run_name):
    """Sauvegarde la courbe ROC en image et la logue comme artefact du run (si applicable)."""
    if not hasattr(pipeline, "predict_proba"):
        return
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title(f"Courbe ROC — {run_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.close(fig)
    mlflow.log_artifact("roc_curve.png")


def get_model_configs():
    """Définit les 6 pipelines + grilles d'hyperparamètres (identiques au notebook)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

    configs = {
        "logistic_regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("model", LogisticRegression(solver="lbfgs", max_iter=2000,
                                                     class_weight="balanced",
                                                     random_state=RANDOM_STATE))]),
            {"model__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "naive_bayes": (
            Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())]),
            {"model__var_smoothing": np.logspace(-12, -6, 7)},
        ),
        "ridge": (
            Pipeline([("scaler", StandardScaler()),
                      ("model", RidgeClassifier(class_weight="balanced", random_state=RANDOM_STATE))]),
            {"model__alpha": [0.01, 0.1, 1, 10, 100]},
        ),
        "lasso": (
            Pipeline([("scaler", StandardScaler()),
                      ("model", LassoLogReg(penalty="l1", solver="liblinear", max_iter=2000,
                                             class_weight="balanced", random_state=RANDOM_STATE))]),
            {"model__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "knn": (
            Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
            {"model__n_neighbors": [3, 5, 7, 9, 11],
             "model__weights": ["uniform", "distance"],
             "model__metric": ["euclidean", "manhattan"]},
        ),
        "mlp": (
            Pipeline([("scaler", StandardScaler()),
                      ("model", MLPClassifier(max_iter=2000, random_state=RANDOM_STATE))]),
            {"model__hidden_layer_sizes": [(16,), (32,), (16, 8)],
             "model__alpha": [0.0001, 0.001, 0.01]},
        ),
    }
    return configs, cv


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    configs, cv = get_model_configs()
    results = {}

    for name, (pipeline, param_grid) in configs.items():
        with mlflow.start_run(run_name=name):
            grid = GridSearchCV(pipeline, param_grid, scoring="balanced_accuracy",
                                 cv=cv, n_jobs=-1, refit=True)
            grid.fit(X_train, y_train)
            best_pipeline = grid.best_estimator_

            y_pred = best_pipeline.predict(X_test)
            try:
                y_proba = best_pipeline.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_proba = None

            metrics = eval_metrics(y_test, y_pred, y_proba)
            results[name] = {"pipeline": best_pipeline, "metrics": metrics}

            # Params + métriques
            mlflow.log_params({k.replace("model__", ""): v for k, v in grid.best_params_.items()})
            mlflow.log_metric("cv_balanced_accuracy", grid.best_score_)
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            # Artefacts visuels — un coup d'œil suffit pour comparer les runs dans l'UI
            log_confusion_matrix(y_test, y_pred, name)
            log_roc_curve(best_pipeline, X_test, y_test, name)

            # Modèle + signature (types/colonnes attendus en entrée, sortie produite)
            signature = infer_signature(X_train, best_pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                best_pipeline, artifact_path="model",
                signature=signature, input_example=X_train.head(3),
                serialization_format="cloudpickle",
            )

            print(f"[{name}] cv_balanced_accuracy={grid.best_score_:.4f} "
                  f"test_accuracy={metrics['accuracy']:.4f} test_f1={metrics['f1']:.4f}")

    # ── Sélection du meilleur modèle (même logique que le notebook : K-NN retenu
    #    pour son meilleur équilibre accuracy/F1/AUC avec peu de faux positifs) ──
    best_name = "knn"
    best_pipeline = results[best_name]["pipeline"]
    best_metrics = results[best_name]["metrics"]

    # ── Sauvegarde pipeline.pkl (scaler + modèle réunis) pour l'app Streamlit —
    #    un seul fichier, pas de dépendance à un serveur MLflow au moment de
    #    l'inférence, et pas de risque de désynchronisation scaler/modèle. ──
    with open(MODELS_DIR / "pipeline.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)

    # ── Sauvegarde des données de référence pour le monitoring de drift ──
    reference_path = PROJECT_DIR / "monitoring" / "reference_data.csv"
    reference_path.parent.mkdir(exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(reference_path, index=False)

    print("\n✅ Entraînement terminé.")
    print("   6 runs loggés dans MLflow (params, métriques, confusion matrix, ROC, modèle)")
    print(f"   pipeline.pkl mis à jour dans {MODELS_DIR}")
    print(f"   Données de référence (drift) sauvegardées dans {reference_path}")
    print(f"   Métriques test du modèle retenu ({best_name}) : {best_metrics}")
    print("\n   Pour comparer les runs : mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
