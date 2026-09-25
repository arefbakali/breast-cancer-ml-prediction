"""
drift_report.py
----------------
Génère un rapport de data drift avec Evidently, en comparant :
  - les données de référence (monitoring/reference_data.csv, extraites du
    jeu d'entraînement par train/train_model.py)
  - les données courantes (monitoring/predictions_log.csv, journalisées par
    l'app Streamlit à chaque prédiction en production)

Le rapport est exporté en HTML dans monitoring/reports/.

Usage :
    python monitoring/drift_report.py
"""

from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from monitoring.logger import load_predictions_log, FEATURES

PROJECT_DIR = Path(__file__).resolve().parent.parent
REFERENCE_PATH = PROJECT_DIR / "monitoring" / "reference_data.csv"
REPORTS_DIR = PROJECT_DIR / "monitoring" / "reports"

MIN_ROWS_FOR_REPORT = 10  # Evidently a besoin d'un minimum de lignes pour un test statistique fiable


def build_drift_report() -> Path:
    """Construit le rapport de drift et retourne le chemin du fichier HTML généré."""
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            "Données de référence introuvables. Lancez d'abord `python train/train_model.py`."
        )

    reference_df = pd.read_csv(REFERENCE_PATH)[FEATURES]
    current_df = load_predictions_log()

    if len(current_df) < MIN_ROWS_FOR_REPORT:
        raise ValueError(
            f"Pas assez de prédictions journalisées ({len(current_df)}) pour un rapport fiable. "
            f"Minimum recommandé : {MIN_ROWS_FOR_REPORT}. Utilisez l'app pour générer des prédictions."
        )

    current_df = current_df[FEATURES]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    REPORTS_DIR.mkdir(exist_ok=True)
    output_path = REPORTS_DIR / "data_drift_report.html"
    report.save_html(str(output_path))
    return output_path


def get_drift_summary() -> dict:
    """Retourne un résumé synthétique (dict) du drift, pour affichage rapide dans Streamlit."""
    if not REFERENCE_PATH.exists():
        return {"status": "no_reference"}

    reference_df = pd.read_csv(REFERENCE_PATH)[FEATURES]
    current_df = load_predictions_log()

    if len(current_df) < MIN_ROWS_FOR_REPORT:
        return {"status": "not_enough_data", "n_current": len(current_df)}

    current_df = current_df[FEATURES]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    result = report.as_dict()

    drift_metric = result["metrics"][0]["result"]
    return {
        "status": "ok",
        "dataset_drift_detected": drift_metric.get("dataset_drift", False),
        "n_drifted_features": drift_metric.get("number_of_drifted_columns", 0),
        "n_features": drift_metric.get("number_of_columns", len(FEATURES)),
        "n_current": len(current_df),
    }


if __name__ == "__main__":
    path = build_drift_report()
    print(f"✅ Rapport de drift généré : {path}")
