"""
logger.py
---------
Petit utilitaire pour journaliser chaque prédiction faite en production
(depuis l'app Streamlit) dans un fichier CSV. Ces logs servent de "données
courantes" (current data) pour le rapport de drift Evidently, comparées aux
données d'entraînement ("données de référence").

Aucune base de données externe n'est nécessaire : un CSV suffit pour ce
volume de trafic (projet académique / démo), mais la fonction log_prediction
peut être branchée sur SQLite/Postgres sans changer l'appelant.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_DIR / "monitoring" / "predictions_log.csv"

FEATURES = ["Age", "BMI", "Glucose", "Insulin", "HOMA", "Leptin", "Adiponectin", "Resistin", "MCP.1"]
LOG_COLUMNS = ["timestamp"] + FEATURES + ["prediction", "probability"]


def log_prediction(feature_values: dict, prediction: int, probability: Optional[float]) -> None:
    """Ajoute une ligne au log de prédictions (créé si absent)."""
    LOG_PATH.parent.mkdir(exist_ok=True)

    row = {"timestamp": datetime.utcnow().isoformat()}
    row.update({feat: feature_values[feat] for feat in FEATURES})
    row["prediction"] = prediction
    row["probability"] = probability if probability is not None else ""

    row_df = pd.DataFrame([row], columns=LOG_COLUMNS)
    header_needed = not LOG_PATH.exists()
    row_df.to_csv(LOG_PATH, mode="a", index=False, header=header_needed)


def load_predictions_log() -> pd.DataFrame:
    """Charge le log de prédictions ; renvoie un DataFrame vide si absent."""
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(LOG_PATH)
