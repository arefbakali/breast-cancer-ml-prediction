"""
restore_run.py
---------------
Récupère le pipeline (scaler + modèle) d'un run MLflow spécifique et
écrase models/pipeline.pkl avec — utile pour revenir à une version
antérieure si un run plus récent s'avère moins bon (ex: revenir au run #4
alors que le run #5 vient d'être fait).

Usage :
    python restore_run.py <run_id>

Trouver un run_id : lance `mlflow ui --backend-store-uri ./mlruns`,
clique sur le run voulu, copie le "Run ID" affiché en haut de la page.
"""

import sys
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn

PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = PROJECT_DIR / "models" / "pipeline.pkl"


def restore(run_id: str):
    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"✅ {PIPELINE_PATH} restauré depuis le run {run_id}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python restore_run.py <run_id>")
        sys.exit(1)
    restore(sys.argv[1])
