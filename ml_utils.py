"""Módulo de utilidades compartidas para los notebooks de Aprendizaje de Máquina.

Tarea de Ejecución 03 - UNAP (Semestre IX - 2026-II)
Autor: Edward Mamani Zapana
"""

import json
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RANDOM_STATE = 42
REPO_RAW = "https://raw.githubusercontent.com/Edwardmamani/ML-UNAP-Tarea-Ejecucion-03/main"


def setup_environment(seed: int = RANDOM_STATE, dpi: int = 110):
    """Configura advertencias, tema de Seaborn, reproducibilidad y directorios de salida."""
    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = dpi
    np.random.seed(seed)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def load_csv(name: str, repo_raw: str = REPO_RAW) -> pd.DataFrame:
    """Carga un dataset desde data/ (local o en subcarpetas) o desde GitHub (en Colab)."""
    candidates = [
        os.path.join("data", name),
        os.path.join("..", "data", name),
        name,
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    url = f"{repo_raw}/data/{name}"
    return pd.read_csv(url)


def savefig(name: str, dpi: int = 160):
    """Guarda la figura actual en figures/<name>.png con alta resolución."""
    out_dir = "figures" if not os.path.exists("../figures") else "../figures"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=dpi, bbox_inches="tight")


def save_results(obj: dict, name: str):
    """Guarda un diccionario de resultados y métricas en results/<name>.json."""
    out_dir = "results" if not os.path.exists("../results") else "../results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )


def bootstrap_ci(y_true, y_score, metric, n_boot: int = 1000, seed: int = RANDOM_STATE):
    """Calcula el intervalo de confianza bootstrap al 95% para una métrica en el conjunto de prueba."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2 and metric.__name__ == "roc_auc_score":
            continue
        vals.append(metric(y_true[idx], y_score[idx]))
    return np.percentile(vals, [2.5, 97.5])


def plot_roc(ax, model, X_eval, y_eval, name: str, color=None, lw: float = 1.8):
    """Dibuja la curva ROC con su área bajo la curva (AUC) de forma compatible entre versiones."""
    from sklearn.metrics import roc_auc_score, roc_curve

    score = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_eval, score)
    auc_val = roc_auc_score(y_eval, score)
    ax.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC = {auc_val:.3f})")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend(loc="lower right", fontsize=8)


def rmse(y_true, y_pred) -> float:
    """Calcula la raíz del error cuadrático medio (RMSE)."""
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_clean_diabetes_data(repo_raw: str = REPO_RAW):
    """Carga y limpia el dataset Pima Indians Diabetes tratando ceros biológicamente imposibles como NaN."""
    df = load_csv("diabetes.csv", repo_raw=repo_raw)
    zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_nan] = df[zero_as_nan].replace(0, np.nan)
    X = df.drop(columns="Outcome")
    y = df["Outcome"]
    return df, X, y
