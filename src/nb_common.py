"""Utilidades comunes para construir los notebooks de la Tarea de Ejecución 03."""
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

GITHUB_USER = "Edwardmamani"
REPO_NAME = "ML-UNAP-Tarea-Ejecucion-03"
REPO_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}"
RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main"


def colab_url(filename: str) -> str:
    return f"https://colab.research.google.com/github/{GITHUB_USER}/{REPO_NAME}/blob/main/notebooks/{filename}"


def md(text: str):
    return new_markdown_cell(textwrap.dedent(text).strip("\n"))


def code(text: str):
    return new_code_cell(textwrap.dedent(text).strip("\n"))


def header(item: int, title: str, question: str, filename: str):
    return md(f"""
    <div align="center">

    **UNIVERSIDAD NACIONAL DEL ALTIPLANO**<br>
    FACULTAD DE INGENIERÍA MECÁNICA ELÉCTRICA, ELECTRÓNICA Y SISTEMAS<br>
    ESCUELA PROFESIONAL DE INGENIERÍA DE SISTEMAS

    <img src="{RAW_URL}/docs/Logo_UNAP.png" width="130" alt="Logo UNAP"/>

    # Tarea de Ejecución 03 — Ítem {item}
    ## {title}

    **CURSO:** APRENDIZAJE MAQUINA (SEMESTRE IX - 2026-II)<br>
    **PRESENTADO POR:** MAMANI ZAPANA EDWARD<br>
    PUNO – PERÚ, 2026

    </div>

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(filename)})
    &nbsp; Repositorio: [{REPO_URL}]({REPO_URL})

    ---

    > **Enunciado:** {question}
    """)


SETUP = f'''
# Librerías y configuración global de reproducibilidad
import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
REPO_RAW = "{RAW_URL}"


def load_csv(name):
    """Carga un CSV desde data/ (local) o desde el repositorio GitHub (Colab)."""
    local = os.path.join("data", name)
    return pd.read_csv(local if os.path.exists(local) else f"{{REPO_RAW}}/data/{{name}}")


def savefig(name):
    """Guarda la figura actual en figures/ (se reutiliza en el informe .docx)."""
    plt.savefig(f"figures/{{name}}.png", dpi=160, bbox_inches="tight")


def bootstrap_ci(y_true, y_score, metric, n_boot=1000, seed=RANDOM_STATE):
    """Intervalo de confianza bootstrap al 95% de una métrica sobre el conjunto de prueba."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2 and metric.__name__ == "roc_auc_score":
            continue
        vals.append(metric(y_true[idx], y_score[idx]))
    return np.percentile(vals, [2.5, 97.5])


def save_results(obj, name):
    """Guarda los resultados en results/<name>.json (insumo del informe)."""
    with open(f"results/{{name}}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


def plot_roc(ax, model, X_eval, y_eval, name, color=None, lw=1.8):
    """Dibuja la curva ROC de un modelo (compatible con cualquier versión de scikit-learn)."""
    from sklearn.metrics import roc_curve, roc_auc_score
    score = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_eval, score)
    ax.plot(fpr, tpr, color=color, lw=lw, label=f"{{name}} (AUC = {{roc_auc_score(y_eval, score):.3f}})")
    ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend(loc="lower right", fontsize=8)


print("scikit-learn", sklearn.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)
'''


def new_notebook(cells):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    return nb
