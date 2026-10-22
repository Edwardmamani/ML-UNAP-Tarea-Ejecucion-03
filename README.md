# Tarea de Ejecución 03: Árboles de Decisión (CART) y Modelos Ensamblados

<div align="center">

**UNIVERSIDAD NACIONAL DEL ALTIPLANO**  
**FACULTAD DE INGENIERÍA MECÁNICA ELÉCTRICA, ELECTRÓNICA Y SISTEMAS**  
**ESCUELA PROFESIONAL DE INGENIERÍA DE SISTEMAS**

<img src="docs/Logo_UNAP.png" width="140" alt="Logo UNAP"/>

### APRENDIZAJE DE MÁQUINA (SEMESTRE IX - 2026-II)
**Presentado por:** MAMANI ZAPANA EDWARD  
**Puno – Perú, 2026**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB3C00.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Cuadernos de Experimentación (.ipynb)

Cada ítem cuenta con un cuaderno `.ipynb` independiente, completamente documentado, reproducible y pre-ejecutado con todas sus celdas y visualizaciones. Puede acceder a ellos directamente en el repositorio:

| Ítem | Tema / Enunciado | Cuaderno Jupyter (.ipynb) |
|:---:|:---|:---:|
| **Ítem 1** | CART (Gini vs. Entropía) vs. Regresor Logístico en *breast_cancer* | [`Item 01 — notebooks/Item_01_Arboles_Gini_Entropia_vs_RegresionLogistica.ipynb`](notebooks/Item_01_Arboles_Gini_Entropia_vs_RegresionLogistica.ipynb) |
| **Ítem 2** | Hiperparámetros en Árboles de Regresión vs. Regresor Lineal en *auto_mpg* | [`Item 02 — notebooks/Item_02_Arboles_Regresion_Hiperparametros_vs_RegresionLineal.ipynb`](notebooks/Item_02_Arboles_Regresion_Hiperparametros_vs_RegresionLineal.ipynb) |
| **Ítem 3** | Ensamble Clasificador (`VotingClassifier`) vs. Modelos Individuales en *Pima Diabetes* | [`Item 03 — notebooks/Item_03_Ensemble_VotingClassifier_vs_Individuales.ipynb`](notebooks/Item_03_Ensemble_VotingClassifier_vs_Individuales.ipynb) |
| **Ítem 4** | `BaggingClassifier` (con evaluación OOB) vs. Árbol Individual en *breast_cancer* | [`Item 04 — notebooks/Item_04_Bagging_Classifier_OOB_vs_ArbolIndividual.ipynb`](notebooks/Item_04_Bagging_Classifier_OOB_vs_ArbolIndividual.ipynb) |
| **Ítem 5** | Random Forest vs. XGBoost (Eficacia, Eficiencia e Importancias) en *Pima Diabetes* | [`Item 05 — notebooks/Item_05_RandomForest_vs_XGBoost.ipynb`](notebooks/Item_05_RandomForest_vs_XGBoost.ipynb) |

---

## 📄 Documentación Oficial

El informe académico formal se encuentra disponible en la carpeta [`docs/`](docs/):
- **Documento editable Word con Índice de Contenido:** [`docs/Informe_Tarea_Ejecucion_03.docx`](docs/Informe_Tarea_Ejecucion_03.docx)
- **Documento oficial en PDF (20 páginas):** [`docs/Informe_Tarea_Ejecucion_03.pdf`](docs/Informe_Tarea_Ejecucion_03.pdf)

El informe incluye la carátula institucional completa, el **Índice de Contenido estructurado y paginado**, tablas de resultados experimentales, 19 figuras de apoyo, pruebas estadísticas no paramétricas (Wilcoxon) y las respuestas detalladas a cada una de las interrogantes de la tarea.

---

## 📊 Resumen Ejecutivo de Hallazgos y Respuestas a las Preguntas

| Ítem | Pregunta de la Tarea | Modelo Ganador | Métricas Clave | Justificación Científica y Teórica |
|:---:|:---|:---:|:---:|:---|
| **1** | *¿Cuál es mejor entre CART (Gini/Entropía) y Regresión Logística?* | **Regresión Logística** | **CV Acc:** 98.24% ± 1.33%<br>**Test Acc:** 98.25%<br>**ROC-AUC:** 0.9954<br>**FN:** 1 | En espacios de 30 variables continuas de alta colinealidad como *breast_cancer*, la frontera suave de la Regresión Logística supera a las particiones ortogonales y escalonadas de CART ($p < 0.001$). Entre Gini y Entropía, Gini fue marginalmente superior y más veloz. |
| **2** | *¿Cuál es el mejor entre árboles con varios hiperparámetros y Regresión Lineal?* | **CART-3 (depth=5, leaf=5) / CART-5 (GridSearch)** | **Test RMSE:** 2.71 mpg<br>**Test R²:** 0.864<br>**CV RMSE:** 3.19 mpg | La relación entre peso/cilindrada y consumo es fuertemente no lineal. Los árboles podados capturan estas curvaturas superando a la regresión lineal (RMSE 2.89 mpg, R² 0.845). El árbol sin podar memorizó el ruido (RMSE train = 0.0, test = 3.37). |
| **3** | *¿Cuál es mejor entre ensamble Voting y modelos individuales?* | **VotingClassifier (Soft)** | **Test Acc:** 73.38%<br>**Precisión:** 67.57%<br>**ROC-AUC:** 0.8154<br>**std CV:** 0.038 | El ensamble combina modelos con sesgos inductivos disímiles (LR lineal, KNN métrico, CART ortogonal). Al tener errores poco correlacionados ($r \le 0.45$), la votación probabilística suave compensa las fallas individuales y logra la menor dispersión. |
| **4** | *¿Cuál es el mejor entre Bagging (con OOB) y árbol individual?* | **BaggingClassifier (300 árboles base)** | **CV Acc:** 96.27% vs 92.53%<br>**Test ROC-AUC:** 0.9931<br>**OOB Acc:** 96.48% | Bagging reduce drásticamente la varianza muestral promediando árboles bootstrap. En 30 particiones independientes superó al árbol individual en 26 ($p = 1.05 \times 10^{-5}$). La estimación OOB ($36.8\%$ datos no vistos) validó con exactitud el rendimiento de generalización. |
| **5** | *¿Cuál es mejor entre Random Forest y XGBoost?* | **XGBoost (Optimizado)** | **Test ROC-AUC:** 0.8209 vs 0.8035<br>**Log-Loss:** 0.4914 vs 0.5126<br>**Inferencia:** 0.04 ms vs 0.95 ms (~23x)<br>**Fit:** 0.17 s vs 1.54 s | XGBoost optimizado ofrece mejor calibración de probabilidades, menor Log-Loss, manejo nativo de valores faltantes y una velocidad de inferencia 23 veces más rápida que Random Forest gracias a árboles compactos (`max_depth=3`). Random Forest destaca por su robustez out-of-the-box. |

---

## 📁 Estructura del Repositorio

```text
ML-UNAP-Tarea-Ejecucion-03/
├── README.md               # Resumen ejecutivo y tablas comparativas
├── requirements.txt        # Dependencias del proyecto
├── ml_utils.py             # Módulo compartido de utilidades (load_csv, savefig, bootstrap_ci, plot_roc, etc.)
├── .gitignore              # Filtros de exclusión (entorno, cachés y scripts auxiliares)
├── notebooks/              # Los 5 cuadernos interactivos .ipynb pre-ejecutados
│   ├── Item_01_Arboles_Gini_Entropia_vs_RegresionLogistica.ipynb
│   ├── Item_02_Arboles_Regresion_Hiperparametros_vs_RegresionLineal.ipynb
│   ├── Item_03_Ensemble_VotingClassifier_vs_Individuales.ipynb
│   ├── Item_04_Bagging_Classifier_OOB_vs_ArbolIndividual.ipynb
│   └── Item_05_RandomForest_vs_XGBoost.ipynb
├── docs/                   # Informes académicos y recursos oficiales
│   ├── Informe_Tarea_Ejecucion_03.docx   # Documento Word editable con Índice de Contenido
│   ├── Informe_Tarea_Ejecucion_03.pdf    # PDF oficial compilado de 20 páginas
│   ├── Logo_UNAP.png                     # Escudo institucional UNAP
│   ├── Tarea de ejecución 03.pdf         # Guía oficial de la tarea
│   └── 05-Arboles-de-decision-en-clasificacion.md # Resumen teórico de referencia
├── figures/                # 29 figuras diagnósticas en alta resolución (.png)
├── results/                # 5 archivos JSON con todas las métricas experimentales
└── data/                   # Datasets locales auxiliares (auto-mpg.csv, diabetes.csv)
```

---

## ⚙️ Reproducción Local

### 1. Clonar el repositorio
```bash
git clone https://github.com/Edwardmamani/ML-UNAP-Tarea-Ejecucion-03.git
cd ML-UNAP-Tarea-Ejecucion-03
```

### 2. Configurar el entorno virtual e instalar dependencias
Con `uv` (recomendado):
```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

O con `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Explorar y ejecutar los cuadernos
Los 5 cuadernos se encuentran pre-ejecutados y documentados en la carpeta `notebooks/`. Pueden abrirse y ejecutarse interactivamente con JupyterLab, VS Code o cualquier entorno Jupyter.

---

## 🏛️ Información Institucional
- **Institución:** Universidad Nacional del Altiplano (UNAP)
- **Facultad:** Facultad de Ingeniería Mecánica Eléctrica, Electrónica y Sistemas
- **Escuela Profesional:** Escuela Profesional de Ingeniería de Sistemas
- **Curso:** Aprendizaje de Máquina (Semestre IX - 2026-II)
- **Estudiante:** Mamani Zapana Edward
- **Docente:** Mayenka Fernández Chambi
- **Fecha:** Octubre de 2026
