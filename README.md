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
- **Documento oficial en PDF (17 páginas):** [`docs/Informe_Tarea_Ejecucion_03.pdf`](docs/Informe_Tarea_Ejecucion_03.pdf)

El informe incluye la carátula institucional completa, el **Índice de Contenido estructurado y paginado**, tablas de resultados experimentales, 19 figuras de apoyo, pruebas estadísticas no paramétricas (Wilcoxon) y las respuestas detalladas a cada una de las interrogantes de la tarea.

---

## 🏆 Hallazgos Experimentales Principales y Respuestas a la Tarea

Esta práctica evalúa de manera comparativa y sistemática el desempeño de modelos lineales e individuales (CART) frente a familias de ensambles (VotingClassifier, Bagging, Random Forest y XGBoost), aplicando validación cruzada 10-Fold estratificada, pruebas de significancia no paramétricas de Wilcoxon y análisis de complejidad sesgo-varianza.

A continuación se destacan los hallazgos clave obtenidos para cada una de las preguntas de la tarea:

### 1️⃣ Ítem 1: Criterios CART (Gini vs. Entropía) vs. Regresión Logística en *breast_cancer*
> **Pregunta:** *Entrene modelos de árboles de decisión con criterios de impureza diferentes (índice gini y entropía) usando el dataset “breast_cancer” de scikit-learn, compare su eficiencia con respecto a un modelo lineal como el Regresor Logístico. ¿Cuál es mejor?*

* **Modelo Ganador:** **Regresión Logística** (ajustada en Pipeline con estandarización `StandardScaler`).
* **Hallazgo Principal:** En espacios de 30 variables continuas de alta colinealidad, la **frontera continua y sigmoidea** de la Regresión Logística supera con significancia estadística ($p < 0.001$, prueba de Wilcoxon) a las particiones ortogonales y escalonadas de CART.
* **Métricas Clave:**
  * **Regresión Logística:** Exactitud CV: **98.24% ± 1.33%** | Exactitud Test: **98.25%** | ROC-AUC: **0.9954** | F1 Maligno: **0.9762** | Falsos Negativos en Test: **1**.
  * **CART Gini Podado (depth=4, leaf=5):** Exactitud CV: **93.62% ± 3.42%** | Exactitud Test: **92.11%** | ROC-AUC: **0.9263** | Falsos Negativos: **4**.
  * **CART Sin Podar:** Sufrió sobreajuste severo (**100% en entrenamiento**, pero cayó a **91.23% en prueba**).
* **Comparativa Gini vs. Entropía:** El criterio Gini fue ligeramente superior en generalización (93.62% vs. 93.19% en CV) y ~15% más veloz en tiempo de ajuste, al evitar el cómputo de funciones logarítmicas.
* **Evidencia Visual:** [Curva Sesgo-Varianza](figures/item1_complejidad.png) | [Matrices de Confusión](figures/item1_confusion.png) | [Importancia de Características](figures/item1_importancia.png).

---

### 2️⃣ Ítem 2: Hiperparámetros en Árboles de Regresión vs. Regresión Lineal en *auto_mpg*
> **Pregunta:** *Entrene modelos de árboles de decisión con varias configuraciones de hiper parámetros usando el dataset auto_mpg, comparar su eficiencia con respecto a un modelo lineal como el Regresor Lineal. ¿Cuál es el mejor?*

* **Modelo Ganador:** **CART Podado (depth=5, min_samples_leaf=5)** y **CART Optimizado por GridSearch**.
* **Hallazgo Principal:** La relación física entre peso/cilindrada y el consumo en millas por galón (mpg) es fuertemente no lineal. Los árboles podados capturan eficazmente estas curvaturas superando a la regresión lineal multivariada.
* **Métricas Clave:**
  * **CART Podado (depth=5, leaf=5):** RMSE Test: **2.71 mpg** | $R^2$ Test: **0.864** | RMSE CV: **3.32 ± 0.51 mpg**.
  * **Regresión Lineal:** RMSE Test: **2.89 mpg** | $R^2$ Test: **0.845** | RMSE CV: **3.45 ± 0.53 mpg**.
  * **CART Sin Podar (Sobreajuste):** RMSE Train: **0.00 mpg** ($R^2 = 1.0$), pero su error en prueba se degradó a **3.37 mpg** ($R^2 = 0.789$).
* **Importancia de la Poda:** Sin restricciones (`max_depth=None`), el estimador memoriza el ruido. Limitar la profundidad y exigir hojas con al menos 5 a 8 muestras minimiza el error de generalización.
* **Evidencia Visual:** [Curva de Complejidad](figures/item2_complejidad.png) | [Dispersión de Predicciones vs Real](figures/item2_predicciones.png).

---

### 3️⃣ Ítem 3: Ensamble Heterogéneo (`VotingClassifier`) vs. Modelos Individuales en *Pima Diabetes*
> **Pregunta:** *Entrene un modelo ensemble clasificador, y compare su eficiencia con respecto a un modelo individual. ¿Cuál es mejor?*

* **Modelo Ganador:** **VotingClassifier (Votación Suave / Soft Voting)**.
* **Hallazgo Principal:** El ensamble combina modelos con sesgos inductivos heterogéneos (Regresión Logística lineal, KNN métrico y CART ortogonal). Al exhibir baja correlación en sus patrones de error ($r \le 0.45$), la combinación probabilística compensa fallos individuales y reduce la dispersión entre pliegues.
* **Métricas Clave:**
  * **VotingClassifier (Soft):** Exactitud Test: **73.38%** | Precisión: **67.57%** | ROC-AUC: **0.8154** | Mínima desviación en 30 pliegues ($\sigma = 0.038$).
  * **Modelos Individuales:** Regresión Logística (70.78% test, AUC 0.8105), KNN (74.03% test, AUC 0.8038), Árbol Individual (70.13% test, AUC 0.7410).
  * **Soft vs. Hard Voting:** La votación suave superó a la dura (72.08% test) al ponderar la certidumbre continua de las probabilidades a posteriori en lugar de una simple cuenta discreta de votos.
* **Evidencia Visual:** [Correlación de Errores](figures/item3_correlacion_errores.png) | [Comparativa de Modelos](figures/item3_comparacion.png).

---

### 4️⃣ Ítem 4: Reducción de Varianza con Bagging y Validación OOB en *breast_cancer*
> **Pregunta:** *Entrene un modelo bagging regresor o clasificador y compare su eficiencia con respecto a un modelo individual. ¿Cuál es el mejor?*

* **Modelo Ganador:** **BaggingClassifier (300 árboles base)**.
* **Hallazgo Principal:** Bagging erradica la inestabilidad muestral del árbol CART individual promediando árboles no podados sobre réplicas bootstrap.
* **Métricas Clave:**
  * **Bagging (300 árboles):** Exactitud CV: **96.27% ± 3.67%** | Exactitud Test: **96.49%** | ROC-AUC: **0.9931**.
  * **Árbol Individual:** Exactitud CV: **92.53% ± 4.16%** | Exactitud Test: **92.11%** | ROC-AUC: **0.9157**.
* **Significancia en 30 Particiones:** En 30 particiones bootstrap independientes, Bagging superó al árbol en **26 de 30 repeticiones** ($p = 1.05 \times 10^{-5}$, Wilcoxon).
* **Demostración y Validación OOB:** La probabilidad asintótica de que una muestra quede fuera de la bolsa bootstrap es $1/e \approx 36.8\%$. La evaluación interna Out-Of-Bag (**96.48%**) estimó de manera prácticamente idéntica el rendimiento en el conjunto de prueba (**96.49%**), demostrando que permite prescindir de particiones de validación adicionales.
* **Evidencia Visual:** [Convergencia OOB vs Test](figures/item4_oob_convergencia.png) | [Distribución de Varianza en 30 Particiones](figures/item4_distribucion_varianza.png).

---

### 5️⃣ Ítem 5: Random Forest vs. XGBoost en *Pima Diabetes*
> **Pregunta:** *Entrene un modelo RandomForest y compare su eficiencia con respecto a un modelo XGBoost. ¿Cuál es mejor?*

* **Modelo Ganador:** **XGBoost (Optimizado por GridSearch)**.
* **Hallazgo Principal:** Aunque Random Forest es altamente robusto por defecto, XGBoost optimizado ofrece mejor calibración de probabilidades, menor función de pérdida (Log-Loss) y una velocidad de inferencia 23 veces superior.
* **Métricas Clave:**
  * **XGBoost Optimizado:** ROC-AUC Test: **0.8209** | Log-Loss: **0.4914** | Brier Score: **0.1610** | Inferencia: **0.04 ms/muestra** | Búsqueda: **13.1 s**.
  * **Random Forest Optimizado:** ROC-AUC Test: **0.8035** | Log-Loss: **0.5126** | Brier Score: **0.1697** | Inferencia: **0.95 ms/muestra** | Búsqueda: **67.2 s**.
* **Eficiencia Computacional:** XGBoost ejecuta inferencia **23 veces más rápido** que Random Forest gracias a árboles compactos (`max_depth=3`) en contraste con los árboles profundos de Random Forest.
* **Manejo Nativo de Valores Nulos:** XGBoost aprende la dirección de derivación para valores faltantes (glucosa o insulina cero), superando los sesgos de la imputación univariada.
* **Evidencia Visual:** [Curvas ROC y PR](figures/item5_roc_pr.png) | [Importancia de Características](figures/item5_importancias.png) | [Eficacia vs Tiempos](figures/item5_tiempos.png).

---

## 📊 Resumen Ejecutivo Consolidado

| Ítem | Dataset | Problema | Modelo Ganador | Métricas Clave | Justificación Científica y Teórica |
|:---:|:---|:---|:---:|:---|:---|
| **1** | *breast_cancer* | Clasificación (30 vars.) | **Regresión Logística** | **CV Acc:** 98.24% ± 1.33%<br>**Test Acc:** 98.25%<br>**ROC-AUC:** 0.9954<br>**FN:** 1 | La frontera sigmoidea continua supera a las particiones ortogonales en variables fuertemente correlacionadas ($p < 0.001$). Gini fue más veloz y preciso que Entropía. |
| **2** | *auto_mpg* | Regresión (no lineal) | **CART-3 (depth=5, leaf=5)** | **Test RMSE:** 2.71 mpg<br>**Test R²:** 0.864<br>**CV RMSE:** 3.19 mpg | La relación entre peso/potencia y consumo es fuertemente no lineal. Árboles podados superan a la regresión lineal (RMSE 2.89 mpg). Árbol sin podar memorizó ruido (train 0.0). |
| **3** | *Pima Diabetes* | Clasificación (desbalanceada) | **VotingClassifier (Soft)** | **Test Acc:** 73.38%<br>**Precisión:** 67.57%<br>**ROC-AUC:** 0.8154<br>**std CV:** 0.038 | Combina sesgos disímiles (LR, KNN, CART) con errores poco correlacionados ($r \le 0.45$). La votación probabilística suave logra la menor dispersión. |
| **4** | *breast_cancer* | Clasificación (estabilidad) | **BaggingClassifier (300 est.)** | **CV Acc:** 96.27% vs 92.53%<br>**Test ROC-AUC:** 0.9931<br>**OOB Acc:** 96.48% | Promedia réplicas bootstrap reduciendo la varianza en 26 de 30 particiones ($p = 1.05 \times 10^{-5}$). La fracción OOB (36.8%) validó con precisión el test set. |
| **5** | *Pima Diabetes* | Ensambles avanzados | **XGBoost (Optimizado)** | **Test ROC-AUC:** 0.8209 vs 0.8035<br>**Log-Loss:** 0.4914 vs 0.5126<br>**Inferencia:** 0.04 ms vs 0.95 ms (~23x) | Menor Log-Loss, manejo nativo de nulos e inferencia 23x más rápida por árboles compactos (`max_depth=3`). Random Forest destaca por su robustez out-of-the-box. |

---


## 🏛️ Información Institucional
- **Institución:** Universidad Nacional del Altiplano (UNAP)
- **Facultad:** Facultad de Ingeniería Mecánica Eléctrica, Electrónica y Sistemas
- **Escuela Profesional:** Escuela Profesional de Ingeniería de Sistemas
- **Curso:** Aprendizaje de Máquina (Semestre IX - 2026-II)
- **Estudiante:** Mamani Zapana Edward
- **Docente:** Mayenka Fernández Chambi
- **Fecha:** Octubre de 2026
