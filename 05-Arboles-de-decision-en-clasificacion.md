# Resumen: Árboles de Decisión y Ensambles en Clasificación y Regresión

**Fuente:** Elie Kawerk (*Machine Learning with Tree-Based Models in Python*), traducido y adaptado por Mayenka Fernández Chambi  
**Temas centrales:** CART (clasificación y regresión), equilibrio sesgo-varianza, validación cruzada K-Fold, aprendizaje por ensambles (Voting, Bagging, OOB) y Random Forests (importancia de características).

---

## 1. Árboles de Clasificación y Regresión (CART)

Un árbol de decisión es una estructura jerárquica compuesta por una secuencia de preguntas *if-else* evaluadas sobre características individuales para inferir etiquetas o valores continuos.

### Propiedades fundamentales
- **Relaciones no lineales:** Capaz de capturar interacciones y fronteras no lineales complejas sin requerir transformaciones manuales previas.
- **Sin necesidad de escalamiento:** No requiere estandarización ni normalización de características (a diferencia de SVM, KNN o redes neuronales).
- **Regiones de decisión ortogonales:** Genera fronteras rectangulares (hiperrectángulos) perpendiculares a los ejes, ya que cada división (*split*) involucra una sola característica a la vez.

### Anatomía de un árbol
1. **Nodo raíz (Root node):** Nodo superior sin padres donde inicia el flujo de evaluación.
2. **Nodos internos (Internal nodes):** Nodos intermedios con un padre que evalúan una condición ($f \le sp$) y bifurcan en dos nodos hijos.
3. **Nodos hoja (Leaf / Terminal nodes):** Nodos terminales sin hijos que emiten la predicción:
   - *Clasificación:* Clase mayoritaria en la hoja.
   - *Regresión:* Valor promedio de la variable objetivo de las muestras en la hoja.
4. **Profundidad máxima (`max_depth`):** Número máximo de niveles de ramificación entre la raíz y cualquier hoja.

---

## 2. Mecanismo de Aprendizaje y Criterios de División

El árbol se construye de manera recursiva (*greedy search*) seleccionando en cada nodo la característica $f$ y el punto de corte $sp$ que maximicen la **Ganancia de Información (IG)**.

### Ganancia de Información (IG)

$$IG(\text{nodo}) = I(\text{nodo}) - \left( \frac{N_{\text{izq}}}{N} I(\text{nodo}_{\text{izq}}) + \frac{N_{\text{der}}}{N} I(\text{nodo}_{\text{der}}) \right)$$

Donde $N$ es el número de muestras en el nodo padre, y $N_{\text{izq}}, N_{\text{der}}$ son las muestras que se desvían a cada nodo hijo.

### Criterios de impureza $I(\text{nodo})$

| Tarea | Criterio | Expresión matemática | Propósito |
|---|---|---|---|
| **Clasificación** | **Índice Gini** | $I_G = 1 - \sum_{i=1}^C p_i^2$ | Mide la probabilidad de clasificar erróneamente un elemento elegido al azar. Cómputo más veloz. |
| **Clasificación** | **Entropía** | $H = -\sum_{i=1}^C p_i \log_2(p_i)$ | Mide el desorden o incertidumbre de la información. |
| **Regresión** | **MSE** | $\frac{1}{N} \sum_{i=1}^N (y_i - \bar{y})^2$ | Mide la dispersión cuadrática de los valores respecto a la media de la hoja. |

### Criterios de parada
- La ganancia de información $IG = 0$ (nodo perfectamente puro).
- Se alcanza la profundidad máxima configurada (`max_depth`).
- La cantidad de muestras en un nodo es inferior a `min_samples_split` o `min_samples_leaf`.

---

## 3. Implementación de CART en Scikit-Learn

### Árbol de Clasificación (`DecisionTreeClassifier`)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# División estratificada de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

# Instanciación y entrenamiento con criterio Gini
dt = DecisionTreeClassifier(max_depth=2, criterion='gini', random_state=1)
dt.fit(X_train, y_train)

# Predicción y evaluación
y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy en test: {acc:.3f}")
```

### Árbol de Regresión (`DecisionTreeRegressor`)

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3
)

# Regularización con min_samples_leaf (10% de muestras mínimas por hoja)
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1, random_state=3)
dt.fit(X_train, y_train)

# Predicción y cálculo de RMSE
y_pred = dt.predict(X_test)
rmse = MSE(y_test, y_pred) ** 0.5
print(f"RMSE en test: {rmse:.3f}")
```

---

## 4. Error de Generalización y Equilibrio Sesgo-Varianza

En el aprendizaje supervisado $y = f(x) + \epsilon$, el objetivo es encontrar un estimador $\hat{f}$ que aproxime la función real $f$ minimizando el error en datos no vistos.

### Descomposición matemática del error

$$\text{Error de Generalización}(\hat{f}) = \text{Sesgo}^2 + \text{Varianza} + \text{Error Irreducible}$$

- **Sesgo (Bias):** Error promedio entre las predicciones del modelo y la realidad ($E[\hat{f}] \ne f$). Refleja supuestos simplistas.
- **Varianza (Variance):** Grado de dispersión e inconsistencia de $\hat{f}$ cuando se entrena con distintos subconjuntos de datos. Refleja extrema sensibilidad a las muestras de entrenamiento.
- **Error irreducible:** Contribución del ruido aleatorio inherente a los datos ($\epsilon$). Es un límite teórico insuperable.

### Comparativa: Sobreajuste vs. Subajuste

| Concepto | Sesgo / Varianza | Rendimiento en datos | Causa principal | Solución recomendada |
|---|---|---|---|---|
| **Subajuste (Underfitting)** | Alto sesgo, baja varianza | Error alto en train $\approx$ Error alto en test | Modelo demasiado rígido / simple para capturar los patrones. | Aumentar complejidad (mayor `max_depth`, menor `min_samples_leaf`), añadir features relevantes. |
| **Sobreajuste (Overfitting)** | Bajo sesgo, alta varianza | Error muy bajo en train $\ll$ Error alto en test/CV | Modelo memoriza el ruido específico de entrenamiento. | Reducir complejidad (podar árbol, limitar `max_depth`, aumentar `min_samples_leaf`), ensambles, más datos. |

---

## 5. Diagnóstico con Validación Cruzada K-Fold

Para evaluar la capacidad de generalización sin contaminar el conjunto de prueba, se utiliza **K-Fold CV** sobre el conjunto de entrenamiento:

1. Se divide el conjunto de entrenamiento en $K$ pliegues (*folds*).
2. Se entrena el modelo en $K - 1$ pliegues y se evalúa en el pliegue restante.
3. Se repite $K$ veces para obtener una distribución de errores representativa.

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE

# Evaluación con 10-Fold CV
mse_cv_scores = -cross_val_score(
    dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1
)
cv_mse = mse_cv_scores.mean()

# Evaluación en entrenamiento y prueba
dt.fit(X_train, y_train)
train_mse = MSE(y_train, dt.predict(X_train))
test_mse = MSE(y_test, dt.predict(X_test))

# Regla de diagnóstico:
# Si cv_mse > train_mse -> El modelo sufre de alta varianza (sobreajuste).
# Si cv_mse ≈ train_mse >> error deseado -> El modelo sufre de alto sesgo (subajuste).
```

---

## 6. Aprendizaje por Ensambles (Ensemble Learning)

Aunque los árboles CART son interpretables y versátiles, tienen dos debilidades críticas:
1. Fronteras de decisión únicamente ortogonales.
2. Alta sensibilidad a variaciones en los datos de entrenamiento (alta varianza al crecer sin restricciones).

El **aprendizaje por ensambles** entrena múltiples modelos base y combina sus predicciones a través de un metamodelo para compensar errores y reducir el riesgo global.

### Clasificador por Votación (Voting Classifier)
Combina **diferentes algoritmos** entrenados sobre el **mismo conjunto de datos**.

- **Votación dura (Hard voting):** La clase final se elige por mayoría de votos entre los clasificadores.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

lr = LogisticRegression(random_state=1)
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier(random_state=1)

# Ensamble heterogéneo
vc = VotingClassifier(estimators=[
    ('lr', lr),
    ('knn', knn),
    ('dt', dt)
])

vc.fit(X_train, y_train)
print(f"Accuracy Voting Classifier: {vc.score(X_test, y_test):.3f}")
```

---

## 7. Bagging (Bootstrap Aggregating)

Entrena **un único tipo de algoritmo base** sobre **múltiples subconjuntos diferentes** generados mediante **muestreo bootstrap** (muestreo aleatorio con reemplazo de igual tamaño que el conjunto original).

### Características clave
- **Efecto primordial:** Reduce significativamente la varianza de estimadores inestables (como árboles profundos) sin deteriorar el sesgo.
- **Predicción final:**
  - *Clasificación:* Votación por mayoría (`BaggingClassifier`).
  - *Regresión:* Promedio de predicciones individuales (`BaggingRegressor`).

### Evaluación Fuera de la Bolsa (Out-Of-Bag - OOB)
- En un muestreo bootstrap de tamaño $N$, en promedio el **63.2%** de las muestras únicas son seleccionadas.
- El **36.8% restante** nunca es visto por el estimador durante su entrenamiento y conforma las instancias **OOB**.
- **Ventaja práctica:** Las muestras OOB permiten evaluar el error de generalización del ensamble de forma directa y gratuita, sin necesidad de validación cruzada externa.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=1)

# Bagging con 300 árboles y estimación OOB habilitada
bc = BaggingClassifier(
    estimator=dt, n_estimators=300, oob_score=True, n_jobs=-1, random_state=1
)

bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"OOB Accuracy:  {bc.oob_score_:.3f}")
```

---

## 8. Random Forests (Bosques Aleatorios)

Random Forests eleva el concepto de Bagging aplicando una **doble aleatorización** para maximizar la diversidad entre los árboles y minimizar su correlación mutua:

1. **Muestreo de instancias:** Cada árbol se entrena sobre una muestra bootstrap diferente.
2. **Muestreo de características en cada división:** En cada nodo, el árbol solo puede elegir la mejor división a partir de un subconjunto aleatorio de $d$ características ($d < D$, sin reemplazo).

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE

# Instanciación de Random Forest Regressor
rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
rmse = MSE(y_test, y_pred) ** 0.5
print(f"RMSE Random Forest: {rmse:.2f}")
```

---

## 9. Importancia de Características (`feature_importances_`)

En los métodos basados en árboles, la importancia de una característica cuantifica la reducción promedio de impureza (Gini o MSE) ponderada por la cantidad de muestras que atraviesan los nodos donde dicha característica fue seleccionada.

- Se normaliza para que la suma total sea igual a 1.0.
- Permite diagnosticar la relevancia de cada variable y descartar ruido.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Extracción y ordenamiento de importancia de variables
importances_rf = pd.Series(rf.feature_importances_, index=X.columns)
sorted_importances = importances_rf.sort_values()

# Gráfico de barras horizontales
sorted_importances.plot(kind='barh', color='lightgreen')
plt.title('Importancia de Características en Random Forest')
plt.xlabel('Importancia Relativa')
plt.show()
```

---

## 10. Cuadro Comparativo de Métodos

| Método | Estimador base | Muestreo de datos | Muestreo de variables | Objetivo principal | En Scikit-Learn |
|---|---|---|---|---|---|
| **CART individual** | Árbol único | Todos los datos | Todas las variables | Alta interpretabilidad, baseline rápido | `DecisionTreeClassifier` / `Regressor` |
| **Voting Classifier** | Heterogéneo (varios modelos) | Mismo dataset completo | Todas las variables | Reducir error combinando debilidades/fortalezas | `VotingClassifier` |
| **Bagging** | Homogéneo (ej. CART) | Muestras bootstrap ($N$ con reemplazo) | Todas las variables | Reducir varianza de estimadores inestables | `BaggingClassifier` / `Regressor` |
| **Random Forest** | Árboles CART descorrelacionados | Muestras bootstrap | Subconjunto aleatorio $d < D$ en cada división | Mínima varianza, alta robustez frente al sobreajuste | `RandomForestClassifier` / `Regressor` |
