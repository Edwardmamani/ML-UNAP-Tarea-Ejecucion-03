"""Ítem 1: CART (Gini vs Entropía) vs Regresión Logística — dataset breast_cancer."""
from nb_common import SETUP, code, header, md

FILENAME = "Item_01_Arboles_Gini_Entropia_vs_RegresionLogistica.ipynb"
TITLE = "Árboles de decisión (Gini vs. Entropía) vs. Regresión Logística"
QUESTION = ("Entrene modelos de árboles de decisión con criterios de impureza diferentes (índice gini y "
            "entropía) usando el dataset “breast_cancer” de scikit-learn, compare su eficiencia con respecto "
            "a un modelo lineal como el Regresor Logístico. ¿Cuál es mejor?")

A = {}  # textos de análisis (se completan tras la ejecución)


def cells():
    return [
        header(1, TITLE, QUESTION, FILENAME),
        md(r"""
        ## Objetivo y fundamento teórico

        Un árbol **CART** de clasificación divide recursivamente el espacio de variables eligiendo, en cada nodo,
        la característica $f$ y el punto de corte $sp$ que maximizan la **ganancia de información**:

        $$IG(\text{nodo}) = I(\text{nodo}) - \left(\frac{N_{izq}}{N} I(\text{nodo}_{izq}) + \frac{N_{der}}{N} I(\text{nodo}_{der})\right)$$

        donde la impureza $I$ puede medirse con:

        | Criterio | Fórmula | Interpretación |
        |---|---|---|
        | **Índice Gini** | $I_G = 1 - \sum_{i=1}^{C} p_i^2$ | Probabilidad de clasificar mal un elemento al azar (cálculo más rápido). |
        | **Entropía** | $H = -\sum_{i=1}^{C} p_i \log_2 p_i$ | Desorden / incertidumbre de la información. |

        La **Regresión Logística** es un modelo lineal que estima $P(y=1\mid x) = \sigma(w^\top x + b)$; requiere
        estandarizar las variables (se ajusta el `StandardScaler` **solo** con datos de entrenamiento dentro de un `Pipeline`).

        **Plan:** (1) exploración de datos, (2) partición estratificada 80/20, (3) entrenamiento de árboles Gini y Entropía
        (sin podar y podados) y Regresión Logística, (4) validación cruzada estratificada 10-Fold, (5) métricas en prueba,
        matrices de confusión, curvas ROC, curvas de complejidad e importancia de variables, (6) prueba estadística y conclusión.
        """),
        code(SETUP),
        md("## 1. Carga y exploración del dataset"),
        code("""
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target          # 0 = maligno, 1 = benigno
        df = data.frame

        print("Dimensiones de X:", X.shape)
        print("Valores nulos totales:", int(df.isna().sum().sum()))
        print("Filas duplicadas:", int(df.duplicated().sum()))
        print("\\nDistribución de clases:")
        print(y.value_counts().rename(index={0: "maligno (0)", 1: "benigno (1)"}))
        df.describe().T[["mean", "std", "min", "max"]].head(10).round(3)
        """),
        md(A.get("eda1", "_(análisis pendiente)_")),
        code("""
        labels = y.map({0: "Maligno", 1: "Benigno"})
        pal = {"Maligno": "#C44E52", "Benigno": "#4C72B0"}
        fig, axes = plt.subplots(1, 4, figsize=(19, 4))
        sns.countplot(x=labels, hue=labels, palette=pal, legend=False, ax=axes[0])
        axes[0].set_title("Distribución de clases"); axes[0].set_xlabel("")
        for ax, feat in zip(axes[1:], ["mean radius", "mean concave points", "worst area"]):
            sns.kdeplot(data=df.assign(clase=labels), x=feat, hue="clase", fill=True,
                        common_norm=False, palette=pal, ax=ax)
            ax.set_title(f"Densidad de '{feat}' por clase")
        plt.tight_layout(); savefig("item1_eda"); plt.show()
        """),
        md(A.get("eda2", "_(análisis pendiente)_")),
        md("## 2. Partición estratificada entrenamiento / prueba (80/20)"),
        code("""
        from sklearn.model_selection import train_test_split, StratifiedKFold

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        print("Train:", X_train.shape, "| Test:", X_test.shape)
        print("Proporción de benignos  train = %.3f | test = %.3f" % (y_train.mean(), y_test.mean()))
        """),
        md("""
        La estratificación conserva la proporción de clases (≈63 % benignos) en ambos subconjuntos. El conjunto de
        prueba **no se utiliza** para ajustar ningún hiperparámetro: solo se usa para la evaluación final.
        """),
        md("## 3. Definición de modelos"),
        code("""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        models = {
            "CART Gini (sin podar)": DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE),
            "CART Entropía (sin podar)": DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE),
            "CART Gini (podado)": DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=5,
                                                         random_state=RANDOM_STATE),
            "CART Entropía (podado)": DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=5,
                                                             random_state=RANDOM_STATE),
            "Regresión Logística": make_pipeline(StandardScaler(),
                                                 LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        }
        for k, v in models.items():
            print(f"{k:28s} -> {v}")
        """),
        md("""
        Se entrenan **cinco** modelos: dos árboles completamente desarrollados (para observar la alta varianza de CART),
        dos árboles **podados** (`max_depth=4`, `min_samples_leaf=5`) para controlar el sobreajuste, y la Regresión
        Logística estandarizada como modelo lineal de referencia.
        """),
        md("## 4. Entrenamiento, validación cruzada 10-Fold y evaluación en prueba"),
        code("""
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                     roc_auc_score, make_scorer, confusion_matrix)

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc",
                   "f1_maligno": make_scorer(f1_score, pos_label=0)}

        rows, cv_acc = [], {}
        for name, model in models.items():
            cvres = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            t0 = time.perf_counter(); model.fit(X_train, y_train); fit_t = time.perf_counter() - t0
            y_pred = model.predict(X_test); y_proba = model.predict_proba(X_test)[:, 1]
            # filas = real, columnas = predicho; orden de etiquetas [maligno(0), benigno(1)]
            mal_ok, mal_como_ben, ben_como_mal, ben_ok = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            cv_acc[name] = cvres["test_accuracy"]
            est = model if hasattr(model, "get_depth") else None
            rows.append({
                "Modelo": name,
                "Acc. train": accuracy_score(y_train, model.predict(X_train)),
                "Acc. CV (media)": cvres["test_accuracy"].mean(),
                "Acc. CV (std)": cvres["test_accuracy"].std(),
                "ROC-AUC CV": cvres["test_roc_auc"].mean(),
                "Acc. test": accuracy_score(y_test, y_pred),
                "Precisión maligno": precision_score(y_test, y_pred, pos_label=0),
                "Recall maligno": recall_score(y_test, y_pred, pos_label=0),
                "F1 maligno": f1_score(y_test, y_pred, pos_label=0),
                "ROC-AUC test": roc_auc_score(y_test, y_proba),
                "Malignos no detectados (FN)": int(mal_como_ben),
                "Profundidad": est.get_depth() if est else np.nan,
                "Hojas": est.get_n_leaves() if est else np.nan,
                "Tiempo fit (s)": fit_t,
            })
        results = pd.DataFrame(rows).set_index("Modelo")
        results.round(4)
        """),
        md(A.get("res", "_(análisis pendiente)_")),
        code("""
        fig, axes = plt.subplots(1, 2, figsize=(18, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
        plot_df = results[["Acc. train", "Acc. CV (media)", "Acc. test"]]
        plot_df.plot(kind="bar", ax=axes[0], color=["#55A868", "#4C72B0", "#C44E52"], width=0.8)
        axes[0].set_ylim(0.85, 1.02); axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy en entrenamiento, validación cruzada y prueba")
        axes[0].tick_params(axis="x", rotation=20)
        for c in axes[0].containers:
            axes[0].bar_label(c, fmt="%.3f", fontsize=7, padding=1)
        axes[0].legend(loc="lower right")
        cv_df = pd.DataFrame(cv_acc).melt(var_name="Modelo", value_name="Accuracy")
        sns.boxplot(data=cv_df, y="Modelo", x="Accuracy", hue="Modelo", palette="Set2", legend=False, ax=axes[1])
        axes[1].set_title("Distribución de Accuracy en los 10 folds"); axes[1].set_ylabel("")
        plt.tight_layout(); savefig("item1_comparacion"); plt.show()
        """),
        md(A.get("bar", "_(análisis pendiente)_")),
        code("""
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, axes = plt.subplots(1, 5, figsize=(23, 4.3))
        for ax, (name, model) in zip(axes, models.items()):
            ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test), labels=[0, 1],
                                                    display_labels=["Maligno", "Benigno"], cmap="Blues",
                                                    colorbar=False, ax=ax)
            ax.set_title(name, fontsize=10)
        plt.tight_layout(); savefig("item1_confusion"); plt.show()
        """),
        md(A.get("cm", "_(análisis pendiente)_")),
        code("""
        # curvas ROC con el helper plot_roc (definido en la celda de configuración)

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        for (name, model), col in zip(models.items(), sns.color_palette("tab10")):
            plot_roc(ax, model, X_test, y_test, name, color=col)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title("Curvas ROC en el conjunto de prueba")
        plt.tight_layout(); savefig("item1_roc"); plt.show()
        """),
        md(A.get("roc", "_(análisis pendiente)_")),
        md("## 5. Curva de complejidad: Gini vs. Entropía según `max_depth`"),
        code("""
        depths = list(range(1, 13))
        curves = {}
        for crit in ["gini", "entropy"]:
            tr, va = [], []
            for d in depths:
                m = DecisionTreeClassifier(criterion=crit, max_depth=d, random_state=RANDOM_STATE)
                s = cross_validate(m, X_train, y_train, cv=cv, scoring="accuracy",
                                   return_train_score=True, n_jobs=-1)
                tr.append(s["train_score"].mean()); va.append(s["test_score"].mean())
            curves[crit] = (tr, va)

        lr_cv = results.loc["Regresión Logística", "Acc. CV (media)"]
        fig, ax = plt.subplots(figsize=(10, 5))
        for crit, col in [("gini", "#4C72B0"), ("entropy", "#DD8452")]:
            ax.plot(depths, curves[crit][0], "--", color=col, alpha=0.7, label=f"{crit} - entrenamiento")
            ax.plot(depths, curves[crit][1], "-o", color=col, label=f"{crit} - validación (CV)")
        ax.axhline(lr_cv, color="#55A868", ls=":", lw=2, label=f"Regresión Logística CV = {lr_cv:.3f}")
        ax.set_xlabel("max_depth"); ax.set_ylabel("Accuracy"); ax.set_xticks(depths)
        ax.set_title("Sesgo-varianza: Accuracy vs. profundidad del árbol")
        ax.legend(); plt.tight_layout(); savefig("item1_complejidad"); plt.show()

        best_depth = {c: depths[int(np.argmax(curves[c][1]))] for c in curves}
        print("Mejor max_depth por CV:", best_depth,
              "| Acc CV máx:", {c: round(max(curves[c][1]), 4) for c in curves})
        """),
        md(A.get("cx", "_(análisis pendiente)_")),
        md("## 6. Interpretabilidad: estructura del árbol e importancia de variables"),
        code("""
        from sklearn.tree import plot_tree

        fig, ax = plt.subplots(figsize=(24, 9))
        plot_tree(models["CART Gini (podado)"], feature_names=list(X.columns), class_names=["Maligno", "Benigno"],
                  filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax, impurity=True, proportion=False)
        ax.set_title("Árbol CART con criterio Gini (podado, primeros 3 niveles)", fontsize=14)
        savefig("item1_arbol"); plt.show()
        """),
        code("""
        imp = pd.DataFrame({
            "Gini (podado)": models["CART Gini (podado)"].feature_importances_,
            "Entropía (podado)": models["CART Entropía (podado)"].feature_importances_,
        }, index=X.columns)
        coef = pd.Series(np.abs(models["Regresión Logística"][-1].coef_[0]), index=X.columns)

        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        top = imp.max(axis=1).sort_values(ascending=False).head(10).index
        imp.loc[top].iloc[::-1].plot(kind="barh", ax=axes[0], color=["#4C72B0", "#DD8452"])
        axes[0].set_title("Importancia (reducción de impureza) — top 10")
        coef.sort_values().tail(10).plot(kind="barh", ax=axes[1], color="#55A868")
        axes[1].set_title("|Coeficientes| estandarizados de la Regresión Logística — top 10")
        plt.tight_layout(); savefig("item1_importancia"); plt.show()
        print("Variables con importancia > 0  Gini:", int((imp['Gini (podado)'] > 0).sum()),
              "| Entropía:", int((imp['Entropía (podado)'] > 0).sum()))
        """),
        md(A.get("imp", "_(análisis pendiente)_")),
        md("## 7. Significancia estadística (CV repetida 10×3 + prueba de Wilcoxon) e IC bootstrap"),
        code("""
        from scipy.stats import wilcoxon
        from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

        rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)
        names = ["CART Gini (podado)", "CART Entropía (podado)", "CART Gini (sin podar)", "Regresión Logística"]
        rep = {n: cross_val_score(models[n], X_train, y_train, cv=rcv, scoring="accuracy", n_jobs=-1) for n in names}
        pairs = [("Regresión Logística", "CART Gini (podado)"), ("Regresión Logística", "CART Entropía (podado)"),
                 ("CART Gini (podado)", "CART Entropía (podado)"), ("CART Gini (podado)", "CART Gini (sin podar)")]
        stats_rows = []
        for a, b in pairs:
            p = wilcoxon(rep[a], rep[b], zero_method="zsplit").pvalue
            stats_rows.append({"Comparación (A vs B)": f"{a} vs {b}", "Acc. media A": rep[a].mean(),
                               "Acc. media B": rep[b].mean(), "Diferencia A-B": (rep[a] - rep[b]).mean(),
                               "p-valor Wilcoxon": p, "Significativo (α=0.05)": "Sí" if p < 0.05 else "No"})
        stats_df = pd.DataFrame(stats_rows)

        ci = {n: bootstrap_ci(y_test, models[n].predict(X_test), accuracy_score) for n in models}
        ci_df = pd.DataFrame(ci, index=["IC95% inf", "IC95% sup"]).T
        display(stats_df.round(4)); display(ci_df.round(4))
        """),
        md(A.get("stats", "_(análisis pendiente)_")),
        code("""
        save_results({
            "results": results.reset_index().to_dict(orient="records"),
            "stats": stats_df.to_dict(orient="records"),
            "ci": ci_df.reset_index().rename(columns={"index": "Modelo"}).to_dict(orient="records"),
            "best_depth": best_depth,
            "curves": {c: {"train": curves[c][0], "cv": curves[c][1]} for c in curves},
        }, "item1")
        print("Resultados guardados en results/item1.json")
        """),
        md(A.get("summary", "## Conclusiones\n_(pendiente)_")),
    ]
