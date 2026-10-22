"""Ítem 3: Ensamble VotingClassifier vs modelos individuales — dataset Pima Indians Diabetes."""
from nb_common import SETUP, code, header, md

FILENAME = "Item_03_Ensemble_VotingClassifier_vs_Individuales.ipynb"
TITLE = "Ensamble clasificador (VotingClassifier) vs. modelos individuales"
QUESTION = ("Entrene un modelo ensemble clasificador, y compare su eficiencia con respecto a un modelo "
            "individual. ¿Cuál es mejor?")

A = {}

DIABETES_PREP = """
ZERO_AS_NAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df = load_csv("diabetes.csv")
print("Dimensiones:", df.shape, "| duplicados:", int(df.duplicated().sum()))
print("Ceros fisiológicamente imposibles por columna:")
print((df[ZERO_AS_NAN] == 0).sum())
df[ZERO_AS_NAN] = df[ZERO_AS_NAN].replace(0, np.nan)      # un 0 en glucosa/IMC/presión = dato faltante
X, y = df.drop(columns="Outcome"), df["Outcome"]
print("\\nDistribución de clases (1 = diabetes):", y.value_counts().to_dict(),
      f"| prevalencia = {y.mean():.3f}")
"""


def cells():
    return [
        header(3, TITLE, QUESTION, FILENAME),
        md(r"""
        ## Objetivo y fundamento teórico

        El **aprendizaje por ensambles** combina varios modelos base para compensar sus errores individuales.
        El **clasificador por votación** (`VotingClassifier`) entrena **algoritmos distintos sobre el mismo conjunto de datos**:

        * **Votación dura (hard):** $\hat{y} = \operatorname{moda}\{h_1(x), h_2(x), \dots, h_M(x)\}$
        * **Votación suave (soft):** $\hat{y} = \arg\max_c \frac{1}{M}\sum_{m=1}^{M} P_m(y=c\mid x)$

        El ensamble mejora cuando los modelos base son **precisos y diversos** (cometen errores poco correlacionados).
        Se combinan tres sesgos inductivos distintos: **Regresión Logística** (frontera lineal), **K-Vecinos**
        (basado en distancias, no paramétrico) y **Árbol de decisión CART** (cortes ortogonales).

        **Dataset:** *Pima Indians Diabetes* (768 pacientes, 8 variables clínicas, objetivo: diabetes sí/no). Se eligió
        por ser un problema realista y difícil (fronteras solapadas), donde las diferencias entre modelos son visibles.
        """),
        code(SETUP),
        md("## 1. Carga, calidad de datos y tratamiento de faltantes"),
        code(DIABETES_PREP),
        md(A.get("eda1", "_(análisis pendiente)_")),
        code("""
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
        (df.isna().mean() * 100).sort_values().plot(kind="barh", ax=axes[0], color="#8172B3")
        axes[0].set_title("% de valores faltantes (ceros → NaN)"); axes[0].set_xlabel("%")
        lab = y.map({0: "No diabetes", 1: "Diabetes"}); pal = {"No diabetes": "#4C72B0", "Diabetes": "#C44E52"}
        sns.countplot(x=lab, hue=lab, palette=pal, legend=False, ax=axes[1]); axes[1].set_title("Distribución de clases")
        axes[1].set_xlabel("")
        for ax, feat in zip(axes[2:], ["Glucose", "BMI"]):
            sns.boxplot(x=lab, y=df[feat], hue=lab, palette=pal, legend=False, ax=ax); ax.set_xlabel("")
            ax.set_title(f"{feat} por clase")
        plt.tight_layout(); savefig("item3_eda"); plt.show()
        """),
        md(A.get("eda2", "_(análisis pendiente)_")),
        md("## 2. Partición estratificada 80/20"),
        code("""
        from sklearn.model_selection import train_test_split, StratifiedKFold

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                            random_state=RANDOM_STATE)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        print("Train:", X_train.shape, "| Test:", X_test.shape)
        """),
        md("## 3. Modelos individuales (ajuste de hiperparámetros con validación cruzada)"),
        code("""
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV

        def imputer():
            # mediana ajustada solo con train + indicadores binarios de "dato faltante"
            return SimpleImputer(strategy="median", add_indicator=True)

        base = {
            "Regresión Logística": (Pipeline([("imp", imputer()), ("sc", StandardScaler()),
                                              ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
                                    {"clf__C": [0.01, 0.1, 1, 10]}),
            "KNN": (Pipeline([("imp", imputer()), ("sc", StandardScaler()), ("clf", KNeighborsClassifier())]),
                    {"clf__n_neighbors": [5, 11, 15, 21, 27, 35], "clf__weights": ["uniform", "distance"]}),
            "Árbol de decisión": (Pipeline([("imp", imputer()),
                                            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
                                  {"clf__max_depth": [2, 3, 4, 5, 6, 8],
                                   "clf__min_samples_leaf": [1, 5, 10, 20, 0.05, 0.13]}),
        }
        tuned = {}
        for name, (pipe, grid) in base.items():
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1).fit(X_train, y_train)
            tuned[name] = gs.best_estimator_
            print(f"{name:20s} mejores params: {gs.best_params_} | Acc CV = {gs.best_score_:.4f}")
        """),
        md(A.get("tune", "_(análisis pendiente)_")),
        md("## 4. Ensambles por votación (hard y soft)"),
        code("""
        from sklearn.ensemble import VotingClassifier
        from sklearn.base import clone

        estimators = [("lr", clone(tuned["Regresión Logística"])), ("knn", clone(tuned["KNN"])),
                      ("dt", clone(tuned["Árbol de decisión"]))]
        models = dict(tuned)
        models["Voting (hard)"] = VotingClassifier(estimators=estimators, voting="hard")
        models["Voting (soft)"] = VotingClassifier(estimators=[(n, clone(e)) for n, e in estimators], voting="soft")
        list(models)
        """),
        md("## 5. Evaluación: validación cruzada 10-Fold y conjunto de prueba"),
        code("""
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                     roc_auc_score, balanced_accuracy_score)

        rows, cv_acc = [], {}
        for name, model in models.items():
            has_proba = hasattr(model, "predict_proba")
            scoring = ["accuracy", "f1", "balanced_accuracy"] + (["roc_auc"] if has_proba else [])
            cvres = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            cv_acc[name] = cvres["test_accuracy"]
            rows.append({
                "Modelo": name,
                "Acc. train": accuracy_score(y_train, model.predict(X_train)),
                "Acc. CV (media)": cvres["test_accuracy"].mean(),
                "Acc. CV (std)": cvres["test_accuracy"].std(),
                "F1 CV": cvres["test_f1"].mean(),
                "ROC-AUC CV": cvres["test_roc_auc"].mean() if has_proba else np.nan,
                "Acc. test": accuracy_score(y_test, y_pred),
                "Acc. balanceada test": balanced_accuracy_score(y_test, y_pred),
                "Precisión test": precision_score(y_test, y_pred),
                "Recall test": recall_score(y_test, y_pred),
                "F1 test": f1_score(y_test, y_pred),
                "ROC-AUC test": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
            })
        results = pd.DataFrame(rows).set_index("Modelo")
        results.round(4)
        """),
        md(A.get("res", "_(análisis pendiente)_")),
        code("""
        fig, axes = plt.subplots(1, 2, figsize=(18, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
        colors = ["#4C72B0", "#55A868", "#DD8452", "#8172B3", "#C44E52"]
        results[["Acc. CV (media)", "Acc. test", "F1 test", "ROC-AUC test"]].T.plot(
            kind="bar", ax=axes[0], color=colors, width=0.85)
        axes[0].set_ylim(0.5, 0.9); axes[0].set_title("Métricas por modelo"); axes[0].tick_params(axis="x", rotation=0)
        for c in axes[0].containers:
            axes[0].bar_label(c, fmt="%.3f", fontsize=6.5, padding=1, rotation=90)
        axes[0].legend(loc="upper left", fontsize=8, ncol=2)
        cv_df = pd.DataFrame(cv_acc).melt(var_name="Modelo", value_name="Accuracy")
        sns.boxplot(data=cv_df, y="Modelo", x="Accuracy", hue="Modelo", palette=colors, legend=False, ax=axes[1])
        axes[1].set_title("Accuracy en los 10 folds de validación"); axes[1].set_ylabel("")
        plt.tight_layout(); savefig("item3_comparacion"); plt.show()
        """),
        md(A.get("bar", "_(análisis pendiente)_")),
        code("""
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, axes = plt.subplots(1, 5, figsize=(23, 4.3))
        for ax, (name, model) in zip(axes, models.items()):
            ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test), display_labels=["No diab.", "Diab."],
                                                    cmap="Blues", colorbar=False, ax=ax)
            ax.set_title(name, fontsize=10)
        plt.tight_layout(); savefig("item3_confusion"); plt.show()

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        for (name, model), col in zip(models.items(), colors):
            if hasattr(model, "predict_proba"):
                plot_roc(ax, model, X_test, y_test, name, color=col,
                                               lw=3 if "soft" in name else 1.5)
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title("Curvas ROC (prueba)")
        plt.tight_layout(); savefig("item3_roc"); plt.show()
        """),
        md(A.get("cm", "_(análisis pendiente)_")),
        md("## 6. ¿Por qué funciona el ensamble? Diversidad de errores y fronteras de decisión"),
        code("""
        errs = pd.DataFrame({n: (tuned[n].predict(X_test) != y_test).astype(int).values for n in tuned})
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(errs.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
        axes[0].set_title("Correlación de errores entre modelos base (prueba)")
        n_err = errs.sum(axis=1).value_counts().sort_index()
        n_err.plot(kind="bar", ax=axes[1], color="#8172B3")
        axes[1].set_xlabel("Nº de modelos base que se equivocan en la muestra"); axes[1].set_ylabel("Nº de muestras")
        axes[1].set_title("Distribución de errores simultáneos"); axes[1].tick_params(axis="x", rotation=0)
        for c in axes[1].containers:
            axes[1].bar_label(c)
        plt.tight_layout(); savefig("item3_diversidad"); plt.show()
        print("Muestras en que exactamente 1 modelo base falla (la votación mayoritaria puede corregirlas):",
              int((errs.sum(axis=1) == 1).sum()))
        """),
        md(A.get("div", "_(análisis pendiente)_")),
        code("""
        from sklearn.inspection import DecisionBoundaryDisplay

        f2 = ["Glucose", "BMI"]
        X2_tr = SimpleImputer(strategy="median").fit(X_train[f2])
        X2_train = pd.DataFrame(X2_tr.transform(X_train[f2]), columns=f2)
        fig, axes = plt.subplots(1, 5, figsize=(25, 4.8))
        for ax, (name, model) in zip(axes, models.items()):
            m2 = clone(model).fit(X2_train, y_train)
            DecisionBoundaryDisplay.from_estimator(m2, X2_train, response_method="predict", cmap="RdBu_r",
                                                   alpha=0.35, ax=ax, grid_resolution=250)
            ax.scatter(X2_train["Glucose"], X2_train["BMI"], c=y_train, cmap="RdBu_r", s=10, edgecolor="k", lw=0.2)
            ax.set_title(f"{name}\\n(acc train 2D = {m2.score(X2_train, y_train):.3f})", fontsize=10)
        plt.tight_layout(); savefig("item3_fronteras"); plt.show()
        """),
        md(A.get("front", "_(análisis pendiente)_")),
        md("## 7. Significancia estadística (CV repetida 10×3 + Wilcoxon) e IC bootstrap"),
        code("""
        from scipy.stats import wilcoxon
        from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

        rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)
        rep = {n: cross_val_score(models[n], X_train, y_train, cv=rcv, scoring="accuracy", n_jobs=-1) for n in models}
        stats_rows = []
        for ens in ["Voting (soft)", "Voting (hard)"]:
            for ind in tuned:
                p = wilcoxon(rep[ens], rep[ind], zero_method="zsplit").pvalue
                stats_rows.append({"Comparación": f"{ens} vs {ind}", "Acc. ensamble": rep[ens].mean(),
                                   "Acc. individual": rep[ind].mean(), "Diferencia": (rep[ens] - rep[ind]).mean(),
                                   "p-valor Wilcoxon": p, "Significativo (α=0.05)": "Sí" if p < 0.05 else "No"})
        stats_df = pd.DataFrame(stats_rows)
        ci_df = pd.DataFrame({n: bootstrap_ci(y_test, models[n].predict(X_test), accuracy_score) for n in models},
                             index=["IC95% inf", "IC95% sup"]).T
        rep_df = pd.DataFrame({n: [v.mean(), v.std()] for n, v in rep.items()}, index=["Acc. media 30 folds", "std"]).T
        display(rep_df.round(4)); display(stats_df.round(4)); display(ci_df.round(4))
        """),
        md(A.get("stats", "_(análisis pendiente)_")),
        code("""
        save_results({
            "results": results.reset_index().to_dict(orient="records"),
            "best_params": {n: {k: (v if not isinstance(v, float) else float(v)) for k, v in m.get_params().items()
                                if k.startswith("clf__") and k.split("__")[1] in
                                ("C", "n_neighbors", "weights", "max_depth", "min_samples_leaf")}
                            for n, m in tuned.items()},
            "repeated_cv": rep_df.reset_index().rename(columns={"index": "Modelo"}).to_dict(orient="records"),
            "stats": stats_df.to_dict(orient="records"),
            "ci": ci_df.reset_index().rename(columns={"index": "Modelo"}).to_dict(orient="records"),
            "error_corr": errs.corr().round(3).to_dict(),
        }, "item3")
        print("Resultados guardados en results/item3.json")
        """),
        md(A.get("summary", "## Conclusiones\n_(pendiente)_")),
    ]
