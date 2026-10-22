"""Ítem 2: Árboles de regresión con varias configuraciones vs Regresión Lineal — dataset auto_mpg."""
from nb_common import SETUP, code, header, md

FILENAME = "Item_02_Arboles_Regresion_Hiperparametros_vs_RegresionLineal.ipynb"
TITLE = "Árboles de regresión con distintos hiperparámetros vs. Regresión Lineal (auto_mpg)"
QUESTION = ("Entrene modelos de árboles de decisión con varias configuraciones de hiper parámetros usando el "
            "dataset auto_mpg, comparar su eficiencia con respecto a un modelo lineal como el Regresor Lineal. "
            "¿Cuál es el mejor?")

A = {}


def cells():
    return [
        header(2, TITLE, QUESTION, FILENAME),
        md(r"""
        ## Objetivo y fundamento teórico

        Un **árbol de regresión CART** particiona el espacio en regiones rectangulares $R_m$ y predice en cada hoja el
        **promedio** de la variable objetivo. En cada nodo se elige el corte que más reduce el **MSE**:

        $$I_{MSE}(\text{nodo}) = \frac{1}{N}\sum_{i \in \text{nodo}} (y_i - \bar{y}_{\text{nodo}})^2$$

        La **Regresión Lineal** ajusta $\hat{y} = w^\top x + b$ minimizando la suma de errores al cuadrado.

        El error de generalización se descompone en $\text{Sesgo}^2 + \text{Varianza} + \text{Error irreducible}$:
        un árbol sin restricciones tiene **bajo sesgo y alta varianza** (sobreajuste), mientras que limitar
        `max_depth` o aumentar `min_samples_leaf` reduce la varianza a costa de algo de sesgo.

        **Objetivo:** predecir el consumo `mpg` (millas por galón) y comparar 5 configuraciones de árboles
        (incluida una optimizada con `GridSearchCV`) frente a la Regresión Lineal, usando RMSE, MAE y $R^2$.
        """),
        code(SETUP),
        md("## 1. Carga y calidad de datos"),
        code("""
        df = load_csv("auto-mpg.csv")
        df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")   # '?' -> NaN si existiera
        print("Dimensiones:", df.shape)
        print("\\nNulos por columna:"); print(df.isna().sum()[df.isna().sum() > 0])
        print("\\nFilas duplicadas:", int(df.duplicated().sum()))
        print("Valores de 'origin':", df["origin"].value_counts().to_dict())
        display(df[df["horsepower"].isna()])
        df.describe().T.round(2)
        """),
        md(A.get("eda1", "_(análisis pendiente)_")),
        code("""
        fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
        pal = {"usa": "#4C72B0", "japan": "#C44E52", "europe": "#55A868"}
        sns.histplot(df["mpg"], kde=True, ax=axes[0, 0], color="#8172B3"); axes[0, 0].set_title("Distribución de mpg")
        for ax, feat in zip([axes[0, 1], axes[0, 2], axes[1, 0]], ["weight", "horsepower", "displacement"]):
            sns.scatterplot(data=df, x=feat, y="mpg", hue="origin", palette=pal, alpha=0.7, ax=ax)
            ax.set_title(f"mpg vs {feat}")
        sns.boxplot(data=df, x="cylinders", y="mpg", hue="cylinders", palette="viridis", legend=False, ax=axes[1, 1])
        axes[1, 1].set_title("mpg por número de cilindros")
        sns.lineplot(data=df, x="model_year", y="mpg", hue="origin", palette=pal, marker="o", ax=axes[1, 2])
        axes[1, 2].set_title("mpg promedio por año de modelo")
        plt.tight_layout(); savefig("item2_eda"); plt.show()

        corr = df.drop(columns=["name", "origin"]).corr(numeric_only=True)["mpg"].drop("mpg").sort_values()
        print("Correlación de Pearson con mpg:\\n", corr.round(3))
        """),
        md(A.get("eda2", "_(análisis pendiente)_")),
        md("## 2. Partición y preprocesamiento (sin fuga de información)"),
        code("""
        from sklearn.model_selection import train_test_split, KFold

        features_num = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
        features_cat = ["origin"]
        X = df[features_num + features_cat]
        y = df["mpg"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        print("Train:", X_train.shape, "| Test:", X_test.shape)
        print("Nulos de horsepower en train:", int(X_train['horsepower'].isna().sum()),
              "| en test:", int(X_test['horsepower'].isna().sum()))
        """),
        code("""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline

        def make_preprocessor(scale: bool):
            num_steps = [("imputer", SimpleImputer(strategy="median"))]
            if scale:
                num_steps.append(("scaler", StandardScaler()))
            return ColumnTransformer([
                ("num", Pipeline(num_steps), features_num),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features_cat),
            ])
        """),
        md("""
        * Se descarta `name` (identificador del vehículo con ~300 valores distintos: no generaliza y causaría sobreajuste).
        * Los 6 valores faltantes de `horsepower` se **imputan con la mediana del conjunto de entrenamiento** dentro del
          `Pipeline` (se conservan las filas en lugar de eliminarlas).
        * `origin` (nominal) se codifica con **One-Hot**. Solo la Regresión Lineal recibe `StandardScaler`
          (los árboles son invariantes a transformaciones monótonas de las variables).
        """),
        md("## 3. Modelos: Regresión Lineal y 5 configuraciones de árboles"),
        code("""
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import GridSearchCV

        def tree(**kw):
            return Pipeline([("prep", make_preprocessor(False)),
                             ("model", DecisionTreeRegressor(random_state=RANDOM_STATE, **kw))])

        models = {
            "Regresión Lineal": Pipeline([("prep", make_preprocessor(True)), ("model", LinearRegression())]),
            "CART-1 sin podar": tree(),
            "CART-2 max_depth=3": tree(max_depth=3),
            "CART-3 depth=5, leaf=5, split=10": tree(max_depth=5, min_samples_leaf=5, min_samples_split=10),
            "CART-4 depth=4, leaf=10%": tree(max_depth=4, min_samples_leaf=0.1),
        }

        kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        param_grid = {
            "model__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
            "model__min_samples_leaf": [1, 2, 4, 6, 8, 10, 15],
            "model__min_samples_split": [2, 10, 20],
            "model__criterion": ["squared_error", "friedman_mse", "absolute_error"],
        }
        grid = GridSearchCV(tree(), param_grid, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
        grid.fit(X_train, y_train)
        best_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
        print("Combinaciones evaluadas:", len(grid.cv_results_["params"]))
        print("Mejores hiperparámetros:", best_params, "| RMSE CV =", round(-grid.best_score_, 4))
        models["CART-5 GridSearchCV"] = grid.best_estimator_
        """),
        md(A.get("grid", "_(análisis pendiente)_")),
        md("## 4. Evaluación: RMSE / MAE / R² en entrenamiento, validación cruzada 10-Fold y prueba"),
        code("""
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        def rmse(a, b):
            return float(np.sqrt(mean_squared_error(a, b)))

        rows, cv_rmse, preds = [], {}, {}
        for name, model in models.items():
            cvres = cross_validate(model, X_train, y_train, cv=kf, n_jobs=-1,
                                   scoring=["neg_root_mean_squared_error", "r2"])
            model.fit(X_train, y_train)
            p_tr, p_te = model.predict(X_train), model.predict(X_test)
            preds[name] = p_te
            cv_rmse[name] = -cvres["test_neg_root_mean_squared_error"]
            est = model.named_steps["model"]
            rows.append({
                "Modelo": name,
                "RMSE train": rmse(y_train, p_tr),
                "RMSE CV (media)": cv_rmse[name].mean(),
                "RMSE CV (std)": cv_rmse[name].std(),
                "RMSE test": rmse(y_test, p_te),
                "MAE test": mean_absolute_error(y_test, p_te),
                "R² train": r2_score(y_train, p_tr),
                "R² CV": cvres["test_r2"].mean(),
                "R² test": r2_score(y_test, p_te),
                "Profundidad": est.get_depth() if hasattr(est, "get_depth") else np.nan,
                "Hojas": est.get_n_leaves() if hasattr(est, "get_n_leaves") else np.nan,
            })
        results = pd.DataFrame(rows).set_index("Modelo")
        results["CV/train (diagnóstico)"] = results["RMSE CV (media)"] / results["RMSE train"]
        results.round(3)
        """),
        md(A.get("res", "_(análisis pendiente)_")),
        code("""
        fig, axes = plt.subplots(1, 2, figsize=(19, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
        results[["RMSE train", "RMSE CV (media)", "RMSE test"]].plot(
            kind="bar", ax=axes[0], color=["#55A868", "#4C72B0", "#C44E52"], width=0.8)
        axes[0].set_ylabel("RMSE (mpg)"); axes[0].set_title("RMSE en entrenamiento, validación cruzada y prueba")
        axes[0].tick_params(axis="x", rotation=20)
        for c in axes[0].containers:
            axes[0].bar_label(c, fmt="%.2f", fontsize=7, padding=1)
        cv_df = pd.DataFrame(cv_rmse).melt(var_name="Modelo", value_name="RMSE")
        sns.boxplot(data=cv_df, y="Modelo", x="RMSE", hue="Modelo", palette="Set2", legend=False, ax=axes[1])
        axes[1].set_title("RMSE en los 10 folds de validación"); axes[1].set_ylabel("")
        plt.tight_layout(); savefig("item2_comparacion"); plt.show()
        """),
        md(A.get("bar", "_(análisis pendiente)_")),
        code("""
        show = ["Regresión Lineal", "CART-1 sin podar", "CART-5 GridSearchCV"]
        cols = ["#55A868", "#C44E52", "#4C72B0"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        lims = [y_test.min() - 2, y_test.max() + 2]
        for j, (name, col) in enumerate(zip(show, cols)):
            axes[0, j].scatter(y_test, preds[name], alpha=0.75, color=col, edgecolor="k", lw=0.3)
            axes[0, j].plot(lims, lims, "k--"); axes[0, j].set_xlim(lims); axes[0, j].set_ylim(lims)
            axes[0, j].set_title(f"{name}\\nRMSE={results.loc[name,'RMSE test']:.2f}  R²={results.loc[name,'R² test']:.3f}")
            axes[0, j].set_xlabel("mpg real"); axes[0, j].set_ylabel("mpg predicho")
            res = y_test - preds[name]
            axes[1, j].scatter(preds[name], res, alpha=0.75, color=col, edgecolor="k", lw=0.3)
            axes[1, j].axhline(0, color="k", ls="--"); axes[1, j].set_title(f"Residuos — {name}")
            axes[1, j].set_xlabel("mpg predicho"); axes[1, j].set_ylabel("residuo (real - predicho)")
        plt.tight_layout(); savefig("item2_real_vs_pred"); plt.show()
        """),
        md(A.get("pred", "_(análisis pendiente)_")),
        md("## 5. Curva de complejidad (diagnóstico sesgo-varianza)"),
        code("""
        depths = list(range(1, 16))
        tr, va = [], []
        for d in depths:
            s = cross_validate(tree(max_depth=d), X_train, y_train, cv=kf, n_jobs=-1,
                               scoring="neg_root_mean_squared_error", return_train_score=True)
            tr.append(-s["train_score"].mean()); va.append(-s["test_score"].mean())
        lr_cv = results.loc["Regresión Lineal", "RMSE CV (media)"]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(depths, tr, "--o", color="#55A868", label="Árbol - entrenamiento")
        ax.plot(depths, va, "-o", color="#4C72B0", label="Árbol - validación (CV 10-Fold)")
        ax.axhline(lr_cv, color="#C44E52", ls=":", lw=2, label=f"Regresión Lineal CV = {lr_cv:.2f}")
        ax.set_xlabel("max_depth"); ax.set_ylabel("RMSE (mpg)"); ax.set_xticks(depths)
        ax.set_title("Error de entrenamiento vs. validación según la profundidad del árbol"); ax.legend()
        plt.tight_layout(); savefig("item2_complejidad"); plt.show()
        print("Profundidad con menor RMSE CV:", depths[int(np.argmin(va))], "->", round(min(va), 3))
        """),
        md(A.get("cx", "_(análisis pendiente)_")),
        md("## 6. Naturaleza de las predicciones: función escalonada vs. recta (variable `weight`)"),
        code("""
        Xw_tr = X_train[["weight"]]; grid_w = pd.DataFrame({"weight": np.linspace(1200, 6000, 500)})
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.scatter(X_train["weight"], y_train, alpha=0.35, color="gray", label="train")
        for lbl, m, col in [("Regresión Lineal", LinearRegression(), "#C44E52"),
                            ("Árbol max_depth=3", DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE), "#4C72B0"),
                            ("Árbol sin podar", DecisionTreeRegressor(random_state=RANDOM_STATE), "#DD8452")]:
            m.fit(Xw_tr, y_train)
            ax.plot(grid_w["weight"], m.predict(grid_w), color=col, lw=2 if "podar" not in lbl else 1, label=lbl)
        ax.axvspan(X_train["weight"].max(), 6000, color="#FFE4B5", alpha=0.5, label="fuera del rango de train")
        ax.set_xlabel("weight (lb)"); ax.set_ylabel("mpg"); ax.legend()
        ax.set_title("Predicción con una sola variable: piecewise constant (árbol) vs. lineal")
        plt.tight_layout(); savefig("item2_escalonado"); plt.show()
        """),
        md(A.get("step", "_(análisis pendiente)_")),
        md("## 7. Interpretabilidad: árbol de profundidad 3 e importancia de variables"),
        code("""
        from sklearn.tree import plot_tree

        feat_names = features_num + [f"origin_{c}" for c in
                                     models["CART-2 max_depth=3"].named_steps["prep"]
                                     .named_transformers_["cat"].categories_[0]]
        fig, ax = plt.subplots(figsize=(24, 8))
        plot_tree(models["CART-2 max_depth=3"].named_steps["model"], feature_names=feat_names, filled=True,
                  rounded=True, fontsize=9, ax=ax, precision=2)
        ax.set_title("Árbol de regresión CART-2 (max_depth=3)", fontsize=14)
        savefig("item2_arbol"); plt.show()

        imp = pd.Series(models["CART-5 GridSearchCV"].named_steps["model"].feature_importances_, index=feat_names)
        coef = pd.Series(models["Regresión Lineal"].named_steps["model"].coef_, index=feat_names)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        imp.sort_values().plot(kind="barh", ax=axes[0], color="#4C72B0")
        axes[0].set_title("Importancia de variables — CART-5 (GridSearchCV)")
        coef.sort_values().plot(kind="barh", ax=axes[1],
                                color=["#C44E52" if v < 0 else "#55A868" for v in coef.sort_values()])
        axes[1].set_title("Coeficientes estandarizados — Regresión Lineal")
        plt.tight_layout(); savefig("item2_importancia"); plt.show()
        """),
        md(A.get("imp", "_(análisis pendiente)_")),
        md("## 8. Significancia estadística e intervalos de confianza"),
        code("""
        from scipy.stats import wilcoxon
        from sklearn.model_selection import RepeatedKFold, cross_val_score

        rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)
        rep = {n: -cross_val_score(models[n], X_train, y_train, cv=rkf, n_jobs=-1,
                                   scoring="neg_root_mean_squared_error") for n in models}
        stats_rows = []
        for n in models:
            if n == "Regresión Lineal":
                continue
            p = wilcoxon(rep["Regresión Lineal"], rep[n], zero_method="zsplit").pvalue
            stats_rows.append({"Comparación": f"Regresión Lineal vs {n}",
                               "RMSE medio LR": rep["Regresión Lineal"].mean(), "RMSE medio árbol": rep[n].mean(),
                               "Diferencia (árbol - LR)": (rep[n] - rep["Regresión Lineal"]).mean(),
                               "p-valor Wilcoxon": p, "Significativo (α=0.05)": "Sí" if p < 0.05 else "No"})
        stats_df = pd.DataFrame(stats_rows)
        ci_df = pd.DataFrame({n: bootstrap_ci(y_test, preds[n], rmse) for n in models},
                             index=["IC95% inf", "IC95% sup"]).T
        display(stats_df.round(4)); display(ci_df.round(3))
        """),
        md(A.get("stats", "_(análisis pendiente)_")),
        code("""
        save_results({
            "results": results.reset_index().to_dict(orient="records"),
            "best_params": best_params, "grid_best_rmse_cv": -grid.best_score_,
            "stats": stats_df.to_dict(orient="records"),
            "ci": ci_df.reset_index().rename(columns={"index": "Modelo"}).to_dict(orient="records"),
            "complexity": {"depths": depths, "train": tr, "cv": va},
        }, "item2")
        print("Resultados guardados en results/item2.json")
        """),
        md(A.get("summary", "## Conclusiones\n_(pendiente)_")),
    ]
