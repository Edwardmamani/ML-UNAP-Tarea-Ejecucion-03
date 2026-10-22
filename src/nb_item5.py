"""Ítem 5: Random Forest vs XGBoost — dataset Pima Indians Diabetes (+ verificación en breast_cancer)."""
from nb_common import SETUP, code, header, md
from nb_item3 import DIABETES_PREP

FILENAME = "Item_05_RandomForest_vs_XGBoost.ipynb"
TITLE = "Random Forest vs. XGBoost"
QUESTION = ("Entrene un modelo RandomForest y compare su eficiencia con respecto a un modelo XGBoost. "
            "¿Cuál es mejor?")

A = {}


def cells():
    return [
        header(5, TITLE, QUESTION, FILENAME),
        md(r"""
        ## Objetivo y fundamento teórico

        | | **Random Forest** | **XGBoost** |
        |---|---|---|
        | Familia | Bagging (árboles en **paralelo**) | Boosting (árboles **secuenciales**) |
        | Idea | Promediar muchos árboles profundos y descorrelacionados | Cada árbol corrige los residuos (gradiente) del anterior |
        | Aleatorización | Bootstrap de filas + subconjunto aleatorio de $d<D$ variables en cada división | `subsample` de filas y `colsample_bytree` de columnas |
        | Reduce principalmente | **Varianza** | **Sesgo** (y varianza vía regularización) |
        | Regularización | `max_depth`, `min_samples_leaf`, `max_features` | $L_1$ (`reg_alpha`), $L_2$ (`reg_lambda`), `gamma`, `learning_rate`, `max_depth` |
        | Faltantes | Requiere imputación | Manejo **nativo** (aprende la dirección por defecto de los NaN) |

        XGBoost minimiza un objetivo regularizado con una aproximación de Taylor de **segundo orden**:
        $$\mathcal{L}^{(t)} = \sum_i \left[g_i f_t(x_i) + \tfrac{1}{2} h_i f_t^2(x_i)\right] + \gamma T + \tfrac{1}{2}\lambda \sum_j w_j^2$$

        **Dataset principal:** *Pima Indians Diabetes* (problema clínico difícil). Se comparan ambos modelos con
        configuración base y **optimizados con `RandomizedSearchCV`**, midiendo eficacia (Accuracy, F1, ROC-AUC,
        Log-Loss) y **eficiencia computacional** (tiempo de entrenamiento e inferencia). Se verifica además en *breast_cancer*.
        """),
        code(SETUP + """
try:
    import xgboost
except ImportError:                       # Colab ya incluye xgboost; por si acaso:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    import xgboost
from xgboost import XGBClassifier
print("xgboost", xgboost.__version__)
"""),
        md("## 1. Datos (mismo tratamiento de faltantes que el Ítem 3) y partición 80/20"),
        code(DIABETES_PREP + """
from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
print("Train:", X_train.shape, "| Test:", X_test.shape)
"""),
        md("""
        Los ceros imposibles se convierten en `NaN`. **Random Forest** recibe una imputación por mediana (ajustada en
        train) con indicadores de faltante; **XGBoost** recibe los `NaN` directamente porque los maneja de forma nativa.
        """),
        md("## 2. Configuraciones base y búsqueda aleatoria de hiperparámetros"),
        code("""
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.dummy import DummyClassifier
        from sklearn.model_selection import RandomizedSearchCV

        def rf_pipe(**kw):
            return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                             ("clf", RandomForestClassifier(random_state=RANDOM_STATE, **kw))])

        rf_base = rf_pipe(n_estimators=300, max_features="sqrt", n_jobs=-1)
        xgb_base = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)

        cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        rf_space = {"clf__n_estimators": [200, 300, 500], "clf__max_depth": [None, 4, 6, 8, 12],
                    "clf__min_samples_leaf": [1, 2, 4, 8], "clf__max_features": ["sqrt", "log2", 0.5]}
        xgb_space = {"n_estimators": [100, 200, 300, 500], "learning_rate": [0.01, 0.03, 0.05, 0.1],
                     "max_depth": [2, 3, 4, 5, 6], "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
                     "min_child_weight": [1, 3, 5], "reg_lambda": [0.5, 1, 2, 5], "gamma": [0, 0.1, 0.5]}

        t0 = time.perf_counter()
        rf_search = RandomizedSearchCV(rf_pipe(n_jobs=1), rf_space, n_iter=40, cv=cv5, scoring="roc_auc",
                                       n_jobs=-1, random_state=RANDOM_STATE).fit(X_train, y_train)
        rf_search_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        xgb_search = RandomizedSearchCV(XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1),
                                        xgb_space, n_iter=40, cv=cv5, scoring="roc_auc", n_jobs=-1,
                                        random_state=RANDOM_STATE).fit(X_train, y_train)
        xgb_search_t = time.perf_counter() - t0
        rf_best = {k.replace("clf__", ""): v for k, v in rf_search.best_params_.items()}
        xgb_best = xgb_search.best_params_
        print(f"RF  mejores params: {rf_best} | AUC CV5 = {rf_search.best_score_:.4f} | búsqueda {rf_search_t:.1f}s")
        print(f"XGB mejores params: {xgb_best} | AUC CV5 = {xgb_search.best_score_:.4f} | búsqueda {xgb_search_t:.1f}s")

        models = {
            "Baseline (clase mayoritaria)": DummyClassifier(strategy="most_frequent"),
            "Random Forest (base)": rf_base,
            "Random Forest (optimizado)": rf_pipe(n_jobs=-1, **rf_best),
            "XGBoost (base)": xgb_base,
            "XGBoost (optimizado)": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
                                                  **xgb_best),
        }
        """),
        md(A.get("tune", "_(análisis pendiente)_")),
        md("## 3. Evaluación: eficacia predictiva y eficiencia computacional"),
        code("""
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                                     log_loss, balanced_accuracy_score)

        rows, cv_auc = [], {}
        for name, model in models.items():
            cvres = cross_validate(model, X_train, y_train, cv=cv, scoring=["accuracy", "f1", "roc_auc"], n_jobs=1)
            fit_times = []
            for _ in range(3):
                t0 = time.perf_counter(); model.fit(X_train, y_train); fit_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            for _ in range(20):
                y_proba = model.predict_proba(X_test)[:, 1]
            inf_ms = (time.perf_counter() - t0) / 20 / len(X_test) * 1000
            y_pred = model.predict(X_test)
            cv_auc[name] = cvres["test_roc_auc"]
            rows.append({
                "Modelo": name,
                "Acc. CV (media)": cvres["test_accuracy"].mean(), "Acc. CV (std)": cvres["test_accuracy"].std(),
                "F1 CV": cvres["test_f1"].mean(), "ROC-AUC CV": cvres["test_roc_auc"].mean(),
                "Acc. train": accuracy_score(y_train, model.predict(X_train)),
                "Acc. test": accuracy_score(y_test, y_pred),
                "Acc. balanceada test": balanced_accuracy_score(y_test, y_pred),
                "Precisión test": precision_score(y_test, y_pred, zero_division=0),
                "Recall test": recall_score(y_test, y_pred), "F1 test": f1_score(y_test, y_pred),
                "ROC-AUC test": roc_auc_score(y_test, y_proba), "Log-Loss test": log_loss(y_test, y_proba),
                "Tiempo fit (s)": float(np.median(fit_times)), "Inferencia (ms/muestra)": inf_ms,
            })
        results = pd.DataFrame(rows).set_index("Modelo")
        results.round(4)
        """),
        md(A.get("res", "_(análisis pendiente)_")),
        code("""
        main = [m for m in models if "Baseline" not in m]
        colors = ["#7FA7D1", "#4C72B0", "#F0A97A", "#C44E52"]
        fig, axes = plt.subplots(1, 3, figsize=(22, 5.5), gridspec_kw={"width_ratios": [1.6, 1, 1]})
        results.loc[main, ["Acc. test", "F1 test", "ROC-AUC CV", "ROC-AUC test"]].T.plot(
            kind="bar", ax=axes[0], color=colors, width=0.85)
        axes[0].set_ylim(0.5, 0.92); axes[0].tick_params(axis="x", rotation=0)
        axes[0].set_title("Eficacia predictiva"); axes[0].legend(fontsize=8, loc="upper left")
        for c in axes[0].containers:
            axes[0].bar_label(c, fmt="%.3f", fontsize=6.5, rotation=90, padding=1)
        cv_df = pd.DataFrame({k: cv_auc[k] for k in main}).melt(var_name="Modelo", value_name="ROC-AUC")
        sns.boxplot(data=cv_df, y="Modelo", x="ROC-AUC", hue="Modelo", palette=colors, legend=False, ax=axes[1])
        axes[1].set_title("ROC-AUC en los 10 folds"); axes[1].set_ylabel("")
        eff = results.loc[main, ["Tiempo fit (s)", "Inferencia (ms/muestra)"]]
        eff.plot(kind="barh", ax=axes[2], color=["#55A868", "#8172B3"])
        axes[2].set_title("Eficiencia computacional (menor es mejor)"); axes[2].set_ylabel("")
        for c in axes[2].containers:
            axes[2].bar_label(c, fmt="%.3f", fontsize=7, padding=2)
        plt.tight_layout(); savefig("item5_comparacion"); plt.show()
        """),
        md(A.get("bar", "_(análisis pendiente)_")),
        code("""
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, axes = plt.subplots(1, 5, figsize=(25, 4.5), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
        for ax, name in zip(axes[:4], main):
            ConfusionMatrixDisplay.from_predictions(y_test, models[name].predict(X_test),
                                                    display_labels=["No diab.", "Diab."], cmap="Blues",
                                                    colorbar=False, ax=ax)
            ax.set_title(name, fontsize=10)
        for name, col in zip(main, colors):
            plot_roc(axes[4], models[name], X_test, y_test, name, color=col)
        axes[4].plot([0, 1], [0, 1], "k--", lw=1); axes[4].set_title("Curvas ROC (prueba)"); axes[4].legend(fontsize=7)
        plt.tight_layout(); savefig("item5_confusion_roc"); plt.show()
        """),
        md(A.get("cm", "_(análisis pendiente)_")),
        md("## 4. Curvas de aprendizaje: OOB de Random Forest y Log-Loss de XGBoost por iteración"),
        code("""
        X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train,
                                                      random_state=RANDOM_STATE)
        # XGBoost: pérdida en entrenamiento y validación (separada de train; el test no se usa)
        xgb_curve = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
                                  **{**xgb_best, "n_estimators": 600})
        xgb_curve.fit(X_fit, y_fit, eval_set=[(X_fit, y_fit), (X_val, y_val)], verbose=False)
        ev = xgb_curve.evals_result()
        # Random Forest: error OOB al crecer el bosque
        imp_rf = SimpleImputer(strategy="median", add_indicator=True).fit(X_train)
        Xi = imp_rf.transform(X_train)
        rf_oob = RandomForestClassifier(oob_score=True, warm_start=True, random_state=RANDOM_STATE, n_jobs=-1,
                                        **{k: v for k, v in rf_best.items() if k != "n_estimators"})
        oob_err, n_grid = [], list(range(25, 601, 25))
        for n_ in n_grid:
            rf_oob.set_params(n_estimators=n_).fit(Xi, y_train); oob_err.append(1 - rf_oob.oob_score_)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        axes[0].plot(n_grid, oob_err, "-o", color="#4C72B0")
        axes[0].set_title("Random Forest: error OOB vs. número de árboles")
        axes[0].set_xlabel("n_estimators"); axes[0].set_ylabel("Error OOB (1 - accuracy)")
        axes[1].plot(ev["validation_0"]["logloss"], color="#55A868", label="entrenamiento")
        axes[1].plot(ev["validation_1"]["logloss"], color="#C44E52", label="validación")
        best_it = int(np.argmin(ev["validation_1"]["logloss"]))
        axes[1].axvline(best_it, color="k", ls=":", label=f"mínimo val. (iter {best_it})")
        axes[1].axvline(xgb_best["n_estimators"], color="#8172B3", ls="--",
                        label=f"n_estimators elegido = {xgb_best['n_estimators']}")
        axes[1].set_title("XGBoost: Log-Loss por iteración de boosting"); axes[1].set_xlabel("iteración")
        axes[1].set_ylabel("Log-Loss"); axes[1].legend()
        plt.tight_layout(); savefig("item5_curvas"); plt.show()
        print(f"RF error OOB: inicial {oob_err[0]:.4f} -> final {oob_err[-1]:.4f}")
        print(f"XGB log-loss val mínimo {min(ev['validation_1']['logloss']):.4f} en iter {best_it}; "
              f"final (600) {ev['validation_1']['logloss'][-1]:.4f}")
        """),
        md(A.get("curves", "_(análisis pendiente)_")),
        md("## 5. Importancia de variables: MDI (RF), Gain (XGBoost) y permutación (ambos)"),
        code("""
        from sklearn.inspection import permutation_importance

        rf_opt, xgb_opt = models["Random Forest (optimizado)"], models["XGBoost (optimizado)"]
        rf_names = list(X.columns) + [f"falta_{X.columns[i]}" for i in rf_opt.named_steps["imp"].indicator_.features_]
        mdi = pd.Series(rf_opt.named_steps["clf"].feature_importances_, index=rf_names)
        mdi = mdi.groupby(lambda c: c.replace("falta_", "")).sum()   # agrega el indicador a su variable
        gain = pd.Series(xgb_opt.get_booster().get_score(importance_type="gain")).reindex(X.columns).fillna(0)
        gain = gain / gain.sum()
        perm = pd.DataFrame({
            "Random Forest": permutation_importance(rf_opt, X_test, y_test, scoring="roc_auc", n_repeats=30,
                                                    random_state=RANDOM_STATE).importances_mean,
            "XGBoost": permutation_importance(xgb_opt, X_test, y_test, scoring="roc_auc", n_repeats=30,
                                              random_state=RANDOM_STATE).importances_mean}, index=X.columns)

        fig, axes = plt.subplots(1, 3, figsize=(21, 5))
        mdi.sort_values().plot(kind="barh", ax=axes[0], color="#4C72B0"); axes[0].set_title("Random Forest — MDI (Gini)")
        gain.sort_values().plot(kind="barh", ax=axes[1], color="#C44E52"); axes[1].set_title("XGBoost — Gain normalizado")
        perm.sort_values("XGBoost").plot(kind="barh", ax=axes[2], color=["#4C72B0", "#C44E52"])
        axes[2].set_title("Importancia por permutación en test (caída de AUC)")
        plt.tight_layout(); savefig("item5_importancia"); plt.show()
        pd.DataFrame({"MDI RF": mdi, "Gain XGB": gain}).join(perm.add_prefix("Perm. ")).sort_values("MDI RF",
                                                                                                    ascending=False).round(4)
        """),
        md(A.get("imp", "_(análisis pendiente)_")),
        md("## 6. Significancia estadística (CV repetida 10×3 + Wilcoxon) e IC bootstrap de la diferencia de AUC"),
        code("""
        from scipy.stats import wilcoxon
        from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

        rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)
        rep = {n: cross_val_score(models[n], X_train, y_train, cv=rcv, scoring="roc_auc", n_jobs=1) for n in main}
        pairs = [("Random Forest (optimizado)", "XGBoost (optimizado)"), ("Random Forest (base)", "XGBoost (base)"),
                 ("XGBoost (optimizado)", "XGBoost (base)"), ("Random Forest (optimizado)", "Random Forest (base)")]
        stats_rows = []
        for a, b in pairs:
            p = wilcoxon(rep[a], rep[b], zero_method="zsplit").pvalue
            stats_rows.append({"Comparación": f"{a} vs {b}", "AUC A": rep[a].mean(), "AUC B": rep[b].mean(),
                               "Diferencia A-B": (rep[a] - rep[b]).mean(), "p-valor Wilcoxon": p,
                               "Significativo (α=0.05)": "Sí" if p < 0.05 else "No"})
        stats_df = pd.DataFrame(stats_rows)

        rng = np.random.default_rng(RANDOM_STATE)
        p_rf, p_xgb = rf_opt.predict_proba(X_test)[:, 1], xgb_opt.predict_proba(X_test)[:, 1]
        yt = y_test.values; diffs = []
        for _ in range(2000):
            idx = rng.integers(0, len(yt), len(yt))
            if len(np.unique(yt[idx])) == 2:
                diffs.append(roc_auc_score(yt[idx], p_rf[idx]) - roc_auc_score(yt[idx], p_xgb[idx]))
        auc_diff_ci = np.percentile(diffs, [2.5, 97.5])
        ci_df = pd.DataFrame({n: bootstrap_ci(y_test, models[n].predict_proba(X_test)[:, 1], roc_auc_score)
                              for n in main}, index=["AUC IC95% inf", "AUC IC95% sup"]).T
        display(stats_df.round(4)); display(ci_df.round(4))
        print(f"Diferencia AUC test (RF opt - XGB opt) = {np.mean(diffs):.4f}, IC95% = [{auc_diff_ci[0]:.4f}, {auc_diff_ci[1]:.4f}]")
        """),
        md(A.get("stats", "_(análisis pendiente)_")),
        md("## 7. Verificación en un segundo dataset (breast_cancer)"),
        code("""
        from sklearn.datasets import load_breast_cancer

        Xb, yb = load_breast_cancer(return_X_y=True, as_frame=True)
        Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.2, stratify=yb, random_state=RANDOM_STATE)
        bc_models = {"Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
                     "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                              colsample_bytree=0.8, eval_metric="logloss",
                                              random_state=RANDOM_STATE, n_jobs=-1)}
        bc_rows = []
        for n_, m in bc_models.items():
            s = cross_validate(m, Xb_tr, yb_tr, cv=cv, scoring=["accuracy", "roc_auc"])
            t0 = time.perf_counter(); m.fit(Xb_tr, yb_tr); ft = time.perf_counter() - t0
            bc_rows.append({"Modelo": n_, "Acc. CV": s["test_accuracy"].mean(), "AUC CV": s["test_roc_auc"].mean(),
                            "Acc. test": m.score(Xb_te, yb_te),
                            "AUC test": roc_auc_score(yb_te, m.predict_proba(Xb_te)[:, 1]), "Tiempo fit (s)": ft})
        bc_df = pd.DataFrame(bc_rows).set_index("Modelo")
        bc_df.round(4)
        """),
        md(A.get("bc", "_(análisis pendiente)_")),
        code("""
        save_results({
            "results": results.reset_index().to_dict(orient="records"),
            "rf_best": rf_best, "xgb_best": xgb_best,
            "rf_search_auc": rf_search.best_score_, "xgb_search_auc": xgb_search.best_score_,
            "rf_search_time": rf_search_t, "xgb_search_time": xgb_search_t,
            "stats": stats_df.to_dict(orient="records"),
            "ci": ci_df.reset_index().rename(columns={"index": "Modelo"}).to_dict(orient="records"),
            "auc_diff_ci": list(auc_diff_ci), "auc_diff_mean": float(np.mean(diffs)),
            "oob": {"first": oob_err[0], "last": oob_err[-1]},
            "xgb_curve": {"best_iter": best_it, "min_val": min(ev["validation_1"]["logloss"]),
                          "final_val": ev["validation_1"]["logloss"][-1]},
            "breast_cancer": bc_df.reset_index().to_dict(orient="records"),
            "importance": pd.DataFrame({"MDI RF": mdi, "Gain XGB": gain}).join(perm.add_prefix("Perm. "))
                            .reset_index().rename(columns={"index": "Variable"}).to_dict(orient="records"),
        }, "item5")
        print("Resultados guardados en results/item5.json")
        """),
        md(A.get("summary", "## Conclusiones\n_(pendiente)_")),
    ]
