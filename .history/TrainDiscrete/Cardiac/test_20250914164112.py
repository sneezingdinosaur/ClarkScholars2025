#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diabetes-status prediction – Leaderboard + Training edition (v1.5.0)
===================================================================
Adds a compact table (Class | Model | Accuracy | F1-score (Macro) | AUC-ROC (Macro))
with 10-fold CV mean ± std for each model, like your screenshot.

Now also prints:
- Per-model Top-3 features by importance (CV-averaged) for models that support it.
- Top-3 features for the best model after fitting on ALL data.
"""

import warnings
import logging
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.base import clone

# Base models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
    VotingClassifier,
)

# External gradient boosting
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── Quiet logs / warnings ─────────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)

RANDOM_STATE = 42

# ── Config ────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "Rate", "PR", "QRSD", "QT", "QTc",
    "P", "QRS", "T", "SDNN", "RMSSD", "pNN50"
]
ID_COLUMN = "participant_id"

# Input paths (edit as needed)
CSV_FILE_PATH = r"C:\Users\sneez\Downloads\TexasTechAI\0704andbefore\participant_data_with_hrv.csv"
TSV_FILE_PATH = r"C:\Users\sneez\Desktop\OrganizedAI-READI\DataFiles\participants_Original.tsv"

# For the compact table header (so it matches your figure)
TABLE_CLASS_NAME = "CardiacECG"   # change if you want a different label in the first column


# ── Data loading & label mapping ──────────────────────────────────────────
def map_study_group_to_label(series: pd.Series) -> pd.Series:
    mapping = {
        "healthy": "healthy",
        "pre_diabetes_lifestyle_controlled": "pre-diabetic",
        "diabetes": "diabetic",
    }
    return series.map(mapping)


def load_and_merge(csv_path: str, tsv_path: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df_feat = pd.read_csv(csv_path)
    required_cols = [ID_COLUMN] + FEATURE_COLUMNS
    missing = [c for c in required_cols if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Feature CSV missing required columns: {missing}")
    df_feat = df_feat[required_cols].copy()

    df_sg = pd.read_csv(tsv_path, sep="\t")
    if ID_COLUMN not in df_sg.columns or "study_group" not in df_sg.columns:
        raise ValueError("TSV must contain columns: 'participant_id' and 'study_group'")

    df_sg["label_3class"] = map_study_group_to_label(df_sg["study_group"])

    df = df_feat.merge(df_sg[[ID_COLUMN, "label_3class"]], on=ID_COLUMN, how="inner")
    df = df.dropna(subset=FEATURE_COLUMNS + ["label_3class"]).copy()

    le = LabelEncoder()
    y = le.fit_transform(df["label_3class"].astype(str))
    classes = list(le.classes_)

    X = df[FEATURE_COLUMNS].astype(float)
    return X, y, classes


# ── Model zoo ─────────────────────────────────────────────────────────────
def build_models() -> Dict[str, Pipeline]:
    scaler = StandardScaler()

    lr = Pipeline([("scaler", scaler),
                   ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])

    svm_rbf = Pipeline([("scaler", scaler),
                        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))])

    knn7 = Pipeline([("scaler", scaler),
                     ("clf", KNeighborsClassifier(n_neighbors=7))])

    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)

    gbdt = GradientBoostingClassifier(random_state=RANDOM_STATE)

    adb = AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=300)

    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=RANDOM_STATE,
        verbosity=0, n_jobs=-1, tree_method="hist",
    )

    lgbm = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
        verbosity=-1, n_jobs=-1,
    )

    base_estimators = [
        ("LR",   Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])),
        ("SVM",  Pipeline([("scaler", StandardScaler()),
                           ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))])),
        ("KNN7", Pipeline([("scaler", StandardScaler()),
                           ("clf", KNeighborsClassifier(n_neighbors=7))])),
        ("RF",   rf),
        ("XGB",  xgb),
        ("LGBM", lgbm),
    ]

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1
    )

    voting = VotingClassifier(estimators=base_estimators, voting="soft", n_jobs=-1)

    return {
        "StackingEnsemble": stacking,
        "VotingEnsemble":   voting,
        "RandomForest":     rf,
        "XGBoost":          xgb,
        "KNN(7)":           knn7,
        "GradientBoost":    gbdt,
        "LightGBM":         lgbm,
        "SVM-RBF":          svm_rbf,
        "LogisticRegression": lr,
        "AdaBoost":         adb,
    }


# ── Helpers: feature importances ─────────────────────────────────────────
def _estimator_from_model(m):
    """Return the fitted estimator object itself (strip Pipeline wrapper if present)."""
    if isinstance(m, Pipeline):
        # Prefer 'clf' step if present, else last step
        if "clf" in m.named_steps:
            return m.named_steps["clf"]
        return list(m.named_steps.values())[-1]
    return m

def _extract_importances(estimator) -> np.ndarray | None:
    """
    Get a 1D vector of feature importances, or None if unsupported.
    - Tree models: feature_importances_
    - LogisticRegression: mean absolute coef across classes
    """
    if hasattr(estimator, "feature_importances_"):
        imp = getattr(estimator, "feature_importances_")
        if imp is not None:
            return np.asarray(imp).ravel()
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        if coef.ndim == 1:
            return np.abs(coef)
        return np.mean(np.abs(coef), axis=0)
    return None

def _print_topk(model_name: str, feature_names: List[str], importances: np.ndarray, k: int = 3):
    if importances is None or len(importances) != len(feature_names):
        print(f"[{model_name}] Top features: N/A (no importances or mismatch).")
        return
    order = np.argsort(importances)[::-1][:k]
    print(f"[{model_name}] Top {k} features by importance (CV-avg):")
    for r, i in enumerate(order, 1):
        print(f"  {r}. {feature_names[i]} : {importances[i]:.6f}")


# ── Evaluation & printing ─────────────────────────────────────────────────
def evaluate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    models: Dict[str, Pipeline],
    n_splits: int = 10
) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, str, str]]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'auc_macro_ovr': 'roc_auc_ovr'
    }

    leaderboard: List[Tuple[str, float, float]] = []
    table_rows: List[Tuple[str, str, str, str]] = []

    def pm(mean, std, dec=4):  # pretty "mean ± std"
        return f"{mean:.{dec}f} ± {std:.{dec}f}"

    for name, model in models.items():
        try:
            cvres = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
            acc_m, acc_s = float(cvres["test_accuracy"].mean()), float(cvres["test_accuracy"].std())
            f1_m,  f1_s  = float(cvres["test_f1_macro"].mean()), float(cvres["test_f1_macro"].std())
            auc_m, auc_s = float(cvres["test_auc_macro_ovr"].mean()), float(cvres["test_auc_macro_ovr"].std())

            leaderboard.append((name, acc_m, acc_s))
            table_rows.append((name, pm(acc_m, acc_s), pm(f1_m, f1_s), pm(auc_m, auc_s)))
        except Exception as e:
            leaderboard.append((name + " (failed)", np.nan, np.nan))
            table_rows.append((name + " (failed)", "N/A", "N/A", "N/A"))
            print(f"[WARN] {name} failed with error: {e}")
            continue

        # ---- NEW: CV-averaged feature importances per model ----
        imp_sum = None
        imp_count = 0
        for tr_idx, te_idx in cv.split(X, y):
            m = clone(model)
            m.fit(X.iloc[tr_idx], y[tr_idx])
            est = _estimator_from_model(m)
            imp = _extract_importances(est)
            if imp is not None and len(imp) == len(FEATURE_COLUMNS):
                if imp_sum is None:
                    imp_sum = np.zeros_like(imp, dtype=float)
                imp_sum += np.asarray(imp, dtype=float)
                imp_count += 1
        if imp_count > 0:
            imp_avg = imp_sum / float(imp_count)
            _print_topk(name, FEATURE_COLUMNS, imp_avg, k=3)
        else:
            print(f"[{name}] Top features: N/A (no importances available).")
        # ---------------------------------------------------------

    # Sort leaderboard by mean accuracy (descending), NaNs to bottom
    leaderboard.sort(key=lambda t: (-t[1] if not np.isnan(t[1]) else float("inf")))
    name_order = [n for n, _, _ in leaderboard]
    table_rows.sort(key=lambda r: name_order.index(r[0]) if r[0] in name_order else 1_000_000)
    return leaderboard, table_rows


def print_leaderboard(results: List[Tuple[str, float, float]]) -> None:
    print("=== Leaderboard: Mean 10-fold Accuracy ===")
    width_name = max(len(name) for name, _, _ in results) + 2
    for i, (name, mean, std) in enumerate(results, start=1):
        if np.isnan(mean):
            line = f"{i:2d}. {name:<{width_name}} Mean Acc: N/A"
        else:
            line = f"{i:2d}. {name:<{width_name}} Mean Acc: {mean:.4f} ± {std:.4f}"
        print(line)


def print_models_table(rows: List[Tuple[str, str, str, str]], class_name: str = TABLE_CLASS_NAME) -> None:
    h1, h2, h3, h4, h5 = "Class", "Model", "Accuracy", "F1-score (Macro)", "AUC-ROC (Macro)"
    line = "+" + "-"*12 + "+" + "-"*18 + "+" + "-"*20 + "+" + "-"*22 + "+" + "-"*22 + "+"
    print(line)
    print(f"| {h1:<10} | {h2:<16} | {h3:<18} | {h4:<20} | {h5:<20} |")
    print(line)
    for model_name, acc_s, f1_s, auc_s in rows:
        print(f"| {class_name:<10} | {model_name:<16} | {acc_s:<18} | {f1_s:<20} | {auc_s:<20} |")
    print(line)


# ── Training section (fit best model on all data) ─────────────────────────
def train_best_model(
    X: pd.DataFrame,
    y: np.ndarray,
    models: Dict[str, Pipeline],
    leaderboard: List[Tuple[str, float, float]]
) -> Tuple[str, Pipeline]:
    best_name, best_mean, best_std = next((n, m, s) for (n, m, s) in leaderboard if not np.isnan(m))
    best_model = models[best_name]

    print("\n=== Training Section ===")
    print(f"Best model selected from CV leaderboard: {best_name}")
    print(f"Cross-validated accuracy: {best_mean:.4f} ± {best_std:.4f}")
    print(f"Fitting {best_name} on all {len(X)} samples ...")

    best_model.fit(X, y)
    train_acc = best_model.score(X, y)  # sanity check (optimistic)
    print(f"Training complete. In-sample accuracy: {train_acc:.4f}")

    # ---- NEW: Top-3 after full-data fit ----
    est = _estimator_from_model(best_model)
    imp = _extract_importances(est)
    if imp is not None and len(imp) == len(FEATURE_COLUMNS):
        print(f"[{best_name}] Top 3 features by importance (full fit):")
        order = np.argsort(imp)[::-1][:3]
        for r, i in enumerate(order, 1):
            print(f"  {r}. {FEATURE_COLUMNS[i]} : {imp[i]:.6f}")
    else:
        print(f"[{best_name}] Top features: N/A (no importances available).")
    # ----------------------------------------

    return best_name, best_model


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    X, y, classes = load_and_merge(CSV_FILE_PATH, TSV_FILE_PATH)
    models = build_models()

    leaderboard, table_rows = evaluate_models(X, y, models, n_splits=10)

    print_leaderboard(leaderboard)
    print("\n")
    print_models_table(table_rows, class_name=TABLE_CLASS_NAME)

    _best_name, _best_model = train_best_model(X, y, models, leaderboard)


if __name__ == "__main__":
    main()
