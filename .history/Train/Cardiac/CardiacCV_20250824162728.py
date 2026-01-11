#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diabetes-status prediction – Leaderboard edition (v1.3.0)
=========================================================
Trains a suite of classifiers with 10-fold Stratified CV and prints a compact
leaderboard of mean accuracy ± std, sorted best → worst.

- Expects a CSV with columns:
    features: ['Rate','PR','QRSD','QT','QTc','P','QRS','T','SDNN','RMSSD','pNN50']
    label:    'diabetes_category'  (e.g., healthy / pre-diabetic / diabetic)
- Suppresses LightGBM/XGBoost info logs.
- Uses scaling for models that need it (SVM, LR, KNN, meta-learner).
"""

import warnings
import logging
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score

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
LABEL_COLUMN = "st"

# Path to your dataset
CSV_FILE_PATH = r"C:\Users\sneez\Downloads\TexasTechAI\0704andbefore\participant_data_with_hrv.csv"


def load_data(path: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Load CSV, keep required columns, drop rows with NA, encode labels."""
    df = pd.read_csv(path)
    cols_needed = FEATURE_COLUMNS + [LABEL_COLUMN]
    df = df[cols_needed].dropna().copy()

    X = df[FEATURE_COLUMNS].astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COLUMN].astype(str))
    classes = list(le.classes_)
    return X, y, classes


def build_models() -> Dict[str, Pipeline]:
    """Create model dictionary with proper scaling where appropriate."""
    # Common scaler
    scaler = StandardScaler()

    # Individual learners
    lr = Pipeline([
        ("scaler", scaler),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, n_jobs=None))
    ])

    svm_rbf = Pipeline([
        ("scaler", scaler),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ])

    knn7 = Pipeline([
        ("scaler", scaler),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ])

    rf = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    )

    gbdt = GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    adb = AdaBoostClassifier(
        random_state=RANDOM_STATE, n_estimators=300
    )

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
        tree_method="hist",
    )

    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )

    # For ensembles requiring probability outputs, wrap scaled versions when needed
    # Use leaner base set for robustness in stacking
    base_estimators = [
        ("LR", Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])),
        ("SVM", Pipeline([("scaler", StandardScaler()),
                          ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))])),
        ("KNN7", Pipeline([("scaler", StandardScaler()),
                           ("clf", KNeighborsClassifier(n_neighbors=7))])),
        ("RF", rf),
        ("XGB", xgb),
        ("LGBM", lgbm),
    ]

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1
    )

    voting = VotingClassifier(
        estimators=base_estimators,
        voting="soft",
        n_jobs=-1
    )

    models: Dict[str, Pipeline] = {
        "StackingEnsemble": stacking,
        "VotingEnsemble": voting,
        "RandomForest": rf,
        "XGBoost": xgb,
        "KNN(7)": knn7,
        "GradientBoost": gbdt,
        "LightGBM": lgbm,
        "SVM-RBF": svm_rbf,
        "LogisticRegression": lr,
        "AdaBoost": adb,
    }
    return models


def evaluate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    models: Dict[str, Pipeline],
    n_splits: int = 10
) -> List[Tuple[str, float, float]]:
    """Compute mean ± std CV accuracies for all models."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scorer = make_scorer(accuracy_score)
    results: List[Tuple[str, float, float]] = []

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            results.append((name, float(scores.mean()), float(scores.std())))
        except Exception as e:
            # If a model fails, record as NaN to keep format stable
            results.append((name + " (failed)", np.nan, np.nan))
            print(f"[WARN] {name} failed with error: {e}")

    # Sort by mean accuracy (descending), NaNs go to bottom
    results.sort(key=lambda t: (-t[1] if not np.isnan(t[1]) else float("inf")))
    return results


def print_leaderboard(results: List[Tuple[str, float, float]]) -> None:
    """Pretty-print leaderboard in the requested format."""
    print("=== Leaderboard: Mean 10-fold Accuracy ===")
    width_name = max(len(name) for name, _, _ in results) + 2
    for i, (name, mean, std) in enumerate(results, start=1):
        if np.isnan(mean):
            line = f"{i:2d}. {name:<{width_name}} Mean Acc: N/A"
        else:
            line = f"{i:2d}. {name:<{width_name}} Mean Acc: {mean:.4f} ± {std:.4f}"
        print(line)


def main():
    # 1) Load data
    X, y, classes = load_data(CSV_FILE_PATH)

    # 2) Build models
    models = build_models()

    # 3) Evaluate with 10-fold CV
    results = evaluate_models(X, y, models, n_splits=10)

    # 4) Print leaderboard
    print_leaderboard(results)


if __name__ == "__main__":
    main()
