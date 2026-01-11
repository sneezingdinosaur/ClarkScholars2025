#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diabetes-status prediction – Leaderboard + Training edition (v1.4.0)
===================================================================
What this script does:
  1) Loads a feature CSV and a study_group TSV, merges on `participant_id`.
  2) Maps study_group → 3-class label:
       - 'healthy' → 'healthy'
       - 'pre_diabetes_lifestyle_controlled' → 'pre-diabetic'
       - {'insulin_dependent',
          'oral_medication_and_or_non_insulin_injectable_medication_controlled'} → 'diabetic'
  3) Runs 10-fold Stratified CV across a suite of models and prints a leaderboard
     of mean accuracy ± std, sorted best → worst.
  4) Trains (fits) the BEST model on ALL available data and prints a concise
     “training section” summary showing the chosen model and its fit status.

Notes:
  - Expects feature CSV with columns:
      ['participant_id', 'Rate','PR','QRSD','QT','QTc','P','QRS','T','SDNN','RMSSD','pNN50']
  - Expects TSV with columns: ['participant_id', 'study_group']
  - Suppresses LightGBM/XGBoost info logs.
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
ID_COLUMN = "participant_id"

# Input paths (edit as needed)
CSV_FILE_PATH = r"C:\Users\sneez\Downloads\TexasTechAI\0704andbefore\participant_data_with_hrv.csv"
TSV_FILE_PATH = r"C:\Users\sneez\Desktop\OrganizedAI-READI\DataFiles\participants_Original.tsv"


# ── Data loading & label mapping ──────────────────────────────────────────
def map_study_group_to_label(series: pd.Series) -> pd.Series:
    """Map raw study_group values to 3 classes: healthy/pre-diabetic/diabetic."""
    mapping = {
        "healthy": "healthy",
        "pre_diabetes_lifestyle_controlled": "pre-diabetic",
        "insulin_dependent": "diabetic",
    }
    return series.map(mapping)


def load_and_merge(csv_path: str, tsv_path: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load feature CSV and study_group TSV, merge on participant_id, and return:
      X (features), y (encoded labels), classes (label names in encoder order)
    """
    # Load features
    df_feat = pd.read_csv(csv_path)
    required_cols = [ID_COLUMN] + FEATURE_COLUMNS
    missing = [c for c in required_cols if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Feature CSV missing required columns: {missing}")

    df_feat = df_feat[required_cols].copy()

    # Load study_group TSV
    df_sg = pd.read_csv(tsv_path, sep="\t")
    if ID_COLUMN not in df_sg.columns or "study_group" not in df_sg.columns:
        raise ValueError("TSV must contain columns: 'participant_id' and 'study_group'")

    # Map to 3-class label
    df_sg["label_3class"] = map_study_group_to_label(df_sg["study_group"])

    # Merge
    df = df_feat.merge(df_sg[[ID_COLUMN, "label_3class"]], on=ID_COLUMN, how="inner")

    # Drop rows without label or with missing features
    df = df.dropna(subset=FEATURE_COLUMNS + ["label_3class"]).copy()

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["label_3class"].astype(str))
    classes = list(le.classes_)

    # Features
    X = df[FEATURE_COLUMNS].astype(float)
    return X, y, classes


# ── Model zoo ─────────────────────────────────────────────────────────────
def build_models() -> Dict[str, Pipeline]:
    """Create model dictionary with proper scaling where appropriate."""
    scaler = StandardScaler()

    lr = Pipeline([
        ("scaler", scaler),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
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

    # Base estimators for ensembles (use scaled versions where needed)
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


# ── Evaluation & printing ─────────────────────────────────────────────────
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


# ── Training section (fit best model on all data) ─────────────────────────
def train_best_model(
    X: pd.DataFrame,
    y: np.ndarray,
    models: Dict[str, Pipeline],
    leaderboard: List[Tuple[str, float, float]]
) -> Tuple[str, Pipeline]:
    """
    Pick the best model from the leaderboard (highest mean accuracy),
    fit it on ALL available data, and print a concise training summary.
    """
    # Pick first non-NaN entry (already sorted best → worst)
    best_name, best_mean, best_std = next((n, m, s) for (n, m, s) in leaderboard if not np.isnan(m))
    best_model = models[best_name]

    print("\n=== Training Section ===")
    print(f"Best model selected from CV leaderboard: {best_name}")
    print(f"Cross-validated accuracy: {best_mean:.4f} ± {best_std:.4f}")
    print(f"Fitting {best_name} on all {len(X)} samples ...")

    # Fit on full dataset
    best_model.fit(X, y)

    # In-sample score (note: optimistic — shown just to confirm fit completed)
    train_acc = best_model.score(X, y)
    print(f"Training complete. In-sample accuracy (for sanity check): {train_acc:.4f}")

    return best_name, best_model


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    # 1) Load & merge data
    X, y, classes = load_and_merge(CSV_FILE_PATH, TSV_FILE_PATH)

    # 2) Build models
    models = build_models()

    # 3) Evaluate with 10-fold CV → leaderboard
    results = evaluate_models(X, y, models, n_splits=10)

    # 4) Print leaderboard (requested format)
    print_leaderboard(results)

    # 5) Train the best model on ALL data (training section)
    _best_name, _best_model = train_best_model(X, y, models, results)


if __name__ == "__main__":
    main()
