#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CGM-feature models for diabetes-status prediction (3-class, 10-fold CV)
=======================================================================
Without PCA feature reduction and with a soft-voting ensemble + stacking ensemble.
Classes:
  0 → healthy
  1 → pre-diabetic
  2 → diabetes

Now also prints a compact table with:
Class | Model | Accuracy | F1-score (Macro) | AUC-ROC (Macro)
(all as 10-fold CV mean ± std).
"""

# ------------------------------------------------------------------#
# 1) FILE LOCATIONS – EDIT THESE                                    #
# ------------------------------------------------------------------#
CGM_CSV_PATH     = r"C:\Users\sneez\Desktop\Wearable\enhanced_cgm_features_with_labels.csv"
PARTICIPANTS_TSV = r"C:\Users\sneez\Desktop\OrganizedAI-READI\DataFiles\multimodal_OMOPHBOverlap.tsv"
# ------------------------------------------------------------------#

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np

from sklearn.compose       import ColumnTransformer
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model   import LogisticRegression
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.svm            import SVC
from sklearn.ensemble       import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)

import xgboost as xgb
import lightgbm as lgb


# ------------------------------------------------------------------#
# 2) DATA LOADING & MERGE                                           #
# ------------------------------------------------------------------#
print("Loading data …")
df_cgm  = pd.read_csv(CGM_CSV_PATH, low_memory=False)
df_part = pd.read_csv(
    PARTICIPANTS_TSV,
    sep="\t",
    usecols=["participant_id", "study_group"],
    dtype=str,
)

# Align IDs
df_cgm["participant_id_clean"] = df_cgm["participant_id"].str.replace(
    "AIREADI-", "", regex=False
)
df_part["participant_id_clean"] = df_part["participant_id"]

# Map to 3 classes
def collapse_group(g: str) -> str:
    g = str(g).lower()
    if "healthy" in g:
        return "healthy"
    if g.startswith("pre"):
        return "pre-diabetic"
    return "diabetes"

df_part["diabetes_cat"] = df_part["study_group"].apply(collapse_group)

df = pd.merge(
    df_cgm,
    df_part["participant_id_clean"].to_frame().join(df_part["diabetes_cat"]),
    on="participant_id_clean",
    how="inner",
    validate="one_to_one",
)
print(f"Total samples after merge: {len(df)}")


# ------------------------------------------------------------------#
# 3) FEATURES & NUMERIC LABELS                                      #
# ------------------------------------------------------------------#
exclude_cols = {
    "participant_id", "participant_id_clean",
    "diabetes_cat", "label"
}
feature_cols = [c for c in df.columns if c not in exclude_cols]

label_order = ["healthy", "pre-diabetic", "diabetes"]
label_map   = {name: idx for idx, name in enumerate(label_order)}
inv_label   = {idx: name for name, idx in label_map.items()}

y_all = df["diabetes_cat"].map(label_map)
X_all = df[feature_cols]

print("Class distribution:", y_all.value_counts().rename(index=inv_label).to_dict())


# ------------------------------------------------------------------#
# 4) PREPROCESSOR (impute → scale)                                  #
# ------------------------------------------------------------------#
numeric_transformer = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler())
])

preprocessor = ColumnTransformer(
    [("num", numeric_transformer, feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)


# ------------------------------------------------------------------#
# 5) MODEL ZOO                                                      #
# ------------------------------------------------------------------#
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "KNN(7)": KNeighborsClassifier(n_neighbors=7),
    "SVM-RBF": SVC(
        kernel="rbf", probability=True, class_weight="balanced", random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(n_estimators=400, random_state=42),
    "GradientBoost": GradientBoostingClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=len(label_order),
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators       = 500,
        learning_rate      = 0.05,
        objective          = "multiclass",
        num_class          = len(label_order),
        random_state       = 42,
        verbose            = -1,
        min_gain_to_split  = 0.0,
    ),
    "VotingEnsemble": VotingClassifier(
        estimators=[
            ("lr",  LogisticRegression(max_iter=1000, class_weight="balanced")),
            ("xgb", xgb.XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        objective="multi:softprob",
                        num_class=len(label_order),
                        tree_method="hist",
                        eval_metric="mlogloss",
                        random_state=42)),
            ("lgbm", lgb.LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        objective="multiclass",
                        num_class=len(label_order),
                        random_state=42,
                        verbose=-1,
                        min_gain_to_split=0.0)),
        ],
        voting="soft",
        n_jobs=-1
    ),
    "StackingEnsemble": StackingClassifier(
        estimators=[
            ("lr",  LogisticRegression(max_iter=1000, class_weight="balanced")),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
            ("rf",  RandomForestClassifier(
                        n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=42)),
            ("xgb", xgb.XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        objective="multi:softprob",
                        num_class=len(label_order),
                        tree_method="hist",
                        eval_metric="mlogloss",
                        random_state=42)),
            ("lgbm", lgb.LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        objective="multiclass",
                        num_class=len(label_order),
                        random_state=42,
                        verbose=-1,
                        min_gain_to_split=0.0)),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
        passthrough=False
    ),
}

# Pretty "mean ± std" string
def pm(mean, std, decimals=4):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

# Simple ASCII table (like your screenshot)
def print_models_table(rows, class_name="CGM"):
    h1, h2, h3, h4, h5 = "Class", "Model", "Accuracy", "F1-score (Macro)", "AUC-ROC (Macro)"
    line = "+" + "-"*12 + "+" + "-"*18 + "+" + "-"*20 + "+" + "-"*22 + "+" + "-"*22 + "+"
    print(line)
    print(f"| {h1:<10} | {h2:<16} | {h3:<18} | {h4:<20} | {h5:<20} |")
    print(line)
    for model_name, acc_s, f1_s, auc_s in rows:
        print(f"| {class_name:<10} | {model_name:<16} | {acc_s:<18} | {f1_s:<20} | {auc_s:<20} |")
    print(line)


# ------------------------------------------------------------------#
# 6) METRIC UTILITIES                                               #
# ------------------------------------------------------------------#
def decode(arr):
    return [inv_label[int(a)] for a in arr]

# ------------------------------------------------------------------#
# 6) METRIC UTILITIES                                               #
# ------------------------------------------------------------------#
def show_metrics(model_name: str, fold: str, y_true_int, y_pred_int, y_proba=None):
    acc = accuracy_score(y_true_int, y_pred_int)
    try:
        auc = roc_auc_score(y_true_int, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan
    print(f"{model_name:<18} | Fold {fold:<2} | "
          f"Accuracy: {acc:.4f} | AUROC: {auc:.4f}")
    return acc, auc


# ------------------------------------------------------------------#
# 7) TRAIN & 10-FOLD EVALUATE (only AUROC + Accuracy)               #
# ------------------------------------------------------------------#
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = {}

for name, estimator in models.items():
    print(f"\n=== {name} (10-fold CV) ===")
    fold_accs, fold_aucs = [], []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_all, y_all), start=1):
        X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]
        
        pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
        pipe.fit(X_train, y_train)
        
        y_pred  = pipe.predict(X_test)
        try:
            y_proba = pipe.predict_proba(X_test)
        except Exception:
            y_proba = None
        
        acc, auc = show_metrics(name, str(fold_idx), y_test, y_pred, y_proba)
        fold_accs.append(acc)
        fold_aucs.append(auc)
    
    mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
    mean_auc, std_auc = np.nanmean(fold_aucs), np.nanstd(fold_aucs)
    results[name]     = (mean_acc, std_acc, mean_auc, std_auc)
    print(f"{name:<18} | Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f} | "
          f"Mean AUROC: {mean_auc:.4f} ± {std_auc:.4f}")


# ──────────────────────────────────────────────────────────────────────
# 8) PRINT LEADERBOARD & COMPACT TABLE
# ──────────────────────────────────────────────────────────────────────
print("\n=== Leaderboard: Mean 10-fold Accuracy ===")
leaderboard = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
for rank, (model_name, (mean_acc, std_acc, *_)) in enumerate(leaderboard, start=1):
    print(f"{rank:>2}. {model_name:<18}  Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")

# final compact summary (screenshot style)
print("\n")
print_models_table(table_rows, class_name="CGM")
