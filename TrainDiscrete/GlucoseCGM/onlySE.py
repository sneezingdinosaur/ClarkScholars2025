#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CGM-feature model for diabetes-status prediction (3-class, 10-fold CV)
=====================================================================
This version evaluates and prints ONLY the StackingEnsemble.

Classes:
  0 → healthy
  1 → pre-diabetic
  2 → diabetes

It prints:
1) Per-fold metrics for the StackingEnsemble
2) Mean ± std (10-fold) for Accuracy, F1 (macro), AUC-ROC (macro)
3) A compact one-row summary table
4) The base models used by the stacker
5) The top meta-features (base-model class probabilities) the final
   LogisticRegression weights most, ranked by average |coef|
6) The top ORIGINAL CGM features (e.g., 'glucose_spike') used by the stacker.
   (We set passthrough=True so the final estimator sees preprocessed CGM features.)
"""

# ------------------------------------------------------------------#
# 1) FILE LOCATIONS – EDIT THESE                                    #
# ------------------------------------------------------------------#
CGM_CSV_PATH     = r"C:\Users\sneez\Desktop\Wearable\enhanced_cgm_features_with_labels.csv"
PARTICIPANTS_TSV = r"C:\Users\sneez\Desktop\OMOPStudyGroup\participants.label_agree.all_three.tsv"
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
from sklearn.svm            import SVC
from sklearn.ensemble       import (
    RandomForestClassifier,
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
    verbose_feature_names_out=False,  # keeps original names
)


# ------------------------------------------------------------------#
# 5) DEFINE ONLY THE STACKING ENSEMBLE (passthrough=True)           #
# ------------------------------------------------------------------#
stacking_estimator = StackingClassifier(
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
    passthrough=True  # <<— include original (preprocessed) CGM features in the meta-learner
)

# Pretty "mean ± std" string
def pm(mean, std, decimals=4):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

# Simple ASCII table
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

def show_metrics(model_name: str, fold: str, y_true_int, y_pred_int):
    y_true = decode(y_true_int)
    y_pred = decode(y_pred_int)
    acc    = accuracy_score(y_true, y_pred)
    print(f"{model_name:<18} | Fold {fold:<2} | Accuracy: {acc:.4f}")
    print(classification_report(
        y_true, y_pred, target_names=label_order, digits=4
    ))
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=label_order),
        index=[f"True {c}" for c in label_order],
        columns=[f"Pred {c}" for c in label_order],
    )
    print("Confusion matrix:")
    print(cm)
    print("-" * 80)


# ------------------------------------------------------------------#
# 7) TRAIN & 10-FOLD EVALUATE (STACKING ONLY) + SUMMARY             #
# ------------------------------------------------------------------#
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

name = "StackingEnsemble"
estimator = stacking_estimator

print(f"\n=== {name} (10-fold CV) ===")
fold_accs, fold_f1s, fold_aucs = [], [], []

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_all, y_all), start=1):
    X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
    y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]
    
    pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    # per-fold accuracy & F1 (macro)
    fold_accs.append(accuracy_score(y_test, y_pred))
    fold_f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
    
    # per-fold AUC-ROC (macro OVR)
    try:
        proba = pipe.predict_proba(X_test)
        auc = roc_auc_score(
            y_test,
            proba,
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        auc = np.nan
    fold_aucs.append(auc)

    # detailed per-fold printout
    show_metrics(name, str(fold_idx), y_test, y_pred)

mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
mean_f1,  std_f1  = np.mean(fold_f1s),  np.std(fold_f1s)
mean_auc, std_auc = np.nanmean(fold_aucs), np.nanstd(fold_aucs)

print(f"{name:<18} | Mean Acc: {mean_acc:.4f} ± {std_acc:.4f} | "
      f"Mean F1(macro): {mean_f1:.4f} ± {std_f1:.4f} | "
      f"Mean AUC(macro): {mean_auc:.4f} ± {std_auc:.4f}")

# compact one-row table
print("\n")
print_models_table([(
    name,
    pm(mean_acc, std_acc, 4),
    pm(mean_f1,  std_f1,  4),
    pm(mean_auc, std_auc, 4)
)], class_name="CGM")


# ------------------------------------------------------------------#
# 8) STACKING ENSEMBLE INTROSPECTION                                #
# ------------------------------------------------------------------#
print("\n=== Stacking Ensemble: base models, top meta-features, top CGM features ===")

try:
    # Build & fit a single Stacking pipeline on ALL data for introspection
    stack_pipe = Pipeline([("prep", preprocessor), ("clf", stacking_estimator)])
    stack_pipe.fit(X_all, y_all)

    prep = stack_pipe.named_steps["prep"]
    clf  = stack_pipe.named_steps["clf"]  # the fitted StackingClassifier
    fe   = clf.final_estimator_

    # 8a) Print the base models the stacker uses
    print("\nBase estimators used in StackingEnsemble:")
    base_names = [n for n, _ in clf.estimators]  # constructor order
    for n in base_names:
        est = clf.named_estimators_[n]
        print(f" - {n}: {est.__class__.__name__}")

    # 8b) Build readable meta feature names in the same order Stacking feeds them
    classes_for_names = label_order  # ["healthy", "pre-diabetic", "diabetes"]
    meta_feature_names = [
        f"{est_name}_proba_{cls_name}"
        for est_name in base_names
        for cls_name in classes_for_names
    ]

    # 8c) Get ORIGINAL CGM feature names (after preprocessing)
    # Because we used verbose_feature_names_out=False and a numeric-only pipeline,
    # these will match your original column names like "glucose_spike"
    try:
        cgm_feature_names = prep.get_feature_names_out().tolist()
    except Exception:
        # Fallback: use the raw feature list if sklearn version lacks the method
        cgm_feature_names = feature_cols

    if not hasattr(fe, "coef_"):
        print("\nFinal estimator does not expose coefficients; cannot derive feature weights.")
    else:
        coefs = fe.coef_  # shape: (n_classes, n_meta_features + n_cgm_features)
        # Average absolute weight across the 3 one-vs-rest rows
        mean_abs = np.mean(np.abs(coefs), axis=0)

        # Split into meta vs CGM sections. By design, Stacking orders as [meta, X]
        meta_len = len(meta_feature_names)
        total_len = mean_abs.shape[0]
        cgm_len = total_len - meta_len
        if cgm_len != len(cgm_feature_names):
            print(f"\n[Note] Feature count mismatch (meta={meta_len}, total={total_len}, "
                  f"cgm_list={len(cgm_feature_names)}). Proceeding with best-effort mapping.")

        # --- Top meta-features ---
        top_k_meta = min(10, meta_len)
        top_idx_meta = np.argsort(mean_abs[:meta_len])[::-1][:top_k_meta]

        print(f"\nTop {top_k_meta} meta-features (base-model probabilities) "
              f"(by avg |coef| across classes):")
        for rank, j in enumerate(top_idx_meta, start=1):
            print(f"{rank:>2}. {meta_feature_names[j]:<30} |coef|avg={mean_abs[j]:.6f}")

        # --- Top ORIGINAL CGM features (this is what you asked for) ---
        top_k_cgm = min(20, cgm_len)  # adjust as you like
        cgm_start = meta_len
        cgm_scores = mean_abs[cgm_start:cgm_start + cgm_len]
        top_idx_cgm_local = np.argsort(cgm_scores)[::-1][:top_k_cgm]

        print(f"\nTop {top_k_cgm} ORIGINAL CGM features for the Stacking final estimator "
              f"(by avg |coef| across classes):")
        for rank, j_local in enumerate(top_idx_cgm_local, start=1):
            j_global = cgm_start + j_local
            fname = cgm_feature_names[j_local] if j_local < len(cgm_feature_names) else f"f{j_local}"
            print(f"{rank:>2}. {fname:<40} |coef|avg={mean_abs[j_global]:.6f}")

except Exception as e:
    print("\n[Warning] Could not introspect Stacking ensemble details.")
    print(f"Reason: {e}")
