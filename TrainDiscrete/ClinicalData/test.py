#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diabetes-status prediction – Clinical-feature edition (no HbA1c, no hyperparameter tuning)
=========================================================================================
• Removes all HbA1c–related logic
• Baseline 10-fold CV for a suite of models
• Feature importance from base RandomForest
• NEW: compact table (Class | Model | Accuracy | F1-score (Macro) | AUC-ROC (Macro))
  with 10-fold CV mean ± std for each model
• NEW: Per-model Top-3 features by importance (averaged across CV folds)
"""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import make_scorer
from sklearn.base import clone  # <-- added

import xgboost as xgb
import lightgbm as lgb

# suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'


# 1) Load & collapse study groups
df_meas = pd.read_csv(r'C:\Users\sneez\Desktop\ClinicalDataTTU\BESTFILERN.csv')
df_part = pd.read_csv(
    r"C:\Users\sneez\Desktop\OrganizedAI-READI\DataFiles\participants_OMOP_HB_Overlap.tsv",
    sep="\t"
)

def collapse_group(g):
    if isinstance(g, str) and g.startswith("healthy"):
        return "healthy"
    if isinstance(g, str) and g.startswith("pre_diabetes"):
        return "prediabetic"
    if isinstance(g, str) and g.startswith("diabetes"):
        return "diabetic"
    return np.nan

df_part["label"] = df_part["study_group"].apply(collapse_group)
df_part = df_part.dropna(subset=["label"])

df = df_meas.merge(
    df_part,
    left_on="person_id",
    right_on="participant_id",
    how="inner"
)


# 2) Feature engineering (no HbA1c)
def create_engineered_features(df):
    df_eng = df.copy()

    # BMI category
    def bmi_cat(b):
        if pd.isna(b): return np.nan
        return 0 if b < 18.5 else 1 if b < 25 else 2 if b < 30 else 3
    df_eng['bmi_category'] = df_eng['bmi'].apply(bmi_cat)

    # BP category
    def bp_cat(sys, dia):
        if pd.isna(sys) or pd.isna(dia): return np.nan
        if sys < 120 and dia <  80: return 0
        if sys < 130 and dia <  80: return 1
        if sys < 140 or dia <  90: return 2
        return 3
    df_eng['bp_category'] = df_eng.apply(
        lambda r: bp_cat(r.get('systolic_bp_avg_mmhg'), r.get('diastolic_bp_avg_mmhg')),
        axis=1
    )

    # Glucose category (fasting)
    def glu_cat(g):
        if pd.isna(g): return np.nan
        return 0 if g < 100 else 1 if g < 126 else 2
    if 'glucose_mg_dl' in df_eng.columns:
        df_eng['glucose_category'] = df_eng['glucose_mg_dl'].apply(glu_cat)

    # eGFR & category
    def est_gfr(cr, age=50, male=True, black=False):
        g = 175 * (cr ** -1.154) * (age ** -0.203)
        if not male: g *= 0.742
        if black:    g *= 1.212
        return g
    if 'creatinine_mg_dl' in df_eng.columns:
        age_col = next((c for c in df_eng if 'age' in c.lower()), None)
        df_eng['estimated_gfr'] = (
            df_eng.apply(lambda r: est_gfr(r['creatinine_mg_dl'], r[age_col])
                         if age_col else est_gfr(r['creatinine_mg_dl']), axis=1)
        )
        def gfr_cat(g):
            return np.nan if pd.isna(g) else 0 if g >= 90 else 1 if g >= 60 else 2 if g >= 30 else 3
        df_eng['gfr_category'] = df_eng['estimated_gfr'].apply(gfr_cat)

    # Metabolic syndrome score
    cols = ['glucose_mg_dl','systolic_bp_avg_mmhg','waist_circumference_cm','bmi']
    if all(c in df_eng.columns for c in cols[:3]):
        def metab_score(g, s, w, b=None):
            if any(pd.isna(v) for v in [g,s,w]): return np.nan
            sc = (g>=100) + (s>=130) + (w>=88) + (b>=30 if b is not None else 0)
            return sc
        df_eng['metabolic_syndrome_score'] = df_eng.apply(
            lambda r: metab_score(r['glucose_mg_dl'], r['systolic_bp_avg_mmhg'],
                                  r['waist_circumference_cm'], r.get('bmi')), axis=1
        )

    # Pulse & MAP
    if 'systolic_bp_avg_mmhg' in df_eng and 'diastolic_bp_avg_mmhg' in df_eng:
        df_eng['pulse_pressure'] = (
            df_eng['systolic_bp_avg_mmhg'] - df_eng['diastolic_bp_avg_mmhg']
        )
        df_eng['mean_arterial_pressure'] = (
            df_eng['systolic_bp_avg_mmhg'] + 2*df_eng['diastolic_bp_avg_mmhg']
        ) / 3

    # Body shape & waist/height
    if all(c in df_eng for c in ['waist_circumference_cm','height_cm','bmi']):
        df_eng['waist_height_ratio'] = (
            df_eng['waist_circumference_cm'] / df_eng['height_cm']
        )
        df_eng['body_shape_index'] = (
            df_eng['waist_circumference_cm'] /
            (df_eng['bmi']**(2/3) * df_eng['height_cm']**0.5)
        )

    # Variabilities
    for pair, name in [
        (('systolic_bp_1_mmhg','systolic_bp_2_mmhg'),'systolic_bp_variability'),
        (('diastolic_bp_1_mmhg','diastolic_bp_2_mmhg'),'diastolic_bp_variability'),
        (('heart_rate_1_bpm','heart_rate_2_bpm'),'heart_rate_variability')
    ]:
        if all(c in df_eng for c in pair):
            df_eng[name] = (
                df_eng[pair[0]] - df_eng[pair[1]]
            ).abs()

    # Creatinine/BMI & cardio risk
    if all(c in df_eng for c in ['creatinine_mg_dl','bmi']):
        df_eng['creatinine_bmi_ratio'] = (
            df_eng['creatinine_mg_dl'] / df_eng['bmi']
        )
    if all(c in df_eng for c in ['glucose_mg_dl','systolic_bp_avg_mmhg']):
        df_eng['cardio_metabolic_risk'] = (
            df_eng['glucose_mg_dl']/100 + df_eng['systolic_bp_avg_mmhg']/120
        )/2

    return df_eng

df_eng = create_engineered_features(df)


# 3) Feature selection
base_cols = [
    "bmi","diastolic_bp_1_mmhg","systolic_bp_1_mmhg",
    "diastolic_bp_2_mmhg","systolic_bp_2_mmhg",
    "height_cm","hip_circumference_cm","waist_circumference_cm",
    "weight_kg","waist_hip_ratio",
    "heart_rate_1_bpm","heart_rate_2_bpm","heart_rate_avg_bpm",
    "systolic_bp_avg_mmhg","diastolic_bp_avg_mmhg",
    "glucose_mg_dl","creatinine_mg_dl"
]
eng_cols = [
    'bmi_category','bp_category','glucose_category',
    'estimated_gfr','gfr_category','metabolic_syndrome_score',
    'pulse_pressure','mean_arterial_pressure','body_shape_index',
    'waist_height_ratio','systolic_bp_variability',
    'diastolic_bp_variability','heart_rate_variability',
    'creatinine_bmi_ratio','cardio_metabolic_risk'
]
available = [c for c in base_cols+eng_cols if c in df_eng.columns]
print(f"Using {len(available)} features.\n")

X = df_eng[available]
y = df_eng["label"]


# 4) Outlier handling
def cap_outliers(df, t=1.5):
    dfc = df.copy()
    for col in dfc.select_dtypes(float).columns:
        q1,q3 = dfc[col].quantile([0.25,0.75])
        iqr = q3-q1
        dfc[col] = dfc[col].clip(q1 - t*iqr, q3 + t*iqr)
    return dfc

X_clean = cap_outliers(X)


# 5) Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_
print("Classes:", classes, "\n")


# 6) Preprocessing pipeline
def preprocessing_pipeline():
    return Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", RobustScaler()),
        ("selector", SelectKBest(f_classif, k='all'))
    ])


# 7) Define base models
models = {
    "LogisticRegression": Pipeline([
        ("prep", preprocessing_pipeline()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ))
    ]),
    "RandomForest": Pipeline([
        ("prep", preprocessing_pipeline()),
        ("clf", RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=42
        ))
    ]),
    "GradientBoosting": Pipeline([
        ("prep", preprocessing_pipeline()),
        ("clf", GradientBoostingClassifier(random_state=42))
    ]),  # supports feature_importances_
    "XGBoost": Pipeline([
        ("prep", preprocessing_pipeline()),
        ("clf", xgb.XGBClassifier(
            verbosity=0, n_jobs=-1, random_state=42
        ))
    ]),
    "LightGBM": Pipeline([
        ("prep", preprocessing_pipeline()),
        ("clf", lgb.LGBMClassifier(
            class_weight="balanced", verbose=-1, random_state=42
        ))
    ])
}

# Pretty "mean ± std"
def pm(mean, std, decimals=4):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

# ASCII table like the screenshot
def print_models_table(rows, class_name="Clinical-Data"):
    h1, h2, h3, h4, h5 = "Class", "Model", "Accuracy", "F1-score (Macro)", "AUC-ROC (Macro)"
    line = "+" + "-"*14 + "+" + "-"*18 + "+" + "-"*20 + "+" + "-"*22 + "+" + "-"*22 + "+"
    print(line)
    print(f"| {h1:<12} | {h2:<16} | {h3:<18} | {h4:<20} | {h5:<20} |")
    print(line)
    for model_name, acc_s, f1_s, auc_s in rows:
        print(f"| {class_name:<12} | {model_name:<16} | {acc_s:<18} | {f1_s:<20} | {auc_s:<20} |")
    print(line)

# ---- NEW: helpers for feature importances -----------------------------------
def _extract_importances(fitted_estimator):
    """
    Return a 1D numpy array of importances for the fitted estimator, or None if unsupported.
    - Tree models: feature_importances_
    - LogisticRegression: mean abs(coef_) across classes
    """
    if hasattr(fitted_estimator, "feature_importances_"):
        imp = getattr(fitted_estimator, "feature_importances_")
        if imp is not None:
            return np.asarray(imp).ravel()
    if hasattr(fitted_estimator, "coef_"):
        coef = np.asarray(fitted_estimator.coef_)
        if coef.ndim == 1:
            return np.abs(coef)
        return np.mean(np.abs(coef), axis=0)
    return None

def _print_topk(name, feature_names, importances, k=3):
    if importances is None or feature_names is None or len(importances) != len(feature_names):
        print(f"[{name}] Top features: N/A (no importances or mismatch).")
        return
    order = np.argsort(importances)[::-1][:k]
    print(f"[{name}] Top {k} features by importance (CV-avg):")
    for r, idx in enumerate(order, 1):
        print(f"  {r}. {feature_names[idx]} : {importances[idx]:.6f}")
# -----------------------------------------------------------------------------


# 8) Baseline 10-fold CV (Accuracy, F1-macro, AUC-ROC macro OvR)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = {}
table_rows = []

# scorers for cross_validate
scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',         # built-in alias
    'auc_macro_ovr': 'roc_auc_ovr'  # built-in multiclass ROC AUC (macro by default)
}

print("=== BASELINE RESULTS (10-Fold CV) ===\n")
for name, base_pipe in models.items():
    # per-fold metrics via cross_validate (fast, parallel)
    cvres = cross_validate(
        base_pipe, X_clean, y_enc, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False
    )
    acc_mean, acc_std = cvres["test_accuracy"].mean(), cvres["test_accuracy"].std()
    f1_mean,  f1_std  = cvres["test_f1_macro"].mean(), cvres["test_f1_macro"].std()
    auc_mean, auc_std = cvres["test_auc_macro_ovr"].mean(), cvres["test_auc_macro_ovr"].std()

    # pooled predictions for a full report
    preds = cross_val_predict(base_pipe, X_clean, y_enc, cv=skf, n_jobs=-1)
    results[name] = {
        'acc_mean': acc_mean, 'acc_std': acc_std,
        'f1_mean': f1_mean,   'f1_std':  f1_std,
        'auc_mean': auc_mean, 'auc_std': auc_std,
        'preds': preds
    }

    print(f"{name}: Acc {acc_mean:.4f} ± {acc_std:.4f} | "
          f"F1(macro) {f1_mean:.4f} ± {f1_std:.4f} | "
          f"AUC(macro) {auc_mean:.4f} ± {auc_std:.4f}")
    print(classification_report(y_enc, preds, target_names=classes, zero_division=0))

    # ---- NEW: compute CV-averaged feature importances per model -------------
    imp_sum = None
    imp_count = 0
    feat_names = np.array(available)  # unchanged by KNNImputer/RobustScaler/SelectKBest(k='all')
    for tr_idx, te_idx in skf.split(X_clean, y_enc):
        X_tr, y_tr = X_clean.iloc[tr_idx], y_enc[tr_idx]
        pipe = clone(base_pipe)
        pipe.fit(X_tr, y_tr)
        imp = _extract_importances(pipe.named_steps['clf'])
        if imp is not None and len(imp) == len(feat_names):
            if imp_sum is None:
                imp_sum = np.zeros_like(imp, dtype=float)
            imp_sum += np.asarray(imp, dtype=float)
            imp_count += 1

    if imp_count > 0:
        imp_avg = imp_sum / float(imp_count)
        _print_topk(name, feat_names, imp_avg, k=3)
    else:
        print(f"[{name}] Top features: N/A (no importances available).")
    # ------------------------------------------------------------------------

    # add row for compact table
    table_rows.append((
        name,
        pm(acc_mean, acc_std, 4),
        pm(f1_mean,  f1_std,  4),
        pm(auc_mean, auc_std, 4)
    ))


# 9) Feature importance from base RandomForest
print("\n=== FEATURE IMPORTANCE (RandomForest – full fit) ===\n")
rf = models["RandomForest"]
rf.fit(X_clean, y_enc)
importances = rf.named_steps['clf'].feature_importances_
feat_imp = pd.Series(importances, index=available).sort_values(ascending=False)
print(feat_imp.head(15))
# Also show Top-3 explicitly (full-fit)
print("\n[RandomForest] Top 3 features (full fit):")
for i, (fname, val) in enumerate(feat_imp.head(3).items(), start=1):
    print(f"  {i}. {fname} : {val:.6f}")


# 10) Summary tables
print("\n=== MODEL COMPARISON (by Accuracy) ===")
print(f"{'Model':<20}{'Mean Acc':<12}{'Std Dev':<10}")
print("-"*42)
for name, res in sorted(results.items(), key=lambda x: x[1]['acc_mean'], reverse=True):
    print(f"{name:<20}{res['acc_mean']:<12.4f}{res['acc_std']:<10.4f}")

# final compact table (screenshot style)
print("\n")
print_models_table(table_rows, class_name="Clinical-Data")
