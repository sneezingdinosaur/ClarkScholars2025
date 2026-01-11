#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Early (feature-level) fusion of four modality-specific feature sets:
  – Combines HRV, Clinical, CGM, and Wearable features into one unified feature matrix
  – Trains multiple classifiers (Logistic Regression, SVM, KNN, Random Forest, XGBoost)
    with probability calibration
  – Adds Stacking and Voting ensembles (using calibrated base learners)
  – Reports mean ± std for Accuracy, F1-macro, AUC-ROC macro/OvR, and Log-Loss
  – Computes permutation-based feature importance (last fold): Top-20 features and
    modality-level contributions (HRV/CLN/CGM/WRB)
"""

import re, warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")


# ───────────────────────── 1. DATA LOADERS ──────────────────────────
def load_hrv(csv_path):
    df = pd.read_csv(csv_path)
    feats = ['Rate','PR','QRSD','QT','QTc','P','QRS','T','SDNN','RMSSD','pNN50']
    df = df[['participant_id'] + feats].dropna()
    df['participant_id'] = df['participant_id'].astype(str)
    X = df[feats].copy()
    X.columns = [f"HRV:{c}" for c in X.columns]
    return df['participant_id'].tolist(), X


def load_clinical(meas_csv, part_tsv):
    def collapse(g):
        g = str(g).lower()
        if g.startswith('healthy'): return 'healthy'
        if 'pre_diabetes' in g or 'pre-diabetes' in g: return 'prediabetic'
        return 'diabetic'

    df_m = pd.read_csv(meas_csv)
    df_p = pd.read_csv(part_tsv, sep='\t')
    df_p['label'] = df_p['study_group'].apply(collapse)
    df_p = df_p.dropna(subset=['label'])
    df   = df_m.merge(df_p, left_on='person_id', right_on='participant_id')

    # ---- basic feature engineering (adds hba1c_cat) ----
    def eng(df):
        df = df.copy()
        df['bmi_cat'] = df['bmi'].apply(
            lambda b: np.nan if pd.isna(b) else
            (0 if b < 18.5 else 1 if b < 25 else 2 if b < 30 else 3)
        )
        df['bp_cat'] = df.apply(
            lambda r: np.nan if pd.isna(r.systolic_bp_avg_mmhg) or pd.isna(r.diastolic_bp_avg_mmhg)
            else (0 if r.systolic_bp_avg_mmhg < 120 and r.diastolic_bp_avg_mmhg < 80
                  else 1 if r.systolic_bp_avg_mmhg < 130 and r.diastolic_bp_avg_mmhg < 80
                  else 2 if (r.systolic_bp_avg_mmhg < 140 or r.diastolic_bp_avg_mmhg < 90)
                  else 3), axis=1
        )
        if 'hba1c_percent' in df.columns:
            df['hba1c_cat'] = df['hba1c_percent'].apply(
                lambda h: np.nan if pd.isna(h) else (0 if h < 5.7 else 1 if h < 6.5 else 2)
            )
        if 'glucose_mg_dl' in df.columns:
            df['glucose_cat'] = df['glucose_mg_dl'].apply(
                lambda g: np.nan if pd.isna(g) else (0 if g < 100 else 1 if g < 126 else 2)
            )
        if {'systolic_bp_avg_mmhg', 'diastolic_bp_avg_mmhg'}.issubset(df.columns):
            df['pulse_pressure'] = df.systolic_bp_avg_mmhg - df.diastolic_bp_avg_mmhg
            df['map']            = (df.systolic_bp_avg_mmhg + 2*df.diastolic_bp_avg_mmhg) / 3
        return df

    df = eng(df)

    # ---- selected raw columns (no winsorization) ----
    base = [
        "bmi","diastolic_bp_1_mmhg","systolic_bp_1_mmhg","diastolic_bp_2_mmhg",
        "systolic_bp_2_mmhg","height_cm","hip_circumference_cm","waist_circumference_cm",
        "weight_kg","waist_hip_ratio","heart_rate_1_bpm","heart_rate_2_bpm",
        "heart_rate_avg_bpm","systolic_bp_avg_mmhg","diastolic_bp_avg_mmhg",
        "hi","hba1c_percent","glucose_mg_dl","creatinine_mg_dl"
    ]
    eng_cols = ['bmi_cat','bp_cat','pulse_pressure','map','hba1c_cat','glucose_cat']
    cols = [c for c in base + eng_cols if c in df.columns]

    df_clean = df[cols].copy()
    df_clean['participant_id'] = df['participant_id'].astype(str)

    X = df_clean.drop(columns=['participant_id'])
    X.columns = [f"CLN:{c}" for c in X.columns]
    return df_clean['participant_id'].tolist(), X



def load_cgm(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=['label'])
    df['participant_id'] = df['participant_id'].astype(str)
    feats = [c for c in df.columns if c not in {'participant_id','label'}]
    X = df[feats].astype(float).copy()
    X.columns = [f"CGM:{c}" for c in X.columns]
    return df['participant_id'].tolist(), X


def load_wearable(data_dir, part_tsv):
    def map_status(g):
        g = str(g).lower()
        if 'healthy' in g: return 'healthy'
        if 'pre_diabetes' in g or 'pre-diabetes' in g: return 'prediabetic'
        return 'diabetic'

    dfp = pd.read_csv(part_tsv, sep='\t')
    dfp['status'] = dfp['study_group'].apply(map_status)
    dfp['participant_id'] = dfp['participant_id'].astype('str')

    dfs = []
    for p in Path(data_dir).iterdir():
        if p.is_dir() and p.name.startswith('participant_'):
            pid = p.name.replace('participant_','')
            f = p / 'master_features.csv'
            if f.exists() and pid in set(dfp.participant_id):
                tmp = pd.read_csv(f)
                tmp['participant_id'] = tmp.get('participant_id', pid)
                dfs.append(tmp)
    if not dfs:
        raise RuntimeError(f"No wearable feature files found under {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    df['participant_id'] = df['participant_id'].astype(str)
    if 'hr_mean' in df.columns:
        df = df[df['hr_mean'].between(40,200) | df['hr_mean'].isna()]
    feats = [c for c in df.columns if c not in {'participant_id','date','status'}]
    X = df[feats].select_dtypes(include=[np.number]).copy()
    X.columns = [f"WRB:{c}" for c in X.columns]
    return df['participant_id'].tolist(), X


# ───────────── 2. CORRELATION FILTER ─────────────
class CorrelationFilter:
    """Drop one of any pair of highly correlated features (|r| > threshold)."""
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.drop_idx = []
        self.keep_idx = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        corr = df.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.drop_idx = [i for i, col in enumerate(upper.columns) if any(upper[col] > self.threshold)]
        self.keep_idx = [i for i in range(df.shape[1]) if i not in self.drop_idx]
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if not self.drop_idx:
            return df.values if isinstance(X, pd.DataFrame) else df.to_numpy()
        kept = df.iloc[:, self.keep_idx]
        return kept.values if isinstance(X, pd.DataFrame) else kept.to_numpy()


# ───────────── 3. HELPERS ─────────────
def align_features(ids, Xdf, common):
    df = Xdf.copy()
    df['pid'] = [str(x) for x in ids]
    df = df.drop_duplicates('pid').set_index('pid')
    return df.loc[common].reset_index(drop=True)


def collapse_group(g):
    g = str(g).lower()
    if g.startswith('healthy'): return 'healthy'
    if 'pre_diabetes' in g or 'pre-diabetes' in g: return 'prediabetic'
    return 'diabetic'


def table(title, metrics, le):
    print(f"\n--- {title} (mean ± std over {len(metrics['acc'])} folds) ---")
    hdr = ["Accuracy","F1-macro","AUC-ROC macro/OvR","Log-Loss"]
    vals = [
        f"{np.mean(metrics['acc']):.4f} ± {np.std(metrics['acc']):.4f}",
        f"{np.mean(metrics['f1'] ):.4f} ± {np.std(metrics['f1'] ):.4f}",
        f"{np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}",
        f"{np.mean(metrics['ll'] ):.4f} ± {np.std(metrics['ll'] ):.4f}",
    ]
    print(pd.DataFrame([vals], columns=hdr).to_string(index=False))
    print("\nReport (last fold):")
    print(classification_report(
        metrics['y_true_last'], metrics['y_pred_last'],
        target_names=le.classes_, digits=3
    ))
    print("Confusion Matrix (last fold):")
    print(pd.DataFrame(
        confusion_matrix(metrics['y_true_last'], metrics['y_pred_last']),
        index=le.classes_, columns=le.classes_
    ))


def print_feature_importance(name, fitted_model, X_te_df, y_te, top_k=20):
    """
    Permutation importance on last fold test set (neg_log_loss):
      - Top-K individual features
      - Modality-level contribution (HRV/CLN/CGM/WRB)
    """
    print(f"\n>>> Permutation Feature Importance — {name} (last fold)")
    pi = permutation_importance(
        fitted_model, X_te_df, y_te,
        scoring='neg_log_loss',
        n_repeats=10, random_state=42, n_jobs=-1
    )
    importances = pd.Series(pi.importances_mean, index=X_te_df.columns)
    importances = importances.fillna(0.0).sort_values(ascending=False)

    top = importances.head(top_k)
    print("\nTop-20 features (by mean permutation importance):")
    print(pd.DataFrame({'feature': top.index, 'importance': top.values}).to_string(index=False))

    def modality_of(col):
        return col.split(':', 1)[0] if ':' in col else 'UNK'
    modality_sum = importances.groupby(importances.index.map(modality_of)).sum().sort_values(ascending=False)

    total = importances.sum()
    frac = (modality_sum / total).fillna(0.0)
    mod_table = pd.DataFrame({
        'modality': modality_sum.index,
        'total_importance': modality_sum.values,
        'fraction': (frac * 100).round(2).astype(str) + '%'
    })
    print("\nModality contribution (sum of feature importances):")
    print(mod_table.to_string(index=False))


# ───────────── 4. MAIN ─────────────
def main():
    # --- full absolute paths ---
    hrv_csv   = r"C:\Users\sneez\Downloads\TexasTechAI\0704andbefore\participant_data_with_hrv.csv"
    meas_csv  = r"C:\Users\sneez\Desktop\WearableTest\clean_clinical.csv"
    part_tsv  = r"C:\Users\sneez\Desktop\OrganizedAI-READI\DataFiles\participants_HB_Split.tsv"
    cgm_csv   = r"C:\Users\sneez\Desktop\Wearable\enhanced_cgm_features_with_labels.csv"
    wear_dir  = r"C:\Users\sneez\Desktop\Wearable\ml_ready_participants"

    # 1) load each modality
    ids1, X1_df = load_hrv(hrv_csv)
    ids2, X2_df = load_clinical(meas_csv, part_tsv)
    ids3, X3_df = load_cgm(cgm_csv)
    ids4, X4_df = load_wearable(wear_dir, part_tsv)

    # normalize IDs
    norm = lambda arr: [re.search(r'(\d+)$', x).group(1) if re.search(r'(\d+)$', x) else x for x in arr]
    ids1, ids3, ids4 = map(norm, (ids1, ids3, ids4))
    ids2 = [str(x) for x in ids2]

    # find common participants
    common = sorted(set(ids1) & set(ids2) & set(ids3) & set(ids4))
    if not common:
        raise RuntimeError("No overlapping participant_ids!")
    print(f"Found {len(common)} common participants across all 4 modalities.")

    # extract labels
    dfp = pd.read_csv(part_tsv, sep='\t')
    dfp['label'] = dfp['study_group'].apply(collapse_group)
    dfp = dfp.dropna(subset=['label'])
    dfp['participant_id'] = dfp['participant_id'].astype(str)
    y = dfp.set_index('participant_id').loc[common, 'label'].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # align features (keep DataFrames with named columns)
    X1 = align_features(ids1, X1_df, common)
    X2 = align_features(ids2, X2_df, common)
    X3 = align_features(ids3, X3_df, common)
    X4 = align_features(ids4, X4_df, common)

    # early-fusion: concatenate all modalities (columns keep modality prefixes)
    X_df = pd.concat([X1.reset_index(drop=True),
                      X2.reset_index(drop=True),
                      X3.reset_index(drop=True),
                      X4.reset_index(drop=True)], axis=1)

    # cross-validation
    N_SPLITS  = 10
    cv_outer  = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.3, random_state=42)

    # define base estimators
    base_defs = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM'               : SVC(probability=True, random_state=42),
        'KNN'               : KNeighborsClassifier(n_neighbors=5),
        'RandomForest'      : RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42),
        'XGBoost'           : xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            objective='multi:softprob', num_class=3,  # updated later if needed
            eval_metric='mlogloss', reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, n_jobs=-1
        )
    }

    n_classes = len(np.unique(y_enc))
    if hasattr(base_defs['XGBoost'], 'set_params'):
        base_defs['XGBoost'].set_params(num_class=n_classes)

    # Build pipelines for each base estimator, then wrap with calibration
    base_pipes = {}
    for name, clf in base_defs.items():
        base_pipes[name] = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale',  StandardScaler(with_mean=True, with_std=True)),
            ('corr',   CorrelationFilter(threshold=0.95)),
            ('clf',    clf)
        ])

    # Calibrated base learners (these will be used standalone AND inside ensembles)
    calibrated = {name: CalibratedClassifierCV(pipe, cv=5, method='sigmoid')
                  for name, pipe in base_pipes.items()}

    # Start models dict with calibrated single models
    models = dict(calibrated)

    # --- ENSEMBLES ---
    # 1) Stacking ensemble: use calibrated base learners; meta-learner is LR on OOF probs
    stack_estimators = [(name, calibrated[name]) for name in calibrated.keys()]
    meta_lr = LogisticRegression(max_iter=1000, random_state=42)
    models['Stacking'] = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=meta_lr,
        stack_method='predict_proba',
        passthrough=False,
        cv=5
    )

    # 2) Soft voting ensemble over calibrated base learners
    models['VotingSoft'] = VotingClassifier(
        estimators=stack_estimators,
        voting='soft'
    )

    # prepare storage
    metrics = {name: {'acc': [], 'f1': [], 'auc': [], 'll': []} for name in models}

    # placeholders for last-fold data/models for feature importance
    last_fold_models = {}
    X_te_last_df = None
    y_te_last = None

    # outer CV loop
    for fold, (tr, te) in enumerate(cv_outer.split(X_df.values, y_enc), start=1):
        print(f"\n{'='*20} FOLD {fold}/{N_SPLITS} {'='*20}")
        X_tr_df, X_te_df = X_df.iloc[tr].reset_index(drop=True), X_df.iloc[te].reset_index(drop=True)
        y_tr, y_te = y_enc[tr], y_enc[te]

        for name, model in models.items():
            model.fit(X_tr_df, y_tr)
            proba   = model.predict_proba(X_te_df)
            y_pred  = np.argmax(proba, axis=1)

            metrics[name]['acc'].append(accuracy_score(y_te, y_pred))
            metrics[name]['f1' ].append(f1_score(y_te, y_pred, average='macro'))
            metrics[name]['auc'].append(roc_auc_score(y_te, proba, multi_class='ovr', average='macro'))
            metrics[name]['ll' ].append(log_loss(y_te, proba))

            if fold == N_SPLITS:
                metrics[name]['y_true_last'] = y_te
                metrics[name]['y_pred_last'] = y_pred
                last_fold_models[name] = model

        if fold == N_SPLITS:
            X_te_last_df = X_te_df.copy()
            y_te_last = y_te.copy()

    # print final summaries
    print("\n" + "="*50)
    print(" " * 18 + "FINAL RESULTS")
    print("="*50)
    for name in models:
        table(name, metrics[name], le)

    # permutation-based feature importance on the last fold
    if X_te_last_df is not None and y_te_last is not None:
        print("\n" + "="*50)
        print(" " * 14 + "FEATURE IMPORTANCE (last fold)")
        print("="*50)
        for name, fitted in last_fold_models.items():
            try:
                print_feature_importance(name, fitted, X_te_last_df, y_te_last, top_k=20)
            except Exception as e:
                print(f"\n[Warning] Skipped feature importance for {name}: {e}")

if __name__ == "__main__":
    main()
