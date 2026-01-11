#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Late (decision-level) fusion of four modality-specific models:

  1. Soft Voting
  2. Per-Class Threshold Tuning
  3. Stacking   (Logistic-Regression meta-learner)
  4. Probability Calibration for all base learners

Added in this version
─────────────────────
• Top-5 meta-feature importance per predicted class (|coef|).
• Modality-importance two ways:
      – SUM of the top-5 coefficients (for quick reading)
      – SUM over *all* meta-features (full picture).
• Mean ± Std tables for Accuracy, F1-macro, AUC-ROC macro/OvR, Log-Loss.
• Per-fold metrics printed and saved to per_fold_metrics.csv
• NEW: Generates participants.tsv listing the participants used for training (the 4-way overlap) and their label
"""

import re, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedShuffleSplit, StratifiedKFold,
    train_test_split, cross_val_predict
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

warnings.filterwarnings("ignore")

# ───────────────────────── 1. DATA LOADERS ──────────────────────────
def load_hrv(csv_path):
    df = pd.read_csv(csv_path)
    feats = ['Rate','PR','QRSD','QT','QTc','P','QRS','T','SDNN','RMSSD','pNN50']
    df = df[['participant_id'] + feats].dropna()
    df['participant_id'] = df['participant_id'].astype(str)
    return df['participant_id'].tolist(), df[feats].copy()

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

    # basic feature-engineering ✂︎ (same as earlier) -----------------
    def eng(df):
        df = df.copy()
        df['bmi_cat'] = df['bmi'].apply(
            lambda b: np.nan if pd.isna(b) else
            (0 if b<18.5 else 1 if b<25 else 2 if b<30 else 3))
        df['bp_cat'] = df.apply(
            lambda r: np.nan if pd.isna(r.systolic_bp_avg_mmhg) or pd.isna(r.diastolic_bp_avg_mmhg)
            else (0 if r.systolic_bp_avg_mmhg<120 and r.diastolic_bp_avg_mmhg<80
                  else 1 if r.systolic_bp_avg_mmhg<130 and r.diastolic_bp_avg_mmhg<80
                  else 2 if (r.systolic_bp_avg_mmhg<140 or r.diastolic_bp_avg_mmhg<90)
                  else 3), axis=1)
        if 'glucose_mg_dl' in df:
            df['glucose_cat'] = df['glucose_mg_dl'].apply(
                lambda g: np.nan if pd.isna(g) else (0 if g<100 else 1 if g<126 else 2))
        if {'systolic_bp_avg_mmhg','diastolic_bp_avg_mmhg'}.issubset(df):
            df['pulse_pressure'] = df.systolic_bp_avg_mmhg - df.diastolic_bp_avg_mmhg
            df['map']            = (df.systolic_bp_avg_mmhg + 2*df.diastolic_bp_avg_mmhg)/3
        return df
    df = eng(df)

    base = ["bmi","diastolic_bp_1_mmhg","systolic_bp_1_mmhg","diastolic_bp_2_mmhg",
            "systolic_bp_2_mmhg","height_cm","hip_circumference_cm","waist_circumference_cm",
            "weight_kg","waist_hip_ratio","heart_rate_1_bpm","heart_rate_2_bpm",
            "heart_rate_avg_bpm","systolic_bp_avg_mmhg","diastolic_bp_avg_mmhg",
            "hi","glucose_mg_dl","creatinine_mg_dl"]
    eng_cols = ['bmi_cat','bp_cat','pulse_pressure','map','glucose_cat']
    cols = [c for c in base+eng_cols if c in df.columns]

    # No winsorization: use raw selected columns
    df_clean = df[cols].copy()
    df_clean['participant_id'] = df['participant_id'].astype(str)
    return df_clean['participant_id'].tolist(), df_clean.drop(columns=['participant_id'])

def load_cgm(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=['label'])
    df['participant_id'] = df['participant_id'].astype(str)
    feats = [c for c in df.columns if c not in {'participant_id','label'}]
    return df['participant_id'].tolist(), df[feats].astype(float)

def load_wearable(data_dir, part_tsv):
    def map_status(g):
        g=str(g).lower()
        if 'healthy' in g: return 'healthy'
        if 'pre_diabetes' in g or 'pre-diabetes' in g: return 'prediabetic'
        return 'diabetic'
    dfp = pd.read_csv(part_tsv, sep='\t')
    dfp['status'] = dfp['study_group'].apply(map_status)
    dfp['participant_id'] = dfp['participant_id'].astype(str)
    dfs=[]
    for p in Path(data_dir).iterdir():
        if p.is_dir() and p.name.startswith('participant_'):
            pid=p.name.replace('participant_','')
            f=p/'master_features.csv'
            if f.exists() and pid in set(dfp.participant_id):
                tmp=pd.read_csv(f); tmp['participant_id']=tmp.get('participant_id',pid)
                dfs.append(tmp)
    df=pd.concat(dfs,ignore_index=True)
    df['participant_id']=df['participant_id'].astype(str)
    if 'hr_mean' in df.columns:
        df=df[df['hr_mean'].between(40,200)|df['hr_mean'].isna()]
    feats=[c for c in df.columns if c not in {'participant_id','date','status'}]
    return df['participant_id'].tolist(), df[feats].select_dtypes(include=[np.number])

# ───────────── 2. SIMPLE CORRELATION FILTER ─────────────
class CorrelationFilter:
    def __init__(self, threshold=0.95): self.threshold=threshold; self.drop_idx=[]
    def fit(self,X,y=None):
        corr=pd.DataFrame(X).corr().abs()
        upper=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
        self.drop_idx=[i for i,col in enumerate(upper.columns) if any(upper[col]>self.threshold)]
        return self
    def transform(self,X): return pd.DataFrame(X).drop(pd.DataFrame(X).columns[self.drop_idx],axis=1).values

# ───────────── 3. SMALL HELPERS ─────────────
def align_features(ids,Xdf,common):
    df=Xdf.copy(); df['pid']=[str(x) for x in ids]
    return df.drop_duplicates('pid').set_index('pid').loc[common].reset_index(drop=True).values

def collapse_group(g):
    g=str(g).lower()
    if g.startswith('healthy'): return 'healthy'
    if 'pre_diabetes' in g or 'pre-diabetes' in g: return 'prediabetic'
    return 'diabetic'

def table(title,metrics,le):
    print(f"\n--- {title} (mean ± std over 10 folds) ---")
    hdr=["Accuracy","F1-macro","AUC-ROC macro/OvR","Log-Loss"]
    vals=[f"{np.mean(metrics['acc']):.4f} ± {np.std(metrics['acc']):.4f}",
          f"{np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}",
          f"{np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}",
          f"{np.mean(metrics['ll']):.4f} ± {np.std(metrics['ll']):.4f}"]
    print(pd.DataFrame([vals],columns=hdr).to_string(index=False))
    print("\nReport (last fold):")
    print(classification_report(metrics['y_true_last'],metrics['y_pred_last'],
                                target_names=le.classes_,digits=3))
    print("Confusion Matrix (last fold):")
    print(pd.DataFrame(confusion_matrix(metrics['y_true_last'],
                                        metrics['y_pred_last']),
                       index=le.classes_,columns=le.classes_))

# ───────────── 4. MAIN ─────────────
def main():
    # ---- Output artifacts
    OUTPUT_CSV_METRICS = "per_fold_metrics.csv"
    OUTPUT_TSV_PARTICIPANTS = "participants.tsv"  # NEW

    hrv_csv   = r"C:\Users\sneez\Downloads\TexasTechAI\0704andbefore\participant_data_with_hrv.csv"
    meas_csv  = r"C:\Users\sneez\Desktop\WearableTest\clean_clinical.csv"
    part_tsv  = r"C:\Users\sneez\Desktop\OrganizedAI-READI\DataFiles\participants_OMOP_HB_Overlap.tsv"
    cgm_csv   = r"C:\Users\sneez\Desktop\Wearable\enhanced_cgm_features_with_labels.csv"
    wear_dir  = r"C:\Users\sneez\Desktop\Wearable\ml_ready_participants"

    ids1,X1_df = load_hrv(hrv_csv)
    ids2,X2_df = load_clinical(meas_csv,part_tsv)
    ids3,X3_df = load_cgm(cgm_csv)
    ids4,X4_df = load_wearable(wear_dir,part_tsv)

    norm = lambda arr: [re.search(r'(\d+)$',x).group(1) if re.search(r'(\d+)$',x) else x for x in arr]
    ids1,ids3,ids4 = map(norm,(ids1,ids3,ids4))
    ids2=[str(x) for x in ids2]

    common=sorted(set(ids1)&set(ids2)&set(ids3)&set(ids4))
    if not common: raise RuntimeError("No overlapping participant_ids!")
    print(f"Found {len(common)} common participants across all 4 modalities.")

    dfp=pd.read_csv(part_tsv,sep='\t'); dfp['label']=dfp['study_group'].apply(collapse_group)
    dfp=dfp.dropna(subset=['label']); dfp['participant_id']=dfp['participant_id'].astype(str)
    y=dfp.set_index('participant_id').loc[common,'label'].values
    le=LabelEncoder(); y_enc=le.fit_transform(y)

    # ─── NEW: Save the participants used for training (the 4-way overlap) ───
    participants_df = pd.DataFrame({"participant_id": common, "label": y})
    participants_df.to_csv(OUTPUT_TSV_PARTICIPANTS, sep='\t', index=False)
    print(f"Saved participants used for training to: {OUTPUT_TSV_PARTICIPANTS}")
    print(participants_df.head().to_string(index=False))

    X1=align_features(ids1,X1_df,common)
    X2=align_features(ids2,X2_df,common)
    X3=align_features(ids3,X3_df,common)
    X4=align_features(ids4,X4_df,common)

    N_SPLITS=10
    cv_outer=StratifiedShuffleSplit(n_splits=N_SPLITS,test_size=0.3,random_state=42)

    metrics_sv= {'acc':[],'f1':[],'auc':[],'ll':[]}
    metrics_thr={'acc':[],'f1':[],'auc':[],'ll':[]}
    metrics_st= {'acc':[],'f1':[],'auc':[],'ll':[]}
    meta_coeffs=[]

    # accumulate per-fold rows for CSV
    per_fold_rows = []

    for fold,(tr,te) in enumerate(cv_outer.split(common,y_enc),1):
        print(f"\n{'='*20} FOLD {fold}/{N_SPLITS} {'='*20}")
        X1_tr,X1_te=X1[tr],X1[te];  X2_tr,X2_te=X2[tr],X2[te]
        X3_tr,X3_te=X3[tr],X3[te];  X4_tr,X4_te=X4[tr],X4[te]
        y_tr,y_te =y_enc[tr],y_enc[te]

        svm_base=Pipeline([('scale',StandardScaler()),
                           ('svc',SVC(probability=True,random_state=42))])
        rf_base = Pipeline([
            ('impute', KNNImputer(n_neighbors=5)),
            ('scale', RobustScaler()),
            ('sel', SelectKBest(f_classif, k='all')),
            ('rf', RandomForestClassifier(
                n_jobs=-1,
                random_state=42
            ))
        ])
        lr_base =Pipeline([('scale',StandardScaler()),
                           ('lr',LogisticRegression(max_iter=1000,random_state=42))])
        xgb_base=Pipeline([('impute',SimpleImputer(strategy='median')),
                           ('scale',StandardScaler()),
                           ('corr',CorrelationFilter(0.95)),
                           ('xgb',xgb.XGBClassifier(n_estimators=200,max_depth=8,
                                                    learning_rate=0.08,subsample=0.85,
                                                    colsample_bytree=0.85,
                                                    objective='multi:softprob',
                                                    num_class=len(le.classes_),
                                                    eval_metric='mlogloss',
                                                    reg_alpha=0.1,reg_lambda=0.1,
                                                    random_state=42))])

        print("Calibrating base learners…")
        cal_svm=CalibratedClassifierCV(svm_base,cv=5,method='sigmoid').fit(X1_tr,y_tr)
        cal_rf =CalibratedClassifierCV(rf_base, cv=5,method='sigmoid').fit(X2_tr,y_tr)
        cal_lr =CalibratedClassifierCV(lr_base, cv=5,method='sigmoid').fit(X3_tr,y_tr)
        cal_xgb=CalibratedClassifierCV(xgb_base,cv=5,method='sigmoid').fit(X4_tr,y_tr)

        probas_te=[cal_svm.predict_proba(X1_te),
                   cal_rf.predict_proba(X2_te),
                   cal_lr.predict_proba(X3_te),
                   cal_xgb.predict_proba(X4_te)]
        avg_proba=np.mean(np.stack(probas_te,axis=0),axis=0)

        # Soft voting
        y_pred_sv=np.argmax(avg_proba,1)
        sv_acc=accuracy_score(y_te,y_pred_sv)
        sv_f1 =f1_score(y_te,y_pred_sv,average='macro')
        sv_auc=roc_auc_score(y_te,avg_proba,multi_class='ovr',average='macro')
        sv_ll =log_loss(y_te,avg_proba)
        metrics_sv['acc'].append(sv_acc); metrics_sv['f1'].append(sv_f1)
        metrics_sv['auc'].append(sv_auc); metrics_sv['ll'].append(sv_ll)

        print(f"[Fold {fold}] Soft Voting        -> Acc: {sv_acc:.4f} | F1: {sv_f1:.4f} | AUC: {sv_auc:.4f} | LogLoss: {sv_ll:.4f}")
        per_fold_rows.append({
            "fold": fold, "method": "soft_voting",
            "accuracy": sv_acc, "f1_macro": sv_f1, "auc_roc_macro_ovr": sv_auc, "log_loss": sv_ll
        })

        # Thresholded voting
        X1_tr2,X1_val,X2_tr2,X2_val,X3_tr2,X3_val,X4_tr2,X4_val,y_tr2,y_val = \
            train_test_split(X1_tr,X2_tr,X3_tr,X4_tr,y_tr,test_size=0.3,
                             stratify=y_tr,random_state=42)
        cal_svm_thr=CalibratedClassifierCV(svm_base,cv=5).fit(X1_tr2,y_tr2)
        cal_rf_thr =CalibratedClassifierCV(rf_base, cv=5).fit(X2_tr2,y_tr2)
        cal_lr_thr =CalibratedClassifierCV(lr_base, cv=5).fit(X3_tr2,y_tr2)
        cal_xgb_thr=CalibratedClassifierCV(xgb_base,cv=5).fit(X4_tr2,y_tr2)
        val_probs=np.mean(np.stack([
            cal_svm_thr.predict_proba(X1_val),
            cal_rf_thr.predict_proba(X2_val),
            cal_lr_thr.predict_proba(X3_val),
            cal_xgb_thr.predict_proba(X4_val)],axis=0),axis=0)

        thresholds={}
        for c in range(val_probs.shape[1]):
            best_t,best_f1=0.5,-1
            for t in np.linspace(0.05,0.95,19):
                f1=f1_score((y_val==c).astype(int),(val_probs[:,c]>=t).astype(int))
                if f1>best_f1:
                    best_t,best_f1=t,f1
            thresholds[c]=best_t

        y_pred_thr=[]
        for p in avg_proba:
            cand=[c for c,pr in enumerate(p) if pr>=thresholds[c]]
            y_pred_thr.append(max(cand,key=lambda c:p[c]) if cand else np.argmax(p))
        y_pred_thr=np.array(y_pred_thr)
        thr_acc=accuracy_score(y_te,y_pred_thr)
        thr_f1 =f1_score(y_te,y_pred_thr,average='macro')
        thr_auc=roc_auc_score(y_te,avg_proba,multi_class='ovr',average='macro')
        thr_ll =log_loss(y_te,avg_proba)
        metrics_thr['acc'].append(thr_acc); metrics_thr['f1'].append(thr_f1)
        metrics_thr['auc'].append(thr_auc); metrics_thr['ll'].append(thr_ll)

        print(f"[Fold {fold}] Thresholded Voting -> Acc: {thr_acc:.4f} | F1: {thr_f1:.4f} | AUC: {thr_auc:.4f} | LogLoss: {thr_ll:.4f}")
        per_fold_rows.append({
            "fold": fold, "method": "thresholded_soft_voting",
            "accuracy": thr_acc, "f1_macro": thr_f1, "auc_roc_macro_ovr": thr_auc, "log_loss": thr_ll
        })

        # Stacking
        cv_inner=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        oof_svm=cross_val_predict(cal_svm,X1_tr,y_tr,cv=cv_inner,method='predict_proba',n_jobs=-1)
        oof_rf =cross_val_predict(cal_rf ,X2_tr,y_tr,cv=cv_inner,method='predict_proba',n_jobs=-1)
        oof_lr =cross_val_predict(cal_lr ,X3_tr,y_tr,cv=cv_inner,method='predict_proba',n_jobs=-1)
        oof_xgb=cross_val_predict(cal_xgb,X4_tr,y_tr,cv=cv_inner,method='predict_proba',n_jobs=-1)
        X_meta_tr=np.hstack([oof_svm,oof_rf,oof_lr,oof_xgb])
        meta_clf=LogisticRegression(max_iter=1000,random_state=42).fit(X_meta_tr,y_tr)
        meta_coeffs.append(meta_clf.coef_)
        meta_te=np.hstack(probas_te); meta_proba=meta_clf.predict_proba(meta_te)
        y_pred_st=np.argmax(meta_proba,1)
        st_acc=accuracy_score(y_te,y_pred_st)
        st_f1 =f1_score(y_te,y_pred_st,average='macro')
        st_auc=roc_auc_score(y_te,meta_proba,multi_class='ovr',average='macro')
        st_ll =log_loss(y_te,meta_proba)
        metrics_st['acc'].append(st_acc); metrics_st['f1'].append(st_f1)
        metrics_st['auc'].append(st_auc); metrics_st['ll'].append(st_ll)

        print(f"[Fold {fold}] Stacking            -> Acc: {st_acc:.4f} | F1: {st_f1:.4f} | AUC: {st_auc:.4f} | LogLoss: {st_ll:.4f}")
        per_fold_rows.append({
            "fold": fold, "method": "stacking",
            "accuracy": st_acc, "f1_macro": st_f1, "auc_roc_macro_ovr": st_auc, "log_loss": st_ll
        })

        if fold==N_SPLITS:
            metrics_sv['y_true_last'],metrics_sv['y_pred_last']=y_te,y_pred_sv
            metrics_thr['y_true_last'],metrics_thr['y_pred_last']=y_te,y_pred_thr
            metrics_st['y_true_last'],metrics_st['y_pred_last']=y_te,y_pred_st

    # ─── Save per-fold metrics to CSV ───
    per_fold_df = pd.DataFrame(per_fold_rows).sort_values(["fold","method"]).reset_index(drop=True)
    per_fold_df.to_csv(OUTPUT_CSV_METRICS, index=False)
    print(f"\nSaved per-fold metrics to: {OUTPUT_CSV_METRICS}")
    print("\nPer-fold metrics (first few rows):")
    print(per_fold_df.head().to_string(index=False))

    # ─── results ───
    print("\n"+"="*50+"\n"+" "*18+"FINAL RESULTS"+"\n"+"="*50)
    table("Soft Voting",metrics_sv,le)
    table("Thresholded Soft Voting",metrics_thr,le)
    table("Stacking",metrics_st,le)

    # ─── feature / modality importance ───
    n_classes=len(le.classes_); modalities=['HRV_SVM','Clinical_RF','CGM_LR','Wearable_XGB']
    avg_coef=np.mean(np.abs(np.stack(meta_coeffs,axis=0)),axis=0)

    print("\n"+"-"*60+"\nStacking Meta-Learner – Top-5 Features per Class\n"+"-"*60)
    mod_top_tot=defaultdict(float)
    for c,cls in enumerate(le.classes_):
        coefs=avg_coef[c]; top=np.argsort(coefs)[-5:][::-1]
        rows=[]
        for rk,idx in enumerate(top,1):
            mod=modalities[idx//n_classes]; imp=coefs[idx]
            mod_top_tot[mod]+=imp
            rows.append((rk,f"{mod} → prob({le.classes_[idx%n_classes]})",imp))
        print(f"\nClass: {cls}")
        print(pd.DataFrame(rows,columns=['rank','meta_feature','abs_coef']).to_string(index=False))

    print("\n"+"-"*60+"\nModality Importance (sum of |coef| across top-5 lists)\n"+"-"*60)
    tot=sum(mod_top_tot.values()) or 1.0
    print(pd.DataFrame(sorted([(m,v,f"{v/tot:.2%}") for m,v in mod_top_tot.items()],
                              key=lambda t:t[1],reverse=True),
                       columns=['modality','total_abs_coef','fraction']).to_string(index=False))

    # -------- FULL-MATRIX modality importance --------
    print("\n"+"-"*60+"\nModality Importance (all coefficients)\n"+"-"*60)
    mod_full=defaultdict(float)
    for idx in range(avg_coef.shape[1]):
        mod=modalities[idx//n_classes]; mod_full[mod]+=avg_coef[:,idx].sum()
    tot=sum(mod_full.values()) or 1.0
    print(pd.DataFrame(sorted([(m,v,f"{v/tot:.2%}") for m,v in mod_full.items() ],
                              key=lambda t:t[1],reverse=True),
                       columns=['modality','total_abs_coef','fraction']).to_string(index=False))

if __name__=="__main__":
    main()
