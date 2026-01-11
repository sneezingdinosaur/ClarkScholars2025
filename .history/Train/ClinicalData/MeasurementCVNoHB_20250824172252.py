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
