#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diabetes prediction with wearable data (extended models, no SMOTE, full pipeline).
Now also prints a compact leaderboard-style table with:
Class | Model | Accuracy | F1-score (Macro) | AUC-ROC (Macro)
(metrics are 10-fold CV mean ± std).
"""

import warnings
import logging

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='X does not have valid feature names, but LGBMClassifier was fitted with feature names'
)
logging.getLogger('lightgbm').setLevel(logging.ERROR)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    make_scorer, f1_score
)

import xgboost as xgb
import lightgbm as lgb


# mapping from integer label to name
LABEL_NAME_MAP = {0: 'healthy', 1: 'pre-diabetic', 2: 'diabetic'}

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.to_drop_columns_ = None

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_columns_ = [c for c in upper.columns if any(upper[c] > self.threshold)]
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return df.drop(columns=self.to_drop_columns_, errors='ignore')


class EnhancedDiabetesPipeline:
    def __init__(self):
        self.participant_labels = {}
        self.df = None
        self.models = {}
        self.results = {}

    def load_participant_labels(self, labels_path: str):
        df = pd.read_csv(labels_path, sep=None, engine='python')
        if 'participant_id' not in df.columns or 'study_group' not in df.columns:
            raise ValueError(
                f"Expected columns 'participant_id' and 'study_group' in labels file, found: {list(df.columns)}"
            )
        def map_status(g):
            g = str(g).lower()
            if 'healthy' in g:
                return 0
            if 'pre_diabetes' in g:
                return 1
            return 2

        df['diabetes_status'] = df['study_group'].apply(map_status)
        df = df.dropna(subset=['diabetes_status'])
        df['participant_id'] = df['participant_id'].astype(str)
        self.participant_labels = dict(zip(df['participant_id'], df['diabetes_status']))
        return self.participant_labels

    def load_all_participants_data(self, data_directory: str, labels_path: str):
        self.load_participant_labels(labels_path)
        data_dir = Path(data_directory)
        all_data = []
        for d in data_dir.iterdir():
            if not d.is_dir() or not d.name.startswith('participant_'):
                continue
            pid = d.name.replace('participant_', '')
            fp = d / 'master_features.csv'
            if not fp.exists():
                continue
            tmp = pd.read_csv(fp)
            tmp['participant_id'] = tmp.get('participant_id', pid)
            if pid in self.participant_labels:
                tmp['diabetes_status'] = self.participant_labels[pid]
                all_data.append(tmp)

        if not all_data:
            raise RuntimeError("No participant data loaded!")
        self.df = pd.concat(all_data, ignore_index=True)
        return self.df

    def filter_high_quality_data(self, min_participant_coverage: float = 0.3, min_feature_coverage: float = 0.5):
        df = self.df.copy()
        keys = [k for k in ['hr_mean', 'spo2_mean', 'total_sleep_time'] if k in df.columns]
        if 'hr_mean' in keys:
            df = df[df['hr_mean'].between(40, 200) | df['hr_mean'].isna()]
        good_pids = [
            pid for pid, sub in df.groupby('participant_id')
            if sub[keys].notna().any(axis=1).mean() >= min_participant_coverage
        ]
        df = df[df['participant_id'].isin(good_pids)]
        cov = df[keys].notna().sum(axis=1) / len(keys) if keys else np.ones(len(df))
        self.df = df[cov >= min_feature_coverage].reset_index(drop=True)
        return self.df

    def advanced_feature_engineering(self):
        # Placeholder for extra features
        return self.df

    def prepare_features_target(self):
        exclude = ['participant_id', 'date', 'diabetes_status']
        X = self.df.drop(columns=[c for c in exclude if c in self.df], errors='ignore')
        y = self.df['diabetes_status']
        return X, y

    @staticmethod
    def _pm(mean, std, decimals=4):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

    def _print_models_table(self, table_rows, class_name="Wearable"):
        # widths tuned to resemble your screenshot
        h1, h2, h3, h4, h5 = "Class", "Model", "Accuracy", "F1-score (Macro)", "AUC-ROC (Macro)"
        line = "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*20 + "+" + "-"*22 + "+" + "-"*22 + "+"
        print(line)
        print(f"| {h1:<10} | {h2:<10} | {h3:<18} | {h4:<20} | {h5:<20} |")
        print(line)
        for model_name, acc_s, f1_s, auc_s in table_rows:
            print(f"| {class_name:<10} | {model_name:<10} | {acc_s:<18} | {f1_s:<20} | {auc_s:<20} |")
        print(line)

    def train_and_evaluate(self, X, y, random_state: int = 42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )

        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
        numeric_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale',  StandardScaler()),
            ('var',    VarianceThreshold())
        ])

        models_config = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=15,
                class_weight='balanced', random_state=random_state, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85,
                objective='multi:softprob', num_class=3,
                eval_metric='mlogloss', reg_alpha=0.1, reg_lambda=0.1,
                random_state=random_state
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.08,
                num_leaves=31, subsample=0.85, colsample_bytree=0.85,
                objective='multiclass', class_weight='balanced',
                reg_alpha=0.1, reg_lambda=0.1, verbosity=-1,
                random_state=random_state
            ),
            'LogisticReg': LogisticRegression(
                max_iter=1000, class_weight='balanced', n_jobs=-1, random_state=random_state
            ),
            'SVC': SVC(
                probability=True, class_weight='balanced', kernel='rbf', random_state=random_state
            ),
            'KNN': KNeighborsClassifier(n_jobs=-1),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=random_state),
            'GradBoost': GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        }

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

        # multi-metric CV scoring (macro where applicable)
        scoring = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',         # built-in alias
            'auc_macro_ovr': 'roc_auc_ovr'  # built-in multiclass ROC AUC (macro by default)
        }

        printable_rows = []

        for name, clf in models_config.items():
            pipe = Pipeline([
                ('pre', clone(numeric_pipeline)),
                ('corr', CorrelationFilter(threshold=0.95)),
                ('clf', clf)
            ])

            cvres = cross_validate(
                pipe, X_train[numeric_features], y_train,
                cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
            )
            acc_mean, acc_std = cvres['test_accuracy'].mean(), cvres['test_accuracy'].std()
            f1_mean, f1_std   = cvres['test_f1_macro'].mean(), cvres['test_f1_macro'].std()
            auc_mean, auc_std = cvres['test_auc_macro_ovr'].mean(), cvres['test_auc_macro_ovr'].std()

            # fit & report test-set performance (kept from your script)
            pipe.fit(X_train[numeric_features], y_train)
            preds = pipe.predict(X_test[numeric_features])
            probs = pipe.predict_proba(X_test[numeric_features])
            acc = accuracy_score(y_test, preds)
            if probs.shape[1] > 2:
                auc = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
            else:
                auc = roc_auc_score(y_test, probs[:, 1])

            present_labels = sorted(np.unique(y_test))
            present_names = [LABEL_NAME_MAP[l] for l in present_labels]

            self.models[name] = pipe
            self.results[name] = {
                'cv_acc_mean': acc_mean, 'cv_acc_std': acc_std,
                'cv_f1_mean': f1_mean,   'cv_f1_std': f1_std,
                'cv_auc_mean': auc_mean, 'cv_auc_std': auc_std,
                'acc': acc, 'auc': auc
            }

            print(f"{name:10s} | CV Acc: {acc_mean:.3f} ±{acc_std:.3f} | "
                  f"CV F1(macro): {f1_mean:.3f} ±{f1_std:.3f} | CV AUC(macro): {auc_mean:.3f} ±{auc_std:.3f}")
            print(classification_report(y_test, preds,
                                        labels=present_labels,
                                        target_names=present_names,
                                        digits=3))

            # add to printable table (format to 4 decimals to match screenshot style)
            printable_rows.append((
                name,
                self._pm(acc_mean, acc_std, 4),
                self._pm(f1_mean,  f1_std,  4),
                self._pm(auc_mean, auc_std, 4)
            ))

        # Stacking (CV acc only)
        stack = StackingClassifier(
            estimators=[(n, self.models[n]) for n in models_config],
            final_estimator=LogisticRegression(max_iter=1000),
            stack_method='predict_proba', n_jobs=-1
        )
        stack_pipe = Pipeline([('pre', clone(numeric_pipeline)), ('corr', CorrelationFilter(0.95)), ('stack', stack)])
        stack_cv = cross_validate(stack_pipe, X_train[numeric_features], y_train, cv=cv,
                                  scoring=scoring, n_jobs=-1)
        print(f"Stacking   | CV Acc: {stack_cv['test_accuracy'].mean():.3f} ±{stack_cv['test_accuracy'].std():.3f}")
        stack_pipe.fit(X_train[numeric_features], y_train)
        spred = stack_pipe.predict(X_test[numeric_features])
        sprob = stack_pipe.predict_proba(X_test[numeric_features])
        s_auc = roc_auc_score(y_test, sprob, multi_class='ovr', average='macro') \
            if sprob.shape[1] > 2 else roc_auc_score(y_test, sprob[:, 1])
        print("Stacking   | Acc:{:.3f} AUC:{:.3f}".format(accuracy_score(y_test, spred), s_auc))
        self.results['Stacking'] = {'cv_acc_mean': stack_cv['test_accuracy'].mean(),
                                    'cv_acc_std': stack_cv['test_accuracy'].std()}

        # Voting (CV acc only)
        vote = VotingClassifier(estimators=[(n, self.models[n]) for n in models_config], voting='soft', n_jobs=-1)
        vote_pipe = Pipeline([('pre', clone(numeric_pipeline)), ('corr', CorrelationFilter(0.95)), ('vote', vote)])
        vote_cv = cross_validate(vote_pipe, X_train[numeric_features], y_train, cv=cv,
                                 scoring=scoring, n_jobs=-1)
        print(f"Voting     | CV Acc: {vote_cv['test_accuracy'].mean():.3f} ±{vote_cv['test_accuracy'].std():.3f}")
        vote_pipe.fit(X_train[numeric_features], y_train)
        vpred = vote_pipe.predict(X_test[numeric_features])
        vprob = vote_pipe.predict_proba(X_test[numeric_features])
        v_auc = roc_auc_score(y_test, vprob, multi_class='ovr', average='macro') \
            if vprob.shape[1] > 2 else roc_auc_score(y_test, vprob[:, 1])
        print("Voting     | Acc:{:.3f} AUC:{:.3f}".format(accuracy_score(y_test, vpred), v_auc))
        self.results['Voting'] = {'cv_acc_mean': vote_cv['test_accuracy'].mean(),
                                  'cv_acc_std': vote_cv['test_accuracy'].std()}

        # Leaderboard
        print("\nLeaderboard (by CV10 Accuracy):")
        for name, r in sorted(self.results.items(), key=lambda x: x[1].get('cv_acc_mean', 0), reverse=True):
            print(f"{name:10s} : {r['cv_acc_mean']:.3f} ±{r.get('cv_acc_std', 0):.3f}")

        # Final compact table like your screenshot
        print("\n")
        self._print_models_table(printable_rows, class_name="Wearable")


if __name__ == '__main__':
    data_dir   = r"C:\Users\sneez\Desktop\Wearable\ml_ready_participants"
    labels_tsv = r"C:\Users\sneez\Desktop\OrganizedAI-READI\DataFiles\participants_OMOP_HB_Overlap.tsv"

    pipeline = EnhancedDiabetesPipeline()
    pipeline.load_all_participants_data(data_dir, labels_tsv)
    pipeline.filter_high_quality_data()
    pipeline.advanced_feature_engineering()
    X, y = pipeline.prepare_features_target()
    pipeline.train_and_evaluate(X, y)
