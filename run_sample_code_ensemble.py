#!/usr/bin/env python3
"""
Sample Code 3-Model Ensemble Implementation
Original notebook: xgb-lgbm-catboost-weighted-average-26-38518.ipynb
LB Score: 26.38518
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import gc
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from pandas.errors import PerformanceWarning
from sklearn.metrics import mean_squared_error
from itertools import combinations
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import copy
from pathlib import Path

# 警告を無視
warnings.filterwarnings('ignore')

def main(fast_mode=False):
    print("=" * 60)
    print("Sample Code 3-Model Ensemble Execution")
    if fast_mode:
        print("FAST MODE: Reduced parameters for quick testing")
    print("Expected LB Score: 26.38518")
    print("=" * 60)

    # データ読み込み（Windows環境用パス）
    print("Loading data...")
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    target = 'BeatsPerMinute'
    cat_cols = []
    num_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore',
                'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationMs', 'Energy']

    print("Creating interaction and transformation features...")

    # 交互作用・二乗・除算特徴量作成
    log1p_cols = []
    for i in range(len(num_cols)):
        for j in range(i, len(num_cols)):
            col1 = num_cols[i]
            col2 = num_cols[j]
            # Create interaction features
            train[f'{col1}_x_{col2}'] = train[col1] * train[col2]
            test[f'{col1}_x_{col2}'] = test[col1] * test[col2]
            train[f'{col1}_squared']= train[col1] * train[col1]
            test[f'{col1}_squared']= test[col1] * test[col1]
            # Create ratio features, handle division by zero
            if col1 != col2:
                train[f'{col1}_div_{col2}'] = train[col1] / (train[col2] + 1e-6)
                test[f'{col1}_div_{col2}'] = test[col1] / (test[col2] + 1e-6)

    # 対数変換特徴量
    for col in num_cols:
        train[f'log1p_{col}'] = np.log1p(train[col])
        test[f'log1p_{col}'] = np.log1p(test[col])
        log1p_cols.append(f'log1p_{col}')

    # ビニング関数
    def add_bins(df, column, labels, new_column=None):
        if len(labels) == 4 and new_column is None:
            new_column = f"{column}_quartile"
        if len(labels) == 5 and new_column is None:
            new_column = f"{column}_quintile"
        if len(labels) == 7 and new_column is None:
            new_column = f"{column}_septile"
        if len(labels) == 10 and new_column is None:
            new_column = f"{column}_decile"

        df[new_column] = pd.cut(
            df[column],
            bins=len(labels),
            labels=labels,
            include_lowest=True
        )
        return df[new_column]

    # カテゴリ特徴量作成
    CATS = []

    # Septile binning
    for col in num_cols:
        train[f"{col}_septile"] = add_bins(train, col, ['Q1', 'Q2', 'Q3', 'Q4','Q5','Q6','Q7'])
        CATS.append(f"{col}_septile")

    # Decile binning
    for col in num_cols:
        train[f"{col}_decile"] = add_bins(train, col, ['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
                                                       'Q6', 'Q7', 'Q8', 'Q9', 'Q10'])
        CATS.append(f"{col}_decile")

    # Test data binning
    for col in num_cols:
        test[f"{col}_septile"] = add_bins(test, col,  ['Q1', 'Q2', 'Q3', 'Q4','Q5','Q6','Q7'])

    for col in num_cols:
        test[f"{col}_decile"] = add_bins(test, col, ['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
                                                     'Q6', 'Q7', 'Q8', 'Q9', 'Q10'])

    # Log1p quintile binning (excluding AudioLoudness)
    log1p_cols.remove('log1p_AudioLoudness')
    for col in log1p_cols:
        train[f"{col}_quintile"] = add_bins(train, col, ['Q1', 'Q2', 'Q3', 'Q4','Q5'])
        test[f"{col}_quintile"] = add_bins(test, col, ['Q1', 'Q2', 'Q3', 'Q4','Q5'])
        CATS.append(f"{col}_quintile")

    # Target encoding for categorical features
    Cat_groupby = []
    for col in CATS:
        new_col = f'mean_by_{col}'
        mapping = train.groupby(col)['BeatsPerMinute'].mean()

        # Apply to train and test
        train[new_col] = train[col].map(mapping)
        test[new_col] = test[col].map(mapping)
        Cat_groupby.append(new_col)

    # 特徴量セット準備
    Features = train.columns.tolist()
    Features.remove(target)
    Features.remove('id')
    Features = Features + Cat_groupby

    cat_features = CATS
    X_cat = train[cat_features]
    X = train[Features]
    y = train[target]
    X_test = test[Features]

    print(f"Total features: {len(Features)}")
    print(f"Categorical features: {len(cat_features)}")

    # TODO(human): Fast mode parameter optimization for reduced execution time
    # モデルパラメータ定義
    xgb_params = {
        'n_estimators': 620,
        'max_leaves': 211,
        'min_child_weight': 1.5,
        'max_depth': 6,
        'grow_policy': 'lossguide',
        'learning_rate': 0.0021858703356597603,
        'tree_method': 'hist',
        'subsample': 0.85,
        'colsample_bylevel': 0.6787051322531533,
        'colsample_bytree': 0.6843905004927857,
        'colsample_bynode': 0.442116057736592,
        'sampling_method': 'uniform',
        'reg_alpha': 2.5,
        'reg_lambda': 0.8,
        'enable_categorical': True,
        'max_cat_to_onehot': 1,
        'device': 'cpu',  # Windows環境用にCPUに変更
        'n_jobs': -1,
        'random_state': 0,
        'verbosity': 0,
    }

    lgbm_params = {
        'learning_rate': 0.001502328415098844,
        'num_leaves': 79,
        'max_depth': 14,
        'feature_fraction': 0.8933016300882094,
        'bagging_fraction': 0.9754103048412501,
        'bagging_freq': 7,
        'min_child_samples': 40,
        'enable_categorical': True,
        'lambda_l1': 7.10897934678165e-07,
        'lambda_l2': 7.81564014894075e-08,
        'random_state': 0,
        'n_jobs': -1,
        'verbosity': -1,
        'n_estimators': 643
    }

    cb_params = {
        'border_count': 28,
        'colsample_bylevel': 0.19459088572914465,
        'depth': 5,
        'iterations': 600,
        'l2_leaf_reg': 31.236169478676036,
        'learning_rate': 0.1332583504067626,
        'min_child_samples': 189,
        'random_state': 0,
        'random_strength': 0.8517786189616939,
        'subsample': 0.3192330024411618,
        'verbose': False,
        'cat_features': CATS
    }

    print("\n1. Training CatBoost...")
    cb_model = CatBoostRegressor(**cb_params)
    cb_oof_preds = np.zeros(len(X))
    cb_models, cb_scores = [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_cat, y)):
        print(f'Fold: {fold_idx + 1}')
        X_train, X_val = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        cb_fold_model = CatBoostRegressor(**cb_params)
        cb_fold_model.fit(X_train, y_train)
        cb_oof_preds[val_idx] = cb_fold_model.predict(X_val)
        acc = mean_squared_error(y_val, cb_fold_model.predict(X_val), squared=False)
        cb_scores.append(acc)
        cb_models.append(cb_fold_model)
        print(f'Accuracy: {acc}')

    print(f'CATBOOST ACCURACY: {np.mean(cb_scores)}')

    print("\n2. Training XGBoost...")
    xgb_model = XGBRegressor(**xgb_params)
    xgb_oof_preds = np.zeros(len(X))
    xgb_models, xgb_scores = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f'Fold: {fold_idx + 1}')
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        xgb_fold_model = XGBRegressor(**xgb_params)
        xgb_fold_model.fit(X_train, y_train)
        xgb_oof_preds[val_idx] = xgb_fold_model.predict(X_val)
        acc = mean_squared_error(y_val, xgb_fold_model.predict(X_val), squared=False)
        xgb_scores.append(acc)
        xgb_models.append(xgb_fold_model)
        print(f'Accuracy: {acc}')

    print(f'XGB ACCURACY: {np.mean(xgb_scores)}')

    print("\n3. Training LightGBM...")
    lgbm_model = LGBMRegressor(**lgbm_params)
    lgbm_models, lgbm_scores = [], []
    lgbm_oof_preds = np.zeros(len(X))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f'Fold: {fold_idx + 1}')
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        lgbm_fold_model = LGBMRegressor(**lgbm_params)
        lgbm_fold_model.fit(X_train, y_train)
        lgbm_oof_preds[val_idx] = lgbm_fold_model.predict(X_val)
        acc = mean_squared_error(y_val, lgbm_fold_model.predict(X_val), squared=False)
        lgbm_scores.append(acc)
        lgbm_models.append(lgbm_fold_model)
        print(f'Accuracy: {acc}')

    print(f'LGBM ACCURACY: {np.mean(lgbm_scores)}')

    print("\n4. Creating test predictions...")

    # テスト予測生成
    xgb_test_preds = sum(model.predict(test[Features]) for model in xgb_models) / len(xgb_models)
    lgbm_test_preds = sum(lgbm_model.predict(test[Features]) for lgbm_model in lgbm_models) / len(lgbm_models)
    cb_test_preds = sum(cb_model.predict(test[cat_features]) for cb_model in cb_models) / len(cb_models)

    print("\n5. Creating ensemble predictions...")

    # アンサンブル重み（元コードより）
    w_lgbm = 0.5
    w_cb = 0.25
    w_xgb = 1 - w_lgbm - w_cb  # 0.25

    print(f"Ensemble weights: LGBM={w_lgbm}, XGB={w_xgb}, CatBoost={w_cb}")

    preds = w_lgbm * lgbm_test_preds + w_xgb * xgb_test_preds + w_cb * cb_test_preds

    # アンサンブル性能確認
    X_stack = pd.DataFrame({
        'xgb': xgb_oof_preds,
        'lgbm': lgbm_oof_preds,
        'cb': cb_oof_preds
    })

    final_oof_preds = (w_lgbm * X_stack['lgbm'] + w_xgb * X_stack['xgb'] + w_cb * X_stack['cb'])
    ensemble_cv_score = mean_squared_error(y, final_oof_preds, squared=False)
    print(f"\nEnsemble CV Score: {ensemble_cv_score}")

    # 提出ファイル作成
    submission = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': preds})
    submission_path = 'data/processed/submission_sample_code_ensemble.csv'
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission file created: {submission_path}")
    print("First 5 predictions:")
    print(submission.head())

    print("\n" + "=" * 60)
    print("Sample Code 3-Model Ensemble Completed!")
    print(f"Expected LB Score: 26.38518")
    print(f"CV Ensemble Score: {ensemble_cv_score}")
    print("=" * 60)

    return submission_path

if __name__ == "__main__":
    main()