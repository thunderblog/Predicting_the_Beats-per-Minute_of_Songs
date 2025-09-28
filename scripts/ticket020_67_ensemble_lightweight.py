"""
TICKET-020 67特徴量版二元アンサンブル（ローカル実行軽量版）
LightGBM + CatBoost + BPM Stratified戦略
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import warnings
import json
from datetime import datetime

import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings('ignore')

from scripts.my_config import config

def create_bpm_bins(y: pd.Series, n_bins: int = 10) -> np.ndarray:
    """BPM値を層化分割用のビンに分割"""
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    logger.info(f"BPM層化分割ビン作成: {len(np.unique(bins))}ビン")
    return bins

def run_lightweight_ensemble():
    """軽量版67特徴量二元アンサンブル実行"""
    logger.info("TICKET-020 軽量版67特徴量二元アンサンブル開始...")

    # データ読み込み
    train_data_path = config.processed_data_dir / "train_unified_75_features.csv"
    test_data_path = config.processed_data_dir / "test_unified_75_features.csv"

    logger.info("データ読み込み中...")
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    # 特徴量準備
    train_feature_cols = [col for col in train_df.columns if col not in ["id", "BeatsPerMinute"]]
    test_feature_cols = [col for col in test_df.columns if col != "id"]
    common_features = sorted(list(set(train_feature_cols) & set(test_feature_cols)))

    X_train = train_df[common_features]
    y_train = train_df["BeatsPerMinute"]
    X_test = test_df[common_features]

    logger.info(f"特徴量数: {len(common_features)}, 訓練サンプル: {len(X_train)}")

    # BPM Stratified KFold（3フォールドで高速化）
    n_folds = 3
    stratify_labels = create_bpm_bins(y_train)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_state)

    # Out-of-fold予測格納
    oof_lgb = np.zeros(len(X_train))
    oof_cat = np.zeros(len(X_train))
    test_lgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))

    lgb_scores = []
    cat_scores = []

    logger.info("3フォールド高速訓練開始...")

    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X_train, stratify_labels), total=n_folds)):
        logger.info(f"フォールド {fold + 1}/{n_folds}")

        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # LightGBM高速設定
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 20,
            'learning_rate': 0.05,  # 高速化
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'reg_alpha': 2.0,
            'reg_lambda': 2.0,
            'random_state': config.random_state,
            'verbosity': -1
        }

        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

        lgb_model = lgb.train(
            lgb_params, train_data,
            valid_sets=[val_data],
            num_boost_round=500,  # 高速化
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # LightGBM予測
        lgb_pred = lgb_model.predict(X_fold_val, num_iteration=lgb_model.best_iteration)
        oof_lgb[val_idx] = lgb_pred
        test_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / n_folds
        lgb_scores.append(np.sqrt(mean_squared_error(y_fold_val, lgb_pred)))

        # CatBoost高速設定
        cat_params = {
            "loss_function": "RMSE",
            "depth": 4,  # 浅く設定
            "learning_rate": 0.05,
            "iterations": 300,  # 高速化
            "subsample": 0.7,
            "random_seed": config.random_state,
            "verbose": 0
        }

        cat_model = cb.CatBoostRegressor(**cat_params)
        cat_model.fit(X_fold_train, y_fold_train, eval_set=(X_fold_val, y_fold_val), verbose=0)

        # CatBoost予測
        cat_pred = cat_model.predict(X_fold_val)
        oof_cat[val_idx] = cat_pred
        test_cat += cat_model.predict(X_test) / n_folds
        cat_scores.append(np.sqrt(mean_squared_error(y_fold_val, cat_pred)))

        logger.info(f"フォールド {fold + 1} - LGB: {lgb_scores[-1]:.6f}, CAT: {cat_scores[-1]:.6f}")

    # モデル性能
    lgb_cv = np.mean(lgb_scores)
    cat_cv = np.mean(cat_scores)

    logger.info(f"LightGBM CV: {lgb_cv:.6f} ± {np.std(lgb_scores):.6f}")
    logger.info(f"CatBoost CV: {cat_cv:.6f} ± {np.std(cat_scores):.6f}")

    # 簡単な重み最適化（グリッドサーチ）
    logger.info("重み最適化中...")
    best_weight = 0.5
    best_rmse = float('inf')

    for w_lgb in np.arange(0.0, 1.1, 0.1):
        w_cat = 1.0 - w_lgb
        ensemble_pred = w_lgb * oof_lgb + w_cat * oof_cat
        rmse = np.sqrt(mean_squared_error(y_train, ensemble_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = w_lgb

    # 最適アンサンブル
    optimal_weights = {'lightgbm': best_weight, 'catboost': 1.0 - best_weight}
    final_predictions = best_weight * test_lgb + (1.0 - best_weight) * test_cat

    logger.success(f"最適重み: LGB={best_weight:.3f}, CAT={1.0-best_weight:.3f}")
    logger.success(f"アンサンブルCV: {best_rmse:.6f}")

    # ベースライン比較
    baseline_67_cv = 26.463984
    improvement = baseline_67_cv - best_rmse

    logger.info(f"\n=== 性能サマリー ===")
    logger.info(f"67特徴量ベースライン: {baseline_67_cv:.6f}")
    logger.info(f"LightGBM単体:        {lgb_cv:.6f}")
    logger.info(f"CatBoost単体:        {cat_cv:.6f}")
    logger.info(f"二元アンサンブル:     {best_rmse:.6f}")
    logger.info(f"ベースラインからの改善: {improvement:+.6f}")

    # 提出ファイル作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'BeatsPerMinute': final_predictions
    })

    submission_path = config.processed_data_dir / f"submission_ticket020_67_lightweight_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    logger.success(f"提出ファイル作成完了: {submission_path}")

    # 結果サマリー
    result_summary = {
        "experiment": "ticket020_67_lightweight_ensemble",
        "cv_performance": {
            "lightgbm_cv": lgb_cv,
            "catboost_cv": cat_cv,
            "ensemble_cv": best_rmse,
            "improvement_from_baseline": improvement
        },
        "optimal_weights": optimal_weights,
        "submission_file": str(submission_path),
        "n_features": len(common_features),
        "n_folds": n_folds
    }

    logger.info(f"結果サマリー: {result_summary}")

    return submission_path, best_rmse, improvement, optimal_weights

if __name__ == "__main__":
    logger.info("軽量版67特徴量二元アンサンブル開始")

    submission_path, cv_score, improvement, weights = run_lightweight_ensemble()

    logger.success("実験完了！")
    logger.info(f"提出ファイル: {submission_path}")
    logger.info(f"アンサンブルCV: {cv_score:.6f}")
    logger.info(f"改善効果: {improvement:+.6f}")

    if improvement > 0:
        logger.success("ベースラインを上回る性能を達成！Kaggle提出を検討してください")
    else:
        logger.info("ベースライン同等の性能。追加改善が必要です")