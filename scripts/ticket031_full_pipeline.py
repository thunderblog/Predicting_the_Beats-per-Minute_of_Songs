"""
TICKET-031: 精密最適化完全パイプライン
- 最適化済み重み（LightGBM 70.6%, CatBoost 29.4%）でモデル訓練
- テスト予測生成
- Submission作成
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

from scripts.my_config import config
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def create_bpm_bins(y: pd.Series, n_bins: int = 10) -> np.ndarray:
    """BPM値を層化分割用のビンに分割"""
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    logger.info(f"BPM層化分割ビン作成: {len(np.unique(bins))}ビン, 範囲: {y.min():.2f}-{y.max():.2f}")
    return bins


def train_lightgbm_fold(X_train, y_train, X_val, y_val):
    """単一フォールドでLightGBM訓練"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # TICKET017正則化版パラメータ
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 20,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'reg_alpha': 2.0,
        'reg_lambda': 2.0,
        'min_child_samples': 20,
        'random_state': config.random_state,
        'verbosity': -1
    }

    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(0)
        ]
    )

    return model


def train_catboost_fold(X_train, y_train, X_val, y_val):
    """単一フォールドでCatBoost訓練"""
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        random_seed=config.random_state,
        verbose=0,
        early_stopping_rounds=200
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False
    )

    return model


def run_ticket031_pipeline(
    weight_lightgbm: float = 0.7060582514745803,
    cv_folds: int = 5
):
    """
    TICKET-031完全パイプライン実行

    Parameters
    ----------
    weight_lightgbm : float
        LightGBMの重み（最適化済み値）
    cv_folds : int
        クロスバリデーションフォールド数
    """
    weight_catboost = 1.0 - weight_lightgbm

    logger.info("=" * 80)
    logger.info("TICKET-031: 精密最適化完全パイプライン")
    logger.info("=" * 80)
    logger.info(f"アンサンブル重み: LightGBM={weight_lightgbm:.6f}, CatBoost={weight_catboost:.6f}")
    logger.info(f"CVフォールド数: {cv_folds}")

    # データ読み込み
    train_data_path = config.processed_data_dir / "train_unified_75_features.csv"
    test_data_path = config.processed_data_dir / "test_unified_75_features.csv"

    logger.info(f"訓練データ読み込み: {train_data_path}")
    train_df = pd.read_csv(train_data_path)

    logger.info(f"テストデータ読み込み: {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    # 特徴量とターゲット分離
    X = train_df.drop(columns=["id", "BeatsPerMinute"])
    y = train_df["BeatsPerMinute"]
    X_test = test_df.drop(columns=["id"])

    logger.info(f"訓練データ: {X.shape}, テストデータ: {X_test.shape}")
    logger.info(f"特徴量数: {X.shape[1]}")

    # BPM層化分割
    bpm_bins = create_bpm_bins(y, n_bins=10)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.random_state)

    # OOF予測と テスト予測用配列初期化
    oof_preds_lgb = np.zeros(len(X))
    oof_preds_cat = np.zeros(len(X))
    test_preds_lgb = np.zeros(len(X_test))
    test_preds_cat = np.zeros(len(X_test))

    fold_scores_lgb = []
    fold_scores_cat = []
    models_lgb = []
    models_cat = []

    # クロスバリデーション
    logger.info(f"\n{'=' * 80}")
    logger.info("クロスバリデーション開始...")
    logger.info(f"{'=' * 80}\n")

    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, bpm_bins), total=cv_folds, desc="CV Folds"), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # LightGBM訓練
        lgb_model = train_lightgbm_fold(X_train, y_train, X_val, y_val)
        lgb_val_pred = lgb_model.predict(X_val)
        oof_preds_lgb[val_idx] = lgb_val_pred
        test_preds_lgb += lgb_model.predict(X_test) / cv_folds

        lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
        fold_scores_lgb.append(lgb_rmse)
        models_lgb.append(lgb_model)

        # CatBoost訓練
        cat_model = train_catboost_fold(X_train, y_train, X_val, y_val)
        cat_val_pred = cat_model.predict(X_val)
        oof_preds_cat[val_idx] = cat_val_pred
        test_preds_cat += cat_model.predict(X_test) / cv_folds

        cat_rmse = np.sqrt(mean_squared_error(y_val, cat_val_pred))
        fold_scores_cat.append(cat_rmse)
        models_cat.append(cat_model)

        # アンサンブル性能
        ensemble_val_pred = weight_lightgbm * lgb_val_pred + weight_catboost * cat_val_pred
        ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))

        logger.info(f"Fold {fold}: LGB={lgb_rmse:.6f}, CAT={cat_rmse:.6f}, ENS={ensemble_rmse:.6f}")

    # OOFアンサンブル予測
    oof_ensemble = weight_lightgbm * oof_preds_lgb + weight_catboost * oof_preds_cat
    oof_rmse_lgb = np.sqrt(mean_squared_error(y, oof_preds_lgb))
    oof_rmse_cat = np.sqrt(mean_squared_error(y, oof_preds_cat))
    oof_rmse_ensemble = np.sqrt(mean_squared_error(y, oof_ensemble))

    logger.info(f"\n{'=' * 80}")
    logger.info("クロスバリデーション結果:")
    logger.info(f"{'=' * 80}")
    logger.info(f"LightGBM  OOF RMSE: {oof_rmse_lgb:.6f} (Folds: {np.mean(fold_scores_lgb):.6f} ± {np.std(fold_scores_lgb):.6f})")
    logger.info(f"CatBoost  OOF RMSE: {oof_rmse_cat:.6f} (Folds: {np.mean(fold_scores_cat):.6f} ± {np.std(fold_scores_cat):.6f})")
    logger.info(f"Ensemble  OOF RMSE: {oof_rmse_ensemble:.6f}")
    logger.info(f"{'=' * 80}\n")

    # テスト予測アンサンブル
    test_ensemble = weight_lightgbm * test_preds_lgb + weight_catboost * test_preds_cat

    # Submission作成
    submission = pd.DataFrame({
        "id": test_df["id"],
        "BeatsPerMinute": test_ensemble
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = config.processed_data_dir / f"submission_ticket031_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)

    logger.success(f"Submission保存完了: {submission_path}")
    logger.info(f"予測統計: Mean={test_ensemble.mean():.4f}, Std={test_ensemble.std():.4f}, Min={test_ensemble.min():.4f}, Max={test_ensemble.max():.4f}")

    # モデル保存
    models_path = config.models_dir / f"ticket031_models_{timestamp}.pkl"
    with open(models_path, 'wb') as f:
        pickle.dump({
            'lightgbm_models': models_lgb,
            'catboost_models': models_cat,
            'weights': {'lightgbm': weight_lightgbm, 'catboost': weight_catboost},
            'oof_rmse': oof_rmse_ensemble,
            'test_predictions': test_ensemble
        }, f)

    logger.success(f"モデル保存完了: {models_path}")

    logger.info(f"\n{'=' * 80}")
    logger.info("次のステップ: Kaggle提出")
    logger.info(f"{'=' * 80}")
    logger.info(f"kaggle competitions submit -c playground-series-s5e9 \\")
    logger.info(f"  -f {submission_path} \\")
    logger.info(f"  -m \"TICKET-031: Precision Optuna (LGB 70.6%%, CAT 29.4%%, CV {oof_rmse_ensemble:.6f})\"")

    return {
        'submission_path': submission_path,
        'models_path': models_path,
        'oof_rmse': oof_rmse_ensemble,
        'cv_scores': {
            'lightgbm': fold_scores_lgb,
            'catboost': fold_scores_cat
        }
    }


if __name__ == "__main__":
    results = run_ticket031_pipeline()
    logger.success("TICKET-031パイプライン完了")