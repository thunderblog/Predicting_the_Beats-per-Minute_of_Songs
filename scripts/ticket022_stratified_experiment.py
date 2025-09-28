"""
TICKET-022 BPM Stratified戦略ベースライン実験
CV-LB格差改善効果を検証するための単一LightGBM実験
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
import pickle

warnings.filterwarnings('ignore')

from scripts.my_config import config
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from tqdm import tqdm

# CV戦略のインポート（直接実装）
from sklearn.model_selection import StratifiedKFold

def create_bpm_bins(y: pd.Series, n_bins: int = 10) -> np.ndarray:
    """BPM値を層化分割用のビンに分割"""
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    logger.info(f"BPM層化分割ビン作成: {len(np.unique(bins))}ビン, 範囲: {y.min():.2f}-{y.max():.2f}")
    return bins

def train_lightgbm_fold(X_train, y_train, X_val, y_val):
    """単一フォールドでLightGBM訓練"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 既存の高性能パラメータを使用（TICKET017正則化版ベース）
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
            lgb.log_evaluation(0)  # Silent training
        ]
    )

    return model

def run_stratified_experiment():
    """BPM Stratified戦略でLightGBM実験実行"""
    logger.info("TICKET-022 BPM Stratified LightGBM実験開始...")

    # データ読み込み
    data_path = config.processed_data_dir / "train_ticket017_75_features.csv"
    logger.info(f"データ読み込み: {data_path}")

    df = pd.read_csv(data_path)

    # 特徴量とターゲットを分離
    feature_cols = [col for col in df.columns if col not in ["id", "BeatsPerMinute"]]
    X = df[feature_cols]
    y = df["BeatsPerMinute"]

    logger.info(f"データ形状: {X.shape}, 特徴量数: {len(feature_cols)}")

    # BPM層化分割の設定
    n_folds = 5
    random_state = config.random_state

    # 1. BPM Stratified KFold実験
    logger.info("\n=== BPM Stratified KFold実験 ===")

    stratify_labels = create_bpm_bins(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, stratify_labels), total=n_folds)):
        logger.info(f"フォールド {fold + 1}/{n_folds} 訓練中...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # LightGBM訓練
        model = train_lightgbm_fold(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)

        fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        cv_scores.append(fold_rmse)
        models.append(model)

        logger.info(f"フォールド {fold + 1} RMSE: {fold_rmse:.6f}")

        # フォールド統計
        logger.info(f"- 訓練データBPM: 平均{y_train.mean():.4f}, 標準偏差{y_train.std():.4f}")
        logger.info(f"- 検証データBPM: 平均{y_val.mean():.4f}, 標準偏差{y_val.std():.4f}")

    # 結果集計
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    logger.success(f"BPM Stratified KFold完了")
    logger.info(f"平均CV RMSE: {mean_cv:.6f} ± {std_cv:.6f}")
    logger.info(f"フォールド別スコア: {[round(s, 6) for s in cv_scores]}")

    # 2. 比較用：標準KFold実験
    logger.info("\n=== 比較用：標準KFold実験 ===")

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    standard_cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=n_folds)):
        logger.info(f"フォールド {fold + 1}/{n_folds} 訓練中（標準KFold）...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = train_lightgbm_fold(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)

        fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        standard_cv_scores.append(fold_rmse)

        logger.info(f"フォールド {fold + 1} RMSE: {fold_rmse:.6f}")

    standard_mean_cv = np.mean(standard_cv_scores)
    standard_std_cv = np.std(standard_cv_scores)

    logger.info(f"標準KFold完了")
    logger.info(f"平均CV RMSE: {standard_mean_cv:.6f} ± {standard_std_cv:.6f}")

    # 3. 結果比較と分析
    logger.info("\n=== CV戦略比較結果 ===")
    logger.info(f"BPM Stratified: {mean_cv:.6f} ± {std_cv:.6f}")
    logger.info(f"標準KFold:      {standard_mean_cv:.6f} ± {standard_std_cv:.6f}")
    logger.info(f"性能差:         {mean_cv - standard_mean_cv:+.6f}")
    logger.info(f"安定性改善:     {standard_std_cv / std_cv:.2f}倍")

    # 4. 実験結果の保存
    exp_name = "exp15_ticket022_stratified_lgb_baseline"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "experiment_name": exp_name,
        "timestamp": timestamp,
        "cv_strategy_comparison": {
            "bpm_stratified": {
                "mean_cv_rmse": mean_cv,
                "std_cv_rmse": std_cv,
                "fold_scores": cv_scores
            },
            "standard_kfold": {
                "mean_cv_rmse": standard_mean_cv,
                "std_cv_rmse": standard_std_cv,
                "fold_scores": standard_cv_scores
            }
        },
        "performance_improvement": {
            "rmse_difference": mean_cv - standard_mean_cv,
            "stability_improvement": standard_std_cv / std_cv if std_cv > 0 else float('inf')
        },
        "model_config": {
            "n_features": len(feature_cols),
            "n_samples": len(X),
            "n_folds": n_folds,
            "stratification_bins": len(np.unique(stratify_labels))
        }
    }

    # 結果保存
    results_path = config.processed_data_dir / f"ticket022_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"実験結果保存: {results_path}")

    return results

if __name__ == "__main__":
    logger.info("TICKET-022 BPM Stratified戦略ベースライン実験開始")

    results = run_stratified_experiment()

    if results:
        stratified_rmse = results["cv_strategy_comparison"]["bpm_stratified"]["mean_cv_rmse"]
        improvement = results["performance_improvement"]["stability_improvement"]

        logger.success(f"実験完了！BPM Stratified CV RMSE: {stratified_rmse:.6f}")
        logger.success(f"安定性改善: {improvement:.2f}倍")

        if improvement > 1.5:
            logger.info("🎯 CV戦略改善効果確認！アンサンブル実験に進む準備完了")
        else:
            logger.warning("期待した改善効果が得られませんでした。パラメータ調整が必要です")
    else:
        logger.error("実験が失敗しました")