"""
TICKET-016 統一67特徴量 + BPM Stratified戦略実験
特徴量統一による性能向上検証
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

    # TICKET017正則化版パラメータを使用
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

def run_unified_feature_experiment():
    """統一67特徴量でBPM Stratified実験実行"""
    logger.info("TICKET-016 統一67特徴量 + BPM Stratified実験開始...")

    # 統一特徴量データ読み込み
    train_data_path = config.processed_data_dir / "train_unified_75_features.csv"
    test_data_path = config.processed_data_dir / "test_unified_75_features.csv"

    logger.info(f"訓練データ読み込み: {train_data_path}")
    train_df = pd.read_csv(train_data_path)

    logger.info(f"テストデータ読み込み: {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    # 特徴量とターゲットを分離
    train_feature_cols = [col for col in train_df.columns if col not in ["id", "BeatsPerMinute"]]
    test_feature_cols = [col for col in test_df.columns if col != "id"]

    # 特徴量一致確認
    common_features = sorted(list(set(train_feature_cols) & set(test_feature_cols)))
    logger.info(f"共通特徴量数: {len(common_features)}")
    logger.info(f"訓練特徴量数: {len(train_feature_cols)}, テスト特徴量数: {len(test_feature_cols)}")

    if len(common_features) != len(train_feature_cols) or len(common_features) != len(test_feature_cols):
        logger.warning("特徴量が完全一致していません")
        logger.info(f"共通特徴量を使用: {len(common_features)}特徴量")

    X_train = train_df[common_features]
    y_train = train_df["BeatsPerMinute"]
    X_test = test_df[common_features]

    logger.info(f"特徴量数: {len(common_features)}, 訓練サンプル: {len(X_train)}, テストサンプル: {len(X_test)}")

    # BPM Stratified KFold訓練
    n_folds = 5
    stratify_labels = create_bpm_bins(y_train)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_state)

    models = []
    cv_scores = []
    test_predictions = np.zeros(len(X_test))

    logger.info("BPM Stratified KFold訓練・推論開始...")

    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X_train, stratify_labels), total=n_folds)):
        logger.info(f"フォールド {fold + 1}/{n_folds} 訓練・推論中...")

        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # 訓練
        model = train_lightgbm_fold(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
        models.append(model)

        # 検証スコア
        val_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
        fold_rmse = np.sqrt(np.mean((y_fold_val - val_pred) ** 2))
        cv_scores.append(fold_rmse)

        # テスト予測
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        test_predictions += test_pred / n_folds

        logger.info(f"フォールド {fold + 1} RMSE: {fold_rmse:.6f}")

        # フォールド統計
        logger.info(f"- 訓練データBPM: 平均{y_fold_train.mean():.4f}, 標準偏差{y_fold_train.std():.4f}")
        logger.info(f"- 検証データBPM: 平均{y_fold_val.mean():.4f}, 標準偏差{y_fold_val.std():.4f}")

    # CV結果
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    logger.success(f"BPM Stratified KFold完了")
    logger.info(f"平均CV RMSE: {mean_cv:.6f} ± {std_cv:.6f}")
    logger.info(f"フォールド別スコア: {[round(s, 6) for s in cv_scores]}")

    # 40特徴量版との比較
    baseline_cv = 26.464760
    baseline_std = 0.005038
    cv_improvement = baseline_cv - mean_cv
    stability_comparison = baseline_std / std_cv if std_cv > 0 else float('inf')

    logger.info(f"\n=== 40特徴量版との比較 ===")
    logger.info(f"40特徴量版CV: {baseline_cv:.6f} ± {baseline_std:.6f}")
    logger.info(f"67特徴量版CV: {mean_cv:.6f} ± {std_cv:.6f}")
    logger.info(f"CV改善:       {cv_improvement:+.6f}")
    logger.info(f"安定性比較:   {stability_comparison:.2f}倍{'改善' if stability_comparison > 1 else '劣化'}")

    # 提出ファイル作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'BeatsPerMinute': test_predictions
    })

    submission_path = config.processed_data_dir / f"submission_ticket016_unified67_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    logger.success(f"提出ファイル作成完了: {submission_path}")
    logger.info(f"予測統計:")
    logger.info(f"- 最小値: {test_predictions.min():.4f}")
    logger.info(f"- 最大値: {test_predictions.max():.4f}")
    logger.info(f"- 平均値: {test_predictions.mean():.4f}")
    logger.info(f"- 標準偏差: {test_predictions.std():.4f}")

    # 実験記録
    experiment_record = {
        "experiment_name": "exp16_ticket016_unified67_stratified",
        "timestamp": timestamp,
        "cv_strategy": "bpm_stratified",
        "model_type": "lightgbm",
        "n_features": len(common_features),
        "cv_results": {
            "mean_cv_rmse": mean_cv,
            "std_cv_rmse": std_cv,
            "fold_scores": cv_scores,
            "baseline_comparison": {
                "baseline_cv": baseline_cv,
                "baseline_std": baseline_std,
                "cv_improvement": cv_improvement,
                "stability_comparison": stability_comparison
            }
        },
        "prediction_stats": {
            "min_prediction": test_predictions.min(),
            "max_prediction": test_predictions.max(),
            "mean_prediction": test_predictions.mean(),
            "std_prediction": test_predictions.std()
        },
        "submission_file": str(submission_path),
        "expected_lb_improvement": f"40特徴量版からの改善期待: {cv_improvement:+.6f}"
    }

    record_path = config.processed_data_dir / f"experiment_record_ticket016_{timestamp}.json"
    with open(record_path, 'w') as f:
        json.dump(experiment_record, f, indent=2, default=str)

    logger.info(f"実験記録保存: {record_path}")

    return submission_path, mean_cv, std_cv, cv_improvement

def submit_to_kaggle(submission_path: Path, cv_score: float, improvement: float):
    """Kaggle提出"""
    logger.info("Kaggle提出準備...")

    message = f"TICKET-016 Unified 67 Features + BPM Stratified - CV: {cv_score:.6f} (improvement: {improvement:+.6f})"

    logger.info("Kaggle CLI提出:")
    logger.info(f"ファイル: {submission_path}")
    logger.info(f"メッセージ: {message}")

    # 自動提出
    import subprocess
    try:
        result = subprocess.run([
            'kaggle', 'competitions', 'submit',
            '-c', 'playground-series-s5e9',
            '-f', str(submission_path),
            '-m', message
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            logger.success("Kaggle提出成功！")
            logger.info(result.stdout)
        else:
            logger.error(f"Kaggle提出失敗: {result.stderr}")
            logger.info("手動で提出してください")

    except Exception as e:
        logger.error(f"提出エラー: {e}")
        logger.info("手動で提出してください")

if __name__ == "__main__":
    logger.info("TICKET-016 統一67特徴量実験開始")

    submission_path, cv_score, cv_std, improvement = run_unified_feature_experiment()

    logger.success("実験完了！")
    logger.info(f"CV RMSE: {cv_score:.6f} ± {cv_std:.6f}")
    logger.info(f"40特徴量版からの改善: {improvement:+.6f}")

    submit_to_kaggle(submission_path, cv_score, improvement)

    logger.info("次のステップ:")
    logger.info("1. LBスコア確認")
    logger.info("2. 特徴量増加効果の検証")
    logger.info("3. さらなる改善戦略の検討")