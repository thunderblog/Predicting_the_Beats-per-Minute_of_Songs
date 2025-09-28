"""
TICKET-022-02 GroupKFold戦略実験
音楽類似性グループ別分割によるCV-LB一貫性改善検証
67特徴量版での実験実行
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
from src.modeling.cross_validation import MusicSimilarityGroupKFoldStrategy
from tqdm import tqdm

def train_lightgbm_fold(X_train, y_train, X_val, y_val):
    """単一フォールドでLightGBM訓練"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 67特徴量最適化パラメータ
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

def run_group_kfold_experiment():
    """GroupKFold戦略67特徴量実験実行"""
    logger.info("TICKET-022-02 GroupKFold戦略67特徴量実験開始...")

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

    X_train = train_df[common_features]
    y_train = train_df["BeatsPerMinute"]
    X_test = test_df[common_features]

    logger.info(f"特徴量数: {len(common_features)}, 訓練サンプル: {len(X_train)}, テストサンプル: {len(X_test)}")

    # GroupKFold戦略初期化
    logger.info("音楽類似性GroupKFold戦略初期化...")
    group_strategy = MusicSimilarityGroupKFoldStrategy(
        n_splits=5,
        random_state=config.random_state,
        n_clusters=50  # クラスタ数（調整可能）
    )

    # GroupKFold分割品質検証
    logger.info("GroupKFold分割品質検証実行...")
    group_strategy.validate_split(X_train, y_train)

    # GroupKFold訓練・推論
    models = []
    cv_scores = []
    test_predictions = np.zeros(len(X_test))
    n_folds = 5

    logger.info("GroupKFold訓練・推論開始...")

    fold_groups_info = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(group_strategy.split(X_train, y_train), total=n_folds)):
        logger.info(f"フォールド {fold + 1}/{n_folds} 訓練・推論中...")

        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # フォールド統計記録
        fold_info = {
            'fold': fold + 1,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'train_mean_bpm': y_fold_train.mean(),
            'val_mean_bpm': y_fold_val.mean(),
            'train_std_bpm': y_fold_train.std(),
            'val_std_bpm': y_fold_val.std()
        }
        fold_groups_info.append(fold_info)

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
        logger.info(f"- 訓練データBPM: 平均{y_fold_train.mean():.4f}, 標準偏差{y_fold_train.std():.4f}")
        logger.info(f"- 検証データBPM: 平均{y_fold_val.mean():.4f}, 標準偏差{y_fold_val.std():.4f}")

    # CV結果
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    logger.success(f"GroupKFold完了")
    logger.info(f"平均CV RMSE: {mean_cv:.6f} ± {std_cv:.6f}")
    logger.info(f"フォールド別スコア: {[round(s, 6) for s in cv_scores]}")

    # 他CV戦略との比較
    bpm_stratified_cv = 26.463984
    bmp_stratified_std = 0.006159

    cv_improvement = bmp_stratified_cv - mean_cv
    stability_comparison = bmp_stratified_std / std_cv if std_cv > 0 else float('inf')

    logger.info(f"\n=== BPM Stratified戦略との比較 ===")
    logger.info(f"BPM Stratified: {bmp_stratified_cv:.6f} ± {bmp_stratified_std:.6f}")
    logger.info(f"GroupKFold:     {mean_cv:.6f} ± {std_cv:.6f}")
    logger.info(f"CV改善:         {cv_improvement:+.6f}")
    logger.info(f"安定性比較:     {stability_comparison:.2f}倍{'改善' if stability_comparison > 1 else '劣化'}")

    # GroupKFold品質分析
    fold_df = pd.DataFrame(fold_groups_info)
    mean_consistency = fold_df['val_mean_bpm'].std()
    std_consistency = fold_df['val_std_bpm'].std()

    logger.info(f"\n=== GroupKFold品質分析 ===")
    logger.info(f"フォールド間平均値一貫性: {mean_consistency:.6f} (低いほど良い)")
    logger.info(f"フォールド間標準偏差一貫性: {std_consistency:.6f} (低いほど良い)")

    # 提出ファイル作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'BeatsPerMinute': test_predictions
    })

    submission_path = config.processed_data_dir / f"submission_ticket022_02_group_kfold_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    logger.success(f"提出ファイル作成完了: {submission_path}")
    logger.info(f"予測統計:")
    logger.info(f"- 最小値: {test_predictions.min():.4f}")
    logger.info(f"- 最大値: {test_predictions.max():.4f}")
    logger.info(f"- 平均値: {test_predictions.mean():.4f}")
    logger.info(f"- 標準偏差: {test_predictions.std():.4f}")

    # 実験記録
    experiment_record = {
        "experiment_name": "exp18_ticket022_02_group_kfold",
        "timestamp": timestamp,
        "cv_strategy": "music_similarity_group_kfold",
        "model_type": "lightgbm",
        "n_features": len(common_features),
        "cv_results": {
            "mean_cv_rmse": mean_cv,
            "std_cv_rmse": std_cv,
            "fold_scores": cv_scores,
            "fold_statistics": fold_groups_info,
            "consistency_metrics": {
                "mean_consistency": mean_consistency,
                "std_consistency": std_consistency
            }
        },
        "comparison_with_bpm_stratified": {
            "bpm_stratified_cv": bmp_stratified_cv,
            "bmp_stratified_std": bmp_stratified_std,
            "cv_improvement": cv_improvement,
            "stability_comparison": stability_comparison
        },
        "group_kfold_config": {
            "n_clusters": 50,
            "clustering_method": "kmeans",
            "feature_normalization": "standard_scaler"
        },
        "prediction_stats": {
            "min_prediction": test_predictions.min(),
            "max_prediction": test_predictions.max(),
            "mean_prediction": test_predictions.mean(),
            "std_prediction": test_predictions.std()
        },
        "submission_file": str(submission_path),
        "cv_lb_consistency_hypothesis": "GroupKFoldによるデータリーク防止でCV-LB一貫性向上期待"
    }

    record_path = config.processed_data_dir / f"experiment_record_ticket022_02_{timestamp}.json"
    with open(record_path, 'w') as f:
        json.dump(experiment_record, f, indent=2, default=str)

    logger.info(f"実験記録保存: {record_path}")

    return submission_path, mean_cv, std_cv, cv_improvement

def submit_to_kaggle_manual(submission_path: Path, cv_score: float, improvement: float):
    """Kaggle提出情報表示（手動提出用）"""
    logger.info("Kaggle提出準備...")

    message = f"TICKET-022-02 GroupKFold Strategy 67 Features - CV: {cv_score:.6f} (vs BPM Stratified: {improvement:+.6f})"

    logger.info("=== 手動提出用情報 ===")
    logger.info(f"ファイル: {submission_path}")
    logger.info(f"メッセージ: {message}")
    logger.info("CV-LB一貫性改善効果の検証が目的です")

if __name__ == "__main__":
    logger.info("TICKET-022-02 GroupKFold戦略実験開始")

    submission_path, cv_score, cv_std, improvement = run_group_kfold_experiment()

    logger.success("実験完了！")
    logger.info(f"GroupKFold CV RMSE: {cv_score:.6f} ± {cv_std:.6f}")
    logger.info(f"BPM Stratifiedからの改善: {improvement:+.6f}")

    if abs(improvement) < 0.001:
        logger.info("CV性能は同等レベル。CV-LB一貫性効果に期待")
    elif improvement > 0:
        logger.success("CV性能改善を達成！")
    else:
        logger.warning("CV性能が悪化。BPM Stratifiedが優秀")

    submit_to_kaggle_manual(submission_path, cv_score, improvement)

    logger.info("次のステップ:")
    logger.info("1. LBスコア確認によるCV-LB一貫性検証")
    logger.info("2. 音楽類似性グループ分割の効果分析")
    logger.info("3. 最適CV戦略の決定")