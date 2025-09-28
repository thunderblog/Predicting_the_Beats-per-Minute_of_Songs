"""
TICKET-022-02 GroupKFold戦略実験（軽量版）
音楽類似性グループ別分割によるCV-LB一貫性改善検証
高速実行版（3フォールド + 少数クラスタ）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import warnings
from datetime import datetime

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

from scripts.my_config import config

def create_lightweight_music_groups(X: pd.DataFrame, n_clusters: int = 20, random_state: int = 42) -> np.ndarray:
    """軽量版音楽類似性グループ作成"""
    logger.info(f"軽量版音楽類似性グループ作成開始（{n_clusters}クラスタ）...")

    # 特徴量選択（ID・ターゲット列除外）
    feature_cols = [col for col in X.columns if col not in ['BeatsPerMinute', 'id']]
    features_for_clustering = X[feature_cols]

    # サンプリングによる高速化（大きすぎる場合）
    if len(features_for_clustering) > 100000:
        sample_idx = np.random.choice(len(features_for_clustering), 100000, replace=False)
        clustering_sample = features_for_clustering.iloc[sample_idx]
        logger.info(f"クラスタリング用にサンプリング: {len(clustering_sample)}サンプル")
    else:
        clustering_sample = features_for_clustering

    # 特徴量正規化
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(clustering_sample)

    # K-meansクラスタリング（高速設定）
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=3,  # 高速化
        max_iter=100  # 高速化
    )
    cluster_labels = kmeans.fit_predict(normalized_features)

    # 全データに対してクラスタ割り当て
    if len(features_for_clustering) > 100000:
        # 全特徴量を正規化してクラスタ予測
        all_normalized = scaler.transform(features_for_clustering)
        groups = kmeans.predict(all_normalized)
    else:
        groups = cluster_labels

    logger.info(f"クラスタリング完了:")
    logger.info(f"- 実際のクラスタ数: {len(np.unique(groups))}")

    group_counts = pd.Series(groups).value_counts()
    logger.info(f"- クラスタサイズ統計: 最小{group_counts.min()}, 最大{group_counts.max()}, 平均{group_counts.mean():.1f}")

    return groups

def train_lightgbm_fold_fast(X_train, y_train, X_val, y_val):
    """高速LightGBM訓練"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 高速パラメータ
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

    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500,  # 高速化
        callbacks=[
            lgb.early_stopping(50),  # 高速化
            lgb.log_evaluation(0)
        ]
    )

    return model

def run_lightweight_group_kfold():
    """軽量版GroupKFold実験実行"""
    logger.info("TICKET-022-02 軽量版GroupKFold実験開始...")

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

    # 軽量版音楽類似性グループ作成
    groups = create_lightweight_music_groups(X_train, n_clusters=20)

    # 3フォールドGroupKFold
    n_folds = 3
    gkf = GroupKFold(n_splits=n_folds)

    cv_scores = []
    test_predictions = np.zeros(len(X_test))
    fold_stats = []

    logger.info("軽量版GroupKFold訓練開始...")

    for fold, (train_idx, val_idx) in enumerate(tqdm(gkf.split(X_train, y_train, groups), total=n_folds)):
        logger.info(f"フォールド {fold + 1}/{n_folds}")

        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # フォールド統計
        fold_info = {
            'fold': fold + 1,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'train_mean_bpm': y_fold_train.mean(),
            'val_mean_bpm': y_fold_val.mean(),
            'train_std_bpm': y_fold_train.std(),
            'val_std_bpm': y_fold_val.std()
        }
        fold_stats.append(fold_info)

        # 高速訓練
        model = train_lightgbm_fold_fast(X_fold_train, y_fold_train, X_fold_val, y_fold_val)

        # 検証予測
        val_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
        fold_rmse = np.sqrt(np.mean((y_fold_val - val_pred) ** 2))
        cv_scores.append(fold_rmse)

        # テスト予測
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        test_predictions += test_pred / n_folds

        logger.info(f"フォールド {fold + 1} RMSE: {fold_rmse:.6f}")
        logger.info(f"- 訓練BPM: 平均{y_fold_train.mean():.4f}, 標準偏差{y_fold_train.std():.4f}")
        logger.info(f"- 検証BPM: 平均{y_fold_val.mean():.4f}, 標準偏差{y_fold_val.std():.4f}")

    # CV結果
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    logger.success(f"軽量版GroupKFold完了")
    logger.info(f"平均CV RMSE: {mean_cv:.6f} ± {std_cv:.6f}")
    logger.info(f"フォールド別スコア: {[round(s, 6) for s in cv_scores]}")

    # BPM Stratified戦略との比較
    bmp_stratified_cv = 26.463984
    bmp_stratified_std = 0.006159

    cv_improvement = bmp_stratified_cv - mean_cv
    stability_comparison = bmp_stratified_std / std_cv if std_cv > 0 else float('inf')

    logger.info(f"\n=== BPM Stratified戦略との比較 ===")
    logger.info(f"BPM Stratified: {bmp_stratified_cv:.6f} ± {bmp_stratified_std:.6f}")
    logger.info(f"GroupKFold:     {mean_cv:.6f} ± {std_cv:.6f}")
    logger.info(f"CV改善:         {cv_improvement:+.6f}")
    logger.info(f"安定性比較:     {stability_comparison:.2f}倍{'改善' if stability_comparison > 1 else '劣化'}")

    # GroupKFold品質分析
    fold_df = pd.DataFrame(fold_stats)
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

    submission_path = config.processed_data_dir / f"submission_ticket022_02_lightweight_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    logger.success(f"提出ファイル作成完了: {submission_path}")

    # 結果サマリー
    result_summary = {
        "experiment": "ticket022_02_lightweight_group_kfold",
        "cv_performance": {
            "group_kfold_cv": mean_cv,
            "group_kfold_std": std_cv,
            "improvement_from_bmp_stratified": cv_improvement,
            "stability_comparison": stability_comparison
        },
        "fold_consistency": {
            "mean_consistency": mean_consistency,
            "std_consistency": std_consistency
        },
        "submission_file": str(submission_path),
        "n_features": len(common_features),
        "n_folds": n_folds,
        "n_clusters": 20
    }

    logger.info(f"結果サマリー: {result_summary}")

    return submission_path, mean_cv, cv_improvement, mean_consistency

if __name__ == "__main__":
    logger.info("軽量版GroupKFold実験開始")

    submission_path, cv_score, improvement, consistency = run_lightweight_group_kfold()

    logger.success("実験完了！")
    logger.info(f"GroupKFold CV: {cv_score:.6f}")
    logger.info(f"BPM Stratifiedからの改善: {improvement:+.6f}")
    logger.info(f"フォールド間一貫性: {consistency:.6f}")

    # 結果評価
    if abs(improvement) < 0.001:
        logger.info("CV性能は同等レベル。CV-LB一貫性効果に期待")
    elif improvement > 0:
        logger.success("CV性能改善を達成！")
    else:
        logger.warning("CV性能が悪化。BPM Stratifiedが優秀")

    logger.info("=== 手動提出用情報 ===")
    logger.info(f"ファイル: {submission_path}")
    logger.info(f"メッセージ: TICKET-022-02 Lightweight GroupKFold Strategy - CV: {cv_score:.6f} (vs BPM Stratified: {improvement:+.6f})")
    logger.info("CV-LB一貫性改善効果の検証が目的です")