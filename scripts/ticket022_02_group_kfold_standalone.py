"""
TICKET-022-02 GroupKFold戦略実験（スタンドアロン版）
音楽類似性グループ別分割によるCV-LB一貫性改善検証
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
from typing import Generator, Tuple

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

from scripts.my_config import config

class MusicSimilarityGroupKFoldStrategy:
    """音楽類似性グループ別KFold戦略"""

    def __init__(self, n_splits: int = 5, random_state: int = 42, n_clusters: int = 50):
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_clusters = n_clusters

    def _create_music_groups(self, X: pd.DataFrame) -> np.ndarray:
        """音楽類似性グループを特徴量クラスタリングで作成"""
        logger.info(f"音楽類似性グループ作成開始...")

        # 特徴量選択（ID・ターゲット列除外）
        feature_cols = [col for col in X.columns if col not in ['BeatsPerMinute', 'id']]
        features_for_clustering = X[feature_cols]

        # 特徴量正規化
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_for_clustering)

        # K-meansクラスタリング
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        groups = kmeans.fit_predict(normalized_features)

        logger.info(f"クラスタリング完了:")
        logger.info(f"- クラスタ数: {len(np.unique(groups))}")

        group_counts = pd.Series(groups).value_counts()
        logger.info(f"- 各クラスタサイズ統計:")
        logger.info(f"  - 最小: {group_counts.min()}")
        logger.info(f"  - 最大: {group_counts.max()}")
        logger.info(f"  - 平均: {group_counts.mean():.1f}")
        logger.info(f"  - 標準偏差: {group_counts.std():.1f}")

        return groups

    def split(self, X: pd.DataFrame, y: pd.Series) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """GroupKFold分割を実行"""
        # 音楽類似性グループ作成
        groups = self._create_music_groups(X)

        # GroupKFold実行
        gkf = GroupKFold(n_splits=self.n_splits)
        return gkf.split(X, y, groups)

    def validate_split(self, X: pd.DataFrame, y: pd.Series) -> None:
        """分割品質を検証"""
        logger.info(f"GroupKFold分割品質検証開始...")

        fold_stats = []
        for fold_idx, (train_idx, val_idx) in enumerate(self.split(X, y)):
            train_y = y.iloc[train_idx]
            val_y = y.iloc[val_idx]

            fold_stats.append({
                'fold': fold_idx + 1,
                'train_samples': len(train_idx),
                'val_samples': len(val_idx),
                'train_mean_bpm': train_y.mean(),
                'val_mean_bpm': val_y.mean(),
                'train_std_bpm': train_y.std(),
                'val_std_bpm': val_y.std(),
                'mean_diff': abs(train_y.mean() - val_y.mean()),
                'std_diff': abs(train_y.std() - val_y.std())
            })

        stats_df = pd.DataFrame(fold_stats)
        logger.info(f"フォールド間BPM統計一貫性:")
        logger.info(f"- 平均差の平均: {stats_df['mean_diff'].mean():.4f}")
        logger.info(f"- 標準偏差差の平均: {stats_df['std_diff'].mean():.4f}")
        logger.info(f"- 各フォールド詳細:")
        for _, row in stats_df.iterrows():
            logger.info(f"  フォールド{int(row['fold'])}: 訓練{row['train_samples']}, 検証{row['val_samples']}, "
                       f"BPM平均差{row['mean_diff']:.4f}")

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

    # データ読み込み
    train_data_path = config.processed_data_dir / "train_unified_75_features.csv"
    test_data_path = config.processed_data_dir / "test_unified_75_features.csv"

    logger.info(f"訓練データ読み込み: {train_data_path}")
    train_df = pd.read_csv(train_data_path)

    logger.info(f"テストデータ読み込み: {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    # 特徴量準備
    train_feature_cols = [col for col in train_df.columns if col not in ["id", "BeatsPerMinute"]]
    test_feature_cols = [col for col in test_df.columns if col != "id"]
    common_features = sorted(list(set(train_feature_cols) & set(test_feature_cols)))

    X_train = train_df[common_features]
    y_train = train_df["BeatsPerMinute"]
    X_test = test_df[common_features]

    logger.info(f"特徴量数: {len(common_features)}, 訓練サンプル: {len(X_train)}, テストサンプル: {len(X_test)}")

    # GroupKFold戦略初期化
    logger.info("音楽類似性GroupKFold戦略初期化...")
    group_strategy = MusicSimilarityGroupKFoldStrategy(
        n_splits=5,
        random_state=config.random_state,
        n_clusters=50  # クラスタ数
    )

    # 分割品質検証
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

    # BPM Stratified戦略との比較
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

    return submission_path, mean_cv, std_cv, cv_improvement, mean_consistency, std_consistency

if __name__ == "__main__":
    logger.info("TICKET-022-02 GroupKFold戦略実験開始")

    submission_path, cv_score, cv_std, improvement, mean_consistency, std_consistency = run_group_kfold_experiment()

    logger.success("実験完了！")
    logger.info(f"GroupKFold CV RMSE: {cv_score:.6f} ± {cv_std:.6f}")
    logger.info(f"BPM Stratifiedからの改善: {improvement:+.6f}")
    logger.info(f"フォールド間一貫性: 平均値{mean_consistency:.6f}, 標準偏差{std_consistency:.6f}")

    # 結果評価
    if abs(improvement) < 0.001:
        logger.info("CV性能は同等レベル。CV-LB一貫性効果に期待")
    elif improvement > 0:
        logger.success("CV性能改善を達成！")
    else:
        logger.warning("CV性能が悪化。BPM Stratifiedが優秀")

    logger.info("=== 手動提出用情報 ===")
    logger.info(f"ファイル: {submission_path}")
    logger.info(f"メッセージ: TICKET-022-02 GroupKFold Strategy 67 Features - CV: {cv_score:.6f} (vs BPM Stratified: {improvement:+.6f})")
    logger.info("CV-LB一貫性改善効果の検証が目的です")

    logger.info("次のステップ:")
    logger.info("1. LBスコア確認によるCV-LB一貫性検証")
    logger.info("2. 音楽類似性グループ分割の効果分析")
    logger.info("3. 最適CV戦略の決定")