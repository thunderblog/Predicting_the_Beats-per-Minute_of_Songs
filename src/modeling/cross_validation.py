"""
Cross-validation strategy module for TICKET-022.

Implements multiple CV strategies to improve CV-LB consistency:
- StratifiedKFold with BPM range-based stratification
- GroupKFold with music similarity grouping
- Standard KFold for comparison

Purpose: Reduce CV-LB gap (-0.077) observed in all experiments.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Generator, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.cluster import KMeans


class BaseCVStrategy(ABC):
    """Base class for cross-validation strategies."""

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.strategy_name = self.__class__.__name__

    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/validation indices for each fold."""
        pass

    def validate_split(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate the CV split quality."""
        logger.info(f"{self.strategy_name} 分割品質検証開始...")

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
        logger.info(f"- 各フォールド詳細:\n{stats_df.round(4)}")


class StandardKFoldStrategy(BaseCVStrategy):
    """Standard KFold strategy for baseline comparison."""

    def split(self, X: pd.DataFrame, y: pd.Series) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        return kf.split(X)


class BPMStratifiedKFoldStrategy(BaseCVStrategy):
    """
    StratifiedKFold with BPM range-based stratification.

    Creates BPM bins based on quantiles to ensure balanced distribution
    across folds, addressing the CV-LB consistency issue.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42, n_bins: int = 10):
        super().__init__(n_splits, random_state)
        self.n_bins = n_bins

    def _create_bpm_bins(self, y: pd.Series) -> np.ndarray:
        """Create BPM bins based on quantiles for stratification."""
        # Create bins using quantiles to ensure balanced distribution
        bins = pd.qcut(y, q=self.n_bins, labels=False, duplicates='drop')

        logger.info(f"BPM層化分割用ビン作成完了:")
        logger.info(f"- ビン数: {len(np.unique(bins))}")
        logger.info(f"- BPM範囲: {y.min():.2f} - {y.max():.2f}")

        # Log bin distribution
        bin_counts = pd.Series(bins).value_counts().sort_index()
        logger.info(f"- ビン別サンプル数: {bin_counts.to_dict()}")

        return bins

    def split(self, X: pd.DataFrame, y: pd.Series) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        # Create stratification groups based on BPM ranges
        stratify_labels = self._create_bpm_bins(y)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        return skf.split(X, stratify_labels)


class MusicSimilarityGroupKFoldStrategy(BaseCVStrategy):
    """
    GroupKFold with music similarity-based grouping.

    Groups similar songs together to prevent data leakage and improve
    generalization by ensuring test folds contain truly unseen song types.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42, n_clusters: int = 50):
        super().__init__(n_splits, random_state)
        self.n_clusters = n_clusters

    def _create_music_groups(self, X: pd.DataFrame) -> np.ndarray:
        """Create music similarity groups using feature clustering."""
        logger.info(f"音楽類似性グループ作成開始...")

        # Select features for clustering (exclude target and ID columns)
        feature_cols = [col for col in X.columns if col not in ['BeatsPerMinute', 'id']]
        features_for_clustering = X[feature_cols]

        # Normalize features for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_for_clustering)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        groups = kmeans.fit_predict(normalized_features)

        logger.info(f"クラスタリング完了:")
        logger.info(f"- クラスタ数: {len(np.unique(groups))}")
        logger.info(f"- 各クラスタサイズ: {pd.Series(groups).value_counts().describe()}")

        return groups

    def split(self, X: pd.DataFrame, y: pd.Series) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        # Create groups based on music similarity
        groups = self._create_music_groups(X)

        gkf = GroupKFold(n_splits=self.n_splits)
        return gkf.split(X, y, groups)


class CVStrategyManager:
    """Manager class for comparing multiple CV strategies."""

    def __init__(self):
        self.strategies = {
            'standard_kfold': StandardKFoldStrategy(),
            'bpm_stratified': BPMStratifiedKFoldStrategy(),
            'music_similarity_group': MusicSimilarityGroupKFoldStrategy()
        }

    def compare_strategies(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compare all CV strategies and return analysis results."""
        logger.info("CV戦略比較分析開始...")

        results = []
        for strategy_name, strategy in self.strategies.items():
            logger.info(f"\n=== {strategy_name} 戦略分析 ===")

            # Validate split quality
            strategy.validate_split(X, y)

            # Calculate CV consistency metrics
            fold_means = []
            fold_stds = []

            for fold_idx, (train_idx, val_idx) in enumerate(strategy.split(X, y)):
                val_y = y.iloc[val_idx]
                fold_means.append(val_y.mean())
                fold_stds.append(val_y.std())

            results.append({
                'strategy': strategy_name,
                'mean_consistency': np.std(fold_means),  # Lower is better
                'std_consistency': np.std(fold_stds),    # Lower is better
                'avg_fold_mean': np.mean(fold_means),
                'avg_fold_std': np.mean(fold_stds)
            })

        results_df = pd.DataFrame(results)
        logger.info(f"\nCV戦略比較結果:")
        logger.info(f"{results_df.round(4)}")

        # Recommend best strategy
        best_strategy = results_df.loc[results_df['mean_consistency'].idxmin(), 'strategy']
        logger.info(f"\n推奨CV戦略: {best_strategy}")

        return results_df

    def get_strategy(self, strategy_name: str) -> BaseCVStrategy:
        """Get a specific CV strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(self.strategies.keys())}")
        return self.strategies[strategy_name]


def create_cv_strategy(strategy_type: str = 'bpm_stratified', **kwargs) -> BaseCVStrategy:
    """
    Factory function to create CV strategy instances.

    Args:
        strategy_type: Type of CV strategy ('standard_kfold', 'bpm_stratified', 'music_similarity_group')
        **kwargs: Additional parameters for the strategy

    Returns:
        CV strategy instance
    """
    strategies = {
        'standard_kfold': StandardKFoldStrategy,
        'bpm_stratified': BPMStratifiedKFoldStrategy,
        'music_similarity_group': MusicSimilarityGroupKFoldStrategy
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}. Available: {list(strategies.keys())}")

    return strategies[strategy_type](**kwargs)