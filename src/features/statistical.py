import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseFeatureCreator, FeatureUtils


class StatisticalFeatureCreator(BaseFeatureCreator):
    """統計的特徴量を作成するクラス。"""

    def __init__(self):
        super().__init__("StatisticalFeatures")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """既存の数値列から統計的特徴量を作成する。

        Args:
            df: 入力データフレーム

        Returns:
            統計的特徴量が追加されたデータフレーム
        """
        logger.info("統計的特徴量を作成中...")

        df_features = df.copy()
        original_count = len(df.columns)

        # 数値特徴量を選択（idとターゲットを除く）
        numerical_cols = [
            "RhythmScore",
            "AudioLoudness",
            "VocalContent",
            "AcousticQuality",
            "InstrumentalScore",
            "LivePerformanceLikelihood",
            "MoodScore",
            "Energy",
        ]

        # 存在する特徴量のみを使用
        available_cols = FeatureUtils.filter_available_columns(df, numerical_cols)

        if not available_cols:
            logger.warning("統計的特徴量用の数値列が見つかりません")
            return df_features

        new_features = {}

        # 全スコアの合計
        new_features["total_score"] = df[available_cols].sum(axis=1)

        # 全スコアの平均
        new_features["mean_score"] = df[available_cols].mean(axis=1)

        # スコアの標準偏差
        new_features["std_score"] = df[available_cols].std(axis=1)

        # 最小値と最大値
        new_features["min_score"] = df[available_cols].min(axis=1)
        new_features["max_score"] = df[available_cols].max(axis=1)
        new_features["range_score"] = new_features["max_score"] - new_features["min_score"]

        # 特徴量を安全に追加
        df_features = FeatureUtils.add_features_safely(df_features, new_features)
        self._created_features = list(new_features.keys())

        self.log_feature_creation(original_count, len(df_features.columns))
        return df_features


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """後方互換性のための関数。統計的特徴量を作成する。

    Args:
        df: 入力データフレーム

    Returns:
        統計的特徴量が追加されたデータフレーム
    """
    creator = StatisticalFeatureCreator()
    return creator.create_features(df)