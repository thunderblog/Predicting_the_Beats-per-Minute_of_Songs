import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseFeatureCreator, FeatureUtils


class DurationFeatureCreator(BaseFeatureCreator):
    """TrackDurationMsから時間に基づく特徴量を作成するクラス。"""

    def __init__(self):
        super().__init__("DurationFeatures")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """TrackDurationMsから時間に基づく特徴量を作成する。

        Args:
            df: TrackDurationMs列を含む入力データフレーム

        Returns:
            時間特徴量が追加されたデータフレーム
        """
        logger.info("トラック時間特徴量を作成中...")

        if "TrackDurationMs" not in df.columns:
            logger.warning("TrackDurationMs列が見つかりません")
            return df

        df_features = df.copy()
        original_count = len(df.columns)
        new_features = {}

        # ミリ秒を他の時間単位に変換
        new_features["track_duration_seconds"] = df["TrackDurationMs"] / 1000
        new_features["track_duration_minutes"] = df["TrackDurationMs"] / (1000 * 60)

        # 時間カテゴリ
        new_features["is_short_track"] = (df["TrackDurationMs"] < 180000).astype(int)  # 3分未満
        new_features["is_long_track"] = (df["TrackDurationMs"] > 300000).astype(int)  # 5分超

        # 時間区分のカテゴリ化
        duration_bins = [0, 120000, 180000, 240000, 300000, float("inf")]
        duration_labels = ["very_short", "short", "medium", "long", "very_long"]
        duration_category = pd.cut(
            df["TrackDurationMs"],
            bins=duration_bins,
            labels=duration_labels,
            include_lowest=True
        ).astype(str)

        # 時間カテゴリのワンホットエンコーディング
        duration_dummies = pd.get_dummies(duration_category, prefix="duration")

        # 辞書に追加
        for col in duration_dummies.columns:
            new_features[col] = duration_dummies[col]

        # 特徴量を安全に追加
        df_features = FeatureUtils.add_features_safely(df_features, new_features)
        self._created_features = list(new_features.keys())

        self.log_feature_creation(original_count, len(df_features.columns))
        return df_features


def create_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """後方互換性のための関数。時間特徴量を作成する。

    Args:
        df: TrackDurationMs列を含む入力データフレーム

    Returns:
        時間特徴量が追加されたデータフレーム
    """
    creator = DurationFeatureCreator()
    return creator.create_features(df)