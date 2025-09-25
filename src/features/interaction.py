from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseFeatureCreator, FeatureUtils


class BasicInteractionCreator(BaseFeatureCreator):
    """基本的な交互作用特徴量を作成するクラス。"""

    def __init__(self):
        super().__init__("BasicInteraction")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """既存の変数間の基本的な交互作用特徴量を作成する。

        Args:
            df: 元の特徴量を含むデータフレーム

        Returns:
            交互作用特徴量が追加されたデータフレーム
        """
        logger.info("基本的な交互作用特徴量を作成中...")

        df_features = df.copy()
        original_count = len(df.columns)
        new_features = {}

        # リズムとエネルギーの交互作用
        if all(col in df.columns for col in ["RhythmScore", "Energy"]):
            new_features["rhythm_energy_product"] = df["RhythmScore"] * df["Energy"]
            new_features["rhythm_energy_ratio"] = FeatureUtils.safe_divide(
                df["RhythmScore"], df["Energy"]
            )

        # 音声特徴量の組み合わせ
        if all(col in df.columns for col in ["AudioLoudness", "VocalContent"]):
            new_features["loudness_vocal_product"] = df["AudioLoudness"] * df["VocalContent"]

        if all(col in df.columns for col in ["AcousticQuality", "InstrumentalScore"]):
            new_features["acoustic_instrumental_ratio"] = FeatureUtils.safe_divide(
                df["AcousticQuality"], df["InstrumentalScore"]
            )

        # パフォーマンスとムード特徴量
        if all(col in df.columns for col in ["LivePerformanceLikelihood", "MoodScore"]):
            new_features["live_mood_product"] = df["LivePerformanceLikelihood"] * df["MoodScore"]

        if all(col in df.columns for col in ["Energy", "MoodScore"]):
            new_features["energy_mood_product"] = df["Energy"] * df["MoodScore"]

        # 複雑な交互作用
        if all(col in df.columns for col in ["RhythmScore", "MoodScore", "Energy"]):
            new_features["rhythm_mood_energy"] = df["RhythmScore"] * df["MoodScore"] * df["Energy"]

        # 特徴量を安全に追加
        df_features = FeatureUtils.add_features_safely(df_features, new_features)
        self._created_features = list(new_features.keys())

        self.log_feature_creation(original_count, len(df_features.columns))
        return df_features


class ComprehensiveInteractionCreator(BaseFeatureCreator):
    """包括的な交互作用特徴量を作成するクラス（Kaggleサンプルコード手法）。"""

    def __init__(self):
        super().__init__("ComprehensiveInteraction")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """全数値特徴量ペアの積、二乗項、比率特徴量を系統的に生成する。

        Args:
            df: 元の特徴量を含むデータフレーム

        Returns:
            包括的交互作用特徴量が追加されたデータフレーム
        """
        logger.info("包括的交互作用特徴量を作成中...")

        df_features = df.copy()
        original_count = len(df.columns)

        # 基本数値特徴量を定義
        num_cols = FeatureUtils.get_numerical_columns()
        available_num_cols = FeatureUtils.filter_available_columns(df, num_cols)

        if not available_num_cols:
            logger.warning("数値特徴量が見つかりません")
            return df_features

        logger.info(f"処理対象数値特徴量: {len(available_num_cols)}個")

        new_features = {}

        # 1. 全ペア組み合わせの積特徴量と二乗項を生成
        logger.info("積特徴量と二乗項を生成中...")
        for i in range(len(available_num_cols)):
            for j in range(i, len(available_num_cols)):
                col1 = available_num_cols[i]
                col2 = available_num_cols[j]

                # 積特徴量 (i=jの場合は二乗項)
                new_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]

                # 二乗項は別途明示的に作成
                if i == j:
                    new_features[f'{col1}_squared'] = df[col1] * df[col1]

        # 2. 全ペア組み合わせの比率特徴量を生成
        logger.info("比率特徴量を生成中...")
        for col1 in available_num_cols:
            for col2 in available_num_cols:
                if col1 != col2:  # 自分自身での除算は除外
                    new_features[f'{col1}_div_{col2}'] = FeatureUtils.safe_divide(
                        df[col1], df[col2], epsilon=1e-6
                    )

        # 3. 新特徴量を一括で追加
        logger.info("新特徴量を一括追加中...")
        df_features = FeatureUtils.add_features_safely(df_features, new_features)
        self._created_features = list(new_features.keys())

        feature_count = len(new_features)
        logger.success(f"包括的交互作用特徴量を作成完了: {feature_count}個の新特徴量を追加")
        self.log_feature_creation(original_count, len(df_features.columns))

        return df_features


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """後方互換性のための関数。基本的な交互作用特徴量を作成する。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        交互作用特徴量が追加されたデータフレーム
    """
    creator = BasicInteractionCreator()
    return creator.create_features(df)


def create_comprehensive_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """後方互換性のための関数。包括的な交互作用特徴量を作成する。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        包括的交互作用特徴量が追加されたデータフレーム
    """
    creator = ComprehensiveInteractionCreator()
    return creator.create_features(df)