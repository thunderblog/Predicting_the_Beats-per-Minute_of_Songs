import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import zscore

from .base import BaseFeatureCreator, FeatureUtils


class AdvancedFeatureCreator(BaseFeatureCreator):
    """独立性の高い高次特徴量を作成するクラス。"""

    def __init__(self):
        super().__init__("AdvancedFeatures")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """独立性の高い高次特徴量を作成する。

        Args:
            df: 元の特徴量を含むデータフレーム

        Returns:
            高次特徴量が追加されたデータフレーム
        """
        logger.info("独立性の高い高次特徴量を作成中...")

        df_features = df.copy()
        original_count = len(df.columns)
        new_features = {}

        # 1. 比率ベース特徴量（ゼロ除算対策付き）
        logger.info("比率ベース特徴量を作成中...")
        if all(col in df.columns for col in ["VocalContent", "Energy"]):
            new_features["vocal_energy_ratio"] = FeatureUtils.safe_divide(
                df["VocalContent"], df["Energy"]
            )

        if all(col in df.columns for col in ["AcousticQuality", "AudioLoudness"]):
            new_features["acoustic_loudness_ratio"] = FeatureUtils.safe_divide(
                df["AcousticQuality"], df["AudioLoudness"]
            )

        if all(col in df.columns for col in ["RhythmScore", "TrackDurationMs"]):
            new_features["rhythm_duration_ratio"] = FeatureUtils.safe_divide(
                df["RhythmScore"], np.log1p(df["TrackDurationMs"])
            )

        if all(col in df.columns for col in ["InstrumentalScore", "LivePerformanceLikelihood"]):
            new_features["instrumental_live_ratio"] = FeatureUtils.safe_divide(
                df["InstrumentalScore"], df["LivePerformanceLikelihood"]
            )

        # 2. 対数変換時間特徴量（スケール正規化）
        if "TrackDurationMs" in df.columns:
            logger.info("対数変換時間特徴量を作成中...")
            log_duration = np.log1p(df["TrackDurationMs"])  # log(1+x)でゼロ値対応

            if "RhythmScore" in df.columns:
                new_features["log_duration_rhythm"] = log_duration * df["RhythmScore"]
            if "Energy" in df.columns:
                new_features["log_duration_energy"] = log_duration * df["Energy"]
            if "MoodScore" in df.columns:
                new_features["log_duration_mood"] = log_duration * df["MoodScore"]

            # 時間の3次元カテゴリ化
            duration_percentiles = np.percentile(df["TrackDurationMs"], [33, 67])
            new_features["duration_category"] = pd.cut(
                df["TrackDurationMs"],
                bins=[0] + list(duration_percentiles) + [np.inf],
                labels=[0, 1, 2]  # short, medium, long
            ).astype(int)

        # 3. 標準化済み交互作用特徴量（Z-score正規化後の積）
        logger.info("標準化済み交互作用特徴量を作成中...")

        # 主要特徴量が存在する場合のZ-score正規化
        standardized_features = {}
        for col in ["VocalContent", "Energy", "RhythmScore", "MoodScore", "AcousticQuality", "AudioLoudness"]:
            if col in df.columns:
                standardized_features[col] = zscore(df[col])

        # 標準化済み交互作用
        if "VocalContent" in standardized_features and "MoodScore" in standardized_features:
            new_features["standardized_vocal_mood"] = (
                standardized_features["VocalContent"] * standardized_features["MoodScore"]
            )

        if "Energy" in standardized_features and "RhythmScore" in standardized_features:
            new_features["standardized_energy_rhythm"] = (
                standardized_features["Energy"] * standardized_features["RhythmScore"]
            )

        if "AcousticQuality" in standardized_features and "AudioLoudness" in standardized_features:
            new_features["standardized_acoustic_loudness"] = (
                standardized_features["AcousticQuality"] * standardized_features["AudioLoudness"]
            )

        if "VocalContent" in standardized_features and "Energy" in standardized_features:
            new_features["standardized_vocal_energy"] = (
                standardized_features["VocalContent"] * standardized_features["Energy"]
            )

        if "RhythmScore" in standardized_features and "MoodScore" in standardized_features:
            new_features["standardized_rhythm_mood"] = (
                standardized_features["RhythmScore"] * standardized_features["MoodScore"]
            )

        # 4. 音楽理論ベース複雑指標（BPM予測特化）
        logger.info("音楽理論ベース複雑指標を作成中...")

        # テンポ複雑性指標（リズム×音響品質/エネルギー）
        if all(col in df.columns for col in ["RhythmScore", "AcousticQuality", "Energy"]):
            new_features["tempo_complexity"] = FeatureUtils.safe_divide(
                df["RhythmScore"] * df["AcousticQuality"], df["Energy"]
            )

        # パフォーマンス動的指標（ライブ性×楽器性）
        if all(col in df.columns for col in ["LivePerformanceLikelihood", "InstrumentalScore"]):
            new_features["performance_dynamics"] = (
                df["LivePerformanceLikelihood"] * df["InstrumentalScore"]
            )

        # 音楽密度指標（音量×ボーカル×楽器/時間）
        if all(col in df.columns for col in ["AudioLoudness", "VocalContent", "InstrumentalScore", "TrackDurationMs"]):
            log_duration = np.log1p(df["TrackDurationMs"])
            new_features["music_density"] = FeatureUtils.safe_divide(
                df["AudioLoudness"] * df["VocalContent"] * df["InstrumentalScore"], log_duration
            )

        # ハーモニック複雑性（音響品質×ムード/エネルギー）
        if all(col in df.columns for col in ["AcousticQuality", "MoodScore", "Energy"]):
            new_features["harmonic_complexity"] = FeatureUtils.safe_divide(
                df["AcousticQuality"] * df["MoodScore"], df["Energy"]
            )

        # 楽曲構造推定（リズム×時間×ライブ性）
        if all(col in df.columns for col in ["RhythmScore", "TrackDurationMs", "LivePerformanceLikelihood"]):
            log_duration = np.log1p(df["TrackDurationMs"])
            new_features["song_structure_indicator"] = (
                df["RhythmScore"] * log_duration * df["LivePerformanceLikelihood"]
            )

        # 特徴量を安全に追加
        df_features = FeatureUtils.add_features_safely(df_features, new_features)
        self._created_features = list(new_features.keys())

        n_new_features = len(df_features.columns) - original_count
        logger.success(f"高次特徴量を作成完了: {n_new_features}個の新特徴量を追加")
        self.log_feature_creation(original_count, len(df_features.columns))

        return df_features


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """後方互換性のための関数。高次特徴量を作成する。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        高次特徴量が追加されたデータフレーム
    """
    creator = AdvancedFeatureCreator()
    return creator.create_features(df)