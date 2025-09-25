import pandas as pd
from loguru import logger

from .base import BaseFeatureCreator, FeatureUtils


class MusicGenreFeatureCreator(BaseFeatureCreator):
    """音楽ジャンル推定特徴量を作成するクラス。"""

    def __init__(self):
        super().__init__("MusicGenreFeatures")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """音楽理論に基づいて、特徴量の組み合わせから暗黙的なジャンル特徴量を推定する。

        音楽ジャンルとBPMの関係：
        - ダンス系: 高エネルギー × 高リズム (120-140+ BPM)
        - アコースティック系: 高音響品質 × 高楽器演奏 (60-120 BPM)
        - バラード系: 高ボーカル × 高ムード (70-100 BPM)
        - ロック/ポップ系: 中エネルギー × ライブ演奏 (90-130 BPM)
        - エレクトロニック系: 低ボーカル × 高エネルギー (100-180 BPM)
        - アンビエント系: 低エネルギー × 高音響品質 (60-90 BPM)

        Args:
            df: 元の特徴量を含むデータフレーム

        Returns:
            音楽ジャンル推定特徴量が追加されたデータフレーム
        """
        logger.info("音楽ジャンル推定特徴量を作成中...")

        df_features = df.copy()
        original_count = len(df.columns)
        new_features = {}

        # ダンス系ジャンル特徴量: Energy×RhythmScore
        if all(col in df.columns for col in ["Energy", "RhythmScore"]):
            new_features["dance_genre_score"] = df["Energy"] * df["RhythmScore"]

        # アコースティック系ジャンル特徴量: AcousticQuality×InstrumentalScore
        if all(col in df.columns for col in ["AcousticQuality", "InstrumentalScore"]):
            new_features["acoustic_genre_score"] = df["AcousticQuality"] * df["InstrumentalScore"]

        # バラード系ジャンル特徴量: VocalContent×MoodScore
        if all(col in df.columns for col in ["VocalContent", "MoodScore"]):
            new_features["ballad_genre_score"] = df["VocalContent"] * df["MoodScore"]

        # ロック/ポップ系: 中程度のエネルギー × ライブ演奏っぽさ
        if all(col in df.columns for col in ["Energy", "LivePerformanceLikelihood"]):
            new_features["rock_genre_score"] = df["Energy"] * df["LivePerformanceLikelihood"]

        # エレクトロニック系: 低ボーカル × 高エネルギー
        if all(col in df.columns for col in ["VocalContent", "Energy"]):
            vocal_max = df["VocalContent"].max()
            if vocal_max > 0:
                normalized_vocal = df["VocalContent"] / vocal_max
                new_features["electronic_genre_score"] = (1 - normalized_vocal) * df["Energy"]

        # アンビエント/チルアウト系: 低エネルギー × 高音響品質
        if all(col in df.columns for col in ["Energy", "AcousticQuality"]):
            energy_max = df["Energy"].max()
            if energy_max > 0:
                normalized_energy = df["Energy"] / energy_max
                new_features["ambient_genre_score"] = (1 - normalized_energy) * df["AcousticQuality"]

        # 特徴量を安全に追加
        df_features = FeatureUtils.add_features_safely(df_features, new_features)
        self._created_features = list(new_features.keys())

        self.log_feature_creation(original_count, len(df_features.columns))
        return df_features


def create_music_genre_features(df: pd.DataFrame) -> pd.DataFrame:
    """後方互換性のための関数。音楽ジャンル推定特徴量を作成する。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        音楽ジャンル推定特徴量が追加されたデータフレーム
    """
    creator = MusicGenreFeatureCreator()
    return creator.create_features(df)