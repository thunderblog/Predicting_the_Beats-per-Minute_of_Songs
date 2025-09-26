"""
Log Transformation Feature Creator - TICKET-017-02

対数変換特徴量を作成するモジュール。分布の歪み補正により予測精度向上を図る。
"""

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseFeatureCreator


class LogTransformFeatureCreator(BaseFeatureCreator):
    """対数変換特徴量作成器 (TICKET-017-02)。

    全数値特徴量のlog1p変換と、変換特徴量同士の組み合わせ特徴量を生成して
    分布の歪み補正により予測精度向上を図る。

    特徴:
    - 基本log1p変換特徴量（AudioLoudnessを除く8個）
    - log変換特徴量同士の組み合わせ特徴量
    - 対数空間での統計特徴量
    - 分布正規化指標

    Attributes:
        target_features: 対数変換対象の特徴量リスト
        exclude_features: 除外する特徴量リスト
    """

    def __init__(self, exclude_features=None):
        """初期化。

        Args:
            exclude_features: 除外する特徴量のリスト（デフォルト: AudioLoudness）
        """
        super().__init__("LogTransform")

        # 対象特徴量（AudioLoudnessを除く8個）
        self.target_features = [
            'RhythmScore', 'VocalContent', 'AcousticQuality',
            'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
            'TrackDurationMs', 'Energy'
        ]

        # 除外特徴量設定
        if exclude_features is None:
            exclude_features = ['AudioLoudness']
        self.exclude_features = exclude_features

        # 除外特徴量を対象から削除
        self.target_features = [
            feat for feat in self.target_features
            if feat not in self.exclude_features
        ]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """対数変換特徴量を作成する。

        Args:
            df: 元の特徴量を含むデータフレーム

        Returns:
            対数変換特徴量が追加されたデータフレーム
        """
        # 作成特徴量リストをリセット
        self._created_features = []
        logger.info(f"{self.name}特徴量作成器: 対数変換特徴量を作成中...")

        df_features = df.copy()

        # 存在する特徴量のみ対象とする
        available_features = [col for col in self.target_features if col in df.columns]

        if not available_features:
            logger.warning("対数変換対象の特徴量が見つかりません")
            return df_features

        logger.info(f"対数変換対象: {len(available_features)}特徴量 - {available_features}")

        # 1. 基本log1p変換特徴量の作成
        log_feature_names = self._create_basic_log_features(df_features, available_features)

        # 2. log変換特徴量同士の組み合わせ特徴量
        self._create_log_combination_features(df_features, log_feature_names)

        # 3. 対数空間での統計特徴量
        self._create_log_statistical_features(df_features, log_feature_names)

        # 4. 分布正規化指標
        self._create_distribution_normalization_indicator(df, df_features, available_features)

        # 作成された特徴量数をログ出力
        n_new_features = len(df_features.columns) - len(df.columns)
        logger.success(f"{self.name}特徴量作成完了: {n_new_features}個の新特徴量を追加")

        return df_features

    def _create_basic_log_features(self, df_features: pd.DataFrame, available_features: list) -> list:
        """基本log1p変換特徴量を作成する。

        Args:
            df_features: 特徴量データフレーム
            available_features: 利用可能な特徴量リスト

        Returns:
            作成されたlog特徴量名のリスト
        """
        logger.info("基本log1p変換特徴量を作成中...")

        log_feature_names = []
        for feature in available_features:
            log_feature_name = f"log1p_{feature}"

            # log1p変換（負値対応、1e-8でクリッピング）
            feature_values = df_features[feature].clip(lower=1e-8)
            df_features[log_feature_name] = np.log1p(feature_values)

            log_feature_names.append(log_feature_name)
            self._created_features.append(log_feature_name)

        logger.info(f"基本log1p変換完了: {len(log_feature_names)}特徴量")
        return log_feature_names

    def _create_log_combination_features(self, df_features: pd.DataFrame, log_feature_names: list):
        """log変換特徴量同士の組み合わせ特徴量を作成する。

        Args:
            df_features: 特徴量データフレーム
            log_feature_names: log特徴量名のリスト
        """
        if len(log_feature_names) < 2:
            return

        logger.info("log変換特徴量の組み合わせを作成中...")
        combination_count = 0

        # ペアワイズ積特徴量
        for i, feature1 in enumerate(log_feature_names):
            for j, feature2 in enumerate(log_feature_names[i+1:], i+1):
                # log(A) * log(B) = log(A^B) の近似
                combo_name = f"{feature1}_x_{feature2}"
                df_features[combo_name] = df_features[feature1] * df_features[feature2]

                self._created_features.append(combo_name)
                combination_count += 1

        # 重要な比率特徴量
        if len(log_feature_names) >= 3:
            # log(TrackDurationMs)を基準とした比率
            if 'log1p_TrackDurationMs' in log_feature_names:
                base_log = 'log1p_TrackDurationMs'
                for other_log in log_feature_names:
                    if other_log != base_log:
                        ratio_name = f"{other_log}_div_{base_log}"
                        # ゼロ除算回避
                        df_features[ratio_name] = df_features[other_log] / (df_features[base_log] + 1e-8)

                        self._created_features.append(ratio_name)
                        combination_count += 1

            # Energy - RhythmScore log space関係
            if 'log1p_Energy' in log_feature_names and 'log1p_RhythmScore' in log_feature_names:
                harmony_name = 'log_energy_rhythm_harmony'
                df_features[harmony_name] = (
                    df_features['log1p_Energy'] + df_features['log1p_RhythmScore']
                ) / 2

                self._created_features.append(harmony_name)
                combination_count += 1

        logger.info(f"組み合わせ特徴量完了: {combination_count}特徴量")

    def _create_log_statistical_features(self, df_features: pd.DataFrame, log_feature_names: list):
        """対数空間での統計特徴量を作成する。

        Args:
            df_features: 特徴量データフレーム
            log_feature_names: log特徴量名のリスト
        """
        if len(log_feature_names) < 2:
            return

        logger.info("対数空間統計特徴量を作成中...")

        log_values = df_features[log_feature_names]

        # 対数空間での統計量
        stat_features = {
            'log_features_mean': log_values.mean(axis=1),
            'log_features_std': log_values.std(axis=1),
            'log_features_range': log_values.max(axis=1) - log_values.min(axis=1),
            'log_features_geometric_mean': np.expm1(log_values.mean(axis=1))  # 幾何平均
        }

        for name, values in stat_features.items():
            df_features[name] = values
            self._created_features.append(name)

        logger.info("対数空間統計特徴量完了: 4特徴量")

    def _create_distribution_normalization_indicator(
        self,
        df_original: pd.DataFrame,
        df_features: pd.DataFrame,
        available_features: list
    ):
        """分布正規化指標を作成する。

        Args:
            df_original: 元のデータフレーム
            df_features: 特徴量データフレーム
            available_features: 利用可能な特徴量リスト
        """
        logger.info("分布正規化指標を作成中...")

        # 元特徴量の歪度改善指標
        skewness_improvements = []
        for original_feature in available_features:
            if original_feature in df_original.columns:
                original_skew = abs(df_original[original_feature].skew())
                log_feature = f"log1p_{original_feature}"
                if log_feature in df_features.columns:
                    log_skew = abs(df_features[log_feature].skew())
                    improvement = max(0, original_skew - log_skew)  # 改善度（正値のみ）
                    skewness_improvements.append(improvement)

        if skewness_improvements:
            benefit_name = 'log_transformation_benefit'
            df_features[benefit_name] = np.mean(skewness_improvements)
            self._created_features.append(benefit_name)

            logger.info(f"分布正規化指標完了: 平均改善度 {np.mean(skewness_improvements):.3f}")

    def get_feature_info(self) -> dict:
        """特徴量作成器の情報を取得する。

        Returns:
            特徴量作成器の詳細情報
        """
        return {
            "name": self.name,
            "description": "対数変換特徴量作成器 (TICKET-017-02)",
            "target_features": self.target_features,
            "exclude_features": self.exclude_features,
            "created_features_count": len(self._created_features),
            "feature_types": {
                "basic_log": "基本log1p変換特徴量",
                "combinations": "log変換特徴量同士の組み合わせ",
                "statistics": "対数空間統計特徴量",
                "normalization_indicator": "分布正規化指標"
            }
        }