"""
Binning Feature Creator - TICKET-017-03

カテゴリ・ビニング特徴量を作成するモジュール。
分位数分割による離散化で非線形パターンを捕捉し予測精度向上を図る。
"""

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseFeatureCreator


class BinningFeatureCreator(BaseFeatureCreator):
    """ビニング・カテゴリ特徴量作成器 (TICKET-017-03)。

    数値特徴量を分位数分割（septile、decile、quintile）でカテゴリ化し、
    各カテゴリの統計特徴量を生成して非線形関係を捕捉する。

    特徴:
    - 7分位（septile）・10分位（decile）・5分位（quintile）分割
    - カテゴリ別統計特徴量（平均、分散、カウント）
    - log変換特徴量のビニング
    - ビン間の相互作用特徴量

    Attributes:
        target_features: ビニング対象の数値特徴量リスト
        binning_configs: ビニング設定（分位数、統計量）
    """

    def __init__(self, target_features=None, binning_configs=None):
        """初期化。

        Args:
            target_features: ビニング対象の特徴量リスト（デフォルト: 全数値特徴量）
            binning_configs: ビニング設定辞書
        """
        super().__init__("Binning")

        # デフォルト対象特徴量（全基本数値特徴量）
        if target_features is None:
            target_features = [
                'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
                'TrackDurationMs', 'Energy'
            ]
        self.target_features = target_features

        # デフォルトビニング設定
        if binning_configs is None:
            binning_configs = {
                'septile': {'n_bins': 7, 'stats': ['mean', 'std', 'count']},
                'decile': {'n_bins': 10, 'stats': ['mean', 'count']},
                'quintile': {'n_bins': 5, 'stats': ['mean', 'std']},
            }
        self.binning_configs = binning_configs

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ビニング・カテゴリ特徴量を作成する。

        Args:
            df: 元の特徴量を含むデータフレーム

        Returns:
            ビニング特徴量が追加されたデータフレーム
        """
        # 作成特徴量リストをリセット
        self._created_features = []
        logger.info(f"{self.name}特徴量作成器: ビニング・カテゴリ特徴量を作成中...")

        df_features = df.copy()

        # 存在する特徴量のみ対象とする
        available_features = [col for col in self.target_features if col in df.columns]

        if not available_features:
            logger.warning("ビニング対象の特徴量が見つかりません")
            return df_features

        logger.info(f"ビニング対象: {len(available_features)}特徴量 - {available_features}")

        # 1. 基本数値特徴量のビニング
        self._create_basic_binning_features(df_features, available_features)

        # 2. log変換特徴量のビニング
        self._create_log_binning_features(df_features)

        # 3. ビン統計特徴量
        self._create_bin_statistical_features(df_features, available_features)

        # 4. ビン間相互作用特徴量
        self._create_bin_interaction_features(df_features)

        # 作成された特徴量数をログ出力
        n_new_features = len(df_features.columns) - len(df.columns)
        logger.success(f"{self.name}特徴量作成完了: {n_new_features}個の新特徴量を追加")

        return df_features

    def _create_basic_binning_features(self, df_features: pd.DataFrame, available_features: list):
        """基本数値特徴量のビニング特徴量を作成する。

        Args:
            df_features: 特徴量データフレーム
            available_features: 利用可能な特徴量リスト
        """
        logger.info("基本数値特徴量のビニングを作成中...")

        binning_count = 0

        for binning_type, config in self.binning_configs.items():
            n_bins = config['n_bins']

            logger.info(f"  {binning_type}分割（{n_bins}分位）を実行中...")

            for feature in available_features:
                feature_values = df_features[feature]

                # 分位数でビニング（等頻度分割）
                try:
                    # qcutで等頻度分割、duplicates='drop'で重複ラベル処理
                    binned_feature_name = f"{feature}_{binning_type}_bin"

                    binned_values, bin_edges = pd.qcut(
                        feature_values,
                        q=n_bins,
                        retbins=True,
                        duplicates='drop',
                        labels=False  # 数値ラベル使用
                    )

                    df_features[binned_feature_name] = binned_values
                    self._created_features.append(binned_feature_name)
                    binning_count += 1

                except Exception as e:
                    # 分位数分割が失敗した場合（値の種類が少ない等）
                    logger.warning(f"    {feature}の{binning_type}分割スキップ: {e}")
                    continue

        logger.info(f"基本ビニング完了: {binning_count}特徴量")

    def _create_log_binning_features(self, df_features: pd.DataFrame):
        """log変換特徴量のビニング特徴量を作成する。

        Args:
            df_features: 特徴量データフレーム
        """
        logger.info("log変換特徴量のビニングを作成中...")

        # log変換特徴量を検索
        log_features = [col for col in df_features.columns if col.startswith('log1p_')]

        if not log_features:
            logger.info("  log変換特徴量が見つかりません、スキップします")
            return

        logger.info(f"  {len(log_features)}個のlog変換特徴量を処理中...")

        # quintile（5分位）分割のみ適用
        binning_config = self.binning_configs.get('quintile', {'n_bins': 5})
        n_bins = binning_config['n_bins']

        binning_count = 0

        for log_feature in log_features:
            feature_values = df_features[log_feature]

            try:
                binned_feature_name = f"{log_feature}_quintile_bin"

                binned_values, _ = pd.qcut(
                    feature_values,
                    q=n_bins,
                    retbins=True,
                    duplicates='drop',
                    labels=False
                )

                df_features[binned_feature_name] = binned_values
                self._created_features.append(binned_feature_name)
                binning_count += 1

            except Exception as e:
                logger.warning(f"    {log_feature}の5分位分割スキップ: {e}")
                continue

        logger.info(f"log変換ビニング完了: {binning_count}特徴量")

    def _create_bin_statistical_features(self, df_features: pd.DataFrame, available_features: list):
        """ビン統計特徴量を作成する。

        Args:
            df_features: 特徴量データフレーム
            available_features: 利用可能な特徴量リスト
        """
        logger.info("ビン統計特徴量を作成中...")

        # BPM目的変数がある場合のみ統計特徴量を作成
        if 'BeatsPerMinute' not in df_features.columns:
            logger.info("  BPM目的変数がないため、ビン統計特徴量をスキップします")
            return

        bpm_values = df_features['BeatsPerMinute']
        stat_count = 0

        # 各ビニング特徴量について統計量を計算
        binning_features = [col for col in df_features.columns if col.endswith('_bin')]

        for binning_feature in binning_features:
            try:
                # ビンごとのBPM統計量を計算
                bin_values = df_features[binning_feature]

                # 欠損値を含むビンは除外
                valid_mask = ~(bin_values.isna() | bpm_values.isna())
                if valid_mask.sum() == 0:
                    continue

                valid_bins = bin_values[valid_mask]
                valid_bpm = bpm_values[valid_mask]

                # ビンごとの統計量計算
                bin_stats = valid_bpm.groupby(valid_bins).agg(['mean', 'std', 'count']).fillna(0)

                # 各サンプルに対応するビン統計量をマップ
                base_name = binning_feature.replace('_bin', '')

                # 平均BPM特徴量
                mean_feature_name = f"{base_name}_bin_mean_bpm"
                df_features[mean_feature_name] = bin_values.map(
                    dict(zip(bin_stats.index, bin_stats['mean']))
                ).fillna(valid_bpm.mean())
                self._created_features.append(mean_feature_name)
                stat_count += 1

                # 標準偏差特徴量
                std_feature_name = f"{base_name}_bin_std_bpm"
                df_features[std_feature_name] = bin_values.map(
                    dict(zip(bin_stats.index, bin_stats['std']))
                ).fillna(valid_bpm.std())
                self._created_features.append(std_feature_name)
                stat_count += 1

                # カウント特徴量
                count_feature_name = f"{base_name}_bin_count"
                df_features[count_feature_name] = bin_values.map(
                    dict(zip(bin_stats.index, bin_stats['count']))
                ).fillna(1)
                self._created_features.append(count_feature_name)
                stat_count += 1

            except Exception as e:
                logger.warning(f"    {binning_feature}の統計特徴量作成スキップ: {e}")
                continue

        logger.info(f"ビン統計特徴量完了: {stat_count}特徴量")

    def _create_bin_interaction_features(self, df_features: pd.DataFrame):
        """ビン間相互作用特徴量を作成する。

        Args:
            df_features: 特徴量データフレーム
        """
        logger.info("ビン間相互作用特徴量を作成中...")

        # ビニング特徴量を取得
        binning_features = [col for col in df_features.columns if col.endswith('_bin')]

        if len(binning_features) < 2:
            logger.info("  ビニング特徴量が2個未満のため、相互作用特徴量をスキップします")
            return

        interaction_count = 0

        # 重要な特徴量ペアに限定した相互作用
        important_pairs = [
            ('RhythmScore', 'Energy'),
            ('VocalContent', 'MoodScore'),
            ('AcousticQuality', 'InstrumentalScore'),
            ('AudioLoudness', 'Energy'),
        ]

        for feature1_base, feature2_base in important_pairs:
            # 対応するビニング特徴量を検索
            feature1_bins = [col for col in binning_features if col.startswith(feature1_base)]
            feature2_bins = [col for col in binning_features if col.startswith(feature2_base)]

            for f1_bin in feature1_bins:
                for f2_bin in feature2_bins:
                    try:
                        # ビン間の組み合わせ特徴量（積）
                        interaction_name = f"{f1_bin}_x_{f2_bin}"
                        df_features[interaction_name] = df_features[f1_bin] * df_features[f2_bin]
                        self._created_features.append(interaction_name)
                        interaction_count += 1

                        # ビン間の差分特徴量
                        diff_name = f"{f1_bin}_diff_{f2_bin}"
                        df_features[diff_name] = df_features[f1_bin] - df_features[f2_bin]
                        self._created_features.append(diff_name)
                        interaction_count += 1

                    except Exception as e:
                        logger.warning(f"    {f1_bin} x {f2_bin}の相互作用作成スキップ: {e}")
                        continue

        logger.info(f"ビン相互作用特徴量完了: {interaction_count}特徴量")

    def get_feature_info(self) -> dict:
        """特徴量作成器の情報を取得する。

        Returns:
            特徴量作成器の詳細情報
        """
        return {
            "name": self.name,
            "description": "ビニング・カテゴリ特徴量作成器 (TICKET-017-03)",
            "target_features": self.target_features,
            "binning_configs": self.binning_configs,
            "created_features_count": len(self._created_features),
            "feature_types": {
                "basic_binning": "基本数値特徴量の分位数分割",
                "log_binning": "log変換特徴量の分位数分割",
                "bin_statistics": "ビンごとのBPM統計特徴量",
                "bin_interactions": "ビン間相互作用特徴量"
            }
        }