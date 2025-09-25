from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class BaseFeatureCreator(ABC):
    """特徴量作成の基底クラス。

    全ての特徴量作成クラスが継承すべき共通インターフェース。
    """

    def __init__(self, name: str):
        """
        Args:
            name: 特徴量作成器の名前
        """
        self.name = name
        self._created_features: List[str] = []

    @abstractmethod
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量を作成する抽象メソッド。

        Args:
            df: 入力データフレーム

        Returns:
            特徴量が追加されたデータフレーム
        """
        pass

    @property
    def created_features(self) -> List[str]:
        """作成された特徴量の名前リストを返す。"""
        return self._created_features.copy()

    def log_feature_creation(self, original_count: int, new_count: int):
        """特徴量作成のログを出力する。"""
        added_features = new_count - original_count
        logger.info(f"{self.name}: {added_features}個の新特徴量を追加")


class FeatureUtils:
    """特徴量作成のためのユーティリティ関数集。"""

    @staticmethod
    def safe_divide(numerator: pd.Series, denominator: pd.Series,
                   epsilon: float = 1e-8) -> pd.Series:
        """安全な除算を行う（ゼロ除算対策）。

        Args:
            numerator: 分子
            denominator: 分母
            epsilon: ゼロ除算回避のための小さな値

        Returns:
            除算結果
        """
        return numerator / (denominator + epsilon)

    @staticmethod
    def get_numerical_columns() -> List[str]:
        """基本的な数値特徴量のカラム名リストを返す。"""
        return [
            'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
            'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
            'TrackDurationMs', 'Energy'
        ]

    @staticmethod
    def filter_available_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
        """データフレームに存在する列のみを返す。

        Args:
            df: データフレーム
            columns: 確認したい列名のリスト

        Returns:
            存在する列名のリスト
        """
        return [col for col in columns if col in df.columns]

    @staticmethod
    def add_features_safely(df: pd.DataFrame, new_features: dict) -> pd.DataFrame:
        """特徴量を安全に追加する（DataFrameの断片化を防ぐ）。

        Args:
            df: 元のデータフレーム
            new_features: 新特徴量の辞書 {特徴量名: 値のSeries}

        Returns:
            特徴量が追加されたデータフレーム
        """
        if not new_features:
            return df

        new_features_df = pd.DataFrame(new_features, index=df.index)
        return pd.concat([df, new_features_df], axis=1)

    @staticmethod
    def validate_features(df: pd.DataFrame, feature_names: List[str]) -> dict:
        """特徴量の品質チェックを行う。

        Args:
            df: データフレーム
            feature_names: チェックしたい特徴量名のリスト

        Returns:
            品質チェック結果の辞書
        """
        if not feature_names:
            return {}

        results = {}
        for feature in feature_names:
            if feature not in df.columns:
                continue

            values = df[feature]
            results[feature] = {
                'nan_count': values.isnull().sum(),
                'inf_count': np.isinf(values).sum() if np.issubdtype(values.dtype, np.number) else 0,
                'unique_count': values.nunique(),
                'mean': values.mean() if np.issubdtype(values.dtype, np.number) else None,
                'std': values.std() if np.issubdtype(values.dtype, np.number) else None,
                'min': values.min() if np.issubdtype(values.dtype, np.number) else None,
                'max': values.max() if np.issubdtype(values.dtype, np.number) else None
            }

        return results


class FeaturePipeline:
    """特徴量作成パイプラインを管理するクラス。"""

    def __init__(self):
        """特徴量作成パイプラインを初期化する。"""
        self.creators: List[BaseFeatureCreator] = []
        self.execution_log: List[dict] = []

    def add_creator(self, creator: BaseFeatureCreator):
        """特徴量作成器をパイプラインに追加する。

        Args:
            creator: 特徴量作成器
        """
        self.creators.append(creator)
        logger.info(f"特徴量作成器を追加: {creator.name}")

    def execute(self, df: pd.DataFrame,
                creators_to_run: Optional[List[str]] = None) -> pd.DataFrame:
        """パイプラインを実行して特徴量を作成する。

        Args:
            df: 入力データフレーム
            creators_to_run: 実行したい作成器の名前リスト（None の場合は全て実行）

        Returns:
            全特徴量が追加されたデータフレーム
        """
        logger.info("特徴量作成パイプラインを開始...")

        result_df = df.copy()
        original_features = len(result_df.columns)

        for creator in self.creators:
            if creators_to_run is not None and creator.name not in creators_to_run:
                continue

            logger.info(f"{creator.name}を実行中...")
            features_before = len(result_df.columns)

            try:
                result_df = creator.create_features(result_df)
                features_after = len(result_df.columns)

                self.execution_log.append({
                    'creator': creator.name,
                    'features_added': features_after - features_before,
                    'status': 'success'
                })

                creator.log_feature_creation(features_before, features_after)

            except Exception as e:
                logger.error(f"{creator.name}でエラー発生: {e}")
                self.execution_log.append({
                    'creator': creator.name,
                    'features_added': 0,
                    'status': 'error',
                    'error': str(e)
                })
                continue

        total_added = len(result_df.columns) - original_features
        logger.success(f"パイプライン完了: {total_added}個の特徴量を追加")

        return result_df

    def get_execution_summary(self) -> pd.DataFrame:
        """実行結果のサマリーを返す。

        Returns:
            実行結果サマリーのDataFrame
        """
        if not self.execution_log:
            return pd.DataFrame()

        return pd.DataFrame(self.execution_log)