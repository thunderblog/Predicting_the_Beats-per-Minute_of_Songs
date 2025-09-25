"""
Feature Engineering Module

リファクタリング済みの特徴量エンジニアリングモジュール。
各種特徴量作成機能を機能別に整理したモジュールである。
"""

# Base classes and utilities
from .base import BaseFeatureCreator, FeatureUtils, FeaturePipeline

# Feature creators
from .interaction import (
    BasicInteractionCreator,
    ComprehensiveInteractionCreator,
    create_interaction_features,
    create_comprehensive_interaction_features,
)
from .statistical import StatisticalFeatureCreator, create_statistical_features
from .genre import MusicGenreFeatureCreator, create_music_genre_features
from .duration import DurationFeatureCreator, create_duration_features
from .advanced import AdvancedFeatureCreator, create_advanced_features

# Feature processing
from .selection import select_features
from .scaling import scale_features
from .analysis import (
    analyze_feature_importance,
    compare_genre_features_to_bpm,
    detect_multicollinearity,
)

# 後方互換性のための関数エクスポート
__all__ = [
    # Base classes
    "BaseFeatureCreator",
    "FeatureUtils",
    "FeaturePipeline",
    # Feature creators
    "BasicInteractionCreator",
    "ComprehensiveInteractionCreator",
    "StatisticalFeatureCreator",
    "MusicGenreFeatureCreator",
    "DurationFeatureCreator",
    "AdvancedFeatureCreator",
    # Backward compatibility functions
    "create_interaction_features",
    "create_comprehensive_interaction_features",
    "create_statistical_features",
    "create_music_genre_features",
    "create_duration_features",
    "create_advanced_features",
    # Processing functions
    "select_features",
    "scale_features",
    "analyze_feature_importance",
    "compare_genre_features_to_bpm",
    "detect_multicollinearity",
]


def create_feature_pipeline():
    """デフォルトの特徴量作成パイプラインを作成する。

    Returns:
        設定済みの FeaturePipeline インスタンス
    """
    pipeline = FeaturePipeline()
    
    # 基本的な特徴量作成器を追加
    pipeline.add_creator(BasicInteractionCreator())
    pipeline.add_creator(StatisticalFeatureCreator())
    pipeline.add_creator(MusicGenreFeatureCreator())
    pipeline.add_creator(DurationFeatureCreator())
    
    return pipeline


# バージョン情報
__version__ = "1.0.0"