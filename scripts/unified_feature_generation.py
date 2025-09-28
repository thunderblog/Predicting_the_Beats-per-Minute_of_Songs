"""
統一された75特徴量生成スクリプト
trainとtestで完全に同一の特徴量セットを生成
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

from scripts.my_config import config

def create_unified_75_features(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
    """
    trainとtestで完全に同一の75特徴量を生成

    Args:
        df: 入力データフレーム（元の9特徴量+id）
        is_train: 訓練データかどうか（ターゲット列の有無判定）

    Returns:
        75特徴量版データフレーム
    """
    logger.info("統一75特徴量生成開始...")

    result_df = df.copy()

    # 元の9特徴量確認
    base_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                     'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
                     'TrackDurationMs', 'Energy']

    # 利用可能な特徴量のみ使用
    available_features = [feat for feat in base_features if feat in df.columns]
    logger.info(f"利用可能な基本特徴量: {available_features}")

    # TICKET017-01: 交互作用特徴量
    logger.info("交互作用特徴量生成中...")

    # 1. 乗算交互作用（主要ペア）
    interaction_pairs = [
        ('RhythmScore', 'RhythmScore'),
        ('RhythmScore', 'AudioLoudness'),
        ('RhythmScore', 'VocalContent'),
        ('RhythmScore', 'LivePerformanceLikelihood'),
        ('RhythmScore', 'MoodScore'),
        ('RhythmScore', 'TrackDurationMs'),
        ('AudioLoudness', 'VocalContent'),
        ('AudioLoudness', 'LivePerformanceLikelihood'),
        ('AudioLoudness', 'MoodScore'),
        ('AudioLoudness', 'TrackDurationMs'),
        ('VocalContent', 'VocalContent'),
        ('VocalContent', 'InstrumentalScore'),
        ('VocalContent', 'LivePerformanceLikelihood'),
        ('VocalContent', 'MoodScore'),
        ('VocalContent', 'TrackDurationMs'),
        ('AcousticQuality', 'Energy'),
        ('InstrumentalScore', 'MoodScore'),
        ('LivePerformanceLikelihood', 'MoodScore'),
        ('LivePerformanceLikelihood', 'TrackDurationMs'),
        ('MoodScore', 'MoodScore'),
        ('MoodScore', 'TrackDurationMs'),
        ('TrackDurationMs', 'TrackDurationMs'),
        ('Energy', 'Energy')
    ]

    for feat1, feat2 in interaction_pairs:
        if feat1 in available_features and feat2 in available_features:
            if feat1 == feat2:
                feature_name = f"{feat1}_squared"
            else:
                feature_name = f"{feat1}_x_{feat2}"
            result_df[feature_name] = result_df[feat1] * result_df[feat2]

    # 2. 除算交互作用
    division_pairs = [
        ('RhythmScore', 'LivePerformanceLikelihood'),
        ('AudioLoudness', 'AcousticQuality'),
        ('AudioLoudness', 'LivePerformanceLikelihood'),
        ('VocalContent', 'RhythmScore'),
        ('AcousticQuality', 'AudioLoudness'),
        ('AcousticQuality', 'LivePerformanceLikelihood'),
        ('LivePerformanceLikelihood', 'AcousticQuality'),
        ('MoodScore', 'LivePerformanceLikelihood'),
        ('TrackDurationMs', 'LivePerformanceLikelihood'),
        ('Energy', 'RhythmScore'),
        ('Energy', 'AudioLoudness'),
        ('Energy', 'VocalContent'),
        ('Energy', 'InstrumentalScore'),
        ('Energy', 'LivePerformanceLikelihood'),
        ('Energy', 'TrackDurationMs')
    ]

    for feat1, feat2 in division_pairs:
        if feat1 in available_features and feat2 in available_features:
            feature_name = f"{feat1}_div_{feat2}"
            result_df[feature_name] = result_df[feat1] / (result_df[feat2] + 1e-8)

    # TICKET017-02: 対数変換特徴量
    logger.info("対数変換特徴量生成中...")

    # 対数変換対象
    log_candidates = ['RhythmScore', 'VocalContent', 'MoodScore', 'TrackDurationMs', 'Energy']
    log_features = [feat for feat in log_candidates if feat in available_features]

    # 1. 基本対数変換
    for feat in log_features:
        result_df[f"log1p_{feat}"] = np.log1p(result_df[feat])

    # 2. 対数変換特徴量の交互作用
    log_interaction_pairs = [
        ('log1p_RhythmScore', 'log1p_VocalContent'),
        ('log1p_RhythmScore', 'log1p_LivePerformanceLikelihood'),
        ('log1p_RhythmScore', 'log1p_MoodScore'),
        ('log1p_RhythmScore', 'log1p_TrackDurationMs'),
        ('log1p_VocalContent', 'log1p_AcousticQuality'),
        ('log1p_VocalContent', 'log1p_InstrumentalScore'),
        ('log1p_VocalContent', 'log1p_LivePerformanceLikelihood'),
        ('log1p_VocalContent', 'log1p_MoodScore'),
        ('log1p_VocalContent', 'log1p_TrackDurationMs'),
        ('log1p_AcousticQuality', 'log1p_Energy'),
        ('log1p_InstrumentalScore', 'log1p_MoodScore'),
        ('log1p_LivePerformanceLikelihood', 'log1p_MoodScore'),
        ('log1p_MoodScore', 'log1p_TrackDurationMs'),
        ('log1p_TrackDurationMs', 'log1p_Energy')
    ]

    for feat1, feat2 in log_interaction_pairs:
        # 実際の対数変換特徴量が存在するかチェック
        if feat1 in result_df.columns and feat2 in result_df.columns:
            feature_name = f"{feat1}_x_{feat2}"
            result_df[feature_name] = result_df[feat1] * result_df[feat2]

    # 3. 対数変換特徴量の除算
    log_division_pairs = [
        ('log1p_RhythmScore', 'log1p_TrackDurationMs'),
        ('log1p_VocalContent', 'log1p_TrackDurationMs'),
        ('log1p_MoodScore', 'log1p_TrackDurationMs'),
        ('log1p_Energy', 'log1p_TrackDurationMs')
    ]

    for feat1, feat2 in log_division_pairs:
        # 実際の対数変換特徴量が存在するかチェック
        if feat1 in result_df.columns and feat2 in result_df.columns:
            feature_name = f"{feat1}_div_{feat2}"
            result_df[feature_name] = result_df[feat1] / (result_df[feat2] + 1e-8)

    # 4. 対数特徴量の統計量
    available_log_cols = [f"log1p_{feat}" for feat in log_features if f"log1p_{feat}" in result_df.columns]

    if len(available_log_cols) > 1:
        result_df['log_features_mean'] = result_df[available_log_cols].mean(axis=1)
        result_df['log_features_std'] = result_df[available_log_cols].std(axis=1)
        result_df['log_features_range'] = result_df[available_log_cols].max(axis=1) - result_df[available_log_cols].min(axis=1)
        result_df['log_features_geometric_mean'] = np.exp(result_df[available_log_cols].mean(axis=1)) - 1

    # 特徴量数確認
    if is_train:
        feature_cols = [col for col in result_df.columns if col not in ["id", "BeatsPerMinute"]]
    else:
        feature_cols = [col for col in result_df.columns if col != "id"]

    logger.info(f"生成特徴量数: {len(feature_cols)}")
    logger.info(f"最終データ形状: {result_df.shape}")

    return result_df

def main():
    """メイン処理"""
    logger.info("統一75特徴量生成開始...")

    # 1. 訓練データ処理
    logger.info("=== 訓練データ処理 ===")
    train_path = config.processed_data_dir / "train.csv"
    train_df = pd.read_csv(train_path)
    logger.info(f"元の訓練データ: {train_df.shape}")

    train_features_df = create_unified_75_features(train_df, is_train=True)

    train_output_path = config.processed_data_dir / "train_unified_75_features.csv"
    train_features_df.to_csv(train_output_path, index=False)
    logger.success(f"訓練データ保存: {train_output_path}")

    # 2. テストデータ処理
    logger.info("=== テストデータ処理 ===")
    test_path = config.processed_data_dir / "test.csv"
    test_df = pd.read_csv(test_path)
    logger.info(f"元のテストデータ: {test_df.shape}")

    test_features_df = create_unified_75_features(test_df, is_train=False)

    test_output_path = config.processed_data_dir / "test_unified_75_features.csv"
    test_features_df.to_csv(test_output_path, index=False)
    logger.success(f"テストデータ保存: {test_output_path}")

    # 3. 特徴量一致確認
    logger.info("=== 特徴量一致確認 ===")
    train_feature_cols = [col for col in train_features_df.columns if col not in ["id", "BeatsPerMinute"]]
    test_feature_cols = [col for col in test_features_df.columns if col != "id"]

    common_features = sorted(list(set(train_feature_cols) & set(test_feature_cols)))

    logger.info(f"訓練特徴量数: {len(train_feature_cols)}")
    logger.info(f"テスト特徴量数: {len(test_feature_cols)}")
    logger.info(f"共通特徴量数: {len(common_features)}")

    train_only = set(train_feature_cols) - set(test_feature_cols)
    test_only = set(test_feature_cols) - set(train_feature_cols)

    if train_only:
        logger.warning(f"訓練限定特徴量 ({len(train_only)}個): {sorted(list(train_only))}")
    if test_only:
        logger.warning(f"テスト限定特徴量 ({len(test_only)}個): {sorted(list(test_only))}")

    if len(train_only) == 0 and len(test_only) == 0:
        logger.success("✅ 完全一致！訓練・テストデータの特徴量が統一されました")
    else:
        logger.info(f"共通特徴量 ({len(common_features)}個) で実験を継続します")

    return train_output_path, test_output_path, len(common_features)

if __name__ == "__main__":
    train_path, test_path, n_features = main()
    logger.success(f"統一特徴量生成完了: {n_features}特徴量")
    logger.info("次のステップ: BPM Stratified戦略で実験実行")