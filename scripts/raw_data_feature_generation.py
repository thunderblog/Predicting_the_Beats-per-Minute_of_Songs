"""
完全なdata/rawからの統一特徴量生成スクリプト
524,164サンプルの完全データセットを使用
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

def create_unified_features_from_raw(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
    """
    data/rawから完全な統一特徴量を生成

    Args:
        df: 入力データフレーム（元の9特徴量+id）
        is_train: 訓練データかどうか（ターゲット列の有無判定）

    Returns:
        統一特徴量版データフレーム
    """
    logger.info("完全データセット特徴量生成開始...")

    result_df = df.copy()

    # 元の9特徴量確認
    base_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                     'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
                     'TrackDurationMs', 'Energy']

    # 利用可能な特徴量のみ使用
    available_features = [feat for feat in base_features if feat in df.columns]
    logger.info(f"利用可能な基本特徴量: {available_features}")

    # 1. 交互作用特徴量 (乗算)
    logger.info("交互作用特徴量生成中...")
    interaction_pairs = [
        ('RhythmScore', 'RhythmScore'),  # RhythmScore^2
        ('RhythmScore', 'AudioLoudness'),
        ('RhythmScore', 'VocalContent'),
        ('RhythmScore', 'LivePerformanceLikelihood'),
        ('RhythmScore', 'MoodScore'),
        ('RhythmScore', 'TrackDurationMs'),
        ('AudioLoudness', 'VocalContent'),
        ('AudioLoudness', 'LivePerformanceLikelihood'),
        ('AudioLoudness', 'MoodScore'),
        ('AudioLoudness', 'TrackDurationMs'),
        ('VocalContent', 'VocalContent'),  # VocalContent^2
        ('VocalContent', 'InstrumentalScore'),
        ('VocalContent', 'LivePerformanceLikelihood'),
        ('VocalContent', 'MoodScore'),
        ('VocalContent', 'TrackDurationMs'),
        ('AcousticQuality', 'Energy'),
        ('InstrumentalScore', 'MoodScore'),
        ('LivePerformanceLikelihood', 'MoodScore'),
        ('LivePerformanceLikelihood', 'TrackDurationMs'),
        ('MoodScore', 'MoodScore'),  # MoodScore^2
        ('MoodScore', 'TrackDurationMs'),
        ('TrackDurationMs', 'TrackDurationMs'),  # TrackDurationMs^2
        ('Energy', 'Energy')  # Energy^2
    ]

    for feat1, feat2 in interaction_pairs:
        if feat1 in available_features and feat2 in available_features:
            if feat1 == feat2:
                feature_name = f"{feat1}_squared"
            else:
                feature_name = f"{feat1}_x_{feat2}"
            result_df[feature_name] = result_df[feat1] * result_df[feat2]

    # 2. 除算交互作用 (ゼロ除算対策あり)
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

    # 3. 対数変換特徴量
    logger.info("対数変換特徴量生成中...")
    log_candidates = ['RhythmScore', 'VocalContent', 'MoodScore', 'TrackDurationMs', 'Energy']
    log_features = [feat for feat in log_candidates if feat in available_features]

    # 基本対数変換
    for feat in log_features:
        result_df[f"log1p_{feat}"] = np.log1p(result_df[feat])

    # 対数変換特徴量の交互作用
    log_interaction_pairs = [
        ('log1p_RhythmScore', 'log1p_VocalContent'),
        ('log1p_RhythmScore', 'log1p_MoodScore'),
        ('log1p_RhythmScore', 'log1p_TrackDurationMs'),
        ('log1p_VocalContent', 'log1p_MoodScore'),
        ('log1p_VocalContent', 'log1p_TrackDurationMs'),
        ('log1p_MoodScore', 'log1p_TrackDurationMs'),
        ('log1p_TrackDurationMs', 'log1p_Energy')
    ]

    for feat1, feat2 in log_interaction_pairs:
        if feat1 in result_df.columns and feat2 in result_df.columns:
            feature_name = f"{feat1}_x_{feat2}"
            result_df[feature_name] = result_df[feat1] * result_df[feat2]

    # 対数変換特徴量の除算
    log_division_pairs = [
        ('log1p_RhythmScore', 'log1p_TrackDurationMs'),
        ('log1p_VocalContent', 'log1p_TrackDurationMs'),
        ('log1p_MoodScore', 'log1p_TrackDurationMs'),
        ('log1p_Energy', 'log1p_TrackDurationMs')
    ]

    for feat1, feat2 in log_division_pairs:
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

    # 5. 音楽理論ベース特徴量
    logger.info("音楽理論ベース特徴量生成中...")

    # テンポ×ジャンル特徴量 (疑似)
    if 'RhythmScore' in available_features and 'Energy' in available_features:
        result_df['tempo_energy_balance'] = result_df['RhythmScore'] * result_df['Energy']

    # 楽器バランス特徴量
    if 'VocalContent' in available_features and 'InstrumentalScore' in available_features:
        result_df['vocal_instrumental_ratio'] = result_df['VocalContent'] / (result_df['InstrumentalScore'] + 1e-8)
        result_df['vocal_instrumental_sum'] = result_df['VocalContent'] + result_df['InstrumentalScore']

    # 音響品質×エネルギー
    if 'AcousticQuality' in available_features and 'Energy' in available_features:
        result_df['acoustic_energy_product'] = result_df['AcousticQuality'] * result_df['Energy']

    # 6. 統計的特徴量
    logger.info("統計的特徴量生成中...")

    # 元特徴量の統計量
    numeric_cols = [feat for feat in available_features if feat in result_df.columns]
    if len(numeric_cols) > 1:
        result_df['features_mean'] = result_df[numeric_cols].mean(axis=1)
        result_df['features_std'] = result_df[numeric_cols].std(axis=1)
        result_df['features_min'] = result_df[numeric_cols].min(axis=1)
        result_df['features_max'] = result_df[numeric_cols].max(axis=1)
        result_df['features_range'] = result_df['features_max'] - result_df['features_min']

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
    logger.info("完全データセット特徴量生成開始...")

    # 1. 訓練データ処理 (data/raw使用)
    logger.info("=== 訓練データ処理 (RAW) ===")
    train_raw_path = config.raw_data_dir / "train.csv"
    train_df = pd.read_csv(train_raw_path)
    logger.info(f"元の訓練データ: {train_df.shape}")

    train_features_df = create_unified_features_from_raw(train_df, is_train=True)

    train_output_path = config.processed_data_dir / "train_raw_complete_features.csv"
    train_features_df.to_csv(train_output_path, index=False)
    logger.success(f"訓練データ保存: {train_output_path}")

    # 2. テストデータ処理 (data/raw使用)
    logger.info("=== テストデータ処理 (RAW) ===")
    test_raw_path = config.raw_data_dir / "test.csv"
    test_df = pd.read_csv(test_raw_path)
    logger.info(f"元のテストデータ: {test_df.shape}")

    test_features_df = create_unified_features_from_raw(test_df, is_train=False)

    test_output_path = config.processed_data_dir / "test_raw_complete_features.csv"
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

    # 4. データ品質統計
    logger.info("=== データ品質統計 ===")
    logger.info(f"訓練データ欠損値数: {train_features_df.isnull().sum().sum()}")
    logger.info(f"テストデータ欠損値数: {test_features_df.isnull().sum().sum()}")

    # BPM統計（訓練データのみ）
    if 'BeatsPerMinute' in train_features_df.columns:
        logger.info(f"BPM範囲: {train_features_df['BeatsPerMinute'].min():.2f} - {train_features_df['BeatsPerMinute'].max():.2f}")
        logger.info(f"BPM平均: {train_features_df['BeatsPerMinute'].mean():.2f}")

    return train_output_path, test_output_path, len(common_features)

if __name__ == "__main__":
    train_path, test_path, n_features = main()
    logger.success(f"完全データセット特徴量生成完了: {n_features}特徴量")
    logger.info("🚀 次のステップ: 新しいデータセットでベースライン実験実行")