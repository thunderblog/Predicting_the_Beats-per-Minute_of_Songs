"""
TICKET-030: 境界値変換済みデータからの76特徴量生成

境界値変換で改善されたデータ品質を基盤として、
より高品質な76特徴量を生成する。

パイプライン完成:
data/raw → 境界値変換 → 特徴量生成 → モデリング
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

def create_features_from_boundary_transformed(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
    """
    境界値変換済みデータから76特徴量を生成

    Args:
        df: 境界値変換済みデータフレーム
        is_train: 訓練データかどうか

    Returns:
        76特徴量版データフレーム
    """
    logger.info("境界値変換済みデータからの特徴量生成開始...")

    result_df = df.copy()

    # 基本特徴量（境界値変換済み）
    base_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                     'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
                     'TrackDurationMs', 'Energy']

    available_features = [feat for feat in base_features if feat in df.columns]
    logger.info(f"境界値変換済み基本特徴量: {available_features}")

    # オリジナル特徴量のバックアップがある場合は除外
    original_backup_features = [col for col in df.columns if col.endswith('_original')]
    logger.info(f"バックアップ特徴量（除外対象）: {len(original_backup_features)}個")

    # 1. 交互作用特徴量（乗算）
    logger.info("交互作用特徴量生成中（境界値変換ベース）...")
    interaction_pairs = [
        ('RhythmScore', 'RhythmScore'),  # 変換済みRhythm^2
        ('RhythmScore', 'AudioLoudness'),
        ('RhythmScore', 'VocalContent'),
        ('RhythmScore', 'LivePerformanceLikelihood'),
        ('RhythmScore', 'MoodScore'),
        ('RhythmScore', 'TrackDurationMs'),
        ('AudioLoudness', 'VocalContent'),
        ('AudioLoudness', 'LivePerformanceLikelihood'),
        ('AudioLoudness', 'MoodScore'),
        ('AudioLoudness', 'TrackDurationMs'),
        ('VocalContent', 'VocalContent'),  # ランク変換済みVocal^2
        ('VocalContent', 'InstrumentalScore'),
        ('VocalContent', 'LivePerformanceLikelihood'),
        ('VocalContent', 'MoodScore'),
        ('VocalContent', 'TrackDurationMs'),
        ('AcousticQuality', 'Energy'),
        ('InstrumentalScore', 'MoodScore'),
        ('LivePerformanceLikelihood', 'MoodScore'),  # ランク変換済み特徴量
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

    # 2. 除算交互作用（改善された分散を活用）
    division_pairs = [
        ('RhythmScore', 'LivePerformanceLikelihood'),  # 両方変換済み
        ('AudioLoudness', 'AcousticQuality'),          # 両方対数変換済み
        ('AudioLoudness', 'LivePerformanceLikelihood'),
        ('VocalContent', 'RhythmScore'),               # ランク変換÷逆変換
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

    # 3. 対数変換特徴量（既に変換済みの特徴量を除く）
    logger.info("追加対数変換特徴量生成中...")
    # すでに対数変換された特徴量: InstrumentalScore, AcousticQuality, AudioLoudness
    # 追加で変換する特徴量
    additional_log_candidates = ['RhythmScore', 'VocalContent', 'MoodScore', 'TrackDurationMs', 'Energy']
    additional_log_features = [feat for feat in additional_log_candidates if feat in available_features]

    for feat in additional_log_features:
        # 負値がある場合は最小値でシフト
        min_val = result_df[feat].min()
        if min_val <= 0:
            shift_val = abs(min_val) + 1e-8
            result_df[f"log1p_{feat}"] = np.log1p(result_df[feat] + shift_val)
        else:
            result_df[f"log1p_{feat}"] = np.log1p(result_df[feat])

    # 4. 対数変換特徴量の交互作用
    log_features_in_df = [col for col in result_df.columns if col.startswith('log1p_')]
    logger.info(f"生成された対数特徴量: {log_features_in_df}")

    if len(log_features_in_df) >= 2:
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

    # 5. 境界値変換特有の特徴量
    logger.info("境界値変換特有の特徴量生成中...")

    # ランク変換済み特徴量の組み合わせ
    rank_transformed = ['VocalContent', 'LivePerformanceLikelihood']
    if all(feat in available_features for feat in rank_transformed):
        result_df['rank_vocal_live_ratio'] = result_df['VocalContent'] / (result_df['LivePerformanceLikelihood'] + 1e-8)
        result_df['rank_vocal_live_sum'] = result_df['VocalContent'] + result_df['LivePerformanceLikelihood']

    # 対数変換済み特徴量の組み合わせ（InstrumentalScore, AcousticQuality）
    log_transformed = ['InstrumentalScore', 'AcousticQuality']
    if all(feat in available_features for feat in log_transformed):
        result_df['log_instrumental_acoustic_product'] = result_df['InstrumentalScore'] * result_df['AcousticQuality']

    # 逆変換済みRhythmScoreの活用
    if 'RhythmScore' in available_features:
        # 逆変換特徴量のエネルギーとの関係
        result_df['inverse_rhythm_energy'] = result_df['RhythmScore'] * result_df['Energy']

    # 6. 音楽理論ベース特徴量（改善された特徴量ベース）
    logger.info("改善された音楽理論ベース特徴量生成中...")

    # 改善されたテンポ×エネルギー
    if 'RhythmScore' in available_features and 'Energy' in available_features:
        result_df['enhanced_tempo_energy_balance'] = result_df['RhythmScore'] * result_df['Energy']

    # 改善されたボーカル×楽器バランス
    if 'VocalContent' in available_features and 'InstrumentalScore' in available_features:
        result_df['enhanced_vocal_instrumental_ratio'] = result_df['VocalContent'] / (result_df['InstrumentalScore'] + 1e-8)
        result_df['enhanced_vocal_instrumental_harmony'] = result_df['VocalContent'] * result_df['InstrumentalScore']

    # 改善された音響品質×エネルギー
    if 'AcousticQuality' in available_features and 'Energy' in available_features:
        result_df['enhanced_acoustic_energy_product'] = result_df['AcousticQuality'] * result_df['Energy']

    # 7. 統計的特徴量（変換済み特徴量ベース）
    logger.info("統計的特徴量生成中（境界値変換ベース）...")

    if len(available_features) > 1:
        result_df['enhanced_features_mean'] = result_df[available_features].mean(axis=1)
        result_df['enhanced_features_std'] = result_df[available_features].std(axis=1)
        result_df['enhanced_features_min'] = result_df[available_features].min(axis=1)
        result_df['enhanced_features_max'] = result_df[available_features].max(axis=1)
        result_df['enhanced_features_range'] = result_df['enhanced_features_max'] - result_df['enhanced_features_min']

    # 対数特徴量の統計量
    if len(log_features_in_df) > 1:
        result_df['enhanced_log_features_mean'] = result_df[log_features_in_df].mean(axis=1)
        result_df['enhanced_log_features_std'] = result_df[log_features_in_df].std(axis=1)
        result_df['enhanced_log_features_range'] = result_df[log_features_in_df].max(axis=1) - result_df[log_features_in_df].min(axis=1)

    # 8. 最終的な特徴量確認
    if is_train:
        feature_cols = [col for col in result_df.columns
                       if col not in ["id", "BeatsPerMinute"] and not col.endswith('_original')]
    else:
        feature_cols = [col for col in result_df.columns
                       if col != "id" and not col.endswith('_original')]

    logger.info(f"生成特徴量数: {len(feature_cols)}")
    logger.info(f"最終データ形状: {result_df.shape}")

    # バックアップ特徴量を除去（オプション）
    if len(original_backup_features) > 0:
        logger.info(f"バックアップ特徴量を除去: {len(original_backup_features)}個")
        result_df = result_df.drop(columns=original_backup_features)

    return result_df

def main():
    """メイン処理"""
    logger.info("境界値変換済みデータからの特徴量生成開始...")

    # 1. 境界値変換済み訓練データの特徴量生成
    logger.info("=== 境界値変換済み訓練データ特徴量生成 ===")
    train_boundary_path = config.processed_data_dir / "train_raw_boundary_transformed.csv"
    train_df = pd.read_csv(train_boundary_path)
    logger.info(f"境界値変換済み訓練データ: {train_df.shape}")

    train_features_df = create_features_from_boundary_transformed(train_df, is_train=True)

    train_output_path = config.processed_data_dir / "train_boundary_transformed_76_features.csv"
    train_features_df.to_csv(train_output_path, index=False)
    logger.success(f"特徴量生成済み訓練データ保存: {train_output_path}")

    # 2. 境界値変換済みテストデータの特徴量生成
    logger.info("=== 境界値変換済みテストデータ特徴量生成 ===")
    test_boundary_path = config.processed_data_dir / "test_raw_boundary_transformed.csv"
    test_df = pd.read_csv(test_boundary_path)
    logger.info(f"境界値変換済みテストデータ: {test_df.shape}")

    test_features_df = create_features_from_boundary_transformed(test_df, is_train=False)

    test_output_path = config.processed_data_dir / "test_boundary_transformed_76_features.csv"
    test_features_df.to_csv(test_output_path, index=False)
    logger.success(f"特徴量生成済みテストデータ保存: {test_output_path}")

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
        logger.success("✅ 完全一致！境界値変換済み特徴量が統一されました")
    else:
        logger.info(f"共通特徴量 ({len(common_features)}個) で実験を継続します")

    # 4. データ品質統計
    logger.info("=== データ品質統計 ===")
    logger.info(f"訓練データ欠損値: {train_features_df.isnull().sum().sum()}")
    logger.info(f"テストデータ欠損値: {test_features_df.isnull().sum().sum()}")

    if 'BeatsPerMinute' in train_features_df.columns:
        logger.info(f"BPM範囲: {train_features_df['BeatsPerMinute'].min():.2f} - {train_features_df['BeatsPerMinute'].max():.2f}")

    return train_output_path, test_output_path, len(common_features)

if __name__ == "__main__":
    train_path, test_path, n_features = main()
    logger.success(f"境界値変換ベース特徴量生成完了: {n_features}特徴量")
    logger.info("🚀 次のステップ: 境界値変換+特徴量エンジニアリング版でのベースライン検証")
    logger.info(f"生成訓練データ: {train_path}")
    logger.info(f"生成テストデータ: {test_path}")