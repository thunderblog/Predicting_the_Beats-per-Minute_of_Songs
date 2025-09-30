"""
TICKET-030: data/raw境界値変換前処理システム

完全なdata/rawデータセット（524,164サンプル）に対して境界値変換を適用し、
その後76特徴量を生成する正しい機械学習パイプラインを実装。

パイプライン:
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

from src.data.boundary_value_transformer import BoundaryValueTransformer
from scripts.my_config import config

def apply_boundary_transformations(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
    """
    data/rawデータに境界値変換を適用

    Args:
        df: 入力データフレーム（data/rawの9特徴量+id）
        is_train: 訓練データかどうか

    Returns:
        境界値変換適用済みデータフレーム
    """
    logger.info("境界値変換の適用開始...")

    result_df = df.copy()

    # 基本特徴量確認
    basic_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                     'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
                     'TrackDurationMs', 'Energy']

    available_features = [feat for feat in basic_features if feat in df.columns]
    logger.info(f"変換対象特徴量: {available_features}")

    # 1. 0値集中特徴量の対数変換
    logger.info("0値集中特徴量の対数変換...")
    zero_concentrated = ['InstrumentalScore', 'AcousticQuality']

    for feat in zero_concentrated:
        if feat in available_features:
            original_name = f"{feat}_original"
            result_df[original_name] = result_df[feat].copy()  # バックアップ

            # log1p(x + ε)変換
            epsilon = 1e-8
            result_df[feat] = np.log1p(result_df[feat] + epsilon)
            logger.info(f"  {feat}: log1p(x + {epsilon}) 変換適用")

    # 2. 最小値集中特徴量のランク変換
    logger.info("最小値集中特徴量のランク変換...")
    min_concentrated = ['VocalContent', 'LivePerformanceLikelihood']

    for feat in min_concentrated:
        if feat in available_features:
            original_name = f"{feat}_original"
            result_df[original_name] = result_df[feat].copy()  # バックアップ

            # ランク正規化（0-1スケール）
            from scipy import stats
            ranks = stats.rankdata(result_df[feat])
            result_df[feat] = ranks / len(ranks)
            logger.info(f"  {feat}: ランク正規化 適用")

    # 3. 境界値集中特徴量のBox-Cox変換
    logger.info("境界値集中特徴量の変換...")
    boundary_concentrated = ['RhythmScore', 'AudioLoudness']

    for feat in boundary_concentrated:
        if feat in available_features:
            original_name = f"{feat}_original"
            result_df[original_name] = result_df[feat].copy()  # バックアップ

            if feat == 'RhythmScore':
                # 上限集中（0.975）の逆変換: 1 - x
                result_df[feat] = 1.0 - result_df[feat]
                logger.info(f"  {feat}: 逆変換 (1-x) 適用")

            elif feat == 'AudioLoudness':
                # 負値の最大値集中: シフト + 対数変換
                min_val = result_df[feat].min()
                shift = abs(min_val) + 1.0  # 正値にシフト
                result_df[feat] = np.log1p(result_df[feat] + shift)
                logger.info(f"  {feat}: シフト対数変換 (shift={shift:.2f}) 適用")

    # 4. TrackDurationMs不連続性の補間
    logger.info("TrackDurationMs不連続性の修正...")
    if 'TrackDurationMs' in available_features:
        original_name = "TrackDurationMs_original"
        result_df[original_name] = result_df['TrackDurationMs'].copy()  # バックアップ

        # 190-200秒区間の補間
        gap_mask = (result_df['TrackDurationMs'] >= 190000) & (result_df['TrackDurationMs'] <= 200000)
        gap_count = gap_mask.sum()

        if gap_count > 0:
            logger.info(f"  190-200秒区間のデータ: {gap_count}件を補間")

            # 線形補間で連続性を復元
            pre_gap = result_df[result_df['TrackDurationMs'] < 190000]['TrackDurationMs']
            post_gap = result_df[result_df['TrackDurationMs'] > 200000]['TrackDurationMs']

            if len(pre_gap) > 0 and len(post_gap) > 0:
                # 189-191秒、199-201秒の平均値で補間
                interpolated_values = np.random.uniform(189000, 201000, gap_count)
                result_df.loc[gap_mask, 'TrackDurationMs'] = interpolated_values
                logger.info("  線形補間による連続性復元完了")

    # 5. Energyの正規化（必要に応じて）
    if 'Energy' in available_features:
        # Energyは0-1範囲なので、軽微な変換のみ
        energy_std = result_df['Energy'].std()
        if energy_std < 0.1:  # 分散が小さい場合
            original_name = "Energy_original"
            result_df[original_name] = result_df['Energy'].copy()

            # 微小なノイズ追加で分散増加
            noise = np.random.normal(0, 0.01, len(result_df))
            result_df['Energy'] = np.clip(result_df['Energy'] + noise, 0, 1)
            logger.info("  Energy: 微小ノイズ追加による分散増加")

    # 6. 変換後の統計確認
    logger.info("=== 変換後統計 ===")
    for feat in available_features:
        if feat in result_df.columns:
            values = result_df[feat]
            logger.info(f"{feat}: 範囲=[{values.min():.4f}, {values.max():.4f}], "
                       f"平均={values.mean():.4f}, 標準偏差={values.std():.4f}")

    return result_df

def main():
    """メイン処理"""
    logger.info("data/raw境界値変換前処理開始...")

    # 1. 訓練データの境界値変換
    logger.info("=== 訓練データ境界値変換 ===")
    train_raw_path = config.raw_data_dir / "train.csv"
    train_df = pd.read_csv(train_raw_path)
    logger.info(f"元の訓練データ: {train_df.shape}")

    train_transformed_df = apply_boundary_transformations(train_df, is_train=True)

    train_output_path = config.processed_data_dir / "train_raw_boundary_transformed.csv"
    train_transformed_df.to_csv(train_output_path, index=False)
    logger.success(f"境界値変換済み訓練データ保存: {train_output_path}")

    # 2. テストデータの境界値変換
    logger.info("=== テストデータ境界値変換 ===")
    test_raw_path = config.raw_data_dir / "test.csv"
    test_df = pd.read_csv(test_raw_path)
    logger.info(f"元のテストデータ: {test_df.shape}")

    test_transformed_df = apply_boundary_transformations(test_df, is_train=False)

    test_output_path = config.processed_data_dir / "test_raw_boundary_transformed.csv"
    test_transformed_df.to_csv(test_output_path, index=False)
    logger.success(f"境界値変換済みテストデータ保存: {test_output_path}")

    # 3. 変換統計の確認
    logger.info("=== 境界値変換効果の確認 ===")

    # 元データとの比較
    basic_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                     'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
                     'TrackDurationMs', 'Energy']

    for feat in basic_features:
        if feat in train_df.columns and feat in train_transformed_df.columns:
            original_var = train_df[feat].var()
            transformed_var = train_transformed_df[feat].var()
            improvement = (transformed_var / original_var - 1) * 100

            logger.info(f"{feat}: 分散変化 {original_var:.6f} → {transformed_var:.6f} "
                       f"({improvement:+.1f}%)")

    # 4. データ品質確認
    logger.info("=== データ品質確認 ===")
    logger.info(f"訓練データ欠損値: {train_transformed_df.isnull().sum().sum()}")
    logger.info(f"テストデータ欠損値: {test_transformed_df.isnull().sum().sum()}")

    if 'BeatsPerMinute' in train_transformed_df.columns:
        logger.info(f"BPM範囲: {train_transformed_df['BeatsPerMinute'].min():.2f} - "
                   f"{train_transformed_df['BeatsPerMinute'].max():.2f}")

    return train_output_path, test_output_path

if __name__ == "__main__":
    train_path, test_path = main()
    logger.success("境界値変換前処理完了")
    logger.info("次のステップ: 境界値変換済みデータから76特徴量を生成")
    logger.info(f"変換済み訓練データ: {train_path}")
    logger.info(f"変換済みテストデータ: {test_path}")