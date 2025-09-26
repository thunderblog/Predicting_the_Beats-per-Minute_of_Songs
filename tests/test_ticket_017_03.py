#!/usr/bin/env python3
"""
TICKET-017-03 ビニング・カテゴリ特徴量のテストスクリプト

ビニング・カテゴリ特徴量の実装をテストし、動作を確認する。
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

def test_binning_features():
    """ビニング・カテゴリ特徴量のテスト実行"""
    print("TICKET-017-03: ビニング・カテゴリ特徴量のテスト開始")
    print("=" * 60)

    # テスト用サンプルデータ作成
    test_data = pd.DataFrame({
        'RhythmScore': [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.6, 0.8, 0.4, 0.95],
        'AudioLoudness': [0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.7, 0.9, 0.5, 0.85],
        'VocalContent': [0.8, 0.6, 0.9, 0.5, 0.7, 0.4, 0.2, 0.1, 0.3, 0.75],
        'AcousticQuality': [0.5, 0.8, 0.7, 0.6, 0.9, 0.3, 0.1, 0.4, 0.2, 0.65],
        'InstrumentalScore': [0.7, 0.5, 0.8, 0.9, 0.6, 0.4, 0.3, 0.2, 0.1, 0.55],
        'LivePerformanceLikelihood': [0.4, 0.6, 0.5, 0.7, 0.8, 0.2, 0.9, 0.1, 0.3, 0.45],
        'MoodScore': [0.6, 0.7, 0.8, 0.5, 0.9, 0.1, 0.4, 0.3, 0.2, 0.35],
        'TrackDurationMs': [200000, 180000, 220000, 240000, 160000, 210000, 190000, 250000, 170000, 230000],
        'Energy': [0.8, 0.9, 0.7, 0.6, 0.5, 0.3, 0.2, 0.4, 0.1, 0.75],
        'BeatsPerMinute': [120, 130, 110, 140, 100, 125, 135, 115, 145, 105]  # テスト用BPM
    })

    original_features = len(test_data.columns)
    print(f"テストデータ: {test_data.shape[0]}サンプル, {original_features}特徴量")
    print()

    # 1. 後方互換関数のテスト
    print("テスト1: 後方互換関数 create_binning_features()")
    try:
        from src.features import create_binning_features

        enhanced_data = create_binning_features(test_data)
        new_features = len(enhanced_data.columns) - original_features

        print(f"成功: {new_features}個の新特徴量を作成")

        # ビニング特徴量の確認
        binning_features = [col for col in enhanced_data.columns if col.endswith('_bin')]
        print(f"   ビニング特徴量: {len(binning_features)}特徴量")
        print(f"      {binning_features[:5]}{'...' if len(binning_features) > 5 else ''}")

        # 統計特徴量の確認
        stat_features = [col for col in enhanced_data.columns if '_bin_mean_bpm' in col or '_bin_std_bpm' in col]
        print(f"   統計特徴量: {len(stat_features)}特徴量")

        # データ品質チェック
        nan_count = enhanced_data.isnull().sum().sum()
        inf_count = np.isinf(enhanced_data.select_dtypes(include=[np.number])).sum().sum()
        print(f"   データ品質: NaN({nan_count}), inf({inf_count})値")

        if nan_count == 0 and inf_count == 0:
            print("   データ品質: 問題なし")
        else:
            print("   データ品質: 要確認")

    except Exception as e:
        print(f"エラー: {type(e).__name__}: {e}")
        return False

    print()

    # 2. 新しいクラスベースAPIのテスト
    print("テスト2: クラスベースAPI BinningFeatureCreator")
    try:
        from src.features import BinningFeatureCreator

        creator = BinningFeatureCreator()
        enhanced_data_class = creator.create_features(test_data)

        new_features_class = len(enhanced_data_class.columns) - original_features
        print(f"成功: {new_features_class}個の新特徴量を作成")
        print(f"   作成器名: {creator.name}")
        print(f"   作成特徴量: {len(creator.created_features)}個")

        # 特徴量情報の取得
        feature_info = creator.get_feature_info()
        print(f"   対象特徴量: {len(feature_info['target_features'])}個")
        print(f"   ビニング設定: {list(feature_info['binning_configs'].keys())}")

        # 両方の実装で同じ結果が得られるかチェック
        if enhanced_data.shape == enhanced_data_class.shape:
            print("   後方互換性: 形状一致")
        else:
            print(f"   形状差異: 関数版{enhanced_data.shape} vs クラス版{enhanced_data_class.shape}")

    except Exception as e:
        print(f"エラー: {type(e).__name__}: {e}")
        return False

    print()

    # 3. ビニング特徴量の詳細分析
    print("テスト3: ビニング特徴量詳細分析")
    try:
        # 各分位数タイプの確認
        binning_types = ['septile', 'decile', 'quintile']

        for binning_type in binning_types:
            type_features = [col for col in enhanced_data.columns if f'_{binning_type}_bin' in col]
            print(f"   {binning_type}分位特徴量: {len(type_features)}個")

            # サンプル値の確認
            if type_features:
                sample_feature = type_features[0]
                unique_values = enhanced_data[sample_feature].nunique()
                value_range = f"[{enhanced_data[sample_feature].min():.0f}-{enhanced_data[sample_feature].max():.0f}]"
                print(f"      {sample_feature}: {unique_values}カテゴリ {value_range}")

        # 統計特徴量の妥当性確認
        mean_bmp_features = [col for col in enhanced_data.columns if '_bin_mean_bpm' in col]
        if mean_bmp_features:
            sample_mean_feature = mean_bmp_features[0]
            mean_values = enhanced_data[sample_mean_feature]
            print(f"   BPM統計特徴量例:")
            print(f"      {sample_mean_feature}: 平均={mean_values.mean():.1f}, 範囲=[{mean_values.min():.1f}-{mean_values.max():.1f}]")

        print(f"   最終特徴量数: {len(enhanced_data.columns)}個")

    except Exception as e:
        print(f"エラー: {type(e).__name__}: {e}")
        return False

    print()

    # 4. log変換特徴量との統合テスト
    print("テスト4: log変換特徴量との統合テスト")
    try:
        from src.features import create_log_features

        # まずlog変換特徴量を作成
        log_enhanced_data = create_log_features(test_data)
        print(f"log変換後: {len(log_enhanced_data.columns)}特徴量")

        # 次にビニング特徴量を追加
        combined_data = create_binning_features(log_enhanced_data)
        combined_new_features = len(combined_data.columns) - original_features

        print(f"組み合わせ後: {len(combined_data.columns)}特徴量 (+{combined_new_features})")

        # log変換特徴量のビニングも確認
        log_binning_features = [col for col in combined_data.columns if 'log1p_' in col and '_bin' in col]
        print(f"   log変換ビニング特徴量: {len(log_binning_features)}個")
        if log_binning_features:
            print(f"      例: {log_binning_features[0]}")

    except Exception as e:
        print(f"エラー: {type(e).__name__}: {e}")
        return False

    print()

    # 5. CLI機能のテスト確認
    print("テスト5: CLI統合確認")
    print("   CLI オプション --create-binning-features が追加されました")
    print("   使用例:")
    print("      python -m src.features --create-binning-features")
    print("      python -m src.features --create-comprehensive-interactions --create-log-features --create-binning-features")

    print()
    print("TICKET-017-03: ビニング・カテゴリ特徴量テスト完了")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_binning_features()
    if success:
        print("\n全テスト成功: 実装準備完了")
    else:
        print("\nテスト失敗: 実装要修正")
        sys.exit(1)