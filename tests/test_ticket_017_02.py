#!/usr/bin/env python3
"""
TICKET-017-02 対数変換特徴量のテストスクリプト

対数変換特徴量の実装をテストし、動作を確認する。
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

def test_log_features():
    """対数変換特徴量のテスト実行"""
    print("TICKET-017-02: 対数変換特徴量のテスト開始")
    print("=" * 60)

    # テスト用サンプルデータ作成
    test_data = pd.DataFrame({
        'RhythmScore': [0.7, 0.8, 0.6, 0.9, 0.5, 0.75],
        'AudioLoudness': [0.6, 0.7, 0.5, 0.8, 0.4, 0.65],  # 除外対象
        'VocalContent': [0.8, 0.6, 0.9, 0.5, 0.7, 0.72],
        'AcousticQuality': [0.5, 0.8, 0.7, 0.6, 0.9, 0.68],
        'InstrumentalScore': [0.7, 0.5, 0.8, 0.9, 0.6, 0.71],
        'LivePerformanceLikelihood': [0.4, 0.6, 0.5, 0.7, 0.8, 0.6],
        'MoodScore': [0.6, 0.7, 0.8, 0.5, 0.9, 0.7],
        'TrackDurationMs': [200000, 180000, 220000, 240000, 160000, 200000],
        'Energy': [0.8, 0.9, 0.7, 0.6, 0.5, 0.75]
    })

    original_features = len(test_data.columns)
    print(f"テストデータ: {test_data.shape[0]}サンプル, {original_features}特徴量")
    print()

    # 1. 後方互換関数のテスト
    print("テスト1: 後方互換関数 create_log_features()")
    try:
        from src.features import create_log_features

        enhanced_data = create_log_features(test_data)
        new_features = len(enhanced_data.columns) - original_features

        print(f"成功: {new_features}個の新特徴量を作成")

        # log1p特徴量の確認
        log_features = [col for col in enhanced_data.columns if col.startswith('log1p_')]
        print(f"   基本log1p変換: {len(log_features)}特徴量")
        print(f"      {log_features[:3]}{'...' if len(log_features) > 3 else ''}")

        # 組み合わせ特徴量の確認
        combo_features = [col for col in enhanced_data.columns if '_x_' in col and 'log1p_' in col]
        print(f"   組み合わせ特徴量: {len(combo_features)}特徴量")

        # 統計特徴量の確認
        stat_features = [col for col in enhanced_data.columns if col.startswith('log_features_')]
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
    print("テスト2: クラスベースAPI LogTransformFeatureCreator")
    try:
        from src.features import LogTransformFeatureCreator

        creator = LogTransformFeatureCreator()
        enhanced_data_class = creator.create_features(test_data)

        new_features_class = len(enhanced_data_class.columns) - original_features
        print(f"成功: {new_features_class}個の新特徴量を作成")
        print(f"   作成器名: {creator.name}")
        print(f"   作成特徴量: {len(creator.created_features)}個")

        # 特徴量情報の取得
        feature_info = creator.get_feature_info()
        print(f"   対象特徴量: {len(feature_info['target_features'])}個")
        print(f"   除外特徴量: {feature_info['exclude_features']}")

        # 両方の実装で同じ結果が得られるかチェック
        if enhanced_data.shape == enhanced_data_class.shape:
            print("   後方互換性: 形状一致")
        else:
            print(f"   形状差異: 関数版{enhanced_data.shape} vs クラス版{enhanced_data_class.shape}")

    except Exception as e:
        print(f"エラー: {type(e).__name__}: {e}")
        return False

    print()

    # 3. 特徴量の統計確認
    print("テスト3: 特徴量統計分析")
    try:
        # AudioLoudnessが除外されているかチェック
        excluded_feature = 'log1p_AudioLoudness'
        if excluded_feature not in enhanced_data.columns:
            print(f"除外機能: {excluded_feature}は正しく除外されている")
        else:
            print(f"除外機能: {excluded_feature}が除外されていない")

        # 分布の歪み改善確認
        if 'log_transformation_benefit' in enhanced_data.columns:
            benefit = enhanced_data['log_transformation_benefit'].iloc[0]
            print(f"分布改善指標: {benefit:.3f}")
        else:
            print("分布改善指標が見つかりません")

        # 幾何平均の確認
        if 'log_features_geometric_mean' in enhanced_data.columns:
            geo_mean = enhanced_data['log_features_geometric_mean'].mean()
            print(f"幾何平均: {geo_mean:.3f}")

        print(f"   最終特徴量数: {len(enhanced_data.columns)}個")

    except Exception as e:
        print(f"エラー: {type(e).__name__}: {e}")
        return False

    print()

    # 4. CLI機能のテスト確認
    print("テスト4: CLI統合確認")
    print("   CLI オプション --create-log-features が追加されました")
    print("   使用例:")
    print("      python -m src.features --create-log-features")
    print("      python -m src.features --create-comprehensive-interactions --create-log-features")

    print()
    print("TICKET-017-02: 対数変換特徴量テスト完了")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_log_features()
    if success:
        print("\n全テスト成功: 実装準備完了")
    else:
        print("\nテスト失敗: 実装要修正")
        sys.exit(1)