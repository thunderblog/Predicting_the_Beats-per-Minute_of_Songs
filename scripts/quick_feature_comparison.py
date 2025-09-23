#!/usr/bin/env python3
"""
特徴量組み合わせの迅速比較スクリプト

目的:
- 包括的交互作用特徴量と各組み合わせの効果を小サンプルで比較
- 複数の組み合わせパターンを一括テスト
- 最適な組み合わせを効率的に特定
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
import time
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
proj_root = Path(__file__).parent.parent
sys.path.append(str(proj_root))

from src.features import (
    create_comprehensive_interaction_features,
    create_music_genre_features,
    create_statistical_features,
    create_interaction_features,
    create_duration_features
)
from src.config import PROCESSED_DATA_DIR

def load_quick_sample(sample_size: int = 500):
    """迅速テスト用の小サンプルデータを読み込む"""
    logger.info(f"迅速テスト用サンプルデータを読み込み中（サイズ: {sample_size}）...")

    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

    # 小サンプル抽出
    if len(train_df) > sample_size:
        sample_df = train_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = train_df

    logger.info(f"サンプルデータ準備完了: {len(sample_df)}件")
    return sample_df

def quick_evaluate(X, y, feature_name="Default"):
    """超軽量2-fold CVで特徴量の性能を評価"""

    # 超軽量LightGBM設定
    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # さらに軽量化
        'learning_rate': 0.1,
        'n_estimators': 30,  # 軽量化
        'verbose': -1,
        'random_state': 42
    }

    # 2-fold CV（超軽量）
    kfold = KFold(n_splits=2, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kfold.split(X):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # モデル訓練
        model = lgb.LGBMRegressor(**model_params)
        model.fit(X_train_fold, y_train_fold)

        # 予測・評価
        y_pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        cv_scores.append(rmse)

    avg_rmse = np.mean(cv_scores)
    return {
        'feature_combination': feature_name,
        'n_features': X.shape[1],
        'avg_rmse': avg_rmse,
        'cv_scores': cv_scores
    }

def test_feature_combinations(sample_data):
    """各特徴量組み合わせをテスト"""

    results = []
    y = sample_data['BeatsPerMinute']

    # 基本特徴量
    basic_features = [
        'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
        'TrackDurationMs', 'Energy'
    ]

    logger.info("=== 特徴量組み合わせ比較開始 ===")

    # 1. ベースライン（基本特徴量のみ）
    logger.info("1. ベースライン（基本特徴量）をテスト中...")
    start_time = time.time()
    X_baseline = sample_data[basic_features]
    result_baseline = quick_evaluate(X_baseline, y, "1. Baseline（基本特徴量）")
    result_baseline['processing_time'] = time.time() - start_time
    results.append(result_baseline)
    logger.info(f"   RMSE: {result_baseline['avg_rmse']:.4f}, 特徴量数: {result_baseline['n_features']}, 時間: {result_baseline['processing_time']:.1f}秒")

    # 2. 包括的交互作用のみ
    logger.info("2. 包括的交互作用特徴量をテスト中...")
    start_time = time.time()
    data_comprehensive = create_comprehensive_interaction_features(sample_data)
    feature_cols = [col for col in data_comprehensive.columns if col not in ['id', 'BeatsPerMinute']]
    X_comprehensive = data_comprehensive[feature_cols]
    result_comprehensive = quick_evaluate(X_comprehensive, y, "2. 包括的交互作用特徴量")
    result_comprehensive['processing_time'] = time.time() - start_time
    results.append(result_comprehensive)
    logger.info(f"   RMSE: {result_comprehensive['avg_rmse']:.4f}, 特徴量数: {result_comprehensive['n_features']}, 時間: {result_comprehensive['processing_time']:.1f}秒")

    # 3. 包括的交互作用 + ジャンル特徴量
    logger.info("3. 包括的交互作用 + ジャンル特徴量をテスト中...")
    start_time = time.time()
    data_genre = create_music_genre_features(data_comprehensive)
    feature_cols = [col for col in data_genre.columns if col not in ['id', 'BeatsPerMinute']]
    X_genre = data_genre[feature_cols]
    result_genre = quick_evaluate(X_genre, y, "3. 包括的交互作用 + ジャンル")
    result_genre['processing_time'] = time.time() - start_time
    results.append(result_genre)
    logger.info(f"   RMSE: {result_genre['avg_rmse']:.4f}, 特徴量数: {result_genre['n_features']}, 時間: {result_genre['processing_time']:.1f}秒")

    # 4. 包括的交互作用 + ジャンル + 統計特徴量
    logger.info("4. 包括的交互作用 + ジャンル + 統計特徴量をテスト中...")
    start_time = time.time()
    data_stats = create_statistical_features(data_genre)
    feature_cols = [col for col in data_stats.columns if col not in ['id', 'BeatsPerMinute']]
    X_stats = data_stats[feature_cols]
    result_stats = quick_evaluate(X_stats, y, "4. 包括的交互作用 + ジャンル + 統計")
    result_stats['processing_time'] = time.time() - start_time
    results.append(result_stats)
    logger.info(f"   RMSE: {result_stats['avg_rmse']:.4f}, 特徴量数: {result_stats['n_features']}, 時間: {result_stats['processing_time']:.1f}秒")

    # 5. 包括的交互作用 + ジャンル + 統計 + 時間特徴量
    logger.info("5. 包括的交互作用 + ジャンル + 統計 + 時間特徴量をテスト中...")
    start_time = time.time()
    data_duration = create_duration_features(data_stats)
    feature_cols = [col for col in data_duration.columns if col not in ['id', 'BeatsPerMinute']]
    X_duration = data_duration[feature_cols]
    result_duration = quick_evaluate(X_duration, y, "5. 包括的交互作用 + ジャンル + 統計 + 時間")
    result_duration['processing_time'] = time.time() - start_time
    results.append(result_duration)
    logger.info(f"   RMSE: {result_duration['avg_rmse']:.4f}, 特徴量数: {result_duration['n_features']}, 時間: {result_duration['processing_time']:.1f}秒")

    # 6. 基本交互作用のみ（比較用）
    logger.info("6. 基本交互作用特徴量（比較用）をテスト中...")
    start_time = time.time()
    data_basic_interaction = create_interaction_features(sample_data)
    feature_cols = [col for col in data_basic_interaction.columns if col not in ['id', 'BeatsPerMinute']]
    X_basic_interaction = data_basic_interaction[feature_cols]
    result_basic_interaction = quick_evaluate(X_basic_interaction, y, "6. 基本交互作用特徴量（比較用）")
    result_basic_interaction['processing_time'] = time.time() - start_time
    results.append(result_basic_interaction)
    logger.info(f"   RMSE: {result_basic_interaction['avg_rmse']:.4f}, 特徴量数: {result_basic_interaction['n_features']}, 時間: {result_basic_interaction['processing_time']:.1f}秒")

    return results

def analyze_results(results):
    """結果を分析して最適な組み合わせを特定"""
    logger.info("=== 結果分析 ===")

    # ベースラインを基準とした改善率計算
    baseline_rmse = results[0]['avg_rmse']

    logger.info("\n📊 性能比較結果:")
    logger.info(f"{'組み合わせ':<35} {'RMSE':<8} {'改善率':<8} {'特徴量数':<8} {'時間':<6}")
    logger.info("-" * 75)

    best_result = None
    best_improvement = -float('inf')

    for result in results:
        improvement = baseline_rmse - result['avg_rmse']
        improvement_pct = (improvement / baseline_rmse) * 100

        logger.info(f"{result['feature_combination']:<35} "
                   f"{result['avg_rmse']:<8.4f} "
                   f"{improvement_pct:+6.2f}% "
                   f"{result['n_features']:<8} "
                   f"{result['processing_time']:<6.1f}秒")

        if improvement > best_improvement:
            best_improvement = improvement
            best_result = result

    logger.info("-" * 75)
    logger.success(f"🏆 最良の組み合わせ: {best_result['feature_combination']}")
    logger.success(f"   RMSE: {best_result['avg_rmse']:.4f} (ベースラインより{best_improvement:+.4f}改善)")
    logger.success(f"   改善率: {(best_improvement / baseline_rmse) * 100:+.2f}%")

    # 効率性分析
    logger.info("\n🎯 効率性分析:")
    for result in results:
        efficiency = (baseline_rmse - result['avg_rmse']) / result['processing_time']
        logger.info(f"{result['feature_combination'][:25]:<25}: 効率性 = {efficiency:.6f} (改善/秒)")

    return best_result, results

def main():
    """メイン実行関数"""
    logger.info("=== 特徴量組み合わせ迅速比較テスト ===")

    # 小サンプルデータ読み込み
    sample_data = load_quick_sample(sample_size=500)

    # 各組み合わせをテスト
    results = test_feature_combinations(sample_data)

    # 結果分析
    best_result, all_results = analyze_results(results)

    # 結果保存
    import json
    results_summary = {
        'test_conditions': {
            'sample_size': len(sample_data),
            'cv_folds': 2,
            'lgbm_n_estimators': 30
        },
        'results': all_results,
        'best_combination': best_result
    }

    results_file = Path("results/quick_feature_comparison.json")
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    logger.success(f"比較結果を保存: {results_file}")

    # 推奨事項
    logger.info("\n💡 推奨事項:")
    if best_result['feature_combination'].find('包括的交互作用') != -1:
        logger.info("✅ 包括的交互作用特徴量が有効です！")
        logger.info(f"✅ 推奨組み合わせ: {best_result['feature_combination']}")
        logger.info("📝 次のステップ: より大きなサンプルサイズでの検証")
    else:
        logger.warning("⚠️ 包括的交互作用特徴量の効果が限定的です")
        logger.info("📝 次のステップ: パラメータ調整やデータ品質の確認")

    return results_summary

if __name__ == "__main__":
    main()