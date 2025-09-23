#!/usr/bin/env python3
"""
サンプルサイズ段階的拡大テストスクリプト

目的:
- 包括的交互作用特徴量が効果を発揮するサンプルサイズを特定
- 500, 1000, 2000, 5000件での性能比較
- 特徴量数とサンプル数の最適バランスを解明
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

from src.features import create_comprehensive_interaction_features
from src.config import PROCESSED_DATA_DIR

def load_scaled_sample(sample_size: int):
    """指定サイズのサンプルデータを読み込む"""
    logger.info(f"サンプルデータを読み込み中（サイズ: {sample_size}）...")

    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

    if len(train_df) > sample_size:
        sample_df = train_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = train_df
        logger.warning(f"要求サイズ{sample_size}件に対して実際は{len(sample_df)}件")

    logger.info(f"サンプルデータ準備完了: {len(sample_df)}件")
    return sample_df

def evaluate_with_size(X, y, feature_name="Default", sample_size=0):
    """サンプルサイズに応じた適切な設定でモデル評価"""

    # サンプルサイズに応じてパラメータを調整
    if sample_size <= 1000:
        cv_folds = 3
        n_estimators = 50
        num_leaves = 31
    elif sample_size <= 3000:
        cv_folds = 4
        n_estimators = 100
        num_leaves = 31
    else:
        cv_folds = 5
        n_estimators = 150
        num_leaves = 63

    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': num_leaves,
        'learning_rate': 0.1,
        'n_estimators': n_estimators,
        'verbose': -1,
        'random_state': 42
    }

    logger.info(f"  モデル設定: {cv_folds}-fold CV, n_estimators={n_estimators}, num_leaves={num_leaves}")

    # 適応的CV
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
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

        logger.info(f"    Fold {fold+1}: RMSE = {rmse:.4f}")

    avg_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)

    return {
        'feature_type': feature_name,
        'sample_size': sample_size,
        'n_features': X.shape[1],
        'avg_rmse': avg_rmse,
        'std_rmse': std_rmse,
        'cv_scores': cv_scores,
        'cv_folds': cv_folds,
        'n_estimators': n_estimators
    }

def test_sample_size(sample_size: int):
    """指定サンプルサイズでベースラインvs包括的交互作用を比較"""
    logger.info(f"\n=== サンプルサイズ {sample_size}件でのテスト ===")

    # データ読み込み
    sample_data = load_scaled_sample(sample_size)
    y = sample_data['BeatsPerMinute']

    # 基本特徴量
    basic_features = [
        'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
        'TrackDurationMs', 'Energy'
    ]

    results = []

    # 1. ベースライン評価
    logger.info("1. ベースライン（基本特徴量）をテスト中...")
    start_time = time.time()
    X_baseline = sample_data[basic_features]
    result_baseline = evaluate_with_size(X_baseline, y, "Baseline", sample_size)
    result_baseline['processing_time'] = time.time() - start_time
    results.append(result_baseline)

    # 2. 包括的交互作用特徴量評価
    logger.info("2. 包括的交互作用特徴量をテスト中...")
    start_time = time.time()
    enhanced_data = create_comprehensive_interaction_features(sample_data)
    feature_cols = [col for col in enhanced_data.columns if col not in ['id', 'BeatsPerMinute']]
    X_enhanced = enhanced_data[feature_cols]
    result_enhanced = evaluate_with_size(X_enhanced, y, "Enhanced", sample_size)
    result_enhanced['processing_time'] = time.time() - start_time
    results.append(result_enhanced)

    # 結果比較
    improvement = result_baseline['avg_rmse'] - result_enhanced['avg_rmse']
    improvement_pct = (improvement / result_baseline['avg_rmse']) * 100

    logger.info(f"\n📊 サンプルサイズ {sample_size}件の結果:")
    logger.info(f"  ベースライン: {result_baseline['avg_rmse']:.4f} (±{result_baseline['std_rmse']:.4f})")
    logger.info(f"  拡張特徴量:   {result_enhanced['avg_rmse']:.4f} (±{result_enhanced['std_rmse']:.4f})")
    logger.info(f"  改善:         {improvement:+.4f} ({improvement_pct:+.2f}%)")
    logger.info(f"  特徴量数:     {result_baseline['n_features']} → {result_enhanced['n_features']}")

    return {
        'sample_size': sample_size,
        'baseline': result_baseline,
        'enhanced': result_enhanced,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }

def main():
    """メイン実行関数 - 段階的サンプルサイズテスト"""
    logger.info("=== 段階的サンプルサイズテスト開始 ===")

    # テストするサンプルサイズ
    sample_sizes = [500, 1000, 2000, 5000]
    all_results = []

    # 各サンプルサイズでテスト
    for sample_size in sample_sizes:
        try:
            result = test_sample_size(sample_size)
            all_results.append(result)
        except Exception as e:
            logger.error(f"サンプルサイズ {sample_size} でエラー: {e}")
            continue

    # 結果サマリー
    logger.info("\n=== 段階的サンプルサイズテスト結果サマリー ===")
    logger.info(f"{'サンプル':<8} {'ベースライン':<12} {'拡張特徴量':<12} {'改善':<10} {'改善率':<8} {'特徴量数':<10}")
    logger.info("-" * 70)

    best_sample_size = None
    best_improvement = -float('inf')

    for result in all_results:
        sample_size = result['sample_size']
        baseline_rmse = result['baseline']['avg_rmse']
        enhanced_rmse = result['enhanced']['avg_rmse']
        improvement = result['improvement']
        improvement_pct = result['improvement_pct']
        n_features = result['enhanced']['n_features']

        logger.info(f"{sample_size:<8} {baseline_rmse:<12.4f} {enhanced_rmse:<12.4f} "
                   f"{improvement:<10.4f} {improvement_pct:<8.2f}% {n_features:<10}")

        if improvement > best_improvement:
            best_improvement = improvement
            best_sample_size = sample_size

    logger.info("-" * 70)

    if best_sample_size:
        logger.success(f"🏆 最良の結果: サンプルサイズ {best_sample_size}件 (改善: {best_improvement:+.4f})")

        # 推奨事項
        if best_improvement > 0:
            logger.success("✅ 包括的交互作用特徴量が有効です！")
            logger.info(f"📝 推奨サンプルサイズ: {best_sample_size}件以上")
        else:
            logger.warning("⚠️ すべてのサンプルサイズで改善が見られませんでした")
    else:
        logger.error("❌ テスト実行に失敗しました")

    # 詳細結果保存
    import json
    results_file = Path("results/scaling_sample_test.json")
    results_file.parent.mkdir(exist_ok=True)

    summary = {
        'test_summary': {
            'sample_sizes_tested': sample_sizes,
            'best_sample_size': best_sample_size,
            'best_improvement': best_improvement
        },
        'detailed_results': all_results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.success(f"詳細結果を保存: {results_file}")

    # パフォーマンストレンド分析
    if len(all_results) >= 2:
        logger.info("\n📈 パフォーマンストレンド分析:")
        for i in range(1, len(all_results)):
            prev = all_results[i-1]
            curr = all_results[i]

            trend = curr['improvement_pct'] - prev['improvement_pct']
            logger.info(f"  {prev['sample_size']} → {curr['sample_size']}件: 改善率変化 {trend:+.2f}%ポイント")

    return all_results

if __name__ == "__main__":
    main()