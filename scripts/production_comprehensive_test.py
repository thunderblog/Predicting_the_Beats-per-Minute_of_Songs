#!/usr/bin/env python3
"""
本番環境での包括的交互作用特徴量性能テスト

目的:
- 過去の実験と同等の条件で包括的交互作用特徴量をテスト
- 全データセット + 5-fold CV + 本格的LightGBM設定で評価
- 実際の性能改善効果を正確に測定
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

def load_full_dataset():
    """全データセットを読み込む（過去の実験と同じ条件）"""
    logger.info("全データセットを読み込み中...")

    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

    logger.info(f"データセットサイズ: {len(train_df)}件")
    logger.info(f"特徴量数: {len(train_df.columns) - 2}個")  # id, BeatsPerMinute除く

    return train_df

def production_evaluate(X, y, feature_name="Default"):
    """本番環境と同等の設定でモデル評価"""
    logger.info(f"本番環境評価開始: {feature_name}")

    # 過去の実験と同じLightGBM設定
    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'n_estimators': 10000,  # 過去の実験と同じ
        'verbose': -1,
        'random_state': 42
    }

    # 5-fold CV（過去の実験と同じ）
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    fold_times = []

    logger.info(f"  モデル設定: 5-fold CV, n_estimators=10000, num_leaves=31")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        fold_start_time = time.time()

        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # モデル訓練（Early Stopping付き）
        model = lgb.LGBMRegressor(**model_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        # 予測・評価
        y_pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        cv_scores.append(rmse)

        fold_time = time.time() - fold_start_time
        fold_times.append(fold_time)

        logger.info(f"    Fold {fold+1}: RMSE = {rmse:.6f}, 時間 = {fold_time:.1f}秒")

    avg_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)
    total_time = sum(fold_times)

    logger.success(f"  {feature_name}: 平均RMSE = {avg_rmse:.6f} (±{std_rmse:.6f})")

    return {
        'feature_combination': feature_name,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'avg_rmse': avg_rmse,
        'std_rmse': std_rmse,
        'cv_scores': cv_scores,
        'total_training_time': total_time,
        'avg_fold_time': np.mean(fold_times)
    }

def main():
    """メイン実行関数 - 本番環境での包括的交互作用特徴量テスト"""
    logger.info("=== 本番環境での包括的交互作用特徴量性能テスト ===")

    # 全データセット読み込み
    full_data = load_full_dataset()
    y = full_data['BeatsPerMinute']

    # 基本特徴量（過去の実験のベースライン）
    basic_features = [
        'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
        'TrackDurationMs', 'Energy'
    ]

    results = []

    # 1. ベースライン評価（過去の実験再現）
    logger.info("=== 1. ベースライン評価（過去の実験再現） ===")
    X_baseline = full_data[basic_features]
    baseline_result = production_evaluate(X_baseline, y, "Baseline（基本特徴量）")
    results.append(baseline_result)

    # 2. 包括的交互作用特徴量評価
    logger.info("=== 2. 包括的交互作用特徴量評価 ===")
    logger.info("包括的交互作用特徴量を生成中...")
    enhanced_data = create_comprehensive_interaction_features(full_data)
    feature_cols = [col for col in enhanced_data.columns if col not in ['id', 'BeatsPerMinute']]
    X_enhanced = enhanced_data[feature_cols]

    logger.info(f"特徴量数: {len(basic_features)} → {len(feature_cols)}個")

    enhanced_result = production_evaluate(X_enhanced, y, "包括的交互作用特徴量")
    results.append(enhanced_result)

    # 3. 結果比較・分析
    logger.info("=== 3. 結果比較・分析 ===")

    baseline_rmse = baseline_result['avg_rmse']
    enhanced_rmse = enhanced_result['avg_rmse']
    improvement = baseline_rmse - enhanced_rmse
    improvement_pct = (improvement / baseline_rmse) * 100

    logger.info(f"\\n📊 本番環境での性能比較結果:")
    logger.info(f"  ベースライン:           {baseline_rmse:.6f} (±{baseline_result['std_rmse']:.6f})")
    logger.info(f"  包括的交互作用特徴量:   {enhanced_rmse:.6f} (±{enhanced_result['std_rmse']:.6f})")
    logger.info(f"  改善:                   {improvement:+.6f} ({improvement_pct:+.4f}%)")
    logger.info(f"  特徴量数:               {baseline_result['n_features']} → {enhanced_result['n_features']}")
    logger.info(f"  サンプル数:             {enhanced_result['n_samples']:,}件")

    # 過去の実験との比較
    logger.info(f"\\n🔍 過去の実験との比較:")
    logger.info(f"  過去のベースライン:     26.470000 (exp01/exp005)")
    logger.info(f"  今回のベースライン:     {baseline_rmse:.6f}")
    logger.info(f"  ベースライン差:         {baseline_rmse - 26.47:+.6f}")

    # 統計的有意性検定
    from scipy.stats import ttest_rel
    if len(baseline_result['cv_scores']) == len(enhanced_result['cv_scores']):
        t_stat, p_value = ttest_rel(baseline_result['cv_scores'], enhanced_result['cv_scores'])
        logger.info(f"\\n📈 統計的有意性:")
        logger.info(f"  t統計量:   {t_stat:.4f}")
        logger.info(f"  p値:       {p_value:.6f}")
        logger.info(f"  有意性:    {'有意' if p_value < 0.05 else '有意ではない'} (α=0.05)")

    # 結果保存
    import json
    results_summary = {
        'test_conditions': {
            'environment': 'production',
            'sample_size': enhanced_result['n_samples'],
            'cv_folds': 5,
            'lgbm_n_estimators': 10000,
            'early_stopping_rounds': 50
        },
        'baseline_result': baseline_result,
        'enhanced_result': enhanced_result,
        'improvement_analysis': {
            'absolute_improvement': improvement,
            'percentage_improvement': improvement_pct,
            'statistical_significance': p_value if 'p_value' in locals() else None
        },
        'comparison_with_past_experiments': {
            'past_baseline_rmse': 26.47,
            'current_baseline_rmse': baseline_rmse,
            'baseline_consistency': abs(baseline_rmse - 26.47) < 0.01
        }
    }

    results_file = Path("results/production_comprehensive_test.json")
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)

    logger.success(f"本番環境テスト結果を保存: {results_file}")

    # 結論・推奨事項
    logger.info(f"\\n💡 結論・推奨事項:")
    if improvement > 0 and improvement_pct > 0.1:
        logger.success("✅ 包括的交互作用特徴量が本番環境で有効に機能しています！")
        logger.info(f"📝 推奨: 次の実験で包括的交互作用特徴量を標準的に使用")
        logger.info(f"📝 期待LBスコア: ~{enhanced_rmse:.2f} (改善: {improvement_pct:+.2f}%)")
    elif abs(improvement) < 0.01:
        logger.warning("⚠️ 改善効果は微小です。特徴量選択との組み合わせを検討してください")
    else:
        logger.warning("❌ 包括的交互作用特徴量の効果が確認できませんでした")
        logger.info("📝 推奨: 特徴量エンジニアリング手法の見直し")

    return results_summary

if __name__ == "__main__":
    main()