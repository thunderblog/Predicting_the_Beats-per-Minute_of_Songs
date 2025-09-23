#!/usr/bin/env python3
"""
TICKET-017-01 包括的交互作用特徴量の性能テストスクリプト

目的:
- 包括的交互作用特徴量（126個）の有効性検証
- ベースライン（26.47 RMSE）との比較
- 軽量テストによる迅速な性能評価
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
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
proj_root = Path(__file__).parent.parent
sys.path.append(str(proj_root))

from src.features import create_comprehensive_interaction_features
from src.config import PROCESSED_DATA_DIR

def load_sample_data(sample_size: int = 1000):
    """軽量テスト用のサンプルデータを読み込む"""
    logger.info(f"サンプルデータを読み込み中（サイズ: {sample_size}）...")

    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

    # 軽量化のためサンプリング
    if len(train_df) > sample_size:
        train_sample = train_df.sample(n=sample_size, random_state=42)
    else:
        train_sample = train_df

    logger.info(f"サンプルデータ読み込み完了: {len(train_sample)}件")
    return train_sample

def evaluate_features(X, y, feature_type_name="Default"):
    """軽量3-fold CVで特徴量の性能を評価"""
    logger.info(f"{feature_type_name}特徴量の性能評価中（特徴量数: {X.shape[1]}）...")

    # LightGBM軽量設定
    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 50,  # 軽量化
        'verbose': -1,
        'random_state': 42
    }

    # 3-fold CV（軽量化）
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
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

        logger.info(f"  Fold {fold+1}: RMSE = {rmse:.4f}")

    avg_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)

    logger.info(f"{feature_type_name} 平均RMSE: {avg_rmse:.4f} (±{std_rmse:.4f})")

    return {
        'feature_type': feature_type_name,
        'n_features': X.shape[1],
        'avg_rmse': avg_rmse,
        'std_rmse': std_rmse,
        'cv_scores': cv_scores
    }

def main():
    """メイン実行関数"""
    logger.info("=== TICKET-017-01 性能テスト開始 ===")

    # 1. サンプルデータ読み込み
    sample_data = load_sample_data(sample_size=1000)

    # 基本特徴量の準備
    basic_features = [
        'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
        'TrackDurationMs', 'Energy'
    ]

    X_basic = sample_data[basic_features]
    y = sample_data['BeatsPerMinute']

    # 2. ベースライン評価（基本特徴量のみ）
    logger.info("--- ベースライン評価 ---")
    baseline_result = evaluate_features(X_basic, y, "Baseline（基本特徴量）")

    # 3. 包括的交互作用特徴量を生成
    logger.info("--- 包括的交互作用特徴量生成 ---")
    enhanced_data = create_comprehensive_interaction_features(sample_data)

    # 新特徴量を含む全特徴量を抽出
    feature_cols = [col for col in enhanced_data.columns if col not in ['id', 'BeatsPerMinute']]
    X_enhanced = enhanced_data[feature_cols]

    # 4. 拡張特徴量評価
    logger.info("--- 拡張特徴量評価 ---")
    enhanced_result = evaluate_features(X_enhanced, y, "Enhanced（拡張特徴量）")

    # 5. 結果比較
    logger.info("=== 性能比較結果 ===")
    improvement = baseline_result['avg_rmse'] - enhanced_result['avg_rmse']
    improvement_pct = (improvement / baseline_result['avg_rmse']) * 100

    logger.info(f"ベースライン RMSE: {baseline_result['avg_rmse']:.4f} (±{baseline_result['std_rmse']:.4f})")
    logger.info(f"拡張特徴量 RMSE: {enhanced_result['avg_rmse']:.4f} (±{enhanced_result['std_rmse']:.4f})")
    logger.info(f"改善: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    logger.info(f"特徴量数: {baseline_result['n_features']} → {enhanced_result['n_features']} (+{enhanced_result['n_features'] - baseline_result['n_features']})")

    # 6. 結果保存
    results_summary = {
        'baseline': baseline_result,
        'enhanced': enhanced_result,
        'improvement': improvement,
        'improvement_percentage': improvement_pct,
        'new_features_count': enhanced_result['n_features'] - baseline_result['n_features']
    }

    import json
    results_file = Path("results/ticket017_01_performance_test.json")
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    logger.success(f"性能テスト完了！結果を保存: {results_file}")

    # 判定
    if improvement > 0:
        logger.success("✅ 包括的交互作用特徴量により性能向上を確認！")
    else:
        logger.warning("⚠️ 性能向上が確認できませんでした。追加調査が必要です。")

    return results_summary

if __name__ == "__main__":
    main()