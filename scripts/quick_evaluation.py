"""
クイック評価: 小サンプルでジャンル特徴量の効果を確認
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from loguru import logger
import lightgbm as lgb

# プロジェクトルートをパスに追加
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def quick_evaluation():
    """クイック評価実行"""
    logger.info("クイック評価を開始...")

    # データ読み込み（小サンプル）
    train_df = pd.read_csv("data/processed/train_features.csv")
    sample_size = min(10000, len(train_df))  # 最大1万サンプル
    train_sample = train_df.sample(n=sample_size, random_state=42)

    logger.info(f"サンプルサイズ: {sample_size}")

    # ベースライン特徴量（ジャンルなし）
    baseline_features = [col for col in train_sample.columns
                        if col not in ["id", "BeatsPerMinute"] and "genre_score" not in col]

    # 拡張特徴量（ジャンル込み）
    enhanced_features = [col for col in train_sample.columns
                        if col not in ["id", "BeatsPerMinute"]]

    X_baseline = train_sample[baseline_features]
    X_enhanced = train_sample[enhanced_features]
    y = train_sample["BeatsPerMinute"]

    # 訓練・テスト分割
    X_base_train, X_base_test, y_train, y_test = train_test_split(
        X_baseline, y, test_size=0.2, random_state=42
    )
    X_enh_train, X_enh_test, _, _ = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42
    )

    # モデル設定
    model_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 100,  # 軽量化
        "verbose": -1,
        "random_state": 42
    }

    # ベースライン評価
    logger.info("ベースライン評価中...")
    model_baseline = lgb.LGBMRegressor(**model_params)
    model_baseline.fit(X_base_train, y_train)
    y_pred_baseline = model_baseline.predict(X_base_test)
    rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))

    # 拡張版評価
    logger.info("拡張版評価中...")
    model_enhanced = lgb.LGBMRegressor(**model_params)
    model_enhanced.fit(X_enh_train, y_train)
    y_pred_enhanced = model_enhanced.predict(X_enh_test)
    rmse_enhanced = np.sqrt(mean_squared_error(y_test, y_pred_enhanced))

    # 結果表示
    improvement = rmse_baseline - rmse_enhanced
    improvement_pct = (improvement / rmse_baseline) * 100

    logger.info("=== クイック評価結果 ===")
    logger.info(f"ベースライン RMSE: {rmse_baseline:.4f} ({len(baseline_features)}特徴量)")
    logger.info(f"拡張版 RMSE: {rmse_enhanced:.4f} ({len(enhanced_features)}特徴量)")
    logger.info(f"改善: {improvement:.4f} ({improvement_pct:+.2f}%)")

    # ジャンル特徴量の重要度
    feature_importance = model_enhanced.feature_importances_
    feature_names = enhanced_features

    # ジャンル特徴量のみの重要度
    genre_importance = []
    for i, feature in enumerate(feature_names):
        if "genre_score" in feature:
            genre_importance.append((feature, feature_importance[i]))

    genre_importance.sort(key=lambda x: x[1], reverse=True)

    logger.info("=== ジャンル特徴量重要度 ===")
    for feature, importance in genre_importance:
        logger.info(f"  {feature}: {importance:.4f}")

    return {
        "baseline_rmse": rmse_baseline,
        "enhanced_rmse": rmse_enhanced,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "baseline_features": len(baseline_features),
        "enhanced_features": len(enhanced_features),
        "genre_importance": genre_importance
    }

if __name__ == "__main__":
    quick_evaluation()