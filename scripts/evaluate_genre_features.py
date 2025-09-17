"""
TICKET-008 評価スクリプト: 音楽ジャンル推定特徴量の効果測定

ベースライン（既存特徴量のみ）と新しいジャンル特徴量を含むモデルの
性能を比較評価し、特徴量重要度分析を実行する。
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from loguru import logger
import lightgbm as lgb

# プロジェクトルートをパスに追加
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.features import analyze_feature_importance, compare_genre_features_to_bpm


def load_data():
    """拡張特徴量データセットを読み込む"""
    logger.info("拡張特徴量データセットを読み込み中...")

    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_features.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "validation_features.csv")

    logger.info(f"訓練データ: {train_df.shape}")
    logger.info(f"検証データ: {val_df.shape}")

    return train_df, val_df


def prepare_features(df, use_genre_features=True):
    """特徴量を準備する"""

    # 基本特徴量（元の9個 + 交互作用 + 時間 + 統計的特徴量）
    base_features = [col for col in df.columns if col not in ["id", "BeatsPerMinute"]]

    if not use_genre_features:
        # ジャンル特徴量を除外
        base_features = [col for col in base_features if "genre_score" not in col]

    X = df[base_features]
    y = df["BeatsPerMinute"] if "BeatsPerMinute" in df.columns else None

    return X, y, base_features


def evaluate_model(X, y, model_name="LightGBM", cv_folds=5):
    """クロスバリデーションでモデルを評価する"""
    logger.info(f"{model_name}モデルをクロスバリデーション評価中...")

    # LightGBMモデル設定
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=42
    )

    # クロスバリデーション実行
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X, y, cv=kfold, scoring="neg_root_mean_squared_error", n_jobs=-1
    )

    # RMSEスコア（負値を正値に変換）
    rmse_scores = -cv_scores

    results = {
        "model_name": model_name,
        "cv_folds": cv_folds,
        "rmse_scores": rmse_scores.tolist(),
        "mean_rmse": rmse_scores.mean(),
        "std_rmse": rmse_scores.std(),
        "n_features": X.shape[1],
        "feature_names": X.columns.tolist()
    }

    logger.info(f"{model_name} 平均RMSE: {results['mean_rmse']:.4f} (±{results['std_rmse']:.4f})")
    logger.info(f"特徴量数: {results['n_features']}")

    return results


def compare_models():
    """ベースラインとジャンル特徴量モデルを比較する"""
    logger.info("モデル比較評価を開始...")

    # データ読み込み
    train_df, val_df = load_data()

    # 全データを結合（クロスバリデーション用）
    all_data = pd.concat([train_df, val_df], ignore_index=True)

    # ベースライン（ジャンル特徴量なし）
    logger.info("=== ベースライン評価（ジャンル特徴量なし）===")
    X_baseline, y, baseline_features = prepare_features(all_data, use_genre_features=False)
    baseline_results = evaluate_model(X_baseline, y, "Baseline_LightGBM")

    # ジャンル特徴量込みモデル
    logger.info("=== ジャンル特徴量込み評価 ===")
    X_enhanced, y, enhanced_features = prepare_features(all_data, use_genre_features=True)
    enhanced_results = evaluate_model(X_enhanced, y, "Enhanced_LightGBM")

    # 結果比較
    logger.info("=== 結果比較 ===")
    rmse_improvement = baseline_results["mean_rmse"] - enhanced_results["mean_rmse"]
    improvement_pct = (rmse_improvement / baseline_results["mean_rmse"]) * 100

    logger.info(f"ベースライン RMSE: {baseline_results['mean_rmse']:.4f}")
    logger.info(f"拡張版 RMSE: {enhanced_results['mean_rmse']:.4f}")
    logger.info(f"改善: {rmse_improvement:.4f} ({improvement_pct:+.2f}%)")

    # 特徴量重要度分析
    logger.info("=== 特徴量重要度分析 ===")

    # 全体重要度分析
    importance_all = analyze_feature_importance(X_enhanced, y, "all")

    # ジャンル特徴量の重要度分析
    importance_genre = analyze_feature_importance(X_enhanced, y, "genre")

    # ジャンル特徴量とBPMの関係分析
    genre_bpm_analysis = compare_genre_features_to_bpm(X_enhanced, y)

    # 結果をまとめる
    comparison_results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": baseline_results,
        "enhanced": enhanced_results,
        "improvement": {
            "absolute_rmse_improvement": rmse_improvement,
            "relative_improvement_pct": improvement_pct,
            "feature_count_increase": len(enhanced_features) - len(baseline_features)
        },
        "feature_analysis": {
            "top_10_features": importance_all.head(10).to_dict("records"),
            "genre_features_importance": importance_genre.to_dict("records"),
            "genre_bpm_relationships": genre_bpm_analysis.to_dict("records")
        }
    }

    return comparison_results, importance_all, genre_bpm_analysis


def save_results(results, importance_df, genre_analysis_df, output_dir):
    """評価結果を保存する"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON結果保存
    results_file = output_dir / f"genre_features_evaluation_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # CSVファイル保存
    importance_file = output_dir / f"feature_importance_{timestamp}.csv"
    importance_df.to_csv(importance_file, index=False)

    genre_analysis_file = output_dir / f"genre_bpm_analysis_{timestamp}.csv"
    genre_analysis_df.to_csv(genre_analysis_file, index=False)

    logger.success(f"評価結果を保存しました: {output_dir}")
    logger.info(f"- 総合結果: {results_file}")
    logger.info(f"- 特徴量重要度: {importance_file}")
    logger.info(f"- ジャンル分析: {genre_analysis_file}")

    return results_file, importance_file, genre_analysis_file


def main():
    """メイン評価関数"""
    logger.info("TICKET-008: 音楽ジャンル推定特徴量評価を開始")

    try:
        # モデル比較実行
        results, importance_df, genre_analysis_df = compare_models()

        # 結果保存
        output_dir = MODELS_DIR / "evaluations"
        save_results(results, importance_df, genre_analysis_df, output_dir)

        logger.success("音楽ジャンル推定特徴量評価が完了しました")

        # 重要な結果をサマリー表示
        logger.info("=== 評価サマリー ===")
        improvement = results["improvement"]
        logger.info(f"RMSE改善: {improvement['absolute_rmse_improvement']:.4f}")
        logger.info(f"改善率: {improvement['relative_improvement_pct']:+.2f}%")
        logger.info(f"特徴量数増加: {improvement['feature_count_increase']}個")

        # トップ5特徴量
        logger.info("=== トップ5重要特徴量 ===")
        for i, feature in enumerate(results["feature_analysis"]["top_10_features"][:5], 1):
            logger.info(f"{i}. {feature['feature_name']}: {feature['average_importance']:.4f}")

        return results

    except Exception as e:
        logger.error(f"評価中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()