#!/usr/bin/env python3
"""
TICKET-021: 正則化二元アンサンブル実行スクリプト

exp09_1最高LB性能(26.38534)の正則化設定 + 最適化CatBoostの統合
目標: 単一モデル限界突破による26.385未満達成
"""

import sys
from pathlib import Path
import time
import pandas as pd
import json
import pickle
from loguru import logger

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.modeling.ensemble import EnsembleRegressor


def run_ticket_021_regularized_ensemble(
    data_path: str = "data/processed/train_unified_75_features.csv",
    test_path: str = "data/processed/test_unified_75_features.csv",
    cv_strategy: str = "bpm_stratified",
    n_trials: int = 100
):
    """TICKET-021正則化二元アンサンブル実行"""

    logger.info("=" * 60)
    logger.info("TICKET-021: 正則化二元アンサンブル実行")
    logger.info("=" * 60)
    logger.info(f"データ: {data_path}")
    logger.info(f"CV戦略: {cv_strategy}")
    logger.info(f"重み最適化トライアル: {n_trials}")

    try:
        # データ読み込み
        logger.info("データ読み込み中...")
        train_df = pd.read_csv(data_path)

        # 特徴量とターゲット分離
        feature_cols = [col for col in train_df.columns if col not in ['id', 'BeatsPerMinute']]
        X = train_df[feature_cols]
        y = train_df['BeatsPerMinute']

        logger.success(f"データ準備完了: 特徴量数={len(feature_cols)}, サンプル数={len(X)}")

        # アンサンブル回帰器初期化
        ensemble = EnsembleRegressor(
            n_folds=5,
            random_state=42,
            cv_strategy=cv_strategy
        )

        # Step 1: モデル訓練とOut-of-Fold予測
        logger.info("Step 1: 二元アンサンブルモデル訓練開始...")
        oof_predictions = ensemble.train_fold_models(X, y)

        # Step 2: 重み最適化
        logger.info("Step 2: Optuna重み最適化開始...")
        optimal_weights = ensemble.optimize_ensemble_weights(
            oof_predictions, y, n_trials=n_trials
        )

        # Step 3: テストデータ予測
        logger.info("Step 3: テストデータ予測開始...")
        test_df = pd.read_csv(test_path)
        X_test = test_df[feature_cols]
        predictions = ensemble.predict(X_test)

        # 結果保存
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 提出ファイル作成
        submission_df = pd.DataFrame({
            'id': test_df['id'],
            'BeatsPerMinute': predictions
        })
        submission_path = f"data/processed/submission_ticket021_regularized_{timestamp}.csv"
        submission_df.to_csv(submission_path, index=False)

        # アンサンブル結果保存
        results_path = f"experiments/ticket021_regularized_ensemble_{timestamp}"
        ensemble_results = ensemble.save_ensemble(Path("models"), f"ticket021_regularized_{timestamp}")

        # 詳細結果保存
        detailed_results = {
            "experiment_name": "ticket021_regularized_ensemble",
            "timestamp": timestamp,
            "cv_strategy": cv_strategy,
            "n_features": len(feature_cols),
            "n_samples": len(X),
            "optimal_weights": optimal_weights,
            "cv_scores": ensemble.cv_scores,
            "target_lb_score": 26.385,  # 目標値
            "baseline_lb_score": 26.38534,  # exp09_1ベースライン
        }

        # JSON保存
        json_path = Path(results_path).with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)

        # サマリー表示
        logger.info("\n" + "=" * 60)
        logger.info("TICKET-021 正則化二元アンサンブル完了")
        logger.info("=" * 60)

        # CV性能表示
        for model_type, scores in ensemble.cv_scores.items():
            logger.info(f"{model_type.upper()} CV: {scores['mean']:.4f} ± {scores['std']:.4f}")

        # アンサンブル重み表示
        logger.info("\n最適アンサンブル重み:")
        for model, weight in optimal_weights.items():
            logger.info(f"  {model}: {weight:.3f}")

        logger.info(f"\n設定情報:")
        logger.info(f"  特徴量数: {len(feature_cols)}")
        logger.info(f"  CV戦略: {cv_strategy}")
        logger.info(f"  LightGBM: exp09_1正則化設定統合")
        logger.info(f"  CatBoost: TICKET-022-03最適化設定統合")

        logger.info(f"\nファイル出力:")
        logger.info(f"  提出ファイル: {submission_path}")
        logger.info(f"  結果ファイル: {json_path}")

        # 目標達成判定
        ensemble_cv_rmse = min([scores['mean'] for scores in ensemble.cv_scores.values()])
        target_rmse = 26.385
        if ensemble_cv_rmse < target_rmse:
            logger.success(f"🎯 目標達成! CV RMSE {ensemble_cv_rmse:.4f} < {target_rmse}")
        else:
            logger.info(f"📊 CV RMSE {ensemble_cv_rmse:.4f} (目標: < {target_rmse})")

        # 提出コマンド表示
        logger.info(f"\n提出コマンド:")
        logger.info(f'kaggle competitions submit -c playground-series-s5e9 -f "{submission_path}" -m "TICKET-021 Regularized Binary Ensemble (exp09_1+optimized_catboost)"')

        return True

    except Exception as e:
        logger.error(f"TICKET-021実行エラー: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TICKET-021正則化二元アンサンブル実行")
    parser.add_argument("--data", default="data/processed/train_unified_75_features.csv", help="訓練データパス")
    parser.add_argument("--test", default="data/processed/test_unified_75_features.csv", help="テストデータパス")
    parser.add_argument("--cv", default="bpm_stratified", help="CV戦略")
    parser.add_argument("--trials", type=int, default=100, help="重み最適化トライアル数")

    args = parser.parse_args()

    success = run_ticket_021_regularized_ensemble(
        data_path=args.data,
        test_path=args.test,
        cv_strategy=args.cv,
        n_trials=args.trials
    )

    if success:
        logger.success("TICKET-021 正則化二元アンサンブル成功")
    else:
        logger.error("TICKET-021 正則化二元アンサンブル失敗")
        sys.exit(1)