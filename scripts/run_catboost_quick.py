#!/usr/bin/env python3
"""
TICKET-022-03 CatBoost最適化クイック実行スクリプト
短時間での実行用（トライアル数削減版）
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.modeling.optimization import OptunaCatBoostOptimizer
from loguru import logger
import time
import json
import pickle
import pandas as pd

def run_catboost_quick_optimization(n_trials: int = 10):
    """CatBoost最適化クイック実行（時間短縮版）"""

    logger.info("=" * 60)
    logger.info(f"TICKET-022-03: CatBoost最適化クイック実行（{n_trials}トライアル）")
    logger.info("=" * 60)

    # CatBoost最適化器初期化（トライアル数を削減）
    optimizer = OptunaCatBoostOptimizer(
        n_trials=n_trials,  # デフォルト10トライアル（約16分）
        timeout=1800,  # 30分タイムアウト
        cv_folds=5,
        cv_strategy="bpm_stratified",
        study_name=f"catboost_quick_{n_trials}trials"
    )

    try:
        # 最適化実行
        results = optimizer.optimize("data/processed/train_unified_75_features.csv")

        # 最終モデル訓練と予測
        predictions, final_results = optimizer.train_final_model("data/processed/test_unified_75_features.csv")

        # 提出ファイル作成
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # テストID読み込み
        test_df_original = pd.read_csv("data/processed/test_unified_75_features.csv")
        submission_df = pd.DataFrame({
            'id': test_df_original['id'],
            'BeatsPerMinute': predictions
        })

        # 提出ファイル保存
        submission_path = f"data/processed/submission_catboost_quick_{n_trials}trials_{timestamp}.csv"
        submission_df.to_csv(submission_path, index=False)

        # 結果保存
        results_path = f"experiments/catboost_quick_{n_trials}trials_{timestamp}"

        # JSON結果保存
        json_results = {k: v for k, v in final_results.items() if k != 'models'}
        json_path = Path(results_path).with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)

        # サマリー表示
        logger.info("\n" + "=" * 60)
        logger.info(f"CatBoost最適化完了（{n_trials}トライアル版）")
        logger.info("=" * 60)
        logger.info(f"最高CV RMSE: {final_results['cv_score']:.4f} (±{final_results['cv_std']:.4f})")
        logger.info(f"特徴量数: {final_results['n_features']}")
        logger.info(f"CV戦略: {final_results['cv_strategy']}")
        logger.info(f"提出ファイル: {submission_path}")
        logger.info(f"結果ファイル: {json_path}")

        # 最適パラメータ表示
        logger.info("\nCatBoost最適ハイパーパラメータ:")
        for param, value in final_results['best_params'].items():
            logger.info(f"  {param}: {value}")

        return True

    except Exception as e:
        logger.error(f"CatBoost最適化エラー: {e}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CatBoost最適化クイック実行")
    parser.add_argument("--trials", type=int, default=10, help="最適化トライアル数（デフォルト: 10）")

    args = parser.parse_args()

    success = run_catboost_quick_optimization(args.trials)
    if success:
        logger.success(f"CatBoost最適化成功（{args.trials}トライアル）")
    else:
        logger.error("CatBoost最適化失敗")
        sys.exit(1)