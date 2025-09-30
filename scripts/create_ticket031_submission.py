"""
TICKET-031精密最適化結果を使用したSubmission作成スクリプト

最適化済みアンサンブル重みを使用してテストデータの予測を生成します。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.my_config import CFG

config = CFG()


def create_submission_with_weights(
    weight_lightgbm: float = 0.7060582514745803,
    weight_catboost: float = None
):
    """
    最適化済み重みを使用してsubmissionを作成

    Parameters
    ----------
    weight_lightgbm : float
        LightGBMの重み（デフォルト: Trial 12の最適値）
    weight_catboost : float, optional
        CatBoostの重み（Noneの場合は1-weight_lightgbmを使用）
    """
    if weight_catboost is None:
        weight_catboost = 1.0 - weight_lightgbm

    logger.info(f"アンサンブル重み: LightGBM={weight_lightgbm:.6f}, CatBoost={weight_catboost:.6f}")

    # テストデータ読み込み
    test_unified_path = config.processed_data_dir / "test_unified_75_features.csv"

    if not test_unified_path.exists():
        logger.error(f"テストデータが見つかりません: {test_unified_path}")
        raise FileNotFoundError(f"Test data not found: {test_unified_path}")

    test_df = pd.read_csv(test_unified_path)
    logger.info(f"テストデータ読み込み完了: {test_df.shape}")

    # モデル予測ファイル読み込み
    lgb_pred_path = config.models_dir / "test_predictions_lightgbm_unified_67feat.csv"
    cat_pred_path = config.models_dir / "test_predictions_catboost_unified_67feat.csv"

    if not lgb_pred_path.exists():
        logger.error(f"LightGBM予測ファイルが見つかりません: {lgb_pred_path}")
        raise FileNotFoundError(f"LightGBM predictions not found: {lgb_pred_path}")

    if not cat_pred_path.exists():
        logger.error(f"CatBoost予測ファイルが見つかりません: {cat_pred_path}")
        raise FileNotFoundError(f"CatBoost predictions not found: {cat_pred_path}")

    # 予測値読み込み
    lgb_pred = pd.read_csv(lgb_pred_path)
    cat_pred = pd.read_csv(cat_pred_path)

    logger.info(f"LightGBM予測: {lgb_pred.shape}, CatBoost予測: {cat_pred.shape}")

    # アンサンブル予測計算
    ensemble_pred = (
        weight_lightgbm * lgb_pred["BeatsPerMinute"].values +
        weight_catboost * cat_pred["BeatsPerMinute"].values
    )

    # Submission DataFrame作成
    submission = pd.DataFrame({
        "id": test_df["id"],
        "BeatsPerMinute": ensemble_pred
    })

    # 統計情報表示
    logger.info(f"予測統計:")
    logger.info(f"  Mean: {ensemble_pred.mean():.4f}")
    logger.info(f"  Std:  {ensemble_pred.std():.4f}")
    logger.info(f"  Min:  {ensemble_pred.min():.4f}")
    logger.info(f"  Max:  {ensemble_pred.max():.4f}")

    # 保存
    output_path = config.processed_data_dir / "submission_ticket031_precision.csv"
    submission.to_csv(output_path, index=False)

    logger.success(f"Submission保存完了: {output_path}")
    logger.info(f"サンプル数: {len(submission)}")

    return submission


def main():
    """メイン実行関数"""
    logger.info("=" * 80)
    logger.info("TICKET-031精密最適化Submission作成")
    logger.info("=" * 80)

    # Trial 12の最適重み使用
    submission = create_submission_with_weights(
        weight_lightgbm=0.7060582514745803
    )

    logger.success("Submission作成完了")
    logger.info(f"次のステップ: Kaggle提出")
    logger.info(f"  kaggle competitions submit -c playground-series-s5e9 \\")
    logger.info(f"    -f data/processed/submission_ticket031_precision.csv \\")
    logger.info(f"    -m \"TICKET-031: Precision Optuna (LGB 70.6%, CAT 29.4%, CV 26.458455)\"")

    return submission


if __name__ == "__main__":
    main()