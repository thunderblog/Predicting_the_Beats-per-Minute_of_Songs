"""
TICKET-031結果を使ったKaggle提出ファイル生成

TICKET-031の最適化結果（JSONファイル）または手動パラメータを使用して、
境界値変換済み75特徴量データでアンサンブル予測を実行し、
Kaggle提出用submission.csvを生成する。

使用方法:
1. 最適化結果ファイルから自動生成:
   python scripts/create_submission_from_optuna.py --optuna-result experiments/ticket031_*.json

2. 手動パラメータで生成:
   python scripts/create_submission_from_optuna.py --manual-params
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import warnings
import lightgbm as lgb
import catboost as cat
import json
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')

from scripts.my_config import config
from src.modeling.cross_validation import BPMStratifiedKFoldStrategy

def load_optuna_results(results_path: str) -> dict:
    """Optuna最適化結果を読み込み"""
    logger.info(f"最適化結果読み込み: {results_path}")

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    logger.info("最適化結果:")
    logger.info(f"  LightGBM重み: {results['ensemble_weights']['weight_lightgbm']:.3f}")
    logger.info(f"  CatBoost重み: {1.0 - results['ensemble_weights']['weight_lightgbm']:.3f}")
    logger.info(f"  CV性能: {results['cv_scores']['ensemble']:.6f}")

    return results

def get_manual_params() -> dict:
    """手動パラメータを取得（TICKET-030の最適パラメータベース）"""
    logger.info("手動パラメータを使用...")

    return {
        'lightgbm': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            'num_leaves': 20,
            'learning_rate': 0.03,
            'n_estimators': 2000,
            'reg_alpha': 2.0,
            'reg_lambda': 2.0,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7
        },
        'catboost': {
            'loss_function': 'RMSE',
            'random_state': 42,
            'verbose': False,
            'allow_writing_files': False,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 5,
            'iterations': 2000
        },
        'ensemble_weights': {
            'weight_lightgbm': 0.601,
            'weight_catboost': 0.399
        }
    }

def create_submission_with_params(params: dict, experiment_name: str = "ticket031_submission"):
    """最適パラメータを使ってsubmission作成"""
    logger.info("境界値変換済みデータ読み込み...")

    # データ読み込み
    train_path = config.processed_data_dir / "train_boundary_transformed_76_features.csv"
    test_path = config.processed_data_dir / "test_boundary_transformed_76_features.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 特徴量とターゲット分離
    feature_cols = [col for col in train_df.columns if col not in ["id", "BeatsPerMinute"]]
    X_train = train_df[feature_cols]
    y_train = train_df["BeatsPerMinute"]
    X_test = test_df[feature_cols]

    logger.info(f"データ形状: 訓練{X_train.shape}, テスト{X_test.shape}")
    logger.info(f"特徴量数: {len(feature_cols)}")

    # アンサンブル重み
    weight_lgb = params['ensemble_weights']['weight_lightgbm']
    weight_cat = 1.0 - weight_lgb

    logger.info(f"アンサンブル重み: LGB {weight_lgb:.3f}, CAT {weight_cat:.3f}")

    # LightGBM訓練
    logger.info("LightGBM訓練開始...")
    lgb_model = lgb.LGBMRegressor(**params['lightgbm'])
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    logger.info("LightGBM訓練完了")

    # CatBoost訓練
    logger.info("CatBoost訓練開始...")
    cat_model = cat.CatBoostRegressor(**params['catboost'])
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_test)
    logger.info("CatBoost訓練完了")

    # アンサンブル予測
    ensemble_pred = weight_lgb * lgb_pred + weight_cat * cat_pred

    logger.info(f"予測統計:")
    logger.info(f"  LightGBM: {lgb_pred.mean():.2f} ± {lgb_pred.std():.2f}")
    logger.info(f"  CatBoost: {cat_pred.mean():.2f} ± {cat_pred.std():.2f}")
    logger.info(f"  Ensemble: {ensemble_pred.mean():.2f} ± {ensemble_pred.std():.2f}")

    # Submission作成
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'BeatsPerMinute': ensemble_pred
    })

    # ファイル保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = config.processed_data_dir / f"submission_{experiment_name}_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    logger.success(f"Submission作成完了: {submission_path}")
    logger.info(f"提出コマンド:")
    logger.info(f'kaggle competitions submit -c playground-series-s5e9 -f "{submission_path}" -m "{experiment_name} optimized ensemble"')

    return submission_path, ensemble_pred

def parse_args():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(
        description="TICKET-031最適化結果からKaggle提出ファイル生成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--optuna-result",
        type=str,
        help="Optuna最適化結果JSONファイルパス"
    )
    group.add_argument(
        "--manual-params",
        action='store_true',
        help="手動パラメータを使用（TICKET-030ベース）"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ticket031_optimized",
        help="実験名（ファイル名に使用）"
    )

    return parser.parse_args()

def main():
    """メイン処理"""
    args = parse_args()

    logger.info("TICKET-031最適化結果からSubmission生成開始...")

    # パラメータ取得
    if args.optuna_result:
        if not Path(args.optuna_result).exists():
            logger.error(f"最適化結果ファイルが見つかりません: {args.optuna_result}")
            return
        params = load_optuna_results(args.optuna_result)
    else:
        params = get_manual_params()

    # Submission作成
    submission_path, predictions = create_submission_with_params(params, args.experiment_name)

    logger.success("処理完了")
    logger.info("次のステップ: Kaggle提出コマンドを実行してください")

    return submission_path

if __name__ == "__main__":
    submission_path = main()