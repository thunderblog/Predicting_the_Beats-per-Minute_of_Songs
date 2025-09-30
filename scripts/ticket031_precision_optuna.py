"""
TICKET-031: 精密Optuna最適化システム

TICKET-030の境界値変換済み75特徴量データを基盤として、
500-1000トライアルの精密ハイパーパラメータ最適化を実行し、
現在の最高性能26.38603を超越する。

実装戦略:
1. LightGBM + CatBoost二元アンサンブルの精密最適化
2. 各モデルの詳細パラメータ空間探索
3. アンサンブル重みの微細調整
4. 最適化結果の詳細分析
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import warnings
import optuna
import lightgbm as lgb
import catboost as cat
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pickle
import json
from datetime import datetime
import argparse

warnings.filterwarnings('ignore')

from scripts.my_config import config
from src.modeling.cross_validation import BPMStratifiedKFoldStrategy

class PrecisionOptunaOptimizer:
    """精密Optuna最適化システム"""

    def __init__(self, n_trials: int = 500, cv_folds: int = 5):
        """
        Args:
            n_trials: 最適化トライアル数
            cv_folds: クロスバリデーションフォールド数
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = {}
        self.best_score = float('inf')
        self.study_results = {}

    def load_boundary_transformed_data(self):
        """境界値変換済み75特徴量データの読み込み"""
        logger.info("境界値変換済みデータ読み込み...")

        train_path = config.processed_data_dir / "train_boundary_transformed_76_features.csv"
        test_path = config.processed_data_dir / "test_boundary_transformed_76_features.csv"

        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        # 特徴量とターゲット分離
        feature_cols = [col for col in self.train_df.columns
                       if col not in ["id", "BeatsPerMinute"]]

        self.X_train = self.train_df[feature_cols]
        self.y_train = self.train_df["BeatsPerMinute"]
        self.X_test = self.test_df[feature_cols]

        logger.info(f"訓練データ: {self.X_train.shape}")
        logger.info(f"テストデータ: {self.X_test.shape}")
        logger.info(f"特徴量数: {len(feature_cols)}")

        return feature_cols

    def optimize_lightgbm(self, trial):
        """LightGBM精密最適化"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,

            # 精密探索パラメータ
            'num_leaves': trial.suggest_int('lgb_num_leaves', 10, 100),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1, log=True),
            'n_estimators': trial.suggest_int('lgb_n_estimators', 500, 3000),

            # 正則化パラメータ
            'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0.1, 5.0),
            'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0.1, 5.0),

            # サンプリングパラメータ
            'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 10),

            # 詳細パラメータ
            'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
            'min_child_weight': trial.suggest_float('lgb_min_child_weight', 0.001, 0.1, log=True),
            'subsample_for_bin': trial.suggest_int('lgb_subsample_for_bin', 50000, 300000),
            'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.5, 1.0)
        }

        # BPM Stratified Cross Validation
        cv_strategy = BPMStratifiedKFoldStrategy(n_splits=self.cv_folds, random_state=42)

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(self.X_train, self.y_train)):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )

            y_pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    def optimize_catboost(self, trial):
        """CatBoost精密最適化"""
        params = {
            'loss_function': 'RMSE',
            'random_state': 42,
            'verbose': False,
            'allow_writing_files': False,

            # 精密探索パラメータ
            'iterations': trial.suggest_int('cat_iterations', 500, 3000),
            'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('cat_depth', 4, 10),

            # 正則化パラメータ
            'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1.0, 20.0),
            'border_count': trial.suggest_int('cat_border_count', 32, 255),

            # サンプリングパラメータ
            'rsm': trial.suggest_float('cat_rsm', 0.5, 1.0),
            'subsample': trial.suggest_float('cat_subsample', 0.5, 1.0),

            # 詳細パラメータ
            'min_data_in_leaf': trial.suggest_int('cat_min_data_in_leaf', 1, 50),
            'one_hot_max_size': trial.suggest_int('cat_one_hot_max_size', 2, 10)
        }

        # BPM Stratified Cross Validation
        cv_strategy = BPMStratifiedKFoldStrategy(n_splits=self.cv_folds, random_state=42)

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(self.X_train, self.y_train)):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            model = cat.CatBoostRegressor(**params)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=(X_fold_val, y_fold_val),
                early_stopping_rounds=100
            )

            y_pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    def optimize_ensemble_weights(self, lgb_params, cat_params, trial):
        """アンサンブル重み精密最適化"""
        weight_lgb = trial.suggest_float('weight_lightgbm', 0.3, 0.8)
        weight_cat = 1.0 - weight_lgb

        # BPM Stratified Cross Validation
        cv_strategy = BPMStratifiedKFoldStrategy(n_splits=self.cv_folds, random_state=42)

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(self.X_train, self.y_train)):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # LightGBM
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            lgb_pred = lgb_model.predict(X_fold_val)

            # CatBoost
            cat_model = cat.CatBoostRegressor(**cat_params)
            cat_model.fit(
                X_fold_train, y_fold_train,
                eval_set=(X_fold_val, y_fold_val),
                early_stopping_rounds=100
            )
            cat_pred = cat_model.predict(X_fold_val)

            # アンサンブル予測
            ensemble_pred = weight_lgb * lgb_pred + weight_cat * cat_pred
            rmse = np.sqrt(mean_squared_error(y_fold_val, ensemble_pred))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    def run_precision_optimization(self):
        """精密最適化の実行"""
        logger.info(f"TICKET-031精密Optuna最適化開始（{self.n_trials}トライアル）...")

        # データ読み込み
        feature_cols = self.load_boundary_transformed_data()

        # Phase 1: LightGBM単体最適化
        logger.info("=== Phase 1: LightGBM精密最適化 ===")
        lgb_study = optuna.create_study(direction='minimize', study_name='lgb_precision')
        lgb_study.optimize(self.optimize_lightgbm, n_trials=self.n_trials//3)

        best_lgb_params = lgb_study.best_params
        best_lgb_score = lgb_study.best_value
        logger.success(f"LightGBM最適化完了: {best_lgb_score:.6f}")

        # Phase 2: CatBoost単体最適化
        logger.info("=== Phase 2: CatBoost精密最適化 ===")
        cat_study = optuna.create_study(direction='minimize', study_name='cat_precision')
        cat_study.optimize(self.optimize_catboost, n_trials=self.n_trials//3)

        best_cat_params = cat_study.best_params
        best_cat_score = cat_study.best_value
        logger.success(f"CatBoost最適化完了: {best_cat_score:.6f}")

        # Phase 3: アンサンブル重み最適化
        logger.info("=== Phase 3: アンサンブル重み精密最適化 ===")

        # パラメータをモデル用に変換
        lgb_model_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42
        }
        lgb_model_params.update({k.replace('lgb_', ''): v for k, v in best_lgb_params.items()})

        cat_model_params = {
            'loss_function': 'RMSE',
            'random_state': 42,
            'verbose': False,
            'allow_writing_files': False
        }
        cat_model_params.update({k.replace('cat_', ''): v for k, v in best_cat_params.items()})

        def ensemble_objective(trial):
            return self.optimize_ensemble_weights(lgb_model_params, cat_model_params, trial)

        ensemble_study = optuna.create_study(direction='minimize', study_name='ensemble_precision')
        ensemble_study.optimize(ensemble_objective, n_trials=self.n_trials//3)

        best_ensemble_score = ensemble_study.best_value
        best_weights = ensemble_study.best_params

        logger.success(f"アンサンブル最適化完了: {best_ensemble_score:.6f}")

        # 結果保存
        self.best_params = {
            'lightgbm': lgb_model_params,
            'catboost': cat_model_params,
            'ensemble_weights': best_weights,
            'cv_scores': {
                'lightgbm_single': best_lgb_score,
                'catboost_single': best_cat_score,
                'ensemble': best_ensemble_score
            }
        }
        self.best_score = best_ensemble_score

        # 結果ファイル保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = config.experiments_dir / f"ticket031_precision_optuna_{timestamp}.json"

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.best_params, f, indent=2, ensure_ascii=False)

        logger.success(f"最適化結果保存: {results_path}")

        return self.best_params, self.best_score

def parse_args():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(
        description="TICKET-031: 精密Optuna最適化システム",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=50,
        help="最適化トライアル数（推奨: 軽量=50, 標準=200, 精密=500）"
    )

    parser.add_argument(
        "--cv-folds", "-f",
        type=int,
        default=5,
        help="クロスバリデーションフォールド数"
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["light", "standard", "precision"],
        default="light",
        help="最適化モード（light=50trials, standard=200trials, precision=500trials）"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="結果保存ディレクトリ（デフォルト: experiments/）"
    )

    return parser.parse_args()

def main():
    """メイン処理"""
    args = parse_args()

    # モードによるトライアル数の自動設定
    mode_trials = {
        "light": 50,
        "standard": 200,
        "precision": 500
    }

    # trialが明示的に指定されていない場合はモードから設定
    if args.trials == 50 and args.mode != "light":
        trials = mode_trials[args.mode]
    else:
        trials = args.trials

    logger.info("TICKET-031: 精密Optuna最適化システム開始...")
    logger.info(f"最適化設定:")
    logger.info(f"  - モード: {args.mode}")
    logger.info(f"  - トライアル数: {trials}")
    logger.info(f"  - CVフォールド数: {args.cv_folds}")

    # 最適化実行
    optimizer = PrecisionOptunaOptimizer(n_trials=trials, cv_folds=args.cv_folds)
    best_params, best_score = optimizer.run_precision_optimization()

    logger.success(f"TICKET-031完了")
    logger.info(f"最適CV性能: {best_score:.6f}")
    logger.info(f"LightGBM重み: {best_params['ensemble_weights']['weight_lightgbm']:.3f}")
    logger.info(f"CatBoost重み: {1.0 - best_params['ensemble_weights']['weight_lightgbm']:.3f}")

    # TODO(human): 最適化完了後のKaggle提出ファイル生成機能を追加
    logger.info("次のステップ: 最適化済みパラメータでのKaggle提出実行")

    return best_params, best_score

if __name__ == "__main__":
    best_params, best_score = main()