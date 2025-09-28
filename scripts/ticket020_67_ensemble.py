"""
TICKET-020 67特徴量版二元アンサンブルシステム
LightGBM + CatBoost + BPM Stratified戦略による最高性能追求
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import warnings
import json
from datetime import datetime
import pickle
from typing import Dict, List, Tuple, Optional

import lightgbm as lgb
import catboost as cb
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings('ignore')

from scripts.my_config import config

def create_bpm_bins(y: pd.Series, n_bins: int = 10) -> np.ndarray:
    """BPM値を層化分割用のビンに分割"""
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    logger.info(f"BPM層化分割ビン作成: {len(np.unique(bins))}ビン, 範囲: {y.min():.2f}-{y.max():.2f}")
    return bins

class BinaryEnsembleRegressor:
    """67特徴量版二元アンサンブル回帰器（LightGBM + CatBoost）"""

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = {
            'lightgbm': [],
            'catboost': []
        }
        self.optimal_weights = None
        self.cv_scores = {}

    def train_fold_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, np.ndarray]:
        """BPM Stratified KFoldで二元モデル訓練を実行"""
        logger.info(f"{self.n_folds}フォールドで二元モデル訓練を開始（BPM Stratified戦略）...")

        # BPM Stratified CV戦略
        stratify_labels = create_bpm_bins(y)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Out-of-fold予測格納用
        oof_predictions = {
            'lightgbm': np.zeros(len(X)),
            'catboost': np.zeros(len(X))
        }

        fold_scores = {model_type: [] for model_type in ['lightgbm', 'catboost']}

        for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, stratify_labels), total=self.n_folds)):
            logger.info(f"フォールド {fold + 1}/{self.n_folds} 訓練中...")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # LightGBM訓練（67特徴量最適化パラメータ）
            lgb_model = self._train_lightgbm_fold(X_train, y_train, X_val, y_val)
            lgb_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
            oof_predictions['lightgbm'][val_idx] = lgb_pred
            fold_scores['lightgbm'].append(np.sqrt(mean_squared_error(y_val, lgb_pred)))
            self.models['lightgbm'].append(lgb_model)

            # CatBoost訓練
            cat_model = self._train_catboost_fold(X_train, y_train, X_val, y_val)
            cat_pred = cat_model.predict(X_val)
            oof_predictions['catboost'][val_idx] = cat_pred
            fold_scores['catboost'].append(np.sqrt(mean_squared_error(y_val, cat_pred)))
            self.models['catboost'].append(cat_model)

            logger.info(f"フォールド {fold + 1} - LGB: {fold_scores['lightgbm'][-1]:.6f}, "
                       f"CAT: {fold_scores['catboost'][-1]:.6f}")

        # CV性能サマリー
        for model_type in ['lightgbm', 'catboost']:
            mean_score = np.mean(fold_scores[model_type])
            std_score = np.std(fold_scores[model_type])
            self.cv_scores[model_type] = {
                'scores': fold_scores[model_type],
                'mean': mean_score,
                'std': std_score
            }
            logger.info(f"{model_type.upper()} CV: {mean_score:.6f} ± {std_score:.6f}")

        return oof_predictions

    def optimize_ensemble_weights(self, oof_predictions: Dict[str, np.ndarray], y_true: pd.Series,
                                n_trials: int = 500) -> Dict[str, float]:
        """Optunaを使用してアンサンブル重みを最適化（二元制約）"""
        logger.info(f"Optuna二元重み最適化開始（{n_trials}トライアル）...")

        def objective(trial):
            # 二元制約: w_lgb + w_cat = 1
            w_lgb = trial.suggest_float('weight_lightgbm', 0.0, 1.0)
            w_cat = 1.0 - w_lgb

            # アンサンブル予測
            ensemble_pred = (w_lgb * oof_predictions['lightgbm'] +
                           w_cat * oof_predictions['catboost'])

            # RMSE計算
            rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
            return rmse

        # Optuna Study実行
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials)

        # 最適重み設定
        best_params = study.best_params
        w_lgb = best_params['weight_lightgbm']
        w_cat = 1.0 - w_lgb
        self.optimal_weights = {
            'weight_lightgbm': w_lgb,
            'weight_catboost': w_cat
        }

        # 最適アンサンブルCV性能
        ensemble_pred = (self.optimal_weights['weight_lightgbm'] * oof_predictions['lightgbm'] +
                        self.optimal_weights['weight_catboost'] * oof_predictions['catboost'])
        ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))

        logger.success(f"最適重み: LGB={self.optimal_weights['weight_lightgbm']:.3f}, "
                      f"CAT={self.optimal_weights['weight_catboost']:.3f}")
        logger.success(f"アンサンブルCV RMSE: {ensemble_rmse:.6f}")

        # 個別モデルとの比較
        lgb_rmse = np.sqrt(mean_squared_error(y_true, oof_predictions['lightgbm']))
        cat_rmse = np.sqrt(mean_squared_error(y_true, oof_predictions['catboost']))

        logger.info(f"性能比較:")
        logger.info(f"- LightGBM単体: {lgb_rmse:.6f}")
        logger.info(f"- CatBoost単体: {cat_rmse:.6f}")
        logger.info(f"- アンサンブル:  {ensemble_rmse:.6f}")
        logger.info(f"- 最良単体からの改善: {min(lgb_rmse, cat_rmse) - ensemble_rmse:+.6f}")

        return self.optimal_weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """テストデータでアンサンブル予測"""
        if self.optimal_weights is None:
            raise ValueError("重み最適化を先に実行してください")

        logger.info("アンサンブル予測実行中...")

        # 各モデルタイプでフォールド平均予測
        lgb_preds = np.mean([model.predict(X, num_iteration=model.best_iteration)
                            for model in self.models['lightgbm']], axis=0)
        cat_preds = np.mean([model.predict(X)
                            for model in self.models['catboost']], axis=0)

        # 重み付きアンサンブル
        ensemble_pred = (self.optimal_weights['weight_lightgbm'] * lgb_preds +
                        self.optimal_weights['weight_catboost'] * cat_preds)

        logger.success("アンサンブル予測完了")
        return ensemble_pred

    def _train_lightgbm_fold(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series):
        """LightGBMの単一フォールドを訓練（67特徴量最適化設定）"""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 67特徴量版最適化パラメータ
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 20,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'reg_alpha': 2.0,
            'reg_lambda': 2.0,
            'min_child_samples': 20,
            'random_state': self.random_state,
            'verbosity': -1
        }

        model = lgb.train(
            params, train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(0)
            ]
        )
        return model

    def _train_catboost_fold(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series):
        """CatBoostの単一フォールドを訓練"""
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "rsm": 0.7,
            "random_seed": self.random_state,
            "verbose": 0,
            "early_stopping_rounds": 200,
            "iterations": 2000,
        }

        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=0,
            plot=False
        )
        return model

def run_67_ensemble_experiment():
    """67特徴量版二元アンサンブル実験実行"""
    logger.info("TICKET-020 67特徴量版二元アンサンブル実験開始...")

    # データ読み込み
    train_data_path = config.processed_data_dir / "train_unified_75_features.csv"
    test_data_path = config.processed_data_dir / "test_unified_75_features.csv"

    logger.info(f"訓練データ読み込み: {train_data_path}")
    train_df = pd.read_csv(train_data_path)

    logger.info(f"テストデータ読み込み: {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    # 特徴量準備
    train_feature_cols = [col for col in train_df.columns if col not in ["id", "BeatsPerMinute"]]
    test_feature_cols = [col for col in test_df.columns if col != "id"]
    common_features = sorted(list(set(train_feature_cols) & set(test_feature_cols)))

    X_train = train_df[common_features]
    y_train = train_df["BeatsPerMinute"]
    X_test = test_df[common_features]

    logger.info(f"特徴量数: {len(common_features)}, 訓練サンプル: {len(X_train)}, テストサンプル: {len(X_test)}")

    # 二元アンサンブル訓練
    ensemble = BinaryEnsembleRegressor(n_folds=5, random_state=config.random_state)
    oof_predictions = ensemble.train_fold_models(X_train, y_train)

    # 重み最適化
    optimal_weights = ensemble.optimize_ensemble_weights(oof_predictions, y_train, n_trials=500)

    # テストデータで予測
    ensemble_predictions = ensemble.predict(X_test)

    # ベースライン比較
    baseline_67_cv = 26.463984
    best_single_cv = min(ensemble.cv_scores['lightgbm']['mean'], ensemble.cv_scores['catboost']['mean'])

    # アンサンブルCV性能計算
    ensemble_pred = (optimal_weights['weight_lightgbm'] * oof_predictions['lightgbm'] +
                    optimal_weights['weight_catboost'] * oof_predictions['catboost'])
    ensemble_cv = np.sqrt(mean_squared_error(y_train, ensemble_pred))

    improvement_from_baseline = baseline_67_cv - ensemble_cv
    improvement_from_best_single = best_single_cv - ensemble_cv

    logger.info(f"\n=== 性能サマリー ===")
    logger.info(f"67特徴量ベースライン: {baseline_67_cv:.6f}")
    logger.info(f"最良単体モデル:       {best_single_cv:.6f}")
    logger.info(f"二元アンサンブル:     {ensemble_cv:.6f}")
    logger.info(f"ベースラインからの改善: {improvement_from_baseline:+.6f}")
    logger.info(f"最良単体からの改善:     {improvement_from_best_single:+.6f}")

    # 提出ファイル作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'BeatsPerMinute': ensemble_predictions
    })

    submission_path = config.processed_data_dir / f"submission_ticket020_67_ensemble_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    logger.success(f"提出ファイル作成完了: {submission_path}")
    logger.info(f"予測統計:")
    logger.info(f"- 最小値: {ensemble_predictions.min():.4f}")
    logger.info(f"- 最大値: {ensemble_predictions.max():.4f}")
    logger.info(f"- 平均値: {ensemble_predictions.mean():.4f}")
    logger.info(f"- 標準偏差: {ensemble_predictions.std():.4f}")

    # 実験記録
    experiment_record = {
        "experiment_name": "exp17_ticket020_67_binary_ensemble",
        "timestamp": timestamp,
        "cv_strategy": "bmp_stratified",
        "model_type": "binary_ensemble_lgb_cat",
        "n_features": len(common_features),
        "optimal_weights": optimal_weights,
        "cv_results": {
            "ensemble_cv_rmse": ensemble_cv,
            "lightgbm_cv": ensemble.cv_scores['lightgbm'],
            "catboost_cv": ensemble.cv_scores['catboost'],
            "improvement_from_baseline": improvement_from_baseline,
            "improvement_from_best_single": improvement_from_best_single
        },
        "prediction_stats": {
            "min_prediction": ensemble_predictions.min(),
            "max_prediction": ensemble_predictions.max(),
            "mean_prediction": ensemble_predictions.mean(),
            "std_prediction": ensemble_predictions.std()
        },
        "submission_file": str(submission_path),
        "target_performance": "最高LB性能26.38534超越を目指す"
    }

    record_path = config.processed_data_dir / f"experiment_record_ticket020_{timestamp}.json"
    with open(record_path, 'w') as f:
        json.dump(experiment_record, f, indent=2, default=str)

    logger.info(f"実験記録保存: {record_path}")

    return submission_path, ensemble_cv, improvement_from_baseline, optimal_weights

def submit_to_kaggle(submission_path: Path, cv_score: float, improvement: float, weights: Dict[str, float]):
    """Kaggle提出（手動確認用）"""
    logger.info("Kaggle提出準備...")

    message = f"TICKET-020 67-Feature Binary Ensemble (LGB+CAT) - CV: {cv_score:.6f} (improvement: {improvement:+.6f}) Weights: LGB={weights['weight_lightgbm']:.3f}"

    logger.info("=== 提出用情報 ===")
    logger.info(f"ファイル: {submission_path}")
    logger.info(f"メッセージ: {message}")
    logger.info("手動でKaggle提出を実行してください")

if __name__ == "__main__":
    logger.info("TICKET-020 67特徴量版二元アンサンブル実験開始")

    submission_path, cv_score, improvement, optimal_weights = run_67_ensemble_experiment()

    logger.success("実験完了！")
    logger.info(f"アンサンブルCV RMSE: {cv_score:.6f}")
    logger.info(f"67特徴量ベースラインからの改善: {improvement:+.6f}")
    logger.info(f"最適重み: {optimal_weights}")

    submit_to_kaggle(submission_path, cv_score, improvement, optimal_weights)

    logger.info("次のステップ:")
    logger.info("1. 手動Kaggle提出後のLB確認")
    logger.info("2. 最高LB性能26.38534との比較")
    logger.info("3. さらなる改善戦略の検討")