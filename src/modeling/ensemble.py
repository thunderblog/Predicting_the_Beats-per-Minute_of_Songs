from datetime import datetime
import json
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from loguru import logger
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.my_config import config
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
import typer

# パス設定をconfigから取得
MODELS_DIR = config.models_dir
PROCESSED_DATA_DIR = config.processed_data_dir

app = typer.Typer()


class EnsembleRegressor:
    """3モデル（LightGBM, XGBoost, CatBoost）アンサンブル回帰器"""

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = {
            'lightgbm': [],
            'xgboost': [],
            'catboost': []
        }
        self.optimal_weights = None
        self.cv_scores = {}

    def train_fold_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List]:
        """全モデルタイプでクロスバリデーション訓練を実行"""
        logger.info(f"{self.n_folds}フォールドで3モデル訓練を開始...")

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Out-of-fold予測格納用
        oof_predictions = {
            'lightgbm': np.zeros(len(X)),
            'xgboost': np.zeros(len(X)),
            'catboost': np.zeros(len(X))
        }

        fold_scores = {model_type: [] for model_type in ['lightgbm', 'xgboost', 'catboost']}

        for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(X), total=self.n_folds)):
            logger.info(f"フォールド {fold + 1}/{self.n_folds} 訓練中...")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # LightGBM訓練
            lgb_model = self._train_lightgbm_fold(X_train, y_train, X_val, y_val)
            lgb_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
            oof_predictions['lightgbm'][val_idx] = lgb_pred
            fold_scores['lightgbm'].append(np.sqrt(mean_squared_error(y_val, lgb_pred)))
            self.models['lightgbm'].append(lgb_model)

            # XGBoost訓練
            xgb_model = self._train_xgboost_fold(X_train, y_train, X_val, y_val)
            xgb_pred = xgb_model.predict(xgb.DMatrix(X_val))
            oof_predictions['xgboost'][val_idx] = xgb_pred
            fold_scores['xgboost'].append(np.sqrt(mean_squared_error(y_val, xgb_pred)))
            self.models['xgboost'].append(xgb_model)

            # CatBoost訓練
            cat_model = self._train_catboost_fold(X_train, y_train, X_val, y_val)
            cat_pred = cat_model.predict(X_val)
            oof_predictions['catboost'][val_idx] = cat_pred
            fold_scores['catboost'].append(np.sqrt(mean_squared_error(y_val, cat_pred)))
            self.models['catboost'].append(cat_model)

            logger.info(f"フォールド {fold + 1} - LGB: {fold_scores['lightgbm'][-1]:.4f}, "
                       f"XGB: {fold_scores['xgboost'][-1]:.4f}, CAT: {fold_scores['catboost'][-1]:.4f}")

        # CV性能サマリー
        for model_type in ['lightgbm', 'xgboost', 'catboost']:
            mean_score = np.mean(fold_scores[model_type])
            std_score = np.std(fold_scores[model_type])
            self.cv_scores[model_type] = {
                'scores': fold_scores[model_type],
                'mean': mean_score,
                'std': std_score
            }
            logger.info(f"{model_type.upper()} CV: {mean_score:.4f} ± {std_score:.4f}")

        return oof_predictions

    def optimize_ensemble_weights(self, oof_predictions: Dict[str, np.ndarray], y_true: pd.Series,
                                n_trials: int = 500) -> Dict[str, float]:
        """Optunaを使用してアンサンブル重みを最適化"""
        logger.info(f"Optuna重み最適化開始（{n_trials}トライアル）...")

        def objective(trial):
            # 重み提案（合計=1制約）
            w_lgb = trial.suggest_float('weight_lightgbm', 0.0, 1.0)
            w_xgb = trial.suggest_float('weight_xgboost', 0.0, 1.0)
            w_cat = trial.suggest_float('weight_catboost', 0.0, 1.0)

            # 正規化
            total_weight = w_lgb + w_xgb + w_cat
            if total_weight == 0:
                return float('inf')

            w_lgb /= total_weight
            w_xgb /= total_weight
            w_cat /= total_weight

            # アンサンブル予測
            ensemble_pred = (w_lgb * oof_predictions['lightgbm'] +
                           w_xgb * oof_predictions['xgboost'] +
                           w_cat * oof_predictions['catboost'])

            # RMSE計算
            rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
            return rmse

        # Optuna Study実行
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials)

        # 最適重みを正規化
        best_params = study.best_params
        total_weight = sum(best_params.values())
        self.optimal_weights = {k: v/total_weight for k, v in best_params.items()}

        # 最適アンサンブルCV性能
        ensemble_pred = (self.optimal_weights['weight_lightgbm'] * oof_predictions['lightgbm'] +
                        self.optimal_weights['weight_xgboost'] * oof_predictions['xgboost'] +
                        self.optimal_weights['weight_catboost'] * oof_predictions['catboost'])
        ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))

        logger.success(f"最適重み: LGB={self.optimal_weights['weight_lightgbm']:.3f}, "
                      f"XGB={self.optimal_weights['weight_xgboost']:.3f}, "
                      f"CAT={self.optimal_weights['weight_catboost']:.3f}")
        logger.success(f"アンサンブルCV RMSE: {ensemble_rmse:.4f}")

        return self.optimal_weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """テストデータでアンサンブル予測"""
        if self.optimal_weights is None:
            raise ValueError("重み最適化を先に実行してください")

        logger.info("アンサンブル予測実行中...")

        # 各モデルタイプでフォールド平均予測
        lgb_preds = np.mean([model.predict(X, num_iteration=model.best_iteration)
                            for model in self.models['lightgbm']], axis=0)
        xgb_preds = np.mean([model.predict(xgb.DMatrix(X))
                            for model in self.models['xgboost']], axis=0)
        cat_preds = np.mean([model.predict(X)
                            for model in self.models['catboost']], axis=0)

        # 重み付きアンサンブル
        ensemble_pred = (self.optimal_weights['weight_lightgbm'] * lgb_preds +
                        self.optimal_weights['weight_xgboost'] * xgb_preds +
                        self.optimal_weights['weight_catboost'] * cat_preds)

        logger.success("アンサンブル予測完了")
        return ensemble_pred

    def _train_lightgbm_fold(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series):
        """LightGBMの単一フォールドを訓練"""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": config.objective,
            "metric": config.metric,
            "boosting_type": "gbdt",
            "num_leaves": config.num_leaves,
            "learning_rate": config.learning_rate,
            "feature_fraction": config.feature_fraction,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": config.random_state,
        }

        model = lgb.train(
            params, train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "eval"],
            num_boost_round=config.n_estimators,
            callbacks=[
                lgb.early_stopping(config.stopping_rounds),
                lgb.log_evaluation(0),  # ログ出力を抑制
            ],
        )
        return model

    def _train_xgboost_fold(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series):
        """XGBoostの単一フォールドを訓練"""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 6,
            "learning_rate": config.learning_rate,
            "subsample": 0.8,
            "colsample_bytree": config.feature_fraction,
            "random_state": config.random_state,
            "verbosity": 0,
        }

        model = xgb.train(
            params, dtrain,
            num_boost_round=config.n_estimators,
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=config.stopping_rounds,
            verbose_eval=0,  # ログ出力を抑制
        )
        return model

    def _train_catboost_fold(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series):
        """CatBoostの単一フォールドを訓練"""
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "depth": 6,
            "learning_rate": config.learning_rate,
            "subsample": 0.8,
            "rsm": config.feature_fraction,
            "random_seed": config.random_state,
            "verbose": 0,  # ログ出力を抑制
            "early_stopping_rounds": config.stopping_rounds,
            "iterations": config.n_estimators,
        }

        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=0,  # ログ出力を抑制
            plot=False
        )
        return model

    def save_ensemble(self, model_dir: Path, exp_name: str) -> Dict:
        """アンサンブルモデルと結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 結果サマリー
        results = {
            "experiment_name": exp_name,
            "model_type": "ensemble_lgb_xgb_cat",
            "timestamp": timestamp,
            "n_folds": self.n_folds,
            "optimal_weights": self.optimal_weights,
            "cv_scores": self.cv_scores,
            "ensemble_cv_rmse": None,  # 後で計算
            "feature_count": None,  # 後で設定
        }

        # アンサンブルモデル保存
        ensemble_path = model_dir / f"{exp_name}_ensemble_{timestamp}.pkl"
        with open(ensemble_path, "wb") as f:
            pickle.dump({
                'models': self.models,
                'optimal_weights': self.optimal_weights,
                'cv_scores': self.cv_scores
            }, f)

        # 結果JSON保存
        results_path = model_dir / f"{exp_name}_ensemble_results_{timestamp}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.success(f"アンサンブル保存完了: {ensemble_path}")
        logger.success(f"結果保存完了: {results_path}")

        return results


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train_ticket017_combined.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    model_dir: Path = MODELS_DIR,
    exp_name: str = "ticket018_ensemble",
    n_folds: int = 5,
    n_trials: int = 500,
):
    """アンサンブル回帰システムを実行する"""
    logger.info(f"TICKET-018アンサンブルシステム開始 (実験名: {exp_name})...")

    # ディレクトリ作成
    model_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 訓練データ読み込み
    logger.info(f"訓練データ読み込み: {train_path}")
    train_df = pd.read_csv(train_path)

    # 特徴量とターゲット分離
    feature_cols = [col for col in train_df.columns if col not in ["id", config.target]]
    X_train = train_df[feature_cols]
    y_train = train_df[config.target]

    logger.info(f"特徴量数: {len(feature_cols)}, サンプル数: {len(X_train)}")

    # アンサンブル訓練
    ensemble = EnsembleRegressor(n_folds=n_folds)
    oof_predictions = ensemble.train_fold_models(X_train, y_train)

    # 重み最適化
    optimal_weights = ensemble.optimize_ensemble_weights(oof_predictions, y_train, n_trials=n_trials)

    # テストデータで予測
    logger.info(f"テストデータ読み込み: {test_path}")
    test_df = pd.read_csv(test_path)
    X_test = test_df[feature_cols]

    ensemble_predictions = ensemble.predict(X_test)

    # 提出ファイル作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        config.target: ensemble_predictions
    })

    submission_path = output_dir / f"submission_{exp_name}_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    logger.success(f"提出ファイル保存: {submission_path}")

    # モデル保存
    results = ensemble.save_ensemble(model_dir, exp_name)
    results["feature_count"] = len(feature_cols)

    # 最終ログ
    logger.success(f"TICKET-018アンサンブルシステム完了")
    logger.info(f"最適重み: {optimal_weights}")
    for model_type, score_info in ensemble.cv_scores.items():
        logger.info(f"{model_type.upper()}: {score_info['mean']:.4f} ± {score_info['std']:.4f}")


if __name__ == "__main__":
    app()