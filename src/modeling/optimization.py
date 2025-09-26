#!/usr/bin/env python3
"""
TICKET-013: Optuna最適化システム実装

全モデル対応の統一的最適化フレームワーク
- ベイジアン最適化によるハイパーパラメータ探索
- 早期停止とトライアル履歴管理
- CV性能向上の追跡・可視化
"""

import sys
import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
import time
import gc
from typing import Dict, Any, Optional, Tuple
import pickle
import json
from loguru import logger

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features import (
    create_comprehensive_interaction_features,
    create_log_features
)


class OptunaLightGBMOptimizer:
    """LightGBM用Optuna最適化器"""

    def __init__(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        study_name: str = "lightgbm_optimization"
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.study_name = study_name

        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.best_params = None
        self.study = None

        # CV設定
        self.kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def create_ticket_017_01_02_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """TICKET-017-01+02特徴量を作成（最高性能構成）"""
        logger.info("TICKET-017-01+02特徴量生成中...")

        # Step 1: 包括的交互作用特徴量
        logger.info("  包括的交互作用特徴量作成中...")
        df_with_interaction = create_comprehensive_interaction_features(df)

        # メモリ最適化
        gc.collect()

        # Step 2: 対数変換特徴量
        logger.info("  対数変換特徴量作成中...")
        df_complete = create_log_features(df_with_interaction)

        # メモリクリーンアップ
        del df_with_interaction
        gc.collect()

        logger.success(f"特徴量生成完了: {len(df_complete.columns)}次元")
        return df_complete

    def prepare_data(self, train_path: str, validation_path: str, n_features_select: int = 75):
        """データ準備と特徴量選択"""
        logger.info("データ準備開始")

        # データ読み込み
        train_df = pd.read_csv(train_path)
        validation_df = pd.read_csv(validation_path)

        # 訓練・検証データ結合
        full_train_df = pd.concat([train_df, validation_df], ignore_index=True)
        logger.info(f"結合後データ形状: {full_train_df.shape}")

        # メモリクリーンアップ
        del train_df, validation_df
        gc.collect()

        # 特徴量生成
        train_features = self.create_ticket_017_01_02_features(full_train_df)
        del full_train_df
        gc.collect()

        # 特徴量とターゲット分離
        feature_cols = [col for col in train_features.columns if col not in ['id', 'BeatsPerMinute']]
        X_full = train_features[feature_cols]
        y = train_features['BeatsPerMinute']

        # 特徴量選択
        logger.info(f"特徴量選択: {len(feature_cols)} -> {n_features_select}")
        selector = SelectKBest(score_func=f_regression, k=n_features_select)
        X_selected = selector.fit_transform(X_full, y)

        # 選択された特徴量名を取得
        selected_features = selector.get_support()
        selected_feature_names = [name for name, selected in zip(feature_cols, selected_features) if selected]

        # データ保存
        self.X_train = X_selected
        self.y_train = y
        self.feature_names = selected_feature_names

        logger.success(f"データ準備完了: {X_selected.shape}")

        # メモリクリーンアップ
        del train_features, X_full
        gc.collect()

        return X_selected, y, selected_feature_names

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna目的関数"""

        # ハイパーパラメータのサンプリング
        params = {
            # 基本パラメータ
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': self.random_state,
            'force_col_wise': True,

            # 最適化対象パラメータ
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 50, 300),

            # 正則化パラメータ（重要）
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),

            # サンプリングパラメータ
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),

            # その他重要パラメータ
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample_for_bin': trial.suggest_int('subsample_for_bin', 50000, 300000),
        }

        # クロスバリデーション
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(self.X_train)):
            X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # モデル訓練
            model = lgb.LGBMRegressor(**params)

            try:
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    callbacks=[lgb.log_evaluation(0)]  # ログ非表示
                )

                # 予測とスコア計算
                val_pred = model.predict(X_fold_val)
                fold_rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
                cv_scores.append(fold_rmse)

            except Exception as e:
                logger.warning(f"Fold {fold} でエラー: {e}")
                return float('inf')  # 失敗時は無限大を返す

        # CV平均スコア
        mean_cv_score = np.mean(cv_scores)

        # メモリクリーンアップ
        del model
        gc.collect()

        return mean_cv_score

    def optimize(
        self,
        train_path: str = "data/processed/train.csv",
        validation_path: str = "data/processed/validation.csv",
        n_features_select: int = 75
    ) -> Dict[str, Any]:
        """ハイパーパラメータ最適化実行"""

        logger.info("TICKET-013 Optuna最適化開始")
        logger.info(f"トライアル数: {self.n_trials}, CV分割: {self.cv_folds}")

        # データ準備
        self.prepare_data(train_path, validation_path, n_features_select)

        # Optuna Study作成
        self.study = optuna.create_study(
            direction='minimize',  # RMSEを最小化
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        # 最適化実行
        start_time = time.time()
        logger.info("最適化実行中...")

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        optimization_time = time.time() - start_time

        # 最適パラメータ保存
        self.best_params = self.study.best_params

        # 結果サマリー
        results = {
            'best_score': self.study.best_value,
            'best_params': self.best_params,
            'n_trials': len(self.study.trials),
            'optimization_time_sec': optimization_time,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0
        }

        logger.success(f"最適化完了: Best RMSE = {self.study.best_value:.4f}")
        logger.info(f"最適化時間: {optimization_time/60:.1f}分")

        return results

    def train_final_model(self, test_path: str = "data/processed/test.csv") -> Tuple[np.ndarray, Dict[str, Any]]:
        """最適パラメータで最終モデル訓練と予測"""

        if self.best_params is None:
            raise ValueError("最適化を先に実行してください")

        logger.info("最終モデル訓練開始")

        # テストデータ準備
        test_df = pd.read_csv(test_path)
        test_features = self.create_ticket_017_01_02_features(test_df)

        # テストデータの特徴量を訓練データと合わせる
        common_features = [col for col in self.feature_names if col in test_features.columns]
        X_test = test_features[common_features].values

        logger.info(f"テストデータ形状: {X_test.shape}")

        # 5-Fold CV で最終予測
        models = []
        cv_scores = []
        predictions = np.zeros(len(X_test))

        final_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': self.random_state,
            'force_col_wise': True,
            **self.best_params
        }

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(self.X_train)):
            logger.info(f"Fold {fold+1}/{self.cv_folds} 訓練中...")

            X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # モデル訓練
            model = lgb.LGBMRegressor(**final_params)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=[lgb.log_evaluation(0)]
            )

            # 検証スコア
            val_pred = model.predict(X_fold_val)
            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
            cv_scores.append(fold_rmse)

            # テスト予測
            test_pred = model.predict(X_test)
            predictions += test_pred / self.cv_folds

            models.append(model)
            logger.info(f"  Fold {fold+1} RMSE: {fold_rmse:.4f}")

        final_cv_score = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        logger.success(f"最終CV RMSE: {final_cv_score:.4f} (±{cv_std:.4f})")

        # 最終結果
        final_results = {
            'cv_score': final_cv_score,
            'cv_std': cv_std,
            'cv_scores': cv_scores,
            'best_params': self.best_params,
            'models': models,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names)
        }

        # メモリクリーンアップ
        del test_features
        gc.collect()

        return predictions, final_results

    def save_optimization_results(self, results: Dict[str, Any], save_path: str):
        """最適化結果の保存"""

        # モデル以外の結果をJSONで保存
        json_results = {k: v for k, v in results.items() if k != 'models'}

        json_path = Path(save_path).with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)

        logger.info(f"最適化結果保存: {json_path}")

        # モデルはPickleで別途保存
        if 'models' in results:
            model_path = Path(save_path).with_suffix('.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(results['models'], f)
            logger.info(f"モデル保存: {model_path}")


def run_ticket_013_optimization():
    """TICKET-013 Optuna最適化メイン実行"""

    logger.info("=" * 60)
    logger.info("TICKET-013: Optuna最適化システム実行")
    logger.info("=" * 60)

    # 最適化器初期化
    optimizer = OptunaLightGBMOptimizer(
        n_trials=50,  # 試行回数（時間に応じて調整）
        timeout=3600,  # 1時間タイムアウト
        cv_folds=5,
        study_name="ticket_013_lightgbm_optimization"
    )

    try:
        # 最適化実行
        results = optimizer.optimize(
            train_path="data/processed/train.csv",
            validation_path="data/processed/validation.csv",
            n_features_select=75  # 最高性能時の特徴量数
        )

        # 最終モデル訓練と予測
        predictions, final_results = optimizer.train_final_model("data/processed/test.csv")

        # 提出ファイル作成
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # テストID読み込み
        test_df_original = pd.read_csv("data/processed/test.csv")
        submission_df = pd.DataFrame({
            'id': test_df_original['id'],
            'BeatsPerMinute': predictions
        })

        # 提出ファイル保存
        submission_path = f"data/processed/submission_ticket013_optuna_{timestamp}.csv"
        submission_df.to_csv(submission_path, index=False)

        # 結果保存
        results_path = f"experiments/ticket013_optuna_results_{timestamp}"
        optimizer.save_optimization_results(final_results, results_path)

        # サマリー表示
        logger.info("\n" + "=" * 60)
        logger.info("TICKET-013 Optuna最適化完了")
        logger.info("=" * 60)
        logger.info(f"最高CV RMSE: {final_results['cv_score']:.4f} (±{final_results['cv_std']:.4f})")
        logger.info(f"特徴量数: {final_results['n_features']}")
        logger.info(f"最適パラメータ数: {len(final_results['best_params'])}")
        logger.info(f"提出ファイル: {submission_path}")
        logger.info(f"結果ファイル: {results_path}.json")

        # 最適パラメータ表示
        logger.info("\n最適ハイパーパラメータ:")
        for param, value in final_results['best_params'].items():
            logger.info(f"  {param}: {value}")

        # 提出コマンド表示
        logger.info(f"\n提出コマンド:")
        cv_score = final_results['cv_score']
        logger.info(f'kaggle competitions submit -c playground-series-s5e9 -f "{submission_path}" -m "TICKET-013 Optuna Optimized (CV: {cv_score:.4f}, Features: {final_results["n_features"]})"')

        return True

    except Exception as e:
        logger.error(f"TICKET-013最適化エラー: {e}")
        return False


if __name__ == "__main__":
    success = run_ticket_013_optimization()
    if success:
        logger.success("TICKET-013 Optuna最適化成功")
    else:
        logger.error("TICKET-013 Optuna最適化失敗")
        sys.exit(1)