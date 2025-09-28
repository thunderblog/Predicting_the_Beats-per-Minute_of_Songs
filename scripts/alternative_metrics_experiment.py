"""
代替評価指標実験スクリプト

RMSE以外の評価指標で訓練し、最終的なRMSE性能を比較する実験
音楽BPM予測タスクにおける評価指標最適化の効果を検証
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from datetime import datetime

# loguru設定を安全に初期化
from loguru import logger
logger.remove()  # 全ハンドラーを削除
logger.add(sys.stderr, level="INFO")  # 標準エラー出力に設定
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from typing import Dict, List, Tuple, Any
import lightgbm as lgb
import catboost as cb

# 設定を直接定義（config.pyのloguru問題を回避）
class Config:
    target = "BeatsPerMinute"
    num_leaves = 31
    learning_rate = 0.1
    feature_fraction = 0.8
    n_estimators = 1000
    stopping_rounds = 100
    random_state = 42

config = Config()

# CV戦略を直接実装
from sklearn.model_selection import StratifiedKFold

def create_bpm_stratified_cv(n_splits=5, random_state=42):
    """BPM Stratified KFold作成"""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def huber_loss_sklearn(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """Huber Loss計算（scikit-learn風）"""
    residual = np.abs(y_true - y_pred)
    return np.mean(np.where(residual <= delta,
                           0.5 * residual**2,
                           delta * (residual - 0.5 * delta)))

def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5) -> float:
    """Quantile Loss計算 (Pinball Loss)"""
    residual = y_true - y_pred
    return np.mean(np.maximum(alpha * residual, (alpha - 1) * residual))


class AlternativeMetricsExperiment:
    """代替評価指標実験クラス"""

    def __init__(self, data_path: str = None):
        self.data_path = data_path or "data/processed/train_unified_75_features.csv"
        self.results = {}

        # 実験する評価指標の定義
        self.metrics_config = {
            "rmse": {
                "lgb_metric": "rmse",
                "catboost_metric": "RMSE",
                "eval_func": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                "description": "Root Mean Squared Error (ベースライン)"
            },
            "mae": {
                "lgb_metric": "mae",
                "catboost_metric": "MAE",
                "eval_func": mean_absolute_error,
                "description": "Mean Absolute Error (外れ値耐性)"
            },
            "mape": {
                "lgb_metric": "mape",
                "catboost_metric": "MAPE",
                "eval_func": mean_absolute_percentage_error,
                "description": "Mean Absolute Percentage Error (相対誤差)"
            },
            "huber": {
                "lgb_metric": "huber",
                "catboost_metric": "Huber",
                "eval_func": lambda y_true, y_pred: huber_loss_sklearn(y_true, y_pred, delta=1.0),
                "description": "Huber Loss (RMSE+MAEハイブリッド)"
            },
            "quantile_50": {
                "lgb_metric": "quantile",
                "catboost_metric": "Quantile",
                "eval_func": lambda y_true, y_pred: quantile_loss(y_true, y_pred, alpha=0.5),
                "description": "Quantile Loss α=0.5 (中央値回帰)",
                "lgb_params": {"alpha": 0.5},
                "catboost_params": {"delta": 0.5}
            }
        }

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """データ読み込み"""
        logger.info(f"データ読み込み: {self.data_path}")
        df = pd.read_csv(self.data_path)

        feature_cols = [col for col in df.columns if col not in ["id", config.target]]
        X = df[feature_cols]
        y = df[config.target]

        logger.info(f"特徴量数: {len(feature_cols)}, サンプル数: {len(X)}")
        return X, y

    def train_lightgbm_with_metric(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric_name: str
    ) -> lgb.Booster:
        """指定評価指標でLightGBMを訓練"""
        metric_config = self.metrics_config[metric_name]

        # ベースパラメータ
        params = {
            "objective": "regression",
            "metric": metric_config["lgb_metric"],
            "boosting_type": "gbdt",
            "num_leaves": config.num_leaves,
            "learning_rate": config.learning_rate,
            "feature_fraction": config.feature_fraction,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": config.random_state,
        }

        # メトリック固有パラメータの追加
        if "lgb_params" in metric_config:
            params.update(metric_config["lgb_params"])

        # データセット作成
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 訓練実行
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "eval"],
            num_boost_round=config.n_estimators,
            callbacks=[
                lgb.early_stopping(config.stopping_rounds),
                lgb.log_evaluation(0)  # ログ無効化
            ],
        )

        return model

    def train_catboost_with_metric(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric_name: str
    ) -> cb.CatBoostRegressor:
        """指定評価指標でCatBoostを訓練"""
        metric_config = self.metrics_config[metric_name]

        # ベースパラメータ
        params = {
            "loss_function": metric_config["catboost_metric"],
            "eval_metric": metric_config["catboost_metric"],
            "depth": 6,
            "learning_rate": config.learning_rate,
            "subsample": 0.8,
            "rsm": config.feature_fraction,
            "random_seed": config.random_state,
            "verbose": 0,
            "early_stopping_rounds": config.stopping_rounds,
            "iterations": config.n_estimators,
        }

        # メトリック固有パラメータの追加
        if "catboost_params" in metric_config:
            params.update(metric_config["catboost_params"])

        # モデル作成・訓練
        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False,
            plot=False
        )

        return model

    def evaluate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """全評価指標でスコア計算"""
        results = {}
        for metric_name, metric_config in self.metrics_config.items():
            try:
                score = metric_config["eval_func"](y_true, y_pred)
                results[metric_name] = score
            except Exception as e:
                logger.warning(f"メトリック {metric_name} の計算エラー: {e}")
                results[metric_name] = np.nan
        return results

    def run_cv_experiment(
        self,
        model_type: str = "lightgbm",
        metric_name: str = "rmse",
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """クロスバリデーション実験実行"""
        logger.info(f"実験開始: {model_type} + {metric_name}")

        # データ読み込み
        X, y = self.load_data()

        # CV戦略作成（BPM値で層化分割）
        # BPM値をビンに分けて層化
        y_binned = pd.cut(y, bins=10, labels=False)
        cv_splitter = create_bpm_stratified_cv(n_splits=n_folds, random_state=config.random_state)

        fold_results = []
        models = []

        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y_binned)):
            logger.info(f"フォールド {fold + 1}/{n_folds}")

            # データ分割
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # モデル訓練
            if model_type == "lightgbm":
                model = self.train_lightgbm_with_metric(X_train, y_train, X_val, y_val, metric_name)
                y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            elif model_type == "catboost":
                model = self.train_catboost_with_metric(X_train, y_train, X_val, y_val, metric_name)
                y_pred = model.predict(X_val)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            # 全メトリックで評価
            fold_scores = self.evaluate_all_metrics(y_val.values, y_pred)
            fold_results.append(fold_scores)
            models.append(model)

            # フォールド結果ログ
            logger.info(f"  RMSE: {fold_scores['rmse']:.4f}, 訓練メトリック({metric_name}): {fold_scores[metric_name]:.4f}")

        # 結果集約
        aggregated_results = {}
        for metric in self.metrics_config.keys():
            scores = [fold[metric] for fold in fold_results if not np.isnan(fold[metric])]
            if scores:
                aggregated_results[f"{metric}_mean"] = np.mean(scores)
                aggregated_results[f"{metric}_std"] = np.std(scores)

        experiment_result = {
            "model_type": model_type,
            "training_metric": metric_name,
            "n_folds": n_folds,
            "fold_results": fold_results,
            "aggregated_results": aggregated_results,
            "models": models,
            "timestamp": datetime.now().isoformat()
        }

        logger.success(f"実験完了: RMSE {aggregated_results['rmse_mean']:.4f}±{aggregated_results['rmse_std']:.4f}")
        return experiment_result

    def run_full_experiment(self) -> Dict[str, Any]:
        """全メトリック・全モデルでの完全実験"""
        logger.info("代替評価指標実験開始")

        all_results = {}

        # 実験対象の組み合わせ
        model_types = ["lightgbm", "catboost"]
        metric_names = ["rmse", "mae", "huber", "mape"]  # quantile_50は後回し

        for model_type in model_types:
            all_results[model_type] = {}

            for metric_name in metric_names:
                try:
                    result = self.run_cv_experiment(model_type, metric_name)
                    all_results[model_type][metric_name] = result
                except Exception as e:
                    logger.error(f"実験失敗 {model_type}+{metric_name}: {e}")
                    all_results[model_type][metric_name] = {"error": str(e)}

        # 結果保存
        self.save_results(all_results)

        # サマリーレポート
        self.generate_summary_report(all_results)

        return all_results

    def save_results(self, results: Dict[str, Any]):
        """実験結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"experiments/alternative_metrics_experiment_{timestamp}.json"

        # modelsオブジェクトは保存できないので除外
        saveable_results = {}
        for model_type, model_results in results.items():
            saveable_results[model_type] = {}
            for metric, result in model_results.items():
                if isinstance(result, dict) and "models" in result:
                    # modelsを除外してコピー
                    saveable_result = {k: v for k, v in result.items() if k != "models"}
                    saveable_results[model_type][metric] = saveable_result
                else:
                    saveable_results[model_type][metric] = result

        Path("experiments").mkdir(exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(saveable_results, f, indent=2, ensure_ascii=False)

        logger.success(f"実験結果保存: {results_path}")

    def generate_summary_report(self, results: Dict[str, Any]):
        """サマリーレポート生成"""
        logger.info("\n" + "="*60)
        logger.info("代替評価指標実験 - サマリーレポート")
        logger.info("="*60)

        # 結果比較テーブル
        comparison_data = []

        for model_type, model_results in results.items():
            for metric_name, result in model_results.items():
                if isinstance(result, dict) and "aggregated_results" in result:
                    agg = result["aggregated_results"]
                    comparison_data.append({
                        "Model": model_type.upper(),
                        "Training_Metric": metric_name.upper(),
                        "CV_RMSE": f"{agg['rmse_mean']:.4f}±{agg['rmse_std']:.4f}",
                        "CV_MAE": f"{agg['mae_mean']:.4f}±{agg['mae_std']:.4f}" if 'mae_mean' in agg else "N/A",
                        "Training_Score": f"{agg[f'{metric_name}_mean']:.4f}±{agg[f'{metric_name}_std']:.4f}"
                    })

        # パフォーマンス比較
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print("\n📊 性能比較表:")
            print(df_comparison.to_string(index=False))

            # ベストパフォーマンス特定
            rmse_scores = []
            for data in comparison_data:
                rmse_str = data["CV_RMSE"].split("±")[0]
                rmse_scores.append((float(rmse_str), data["Model"], data["Training_Metric"]))

            best_rmse = min(rmse_scores)
            logger.info(f"\n🏆 最高RMSE性能: {best_rmse[0]:.4f} ({best_rmse[1]} + {best_rmse[2]})")


def main():
    """メイン実行関数"""
    # ログディレクトリ作成
    Path("logs").mkdir(exist_ok=True)
    # ファイルログ追加
    logger.add("logs/alternative_metrics_experiment_{time}.log", level="INFO")

    experiment = AlternativeMetricsExperiment()

    # 完全実験を実行（全メトリック × 全モデル）
    results = experiment.run_full_experiment()

    logger.success("代替評価指標実験完了!")


if __name__ == "__main__":
    main()