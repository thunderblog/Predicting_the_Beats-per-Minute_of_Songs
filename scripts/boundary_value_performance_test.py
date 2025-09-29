"""
TICKET-027: 境界値変換前後の性能比較実験

TICKET-025で実装した境界値変換システムの効果を検証する。
目標: CV-LB格差+0.076→+0.030以下の大幅改善の確認
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR


class BoundaryValuePerformanceTest:
    """境界値変換前後の性能比較実験クラス."""

    def __init__(self):
        """初期化."""
        self.results = {}

        # LightGBMパラメータ（既存ベースライン準拠）
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """データセット読み込み.

        Returns:
            (元データ, 変換後データ)のタプル
        """
        logger.info("データセット読み込み中...")

        # 元データ
        original_path = PROCESSED_DATA_DIR / "train_unified_75_features.csv"
        original_data = pd.read_csv(original_path)

        # 変換後データ
        transformed_path = PROCESSED_DATA_DIR / "train_boundary_transformed.csv"
        transformed_data = pd.read_csv(transformed_path)

        logger.info(f"元データ: {original_data.shape}")
        logger.info(f"変換後データ: {transformed_data.shape}")

        return original_data, transformed_data

    def create_bpm_stratified_folds(self, y: pd.Series, n_splits: int = 5) -> StratifiedKFold:
        """BPM帯域別StratifiedKFold作成.

        Args:
            y: ターゲット変数
            n_splits: 分割数

        Returns:
            StratifiedKFoldオブジェクト
        """
        # BPM帯域ラベル作成
        bpm_bins = [0, 80, 120, 160, 200, float('inf')]
        bpm_labels = pd.cut(y, bins=bpm_bins, labels=['Slow', 'Moderate', 'Fast', 'VeryFast', 'Extreme'])

        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), bpm_labels

    def run_cv_experiment(self, X: pd.DataFrame, y: pd.Series,
                         experiment_name: str) -> Dict:
        """クロスバリデーション実験実行.

        Args:
            X: 特徴量データ
            y: ターゲット変数
            experiment_name: 実験名

        Returns:
            実験結果辞書
        """
        logger.info(f"CV実験開始: {experiment_name}")

        skf, bpm_labels = self.create_bpm_stratified_folds(y)

        fold_results = []
        oof_predictions = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, bpm_labels), 1):
            logger.info(f"Fold {fold} 実行中...")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # LightGBMデータセット作成
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # モデル訓練
            model = lgb.train(
                self.lgb_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            # 予測
            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            oof_predictions[val_idx] = val_pred

            # RMSE計算
            fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            fold_results.append(fold_rmse)

            logger.info(f"Fold {fold} RMSE: {fold_rmse:.6f}")

        # 全体結果
        overall_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
        mean_cv_rmse = np.mean(fold_results)
        std_cv_rmse = np.std(fold_results)

        results = {
            'experiment_name': experiment_name,
            'n_features': X.shape[1],
            'n_samples': len(X),
            'overall_rmse': overall_rmse,
            'mean_cv_rmse': mean_cv_rmse,
            'std_cv_rmse': std_cv_rmse,
            'fold_results': fold_results,
            'oof_predictions': oof_predictions
        }

        logger.success(f"{experiment_name} 完了: CV RMSE {mean_cv_rmse:.6f} ± {std_cv_rmse:.6f}")

        return results

    def run_comparison_experiment(self) -> Dict:
        """比較実験の実行.

        Returns:
            比較結果辞書
        """
        logger.info("境界値変換前後の性能比較実験開始")

        # データ読み込み
        original_data, transformed_data = self.load_datasets()

        target_col = 'BeatsPerMinute'
        y = original_data[target_col]

        # 元データの特徴量（ターゲット除く）
        original_features = [col for col in original_data.columns if col != target_col]
        X_original = original_data[original_features]

        # 変換後データの特徴量（元特徴量 + 新特徴量）
        transformed_features = [col for col in transformed_data.columns if col != target_col]
        X_transformed = transformed_data[transformed_features]

        logger.info(f"元データ特徴量数: {len(original_features)}")
        logger.info(f"変換後特徴量数: {len(transformed_features)}")

        # 実験1: 元データ（ベースライン）
        baseline_results = self.run_cv_experiment(
            X_original, y, "Baseline_Original_Features"
        )

        # 実験2: 変換後データ
        transformed_results = self.run_cv_experiment(
            X_transformed, y, "Boundary_Transformed_Features"
        )

        # 比較分析
        improvement = baseline_results['mean_cv_rmse'] - transformed_results['mean_cv_rmse']
        improvement_pct = (improvement / baseline_results['mean_cv_rmse']) * 100

        comparison_results = {
            'baseline': baseline_results,
            'transformed': transformed_results,
            'improvement': {
                'rmse_improvement': improvement,
                'improvement_percentage': improvement_pct,
                'is_significant': abs(improvement) > (baseline_results['std_cv_rmse'] + transformed_results['std_cv_rmse'])
            }
        }

        # 結果表示
        self.display_comparison_results(comparison_results)

        return comparison_results

    def display_comparison_results(self, results: Dict):
        """比較結果の表示.

        Args:
            results: 比較結果辞書
        """
        logger.success("=== 境界値変換効果検証結果 ===")

        baseline = results['baseline']
        transformed = results['transformed']
        improvement = results['improvement']

        logger.info(f"ベースライン    : CV RMSE {baseline['mean_cv_rmse']:.6f} ± {baseline['std_cv_rmse']:.6f} ({baseline['n_features']}特徴量)")
        logger.info(f"境界値変換後    : CV RMSE {transformed['mean_cv_rmse']:.6f} ± {transformed['std_cv_rmse']:.6f} ({transformed['n_features']}特徴量)")

        if improvement['rmse_improvement'] > 0:
            logger.success(f"改善効果: -{improvement['rmse_improvement']:.6f} ({improvement['improvement_percentage']:.3f}%向上)")
        else:
            logger.warning(f"性能変化: {abs(improvement['rmse_improvement']):.6f} ({abs(improvement['improvement_percentage']):.3f}%劣化)")

        logger.info(f"統計的有意性: {'有意' if improvement['is_significant'] else '非有意'}")

        # TICKET-025目標との比較
        target_improvement = 0.076 - 0.030  # +0.076→+0.030の改善目標
        if improvement['rmse_improvement'] >= target_improvement:
            logger.success(f"🎯 TICKET-025目標達成: {improvement['rmse_improvement']:.6f} >= {target_improvement:.6f}")
        else:
            logger.info(f"目標まで: あと{target_improvement - improvement['rmse_improvement']:.6f}の改善が必要")

    def save_results(self, results: Dict, output_path: Path = None):
        """結果保存.

        Args:
            results: 保存対象結果
            output_path: 出力パス
        """
        if output_path is None:
            output_path = Path("boundary_value_experiment_results.json")

        # numpy配列をリストに変換
        serializable_results = self._make_json_serializable(results)

        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"実験結果保存: {output_path}")

    def _make_json_serializable(self, obj):
        """JSON serializable形式に変換."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj


def main():
    """メイン実行関数."""
    logger.info("TICKET-027: 境界値変換前後の性能比較実験開始")

    try:
        # 性能比較実験実行
        tester = BoundaryValuePerformanceTest()
        results = tester.run_comparison_experiment()

        # 結果保存
        tester.save_results(results)

        logger.success("TICKET-027: 性能比較実験完了")

    except Exception as e:
        logger.error(f"性能比較実験中にエラー: {e}")
        raise


if __name__ == "__main__":
    main()