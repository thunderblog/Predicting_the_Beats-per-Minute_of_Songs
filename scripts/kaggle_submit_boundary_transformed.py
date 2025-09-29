"""
境界値変換データでのKaggle提出スクリプト

TICKET-025で実装した境界値変換をtest_dataにも適用し、
変換済み特徴量でモデル訓練・予測・提出を行う。
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from loguru import logger
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.data.boundary_value_transformer import BoundaryValueTransformer


class KaggleBoundarySubmission:
    """境界値変換データでのKaggle提出クラス."""

    def __init__(self):
        """初期化."""
        # LightGBMパラメータ（ベースライン準拠）
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

        self.target_col = 'BeatsPerMinute'

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """元データの読み込み.

        Returns:
            (train_data, test_data)のタプル
        """
        logger.info("元データ読み込み中...")

        # 訓練データ
        train_path = PROCESSED_DATA_DIR / "train_unified_75_features.csv"
        train_data = pd.read_csv(train_path)

        # テストデータ
        test_path = PROCESSED_DATA_DIR / "test_unified_75_features.csv"
        test_data = pd.read_csv(test_path)

        logger.info(f"訓練データ: {train_data.shape}")
        logger.info(f"テストデータ: {test_data.shape}")

        return train_data, test_data

    def apply_boundary_transformation_to_test(self, train_data: pd.DataFrame,
                                            test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """テストデータへの境界値変換適用.

        Args:
            train_data: 訓練データ
            test_data: テストデータ

        Returns:
            (変換済み訓練データ, 変換済みテストデータ)
        """
        logger.info("テストデータへの境界値変換適用中...")

        # 訓練データで変換パラメータを学習
        transformer = BoundaryValueTransformer()

        # 訓練データの変換（既存の変換済みデータを使用）
        transformed_train_path = PROCESSED_DATA_DIR / "train_boundary_transformed.csv"
        if transformed_train_path.exists():
            logger.info("既存の変換済み訓練データを使用")
            transformed_train = pd.read_csv(transformed_train_path)
        else:
            logger.info("訓練データの境界値変換を実行")
            transformed_train = transformer.transform_all_boundary_issues(train_data)

        # テストデータに同じ変換パラメータを適用
        logger.info("テストデータに境界値変換を適用中...")
        transformed_test = self._apply_same_transforms_to_test(test_data, transformer)

        logger.success(f"変換完了 - 訓練: {transformed_train.shape}, テスト: {transformed_test.shape}")

        return transformed_train, transformed_test

    def _apply_same_transforms_to_test(self, test_data: pd.DataFrame,
                                     transformer: BoundaryValueTransformer) -> pd.DataFrame:
        """テストデータに同一の変換パラメータを適用.

        Args:
            test_data: テストデータ
            transformer: 訓練済み変換器

        Returns:
            変換済みテストデータ
        """
        result_data = test_data.copy()

        # 1. 0値集中特徴量の対数変換
        for feature in transformer.zero_concentrated_features.keys():
            if feature in result_data.columns:
                epsilon = 1e-8
                transformed_data = np.log1p(result_data[feature] + epsilon)
                new_feature_name = f"log_transform_{feature}"
                result_data[new_feature_name] = transformed_data
                logger.info(f"テスト変換: {feature} → {new_feature_name}")

        # 2. 最小値集中特徴量のランク正規化
        for feature in transformer.min_value_concentrated_features.keys():
            if feature in result_data.columns:
                # テストデータの独立ランク正規化
                from scipy import stats
                ranks = stats.rankdata(result_data[feature], method='average')
                normalized_ranks = ranks / len(result_data)
                new_feature_name = f"rank_normalized_{feature}"
                result_data[new_feature_name] = normalized_ranks
                logger.info(f"テスト変換: {feature} → {new_feature_name}")

        # 3. 境界値集中特徴量の変換
        for feature in transformer.boundary_concentrated_features.keys():
            if feature in result_data.columns:
                if feature == 'RhythmScore':
                    # 逆変換
                    transformed_data = 1.0 - result_data[feature]
                elif feature == 'AudioLoudness':
                    # Shifted log変換
                    shifted_data = result_data[feature] - result_data[feature].min() + 1.0
                    transformed_data = np.log(shifted_data)
                else:
                    # その他はそのまま
                    transformed_data = result_data[feature]

                new_feature_name = f"boundary_transform_{feature}"
                result_data[new_feature_name] = transformed_data
                logger.info(f"テスト変換: {feature} → {new_feature_name}")

        # 4. TrackDurationMs不連続性対応（テストデータはそのまま）
        if 'TrackDurationMs' in result_data.columns:
            result_data['interpolated_TrackDurationMs'] = result_data['TrackDurationMs']
            logger.info("テスト変換: TrackDurationMs → interpolated_TrackDurationMs")

        return result_data

    def train_model_with_cv(self, X: pd.DataFrame, y: pd.Series) -> Tuple[list, np.ndarray]:
        """CVでモデル訓練.

        Args:
            X: 特徴量データ
            y: ターゲット変数

        Returns:
            (モデルリスト, OOF予測)
        """
        logger.info("境界値変換データでのモデル訓練開始...")

        # BPM帯域別StratifiedKFold
        bpm_bins = [0, 80, 120, 160, 200, float('inf')]
        bpm_labels = pd.cut(y, bins=bpm_bins, labels=['Slow', 'Moderate', 'Fast', 'VeryFast', 'Extreme'])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        models = []
        oof_predictions = np.zeros(len(y))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, bpm_labels), 1):
            logger.info(f"Fold {fold} 訓練中...")

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

            models.append(model)

            # OOF予測
            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            oof_predictions[val_idx] = val_pred

            # スコア計算
            fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            fold_scores.append(fold_rmse)

            logger.info(f"Fold {fold} RMSE: {fold_rmse:.6f}")

        # 全体スコア
        overall_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
        mean_cv_rmse = np.mean(fold_scores)

        logger.success(f"CV完了: 平均RMSE {mean_cv_rmse:.6f}, 全体RMSE {overall_rmse:.6f}")

        return models, oof_predictions

    def predict_test(self, models: list, X_test: pd.DataFrame) -> np.ndarray:
        """テストデータ予測.

        Args:
            models: 訓練済みモデルリスト
            X_test: テスト特徴量

        Returns:
            予測値配列
        """
        logger.info("テストデータ予測中...")

        test_predictions = np.zeros(len(X_test))

        for i, model in enumerate(models):
            pred = model.predict(X_test, num_iteration=model.best_iteration)
            test_predictions += pred
            logger.info(f"モデル {i+1}/{len(models)} 予測完了")

        # アンサンブル平均
        test_predictions /= len(models)

        logger.success(f"テスト予測完了: {len(test_predictions):,}サンプル")

        return test_predictions

    def create_submission(self, test_predictions: np.ndarray,
                         test_ids: pd.Series = None,
                         output_path: Path = None) -> Path:
        """提出ファイル作成.

        Args:
            test_predictions: テスト予測値
            test_ids: テストデータのID（実際のID使用）
            output_path: 出力パス

        Returns:
            提出ファイルパス
        """
        if output_path is None:
            output_path = PROCESSED_DATA_DIR / "submission_boundary_transformed.csv"

        # 実際のテストデータIDを使用
        if test_ids is not None:
            ids_to_use = test_ids
        else:
            ids_to_use = range(len(test_predictions))

        # 提出フォーマット作成
        submission_df = pd.DataFrame({
            'id': ids_to_use,
            'BeatsPerMinute': test_predictions
        })

        submission_df.to_csv(output_path, index=False)

        logger.success(f"提出ファイル作成: {output_path}")
        logger.info(f"予測統計: mean={test_predictions.mean():.2f}, "
                   f"min={test_predictions.min():.2f}, max={test_predictions.max():.2f}")

        return output_path

    def run_full_pipeline(self) -> Path:
        """完全パイプライン実行.

        Returns:
            提出ファイルパス
        """
        logger.info("境界値変換データでの Kaggle提出パイプライン開始")

        try:
            # 1. データ読み込み
            train_data, test_data = self.load_raw_data()

            # 2. 境界値変換適用
            transformed_train, transformed_test = self.apply_boundary_transformation_to_test(
                train_data, test_data
            )

            # 3. 特徴量・ターゲット分離
            feature_cols = [col for col in transformed_train.columns if col != self.target_col]
            X_train = transformed_train[feature_cols]
            y_train = transformed_train[self.target_col]
            X_test = transformed_test[feature_cols]

            # テストデータのIDを取得
            test_ids = test_data['id'] if 'id' in test_data.columns else None

            logger.info(f"使用特徴量数: {len(feature_cols)}")

            # 4. モデル訓練
            models, oof_predictions = self.train_model_with_cv(X_train, y_train)

            # 5. テスト予測
            test_predictions = self.predict_test(models, X_test)

            # 6. 提出ファイル作成（実際のIDを使用）
            submission_path = self.create_submission(test_predictions, test_ids)

            logger.success("境界値変換データでのKaggle提出パイプライン完了")

            return submission_path

        except Exception as e:
            logger.error(f"提出パイプライン実行中にエラー: {e}")
            raise


def main():
    """メイン実行関数."""
    logger.info("境界値変換データでのKaggle提出開始")

    try:
        submitter = KaggleBoundarySubmission()
        submission_path = submitter.run_full_pipeline()

        logger.success(f"Kaggle提出準備完了: {submission_path}")
        logger.info("次のコマンドでKaggleに提出してください:")
        logger.info(f'kaggle competitions submit -c playground-series-s4e9 -f "{submission_path}" -m "TICKET-025 Boundary Value Transformation"')

    except Exception as e:
        logger.error(f"Kaggle提出処理中にエラー: {e}")
        raise


if __name__ == "__main__":
    main()