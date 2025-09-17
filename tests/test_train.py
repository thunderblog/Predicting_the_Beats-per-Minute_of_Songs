"""
TICKET-006: src/modeling/train.py用ユニットテスト
LightGBM回帰モデル訓練機能のテストケース
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import json
import pickle

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb

# プロジェクトルートをパスに追加
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.modeling.train import (
    train_with_cross_validation,
    train_single_model,
    save_cv_results,
    save_single_model,
)


class TestTrainWithCrossValidation:
    """クロスバリデーション訓練機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        np.random.seed(42)
        self.X = pd.DataFrame({
            f"feature_{i}": np.random.randn(100) for i in range(10)
        })
        self.y = pd.Series(np.random.uniform(100, 200, 100))

    @patch("src.modeling.train.config")
    @patch("src.modeling.train.lgb.train")
    def test_cross_validation_basic(self, mock_lgb_train, mock_config):
        """基本的なクロスバリデーションテスト"""
        # configのモック設定
        mock_config.random_state = 42
        mock_config.objective = "regression"
        mock_config.metric = "rmse"
        mock_config.num_leaves = 31
        mock_config.learning_rate = 0.1
        mock_config.feature_fraction = 0.8
        mock_config.n_estimators = 100
        mock_config.stopping_rounds = 10
        mock_config.log_evaluation = 100

        # LightGBMモデルのモック
        mock_model = MagicMock()
        mock_model.best_iteration = 50
        mock_model.predict.return_value = np.random.uniform(100, 200, 20)  # フォールドサイズに合わせる
        mock_lgb_train.return_value = mock_model

        # テスト実行
        cv_scores, models = train_with_cross_validation(self.X, self.y, n_folds=5)

        # アサーション
        assert len(cv_scores) == 5
        assert len(models) == 5
        assert all(isinstance(score, float) for score in cv_scores)
        assert all(score > 0 for score in cv_scores)  # RMSEは正の値
        assert mock_lgb_train.call_count == 5

    @patch("src.modeling.train.config")
    @patch("src.modeling.train.lgb.train")
    def test_cross_validation_different_folds(self, mock_lgb_train, mock_config):
        """異なるフォールド数でのテスト"""
        # configのモック設定
        mock_config.random_state = 42
        mock_config.objective = "regression"
        mock_config.metric = "rmse"
        mock_config.num_leaves = 31
        mock_config.learning_rate = 0.1
        mock_config.feature_fraction = 0.8
        mock_config.n_estimators = 100
        mock_config.stopping_rounds = 10
        mock_config.log_evaluation = 100

        # LightGBMモデルのモック
        mock_model = MagicMock()
        mock_model.best_iteration = 50
        mock_model.predict.return_value = np.random.uniform(100, 200, 34)  # 100/3の約1/3
        mock_lgb_train.return_value = mock_model

        # 3フォールドでテスト
        cv_scores, models = train_with_cross_validation(self.X, self.y, n_folds=3)

        assert len(cv_scores) == 3
        assert len(models) == 3
        assert mock_lgb_train.call_count == 3

    @patch("src.modeling.train.config")
    def test_cross_validation_empty_data(self, mock_config):
        """空のデータでのテスト"""
        mock_config.random_state = 42

        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)

        with pytest.raises(Exception):  # 適切な例外が発生することを確認
            train_with_cross_validation(empty_X, empty_y, n_folds=5)


class TestTrainSingleModel:
    """単一モデル訓練機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            f"feature_{i}": np.random.randn(80) for i in range(10)
        })
        self.y_train = pd.Series(np.random.uniform(100, 200, 80))

        self.X_val = pd.DataFrame({
            f"feature_{i}": np.random.randn(20) for i in range(10)
        })
        self.y_val = pd.Series(np.random.uniform(100, 200, 20))

    @patch("src.modeling.train.config")
    @patch("src.modeling.train.lgb.train")
    def test_train_single_model_basic(self, mock_lgb_train, mock_config):
        """基本的な単一モデル訓練テスト"""
        # configのモック設定
        mock_config.objective = "regression"
        mock_config.metric = "rmse"
        mock_config.num_leaves = 31
        mock_config.learning_rate = 0.1
        mock_config.feature_fraction = 0.8
        mock_config.n_estimators = 100
        mock_config.stopping_rounds = 10
        mock_config.log_evaluation = 100
        mock_config.random_state = 42

        # LightGBMモデルのモック
        mock_model = MagicMock()
        mock_model.best_iteration = 50
        mock_model.predict.side_effect = [
            np.random.uniform(100, 200, 80),  # 訓練データ予測
            np.random.uniform(100, 200, 20)   # 検証データ予測
        ]
        mock_lgb_train.return_value = mock_model

        # テスト実行
        model, train_score, val_score = train_single_model(
            self.X_train, self.y_train, self.X_val, self.y_val
        )

        # アサーション
        assert model == mock_model
        assert isinstance(train_score, float)
        assert isinstance(val_score, float)
        assert train_score > 0
        assert val_score > 0
        assert mock_lgb_train.call_count == 1

    @patch("src.modeling.train.config")
    @patch("src.modeling.train.lgb.train")
    def test_train_single_model_perfect_prediction(self, mock_lgb_train, mock_config):
        """完全予測の場合のテスト"""
        mock_config.objective = "regression"
        mock_config.metric = "rmse"
        mock_config.num_leaves = 31
        mock_config.learning_rate = 0.1
        mock_config.feature_fraction = 0.8
        mock_config.n_estimators = 100
        mock_config.stopping_rounds = 10
        mock_config.log_evaluation = 100
        mock_config.random_state = 42

        # 完全な予測を返すモデル
        mock_model = MagicMock()
        mock_model.best_iteration = 50
        mock_model.predict.side_effect = [
            self.y_train.values,  # 訓練データの完全予測
            self.y_val.values     # 検証データの完全予測
        ]
        mock_lgb_train.return_value = mock_model

        model, train_score, val_score = train_single_model(
            self.X_train, self.y_train, self.X_val, self.y_val
        )

        # 完全予測なのでRMSEは0に近い
        assert train_score < 1e-10
        assert val_score < 1e-10


class TestSaveCvResults:
    """クロスバリデーション結果保存機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        self.cv_scores = [25.5, 26.2, 24.8, 25.9, 26.1]
        self.mock_models = [MagicMock() for _ in range(5)]
        self.feature_cols = [f"feature_{i}" for i in range(10)]

    @patch("src.modeling.train.config")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.modeling.train.json.dump")
    @patch("src.modeling.train.pickle.dump")
    def test_save_cv_results(self, mock_pickle_dump, mock_json_dump,
                           mock_file_open, mock_config):
        """クロスバリデーション結果保存テスト"""
        # configのモック設定
        mock_config.objective = "regression"
        mock_config.metric = "rmse"
        mock_config.num_leaves = 31
        mock_config.learning_rate = 0.1
        mock_config.feature_fraction = 0.8
        mock_config.n_estimators = 100
        mock_config.stopping_rounds = 10

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            # テスト実行
            save_cv_results(
                self.cv_scores, self.mock_models, model_dir, "test_exp", self.feature_cols
            )

            # アサーション
            assert mock_json_dump.called
            assert mock_pickle_dump.call_count == 5  # 5つのモデル

            # JSON保存の内容確認
            saved_data = mock_json_dump.call_args[0][0]
            assert saved_data["experiment_name"] == "test_exp"
            assert saved_data["cv_scores"] == self.cv_scores
            assert saved_data["n_folds"] == 5
            assert saved_data["feature_count"] == 10
            assert np.isclose(saved_data["mean_cv_score"], np.mean(self.cv_scores))

    def test_save_cv_results_directory_creation(self):
        """ディレクトリ作成のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "non_existent_dir"

            # ディレクトリが存在しない場合でも実行可能
            with patch("src.modeling.train.config") as mock_config:
                mock_config.objective = "regression"
                mock_config.metric = "rmse"
                mock_config.num_leaves = 31
                mock_config.learning_rate = 0.1
                mock_config.feature_fraction = 0.8
                mock_config.n_estimators = 100
                mock_config.stopping_rounds = 10

                # 例外が発生しないことを確認
                save_cv_results(
                    self.cv_scores, self.mock_models, model_dir, "test_exp", self.feature_cols
                )


class TestSaveSingleModel:
    """単一モデル結果保存機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        self.mock_model = MagicMock()
        self.train_score = 24.5
        self.val_score = 26.3
        self.feature_cols = [f"feature_{i}" for i in range(15)]

    @patch("src.modeling.train.config")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.modeling.train.json.dump")
    @patch("src.modeling.train.pickle.dump")
    def test_save_single_model(self, mock_pickle_dump, mock_json_dump,
                             mock_file_open, mock_config):
        """単一モデル結果保存テスト"""
        # configのモック設定
        mock_config.objective = "regression"
        mock_config.metric = "rmse"
        mock_config.num_leaves = 31
        mock_config.learning_rate = 0.1
        mock_config.feature_fraction = 0.8
        mock_config.n_estimators = 100
        mock_config.stopping_rounds = 10

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            # テスト実行
            save_single_model(
                self.mock_model, self.train_score, self.val_score,
                model_dir, "single_exp", self.feature_cols
            )

            # アサーション
            assert mock_json_dump.called
            assert mock_pickle_dump.called

            # JSON保存の内容確認
            saved_data = mock_json_dump.call_args[0][0]
            assert saved_data["experiment_name"] == "single_exp"
            assert saved_data["train_rmse"] == self.train_score
            assert saved_data["val_rmse"] == self.val_score
            assert saved_data["feature_count"] == 15

    @patch("src.modeling.train.config")
    def test_save_single_model_with_zero_scores(self, mock_config):
        """スコアが0の場合のテスト"""
        mock_config.objective = "regression"
        mock_config.metric = "rmse"
        mock_config.num_leaves = 31
        mock_config.learning_rate = 0.1
        mock_config.feature_fraction = 0.8
        mock_config.n_estimators = 100
        mock_config.stopping_rounds = 10

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            # スコアが0でもエラーが発生しないことを確認
            save_single_model(
                self.mock_model, 0.0, 0.0,
                model_dir, "zero_exp", self.feature_cols
            )


# フィクスチャとヘルパー関数
@pytest.fixture
def sample_training_data():
    """訓練用データのフィクスチャ"""
    np.random.seed(42)
    X = pd.DataFrame({
        "RhythmScore": np.random.uniform(30, 80, 50),
        "AudioLoudness": np.random.uniform(10, 50, 50),
        "Energy": np.random.uniform(20, 80, 50),
        "MoodScore": np.random.uniform(25, 75, 50),
    })
    # ターゲットに相関を持たせる
    y = pd.Series(
        X["RhythmScore"] * 1.5 +
        X["Energy"] * 0.8 +
        np.random.normal(0, 5, 50) + 100
    )
    return X, y


@pytest.fixture
def sample_config():
    """設定のフィクスチャ"""
    config = MagicMock()
    config.random_state = 42
    config.objective = "regression"
    config.metric = "rmse"
    config.num_leaves = 31
    config.learning_rate = 0.1
    config.feature_fraction = 0.8
    config.n_estimators = 100
    config.stopping_rounds = 10
    config.log_evaluation = 100
    config.target = "BeatsPerMinute"
    return config


def test_integration_training_pipeline(sample_training_data, sample_config):
    """訓練パイプライン統合テスト"""
    X, y = sample_training_data

    with patch("src.modeling.train.config", sample_config):
        with patch("src.modeling.train.lgb.train") as mock_lgb_train:
            # LightGBMモデルのモック
            mock_model = MagicMock()
            mock_model.best_iteration = 50
            mock_model.predict.return_value = np.random.uniform(100, 200, 10)
            mock_lgb_train.return_value = mock_model

            # クロスバリデーション実行
            cv_scores, models = train_with_cross_validation(X, y, n_folds=5)

            # 基本的な整合性チェック
            assert len(cv_scores) == 5
            assert len(models) == 5
            assert all(isinstance(score, float) for score in cv_scores)
            assert all(score > 0 for score in cv_scores)

            # 保存機能テスト
            with tempfile.TemporaryDirectory() as temp_dir:
                save_cv_results(cv_scores, models, Path(temp_dir), "integration_test", list(X.columns))