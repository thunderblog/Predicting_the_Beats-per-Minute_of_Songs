"""
TICKET-006: データパイプライン統合テスト
データ処理→特徴量エンジニアリング→モデル訓練→予測の全体フローをテスト
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

import numpy as np
import pandas as pd
import pytest

# プロジェクトルートをパスに追加
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import (
    load_raw_data,
    validate_data,
    analyze_target_distribution,
    create_feature_summary,
    split_train_validation,
    save_processed_data,
)
from src.features import (
    create_interaction_features,
    create_duration_features,
    create_statistical_features,
    select_features,
    scale_features,
)
from src.modeling.train import (
    train_with_cross_validation,
    save_cv_results,
)
from src.modeling.predict import (
    make_ensemble_predictions,
    process_predictions,
    save_submission,
)


class TestDataProcessingPipeline:
    """データ処理パイプラインの統合テスト"""

    def setup_method(self):
        """テスト用データセットの準備"""
        # Kaggle BPMコンペに近いリアルなデータを作成
        np.random.seed(42)

        self.train_data = pd.DataFrame({
            "id": range(1, 201),
            "RhythmScore": np.random.uniform(30, 80, 200),
            "AudioLoudness": np.random.uniform(10, 50, 200),
            "VocalContent": np.random.uniform(0, 40, 200),
            "AcousticQuality": np.random.uniform(20, 70, 200),
            "InstrumentalScore": np.random.uniform(15, 60, 200),
            "LivePerformanceLikelihood": np.random.uniform(5, 45, 200),
            "MoodScore": np.random.uniform(25, 75, 200),
            "TrackDurationMs": np.random.uniform(120000, 400000, 200),
            "Energy": np.random.uniform(20, 80, 200),
        })

        # リアルなBPM値を特徴量の組み合わせで生成
        self.train_data["BeatsPerMinute"] = (
            self.train_data["RhythmScore"] * 1.2 +
            self.train_data["Energy"] * 0.8 +
            self.train_data["MoodScore"] * 0.3 +
            np.random.normal(0, 10, 200) + 80
        ).clip(60, 200)  # BPMの現実的な範囲

        self.test_data = pd.DataFrame({
            "id": range(201, 251),
            "RhythmScore": np.random.uniform(30, 80, 50),
            "AudioLoudness": np.random.uniform(10, 50, 50),
            "VocalContent": np.random.uniform(0, 40, 50),
            "AcousticQuality": np.random.uniform(20, 70, 50),
            "InstrumentalScore": np.random.uniform(15, 60, 50),
            "LivePerformanceLikelihood": np.random.uniform(5, 45, 50),
            "MoodScore": np.random.uniform(25, 75, 50),
            "TrackDurationMs": np.random.uniform(120000, 400000, 50),
            "Energy": np.random.uniform(20, 80, 50),
        })

        self.sample_submission = pd.DataFrame({
            "id": range(201, 251),
            "BeatsPerMinute": [0] * 50
        })

    @patch("src.dataset.config")
    def test_full_data_processing_pipeline(self, mock_config):
        """データ処理パイプライン全体のテスト"""
        # configのモック設定
        mock_config.target = "BeatsPerMinute"
        mock_config.features = [
            "RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality",
            "InstrumentalScore", "LivePerformanceLikelihood", "MoodScore",
            "TrackDurationMs", "Energy"
        ]
        mock_config.test_size = 0.2
        mock_config.random_state = 42

        # Step 1: データ検証
        validate_data(self.train_data, self.test_data)

        # Step 2: ターゲット分析
        analyze_target_distribution(self.train_data)

        # Step 3: 特徴量要約
        feature_summary = create_feature_summary(self.train_data)
        assert len(feature_summary) > 0

        # Step 4: データ分割
        train_split, val_split = split_train_validation(self.train_data)

        # データ分割の検証
        assert len(train_split) + len(val_split) == len(self.train_data)
        assert len(train_split) == 160  # 80% of 200
        assert len(val_split) == 40     # 20% of 200

        # 分割後のデータにターゲットが含まれている
        assert mock_config.target in train_split.columns
        assert mock_config.target in val_split.columns

    def test_feature_engineering_pipeline(self):
        """特徴量エンジニアリング全体のパイプラインテスト"""
        # 元の特徴量数を記録
        original_features = len(self.train_data.columns) - 2  # id, BeatsPerMinuteを除く

        # Step 1: 交互作用特徴量
        interaction_data = create_interaction_features(self.train_data)
        assert interaction_data.shape[1] > self.train_data.shape[1]

        # Step 2: 時間特徴量
        duration_data = create_duration_features(interaction_data)
        assert duration_data.shape[1] > interaction_data.shape[1]

        # Step 3: 統計的特徴量
        statistical_data = create_statistical_features(duration_data)
        assert statistical_data.shape[1] > duration_data.shape[1]

        # 最終的に特徴量が大幅に増加している
        final_features = len([col for col in statistical_data.columns if col not in ["id", "BeatsPerMinute"]])
        assert final_features > original_features * 2  # 少なくとも2倍以上

        return statistical_data

    def test_feature_selection_and_scaling_pipeline(self):
        """特徴量選択とスケーリングのパイプラインテスト"""
        # 特徴量エンジニアリング後のデータを取得
        enhanced_data = self.test_feature_engineering_pipeline()

        # 特徴量とターゲットの分離
        feature_cols = [col for col in enhanced_data.columns if col not in ["id", "BeatsPerMinute"]]
        X = enhanced_data[feature_cols]
        y = enhanced_data["BeatsPerMinute"]

        # データ分割
        train_size = int(0.8 * len(X))
        X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]

        # Step 1: 特徴量選択
        n_selected_features = 15
        X_train_selected, X_val_selected = select_features(
            X_train, y_train, X_val, method="kbest", k=n_selected_features
        )

        assert X_train_selected.shape[1] == n_selected_features
        assert X_val_selected.shape[1] == n_selected_features

        # Step 2: スケーリング
        X_train_scaled, X_val_scaled, _, scaler = scale_features(
            X_train_selected, X_val_selected, scaler_type="standard"
        )

        # スケーリング後の統計チェック
        assert np.allclose(X_train_scaled.mean(), 0, atol=1e-10)
        assert np.allclose(X_train_scaled.std(), 1, atol=1e-10)

        return X_train_scaled, X_val_scaled, y_train, y_val


class TestModelTrainingPipeline:
    """モデル訓練パイプラインの統合テスト"""

    def setup_method(self):
        """テスト用データの準備"""
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            f"feature_{i}": np.random.randn(100) for i in range(10)
        })
        self.y_train = pd.Series(
            self.X_train.iloc[:, :3].sum(axis=1) * 20 +
            np.random.normal(0, 5, 100) + 120
        )

    @patch("src.modeling.train.config")
    @patch("src.modeling.train.lgb.train")
    def test_training_and_saving_pipeline(self, mock_lgb_train, mock_config):
        """モデル訓練と保存のパイプラインテスト"""
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
        mock_models = []
        for fold in range(5):
            mock_model = MagicMock()
            mock_model.best_iteration = 50
            mock_model.predict.return_value = self.y_train.iloc[:20].values + np.random.normal(0, 1, 20)
            mock_models.append(mock_model)

        mock_lgb_train.side_effect = mock_models

        # Step 1: クロスバリデーション訓練
        cv_scores, trained_models = train_with_cross_validation(
            self.X_train, self.y_train, n_folds=5
        )

        # 訓練結果の検証
        assert len(cv_scores) == 5
        assert len(trained_models) == 5
        assert all(score > 0 for score in cv_scores)

        # Step 2: モデル保存
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            feature_cols = list(self.X_train.columns)

            save_cv_results(cv_scores, trained_models, model_dir, "test_pipeline", feature_cols)

            # 保存されたファイルの確認
            json_files = list(model_dir.glob("*_cv_results_*.json"))
            model_files = list(model_dir.glob("*_fold_*_*.pkl"))

            assert len(json_files) == 1
            assert len(model_files) == 5


class TestPredictionPipeline:
    """予測パイプラインの統合テスト"""

    def setup_method(self):
        """テスト用データの準備"""
        self.test_data = pd.DataFrame({
            "id": range(1, 26),
            "feature_1": np.random.randn(25),
            "feature_2": np.random.randn(25),
            "feature_3": np.random.randn(25),
        })

        # モックモデルの準備
        self.mock_models = []
        for i in range(3):
            mock_model = MagicMock()
            mock_model.best_iteration = 50
            # 現実的なBPM予測値
            mock_model.predict.return_value = np.random.uniform(80, 200, 25)
            self.mock_models.append(mock_model)

    @patch("src.modeling.predict.config")
    def test_prediction_to_submission_pipeline(self, mock_config):
        """予測から提出ファイル生成までのパイプラインテスト"""
        mock_config.target = "BeatsPerMinute"

        feature_cols = ["feature_1", "feature_2", "feature_3"]

        # Step 1: アンサンブル予測
        predictions = make_ensemble_predictions(self.mock_models, self.test_data, feature_cols)

        assert len(predictions) == len(self.test_data)
        assert isinstance(predictions, np.ndarray)

        # Step 2: 予測値後処理
        processed_predictions = process_predictions(predictions)

        # 後処理後の値が妥当な範囲内
        assert all(30 <= pred <= 300 for pred in processed_predictions)

        # Step 3: 提出ファイル保存
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "submission.csv"

            save_submission(self.test_data["id"], processed_predictions, output_path)

            # 保存されたファイルの検証
            assert output_path.exists()

            submission_df = pd.read_csv(output_path)
            assert len(submission_df) == len(self.test_data)
            assert list(submission_df.columns) == ["id", "BeatsPerMinute"]
            assert submission_df["id"].equals(self.test_data["id"])


class TestEndToEndPipeline:
    """エンドツーエンドパイプラインの統合テスト"""

    @patch("src.modeling.train.config")
    @patch("src.modeling.predict.config")
    @patch("src.dataset.config")
    @patch("src.modeling.train.lgb.train")
    def test_complete_ml_pipeline(
        self, mock_lgb_train, mock_dataset_config,
        mock_predict_config, mock_train_config
    ):
        """完全なMLパイプラインの統合テスト"""
        # 各configのモック設定を統一
        for mock_config in [mock_dataset_config, mock_train_config, mock_predict_config]:
            mock_config.target = "BeatsPerMinute"
            mock_config.features = ["RhythmScore", "AudioLoudness", "Energy"]
            mock_config.test_size = 0.2
            mock_config.random_state = 42
            mock_config.objective = "regression"
            mock_config.metric = "rmse"
            mock_config.num_leaves = 31
            mock_config.learning_rate = 0.1
            mock_config.feature_fraction = 0.8
            mock_config.n_estimators = 100
            mock_config.stopping_rounds = 10
            mock_config.log_evaluation = 100

        # リアルなデータセット作成
        np.random.seed(42)
        full_train_data = pd.DataFrame({
            "id": range(1, 101),
            "RhythmScore": np.random.uniform(30, 80, 100),
            "AudioLoudness": np.random.uniform(10, 50, 100),
            "Energy": np.random.uniform(20, 80, 100),
        })

        full_train_data["BeatsPerMinute"] = (
            full_train_data["RhythmScore"] * 1.2 +
            full_train_data["Energy"] * 0.8 +
            np.random.normal(0, 5, 100) + 100
        ).clip(80, 180)

        test_data = pd.DataFrame({
            "id": range(101, 126),
            "RhythmScore": np.random.uniform(30, 80, 25),
            "AudioLoudness": np.random.uniform(10, 50, 25),
            "Energy": np.random.uniform(20, 80, 25),
        })

        # LightGBMモデルのモック
        mock_models = []
        for fold in range(3):
            mock_model = MagicMock()
            mock_model.best_iteration = 50
            mock_model.predict.return_value = np.random.uniform(100, 150, len(test_data))
            mock_models.append(mock_model)

        mock_lgb_train.side_effect = mock_models

        # Phase 1: データ処理
        validate_data(full_train_data, test_data)
        train_split, val_split = split_train_validation(full_train_data)

        # Phase 2: 特徴量エンジニアリング
        enhanced_train = create_interaction_features(train_split)
        enhanced_train = create_statistical_features(enhanced_train)

        enhanced_test = create_interaction_features(test_data)
        enhanced_test = create_statistical_features(enhanced_test)

        # Phase 3: モデル訓練
        feature_cols = [col for col in enhanced_train.columns if col not in ["id", "BeatsPerMinute"]]
        X_train = enhanced_train[feature_cols]
        y_train = enhanced_train["BeatsPerMinute"]

        cv_scores, trained_models = train_with_cross_validation(X_train, y_train, n_folds=3)

        # Phase 4: 予測と提出
        predictions = make_ensemble_predictions(trained_models, enhanced_test, feature_cols)
        processed_predictions = process_predictions(predictions)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "final_submission.csv"
            save_submission(test_data["id"], processed_predictions, output_path)

            # 最終結果の検証
            assert output_path.exists()
            submission_df = pd.read_csv(output_path)
            assert len(submission_df) == len(test_data)

        # パイプライン全体の統計確認
        assert len(cv_scores) == 3
        assert all(score > 0 for score in cv_scores)
        assert len(predictions) == len(test_data)
        assert all(30 <= pred <= 300 for pred in processed_predictions)


# フィクスチャとヘルパー関数
@pytest.fixture
def kaggle_bpm_dataset():
    """Kaggle BPMコンペのリアルなデータセットフィクスチャ"""
    np.random.seed(42)

    # より多くのサンプルで現実的なデータを生成
    n_samples = 500

    data = pd.DataFrame({
        "id": range(1, n_samples + 1),
        "RhythmScore": np.random.uniform(20, 90, n_samples),
        "AudioLoudness": np.random.uniform(5, 60, n_samples),
        "VocalContent": np.random.uniform(0, 50, n_samples),
        "AcousticQuality": np.random.uniform(10, 80, n_samples),
        "InstrumentalScore": np.random.uniform(10, 70, n_samples),
        "LivePerformanceLikelihood": np.random.uniform(0, 50, n_samples),
        "MoodScore": np.random.uniform(20, 80, n_samples),
        "TrackDurationMs": np.random.uniform(90000, 480000, n_samples),
        "Energy": np.random.uniform(10, 90, n_samples),
    })

    # 複雑な非線形関係でBPMを生成
    data["BeatsPerMinute"] = (
        data["RhythmScore"] * 1.5 +
        data["Energy"] * 1.2 +
        data["MoodScore"] * 0.4 +
        (data["TrackDurationMs"] / 1000) * -0.01 +
        np.random.normal(0, 15, n_samples) + 90
    ).clip(50, 250)

    return data


def test_realistic_pipeline_performance(kaggle_bpm_dataset):
    """リアルなデータセットでのパイプライン性能テスト"""
    with patch("src.dataset.config") as mock_config:
        mock_config.target = "BeatsPerMinute"
        mock_config.features = [
            "RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality",
            "InstrumentalScore", "LivePerformanceLikelihood", "MoodScore",
            "TrackDurationMs", "Energy"
        ]
        mock_config.test_size = 0.2
        mock_config.random_state = 42

        # データ分割
        train_split, val_split = split_train_validation(kaggle_bpm_dataset)

        # 特徴量エンジニアリング
        enhanced_data = create_interaction_features(train_split)
        enhanced_data = create_duration_features(enhanced_data)
        enhanced_data = create_statistical_features(enhanced_data)

        # 特徴量選択とスケーリング
        feature_cols = [col for col in enhanced_data.columns if col not in ["id", "BeatsPerMinute"]]
        X = enhanced_data[feature_cols]
        y = enhanced_data["BeatsPerMinute"]

        X_selected, _ = select_features(X, y, method="correlation", k=20)
        X_scaled, _, _, _ = scale_features(X_selected)

        # 最終的な特徴量の妥当性チェック
        assert X_scaled.shape[1] == 20
        assert not X_scaled.isna().any().any()
        assert np.allclose(X_scaled.mean(), 0, atol=1e-10)

        # ターゲットの分布チェック
        assert y.min() >= 50
        assert y.max() <= 250
        assert y.std() > 0