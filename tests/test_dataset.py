"""
TICKET-006: src/dataset.py用ユニットテスト
データセット処理機能のテストケース
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# プロジェクトルートをパスに追加
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import (
    analyze_target_distribution,
    create_feature_summary,
    load_raw_data,
    save_processed_data,
    split_train_validation,
    validate_data,
)


class TestLoadRawData:
    """生データ読み込み機能のテスト"""

    @patch("src.dataset.pd.read_csv")
    @patch("src.dataset.config")
    def test_load_raw_data_success(self, mock_config, mock_read_csv):
        """正常な生データ読み込みテスト"""
        # テストデータの準備
        mock_train_df = pd.DataFrame({"col1": [1, 2], "BeatsPerMinute": [120, 140]})
        mock_test_df = pd.DataFrame({"col1": [3, 4]})
        mock_sample_df = pd.DataFrame({"id": [1, 2], "BeatsPerMinute": [0, 0]})

        mock_read_csv.side_effect = [mock_train_df, mock_test_df, mock_sample_df]

        # configのモック設定
        mock_config.get_raw_path.side_effect = lambda x: f"path/to/{x}.csv"

        # テスト実行
        train_df, test_df, sample_df = load_raw_data()

        # アサーション
        assert len(train_df) == 2
        assert len(test_df) == 2
        assert len(sample_df) == 2
        assert mock_read_csv.call_count == 3

    @patch("src.dataset.pd.read_csv")
    @patch("src.dataset.config")
    def test_load_raw_data_file_not_found(self, mock_config, mock_read_csv):
        """ファイルが見つからない場合のテスト"""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        mock_config.get_raw_path.side_effect = lambda x: f"path/to/{x}.csv"

        with pytest.raises(FileNotFoundError):
            load_raw_data()


class TestValidateData:
    """データ品質検証機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        self.valid_train_df = pd.DataFrame({
            "id": [1, 2, 3],
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [10, 20, 30],
            "BeatsPerMinute": [120, 140, 160]
        })

        self.valid_test_df = pd.DataFrame({
            "id": [4, 5, 6],
            "feature1": [4.0, 5.0, 6.0],
            "feature2": [40, 50, 60]
        })

    @patch("src.dataset.config")
    def test_validate_data_success(self, mock_config):
        """正常なデータの検証テスト"""
        mock_config.target = "BeatsPerMinute"
        mock_config.features = ["feature1", "feature2"]

        # 正常なケースでは例外が発生しない
        validate_data(self.valid_train_df, self.valid_test_df)

    @patch("src.dataset.config")
    def test_validate_data_missing_values(self, mock_config):
        """欠損値があるデータの検証テスト"""
        mock_config.target = "BeatsPerMinute"
        mock_config.features = ["feature1", "feature2"]

        # 欠損値を含むデータ
        train_with_missing = self.valid_train_df.copy()
        train_with_missing.loc[0, "feature1"] = None

        # 警告は出るが例外は発生しない
        validate_data(train_with_missing, self.valid_test_df)

    @patch("src.dataset.config")
    def test_validate_data_feature_mismatch(self, mock_config):
        """特徴量の不整合があるデータの検証テスト"""
        mock_config.target = "BeatsPerMinute"
        mock_config.features = ["feature1", "feature2"]

        # テストデータから特徴量を削除
        test_missing_feature = self.valid_test_df.drop("feature2", axis=1)

        with pytest.raises(ValueError, match="訓練データとテストデータの特徴量が一致しません"):
            validate_data(self.valid_train_df, test_missing_feature)

    @patch("src.dataset.config")
    def test_validate_data_config_mismatch(self, mock_config):
        """設定ファイルと実データの特徴量不整合テスト"""
        mock_config.target = "BeatsPerMinute"
        mock_config.features = ["feature1", "feature3"]  # feature3はデータに存在しない

        with pytest.raises(ValueError, match="Configと実際のデータの特徴量が一致しません"):
            validate_data(self.valid_train_df, self.valid_test_df)


class TestAnalyzeTargetDistribution:
    """ターゲット変数分析機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        self.train_df = pd.DataFrame({
            "id": range(100),
            "feature1": range(100),
            "BeatsPerMinute": [120 + i * 0.5 for i in range(100)]  # 120から169.5まで
        })

    @patch("src.dataset.config")
    def test_analyze_target_distribution(self, mock_config):
        """ターゲット分析の正常実行テスト"""
        mock_config.target = "BeatsPerMinute"

        # 例外が発生しないことを確認
        analyze_target_distribution(self.train_df)

    @patch("src.dataset.config")
    def test_analyze_target_distribution_with_outliers(self, mock_config):
        """外れ値を含むターゲット分析テスト"""
        mock_config.target = "BeatsPerMinute"

        # 外れ値を追加
        outlier_df = self.train_df.copy()
        outlier_df.loc[len(outlier_df)] = {"id": 999, "feature1": 999, "BeatsPerMinute": 500}

        # 例外が発生しないことを確認
        analyze_target_distribution(outlier_df)


class TestSplitTrainValidation:
    """データ分割機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        self.train_df = pd.DataFrame({
            "id": range(100),
            "feature1": range(100),
            "BeatsPerMinute": range(100, 200)
        })

    @patch("src.dataset.config")
    def test_split_train_validation(self, mock_config):
        """データ分割の正常実行テスト"""
        mock_config.test_size = 0.2
        mock_config.random_state = 42
        mock_config.target = "BeatsPerMinute"

        train_split, val_split = split_train_validation(self.train_df)

        # 分割サイズの確認
        assert len(train_split) == 80
        assert len(val_split) == 20
        assert len(train_split) + len(val_split) == len(self.train_df)

        # 重複がないことを確認
        train_ids = set(train_split["id"])
        val_ids = set(val_split["id"])
        assert len(train_ids.intersection(val_ids)) == 0

    @patch("src.dataset.config")
    def test_split_train_validation_different_ratio(self, mock_config):
        """異なる分割比率でのテスト"""
        mock_config.test_size = 0.3
        mock_config.random_state = 42
        mock_config.target = "BeatsPerMinute"

        train_split, val_split = split_train_validation(self.train_df)

        # 分割サイズの確認（30%が検証データ）
        assert len(train_split) == 70
        assert len(val_split) == 30


class TestCreateFeatureSummary:
    """特徴量要約作成機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        self.df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "numeric_feature": [1.5, 2.5, None, 4.5],
            "int_feature": [10, 20, 30, 40],
            "string_feature": ["A", "B", "A", "C"],
        })

    def test_create_feature_summary(self):
        """特徴量要約作成の正常実行テスト"""
        summary_df = create_feature_summary(self.df)

        # 基本的な構造の確認
        assert len(summary_df) == 4  # 4つの特徴量
        expected_columns = ["data_type", "missing_count", "unique_count", "min_value", "max_value", "mean_value"]
        assert all(col in summary_df.columns for col in expected_columns)

        # 特定の特徴量の確認
        assert summary_df.loc["numeric_feature", "missing_count"] == 1
        assert summary_df.loc["int_feature", "missing_count"] == 0
        assert summary_df.loc["string_feature", "unique_count"] == 3

    def test_create_feature_summary_empty_df(self):
        """空のデータフレームでのテスト"""
        empty_df = pd.DataFrame()

        # 空のデータフレームでは例外が発生することを期待
        with pytest.raises(ValueError):
            create_feature_summary(empty_df)

    def test_create_feature_summary_only_numeric(self):
        """数値特徴量のみのデータフレームテスト"""
        numeric_only_df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [10, 20, 30]
        })

        summary_df = create_feature_summary(numeric_only_df)

        # すべての特徴量で統計値が計算されている
        assert summary_df.loc["feature1", "mean_value"] == 2.0
        assert summary_df.loc["feature2", "mean_value"] == 20.0


class TestSaveProcessedData:
    """処理済みデータ保存機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        self.train_df = pd.DataFrame({"id": [1, 2], "feature1": [10, 20], "BeatsPerMinute": [120, 140]})
        self.val_df = pd.DataFrame({"id": [3, 4], "feature1": [30, 40], "BeatsPerMinute": [160, 180]})
        self.test_df = pd.DataFrame({"id": [5, 6], "feature1": [50, 60]})
        self.sample_df = pd.DataFrame({"id": [5, 6], "BeatsPerMinute": [0, 0]})

    @patch("src.dataset.config")
    def test_save_processed_data(self, mock_config):
        """データ保存の正常実行テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_config.processed_data_dir = temp_path
            mock_config.get_processed_path.side_effect = lambda x: temp_path / f"{x}.csv"

            # テスト実行
            save_processed_data(self.train_df, self.val_df, self.test_df, self.sample_df)

            # ファイルが作成されていることを確認
            assert (temp_path / "train.csv").exists()
            assert (temp_path / "validation.csv").exists()
            assert (temp_path / "test.csv").exists()
            assert (temp_path / "sample_submission.csv").exists()

            # 保存されたデータの内容確認
            saved_train = pd.read_csv(temp_path / "train.csv")
            assert len(saved_train) == 2
            assert "BeatsPerMinute" in saved_train.columns


# フィクスチャとヘルパー関数
@pytest.fixture
def sample_train_data():
    """テスト用訓練データのフィクスチャ"""
    return pd.DataFrame({
        "id": range(1, 101),
        "RhythmScore": [50 + i * 0.5 for i in range(100)],
        "AudioLoudness": [30 + i * 0.3 for i in range(100)],
        "VocalContent": [20 + i * 0.4 for i in range(100)],
        "BeatsPerMinute": [100 + i * 0.8 for i in range(100)]
    })


@pytest.fixture
def sample_test_data():
    """テスト用テストデータのフィクスチャ"""
    return pd.DataFrame({
        "id": range(101, 151),
        "RhythmScore": [60 + i * 0.5 for i in range(50)],
        "AudioLoudness": [40 + i * 0.3 for i in range(50)],
        "VocalContent": [30 + i * 0.4 for i in range(50)],
    })


def test_integration_data_processing_pipeline(sample_train_data, sample_test_data):
    """データ処理パイプライン統合テスト"""
    with patch("src.dataset.config") as mock_config:
        mock_config.target = "BeatsPerMinute"
        mock_config.features = ["RhythmScore", "AudioLoudness", "VocalContent"]
        mock_config.test_size = 0.2
        mock_config.random_state = 42

        # データ検証
        validate_data(sample_train_data, sample_test_data)

        # ターゲット分析
        analyze_target_distribution(sample_train_data)

        # 特徴量要約
        summary = create_feature_summary(sample_train_data)
        assert len(summary) > 0

        # データ分割
        train_split, val_split = split_train_validation(sample_train_data)
        assert len(train_split) + len(val_split) == len(sample_train_data)