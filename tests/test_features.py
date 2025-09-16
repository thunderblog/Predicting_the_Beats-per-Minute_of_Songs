"""
TICKET-006: src/features.py用ユニットテスト
特徴量エンジニアリング機能のテストケース
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# プロジェクトルートをパスに追加
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.features import (
    create_interaction_features,
    create_duration_features,
    create_statistical_features,
    select_features,
    scale_features,
)


class TestCreateInteractionFeatures:
    """交互作用特徴量作成機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        self.df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "RhythmScore": [50.0, 60.0, 70.0, 80.0],
            "Energy": [30.0, 40.0, 50.0, 60.0],
            "AudioLoudness": [20.0, 25.0, 30.0, 35.0],
            "VocalContent": [10.0, 15.0, 20.0, 25.0],
            "AcousticQuality": [40.0, 45.0, 50.0, 55.0],
            "InstrumentalScore": [35.0, 40.0, 45.0, 50.0],
            "LivePerformanceLikelihood": [25.0, 30.0, 35.0, 40.0],
            "MoodScore": [45.0, 50.0, 55.0, 60.0],
            "BeatsPerMinute": [120, 130, 140, 150]
        })

    def test_create_interaction_features_basic(self):
        """基本的な交互作用特徴量作成テスト"""
        result = create_interaction_features(self.df)

        # 元の特徴量が保持されている
        assert all(col in result.columns for col in self.df.columns)

        # 新しい交互作用特徴量が追加されている
        expected_new_features = [
            "rhythm_energy_product",
            "rhythm_energy_ratio",
            "loudness_vocal_product",
            "acoustic_instrumental_ratio",
            "live_mood_product",
            "energy_mood_product",
            "rhythm_mood_energy"
        ]
        assert all(feature in result.columns for feature in expected_new_features)

        # 計算が正しく行われている
        expected_rhythm_energy_product = self.df["RhythmScore"] * self.df["Energy"]
        pd.testing.assert_series_equal(
            result["rhythm_energy_product"],
            expected_rhythm_energy_product,
            check_names=False
        )

    def test_create_interaction_features_division_by_zero(self):
        """ゼロ除算対策のテスト"""
        # Energyが0のケース
        df_zero = self.df.copy()
        df_zero.loc[0, "Energy"] = 0.0

        result = create_interaction_features(df_zero)

        # ゼロ除算対策で1e-8が足されているので有限値になる
        assert not result["rhythm_energy_ratio"].isna().any()
        assert np.isfinite(result["rhythm_energy_ratio"]).all()

    def test_create_interaction_features_empty_df(self):
        """空のデータフレームでのテスト"""
        empty_df = pd.DataFrame(columns=self.df.columns)
        result = create_interaction_features(empty_df)

        # 新しい特徴量が追加されているが、行数は0
        assert len(result) == 0
        assert "rhythm_energy_product" in result.columns


class TestCreateDurationFeatures:
    """時間特徴量作成機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        self.df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "TrackDurationMs": [120000, 180000, 240000, 300000, 360000],  # 2分, 3分, 4分, 5分, 6分
            "BeatsPerMinute": [120, 130, 140, 150, 160]
        })

    def test_create_duration_features_basic(self):
        """基本的な時間特徴量作成テスト"""
        result = create_duration_features(self.df)

        # 元の特徴量が保持されている
        assert all(col in result.columns for col in self.df.columns)

        # 新しい時間特徴量が追加されている
        expected_features = [
            "track_duration_seconds",
            "track_duration_minutes",
            "is_short_track",
            "is_long_track"
        ]
        assert all(feature in result.columns for feature in expected_features)

        # 時間変換の計算が正しい
        expected_seconds = self.df["TrackDurationMs"] / 1000
        pd.testing.assert_series_equal(
            result["track_duration_seconds"],
            expected_seconds,
            check_names=False
        )

        expected_minutes = self.df["TrackDurationMs"] / (1000 * 60)
        pd.testing.assert_series_equal(
            result["track_duration_minutes"],
            expected_minutes,
            check_names=False
        )

    def test_create_duration_features_categories(self):
        """時間カテゴリ分類のテスト"""
        result = create_duration_features(self.df)

        # 短いトラック判定（3分未満）
        assert result.loc[0, "is_short_track"] == 1  # 2分
        assert result.loc[1, "is_short_track"] == 0  # 3分

        # 長いトラック判定（5分超）
        assert result.loc[3, "is_long_track"] == 0  # 5分
        assert result.loc[4, "is_long_track"] == 1  # 6分

    def test_create_duration_features_one_hot_encoding(self):
        """時間カテゴリのワンホットエンコーディングテスト"""
        result = create_duration_features(self.df)

        # ワンホットエンコード列が存在する
        duration_columns = [col for col in result.columns if col.startswith("duration_")]
        assert len(duration_columns) > 0

        # 各行でワンホットエンコードの合計が1になる
        duration_sum = result[duration_columns].sum(axis=1)
        assert (duration_sum == 1).all()

    def test_create_duration_features_edge_cases(self):
        """境界値でのテスト"""
        edge_df = pd.DataFrame({
            "id": [1, 2, 3],
            "TrackDurationMs": [179999, 180000, 180001],  # 3分の境界値
            "BeatsPerMinute": [120, 130, 140]
        })

        result = create_duration_features(edge_df)

        # 境界値での分類が正しい
        assert result.loc[0, "is_short_track"] == 1  # 179999ms < 180000ms
        assert result.loc[1, "is_short_track"] == 0  # 180000ms >= 180000ms
        assert result.loc[2, "is_short_track"] == 0  # 180001ms > 180000ms


class TestCreateStatisticalFeatures:
    """統計的特徴量作成機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        self.df = pd.DataFrame({
            "id": [1, 2, 3],
            "RhythmScore": [50.0, 60.0, 70.0],
            "AudioLoudness": [20.0, 30.0, 40.0],
            "VocalContent": [10.0, 20.0, 30.0],
            "AcousticQuality": [40.0, 50.0, 60.0],
            "InstrumentalScore": [30.0, 40.0, 50.0],
            "LivePerformanceLikelihood": [25.0, 35.0, 45.0],
            "MoodScore": [45.0, 55.0, 65.0],
            "Energy": [35.0, 45.0, 55.0],
            "BeatsPerMinute": [120, 130, 140]
        })

    def test_create_statistical_features_basic(self):
        """基本的な統計的特徴量作成テスト"""
        result = create_statistical_features(self.df)

        # 元の特徴量が保持されている
        assert all(col in result.columns for col in self.df.columns)

        # 新しい統計的特徴量が追加されている
        expected_features = [
            "total_score",
            "mean_score",
            "std_score",
            "min_score",
            "max_score",
            "range_score"
        ]
        assert all(feature in result.columns for feature in expected_features)

    def test_create_statistical_features_calculations(self):
        """統計計算の正確性テスト"""
        result = create_statistical_features(self.df)

        numerical_cols = [
            "RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality",
            "InstrumentalScore", "LivePerformanceLikelihood", "MoodScore", "Energy"
        ]

        # 合計値の確認
        expected_total = self.df[numerical_cols].sum(axis=1)
        pd.testing.assert_series_equal(
            result["total_score"],
            expected_total,
            check_names=False
        )

        # 平均値の確認
        expected_mean = self.df[numerical_cols].mean(axis=1)
        pd.testing.assert_series_equal(
            result["mean_score"],
            expected_mean,
            check_names=False
        )

        # レンジの確認
        expected_range = self.df[numerical_cols].max(axis=1) - self.df[numerical_cols].min(axis=1)
        pd.testing.assert_series_equal(
            result["range_score"],
            expected_range,
            check_names=False
        )

    def test_create_statistical_features_with_nan(self):
        """欠損値を含むデータでのテスト"""
        df_with_nan = self.df.copy()
        df_with_nan.loc[0, "RhythmScore"] = np.nan

        result = create_statistical_features(df_with_nan)

        # NaNがある行での統計計算
        assert not np.isnan(result.loc[0, "total_score"])  # pandasのsum()はNaNを無視
        assert result.loc[0, "mean_score"] > 0  # 平均も計算される


class TestSelectFeatures:
    """特徴量選択機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            f"feature_{i}": np.random.randn(100) for i in range(20)
        })
        # ターゲットと相関のある特徴量を作成
        self.y_train = (
            self.X_train["feature_0"] * 2 +
            self.X_train["feature_1"] * 1.5 +
            np.random.randn(100) * 0.1
        )
        self.X_val = pd.DataFrame({
            f"feature_{i}": np.random.randn(50) for i in range(20)
        })

    def test_select_features_kbest(self):
        """F統計量による特徴量選択テスト"""
        X_train_selected, X_val_selected = select_features(
            self.X_train, self.y_train, self.X_val, method="kbest", k=5
        )

        # 選択された特徴量数が正しい
        assert X_train_selected.shape[1] == 5
        assert X_val_selected.shape[1] == 5

        # 同じ特徴量が選択されている
        assert list(X_train_selected.columns) == list(X_val_selected.columns)

        # 行数は変わらない
        assert X_train_selected.shape[0] == self.X_train.shape[0]
        assert X_val_selected.shape[0] == self.X_val.shape[0]

    def test_select_features_mutual_info(self):
        """相互情報量による特徴量選択テスト"""
        X_train_selected, X_val_selected = select_features(
            self.X_train, self.y_train, self.X_val, method="mutual_info", k=8
        )

        assert X_train_selected.shape[1] == 8
        assert X_val_selected.shape[1] == 8

    def test_select_features_correlation(self):
        """相関係数による特徴量選択テスト"""
        X_train_selected, X_val_selected = select_features(
            self.X_train, self.y_train, self.X_val, method="correlation", k=3
        )

        assert X_train_selected.shape[1] == 3
        assert X_val_selected.shape[1] == 3

        # feature_0とfeature_1が選ばれる可能性が高い（相関が強いため）
        selected_features = list(X_train_selected.columns)
        assert "feature_0" in selected_features or "feature_1" in selected_features

    def test_select_features_combined(self):
        """組み合わせ手法による特徴量選択テスト"""
        X_train_selected, X_val_selected = select_features(
            self.X_train, self.y_train, self.X_val, method="combined", k=6
        )

        assert X_train_selected.shape[1] == 6
        assert X_val_selected.shape[1] == 6

    def test_select_features_without_validation(self):
        """検証データなしでの特徴量選択テスト"""
        X_train_selected, X_val_selected = select_features(
            self.X_train, self.y_train, None, method="kbest", k=4
        )

        assert X_train_selected.shape[1] == 4
        assert X_val_selected is None

    def test_select_features_unknown_method(self):
        """未知の手法でのテスト"""
        X_train_selected, X_val_selected = select_features(
            self.X_train, self.y_train, self.X_val, method="unknown", k=5
        )

        # 元の特徴量がそのまま返される
        assert X_train_selected.shape[1] == self.X_train.shape[1]
        assert X_val_selected.shape[1] == self.X_val.shape[1]


class TestScaleFeatures:
    """特徴量スケーリング機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        self.X_train = pd.DataFrame({
            "feature_1": [1.0, 2.0, 3.0, 4.0],
            "feature_2": [10.0, 20.0, 30.0, 40.0],
            "feature_3": [100.0, 200.0, 300.0, 400.0]
        })
        self.X_val = pd.DataFrame({
            "feature_1": [1.5, 2.5],
            "feature_2": [15.0, 25.0],
            "feature_3": [150.0, 250.0]
        })
        self.X_test = pd.DataFrame({
            "feature_1": [5.0, 6.0],
            "feature_2": [50.0, 60.0],
            "feature_3": [500.0, 600.0]
        })

    def test_scale_features_standard(self):
        """標準化スケーリングテスト"""
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            self.X_train, self.X_val, self.X_test, scaler_type="standard"
        )

        # スケーラが正しい種類
        assert isinstance(scaler, StandardScaler)

        # 訓練データの平均が約0、標準偏差が約1
        assert np.allclose(X_train_scaled.mean(), 0, atol=1e-10)
        assert np.allclose(X_train_scaled.std(ddof=0), 1, atol=1e-10)

        # 元の形状が保持されている
        assert X_train_scaled.shape == self.X_train.shape
        assert X_val_scaled.shape == self.X_val.shape
        assert X_test_scaled.shape == self.X_test.shape

        # 列名とインデックスが保持されている
        assert list(X_train_scaled.columns) == list(self.X_train.columns)
        assert list(X_train_scaled.index) == list(self.X_train.index)

    def test_scale_features_robust(self):
        """ロバストスケーリングテスト"""
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            self.X_train, self.X_val, self.X_test, scaler_type="robust"
        )

        assert isinstance(scaler, RobustScaler)
        assert X_train_scaled.shape == self.X_train.shape

    def test_scale_features_minmax(self):
        """MinMaxスケーリングテスト"""
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            self.X_train, self.X_val, self.X_test, scaler_type="minmax"
        )

        assert isinstance(scaler, MinMaxScaler)

        # MinMaxスケーリング後の値が0-1の範囲内
        assert X_train_scaled.min().min() >= 0
        assert X_train_scaled.max().max() <= 1

    def test_scale_features_without_optional_data(self):
        """オプションデータなしでのスケーリングテスト"""
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            self.X_train, scaler_type="standard"
        )

        assert X_train_scaled is not None
        assert X_val_scaled is None
        assert X_test_scaled is None
        assert isinstance(scaler, StandardScaler)

    def test_scale_features_unknown_scaler(self):
        """未知のスケーラでのテスト"""
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            self.X_train, self.X_val, self.X_test, scaler_type="unknown"
        )

        # デフォルトでStandardScalerが使われる
        assert isinstance(scaler, StandardScaler)


# フィクスチャとヘルパー関数
@pytest.fixture
def sample_audio_data():
    """オーディオデータのフィクスチャ"""
    return pd.DataFrame({
        "id": range(1, 21),
        "RhythmScore": np.random.uniform(30, 80, 20),
        "AudioLoudness": np.random.uniform(10, 50, 20),
        "VocalContent": np.random.uniform(0, 40, 20),
        "AcousticQuality": np.random.uniform(20, 70, 20),
        "InstrumentalScore": np.random.uniform(15, 60, 20),
        "LivePerformanceLikelihood": np.random.uniform(5, 45, 20),
        "MoodScore": np.random.uniform(25, 75, 20),
        "TrackDurationMs": np.random.uniform(120000, 400000, 20),
        "Energy": np.random.uniform(20, 80, 20),
        "BeatsPerMinute": np.random.uniform(80, 200, 20)
    })


def test_integration_feature_engineering_pipeline(sample_audio_data):
    """特徴量エンジニアリングパイプライン統合テスト"""
    # 元のデータ
    original_shape = sample_audio_data.shape

    # 交互作用特徴量作成
    interaction_features = create_interaction_features(sample_audio_data)
    assert interaction_features.shape[1] > original_shape[1]

    # 時間特徴量作成
    duration_features = create_duration_features(interaction_features)
    assert duration_features.shape[1] > interaction_features.shape[1]

    # 統計的特徴量作成
    statistical_features = create_statistical_features(duration_features)
    assert statistical_features.shape[1] > duration_features.shape[1]

    # 特徴量行列の準備
    feature_cols = [col for col in statistical_features.columns if col not in ["id", "BeatsPerMinute"]]
    X = statistical_features[feature_cols]
    y = statistical_features["BeatsPerMinute"]

    # 特徴量選択
    X_selected, _ = select_features(X, y, method="kbest", k=10)
    assert X_selected.shape[1] == 10

    # スケーリング
    X_scaled, _, _, scaler = scale_features(X_selected)
    assert X_scaled.shape == X_selected.shape
    assert isinstance(scaler, StandardScaler)