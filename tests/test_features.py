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
    create_music_genre_features,
    analyze_feature_importance,
    compare_genre_features_to_bpm,
    select_features,
    scale_features,
    detect_multicollinearity,
    remove_correlated_features,
    evaluate_multicollinearity_impact,
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


class TestCreateMusicGenreFeatures:
    """音楽ジャンル推定特徴量作成機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        self.df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "RhythmScore": [80.0, 40.0, 60.0, 30.0],
            "Energy": [90.0, 20.0, 50.0, 70.0],
            "AudioLoudness": [30.0, 25.0, 35.0, 40.0],
            "VocalContent": [10.0, 80.0, 40.0, 20.0],
            "AcousticQuality": [20.0, 90.0, 60.0, 50.0],
            "InstrumentalScore": [30.0, 85.0, 55.0, 45.0],
            "LivePerformanceLikelihood": [60.0, 20.0, 40.0, 80.0],
            "MoodScore": [40.0, 90.0, 70.0, 30.0],
            "BeatsPerMinute": [140, 80, 110, 120]
        })

    def test_create_music_genre_features_basic(self):
        """基本的な音楽ジャンル特徴量作成テスト"""
        result = create_music_genre_features(self.df)

        # 元の特徴量が保持されている
        assert all(col in result.columns for col in self.df.columns)

        # 新しいジャンル特徴量が追加されている
        expected_genre_features = [
            "dance_genre_score",
            "acoustic_genre_score",
            "ballad_genre_score",
            "rock_genre_score",
            "electronic_genre_score",
            "ambient_genre_score"
        ]
        assert all(feature in result.columns for feature in expected_genre_features)

    def test_create_music_genre_features_calculations(self):
        """ジャンル特徴量計算の正確性テスト"""
        result = create_music_genre_features(self.df)

        # ダンス系特徴量: Energy × RhythmScore
        expected_dance = self.df["Energy"] * self.df["RhythmScore"]
        pd.testing.assert_series_equal(
            result["dance_genre_score"],
            expected_dance,
            check_names=False
        )

        # アコースティック系特徴量: AcousticQuality × InstrumentalScore
        expected_acoustic = self.df["AcousticQuality"] * self.df["InstrumentalScore"]
        pd.testing.assert_series_equal(
            result["acoustic_genre_score"],
            expected_acoustic,
            check_names=False
        )

        # バラード系特徴量: VocalContent × MoodScore
        expected_ballad = self.df["VocalContent"] * self.df["MoodScore"]
        pd.testing.assert_series_equal(
            result["ballad_genre_score"],
            expected_ballad,
            check_names=False
        )

        # ロック系特徴量: Energy × LivePerformanceLikelihood
        expected_rock = self.df["Energy"] * self.df["LivePerformanceLikelihood"]
        pd.testing.assert_series_equal(
            result["rock_genre_score"],
            expected_rock,
            check_names=False
        )

    def test_create_music_genre_features_negative_correlation(self):
        """負の相関特徴量のテスト"""
        result = create_music_genre_features(self.df)

        # エレクトロニック系: 低ボーカル × 高エネルギー
        max_vocal = self.df["VocalContent"].max()
        expected_electronic = (1 - self.df["VocalContent"] / (max_vocal + 1e-8)) * self.df["Energy"]
        pd.testing.assert_series_equal(
            result["electronic_genre_score"],
            expected_electronic,
            check_names=False
        )

        # アンビエント系: 低エネルギー × 高音響品質
        max_energy = self.df["Energy"].max()
        expected_ambient = (1 - self.df["Energy"] / (max_energy + 1e-8)) * self.df["AcousticQuality"]
        pd.testing.assert_series_equal(
            result["ambient_genre_score"],
            expected_ambient,
            check_names=False
        )

    def test_create_music_genre_features_edge_cases(self):
        """境界値・特殊ケースのテスト"""
        # 全ての値が0のケース
        edge_df = pd.DataFrame({
            "id": [1, 2],
            "RhythmScore": [0.0, 100.0],
            "Energy": [0.0, 100.0],
            "VocalContent": [0.0, 100.0],
            "AcousticQuality": [0.0, 100.0],
            "InstrumentalScore": [0.0, 100.0],
            "LivePerformanceLikelihood": [0.0, 100.0],
            "MoodScore": [0.0, 100.0],
            "BeatsPerMinute": [60, 180]
        })

        result = create_music_genre_features(edge_df)

        # 0の場合は0、最大値の場合は最大値になる
        assert result.loc[0, "dance_genre_score"] == 0.0  # 0 * 0
        assert result.loc[1, "dance_genre_score"] == 10000.0  # 100 * 100

        # ゼロ除算対策の確認
        assert np.isfinite(result["electronic_genre_score"]).all()
        assert np.isfinite(result["ambient_genre_score"]).all()

    def test_create_music_genre_features_empty_df(self):
        """空のデータフレームでのテスト"""
        empty_df = pd.DataFrame(columns=self.df.columns)
        result = create_music_genre_features(empty_df)

        # ジャンル特徴量が追加されているが、行数は0
        assert len(result) == 0
        assert "dance_genre_score" in result.columns
        assert "acoustic_genre_score" in result.columns


class TestAnalyzeFeatureImportance:
    """特徴量重要度分析機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        np.random.seed(42)
        self.X = pd.DataFrame({
            "dance_genre_score": np.random.uniform(0, 100, 50),
            "acoustic_genre_score": np.random.uniform(0, 100, 50),
            "ballad_genre_score": np.random.uniform(0, 100, 50),
            "rhythm_energy_product": np.random.uniform(0, 100, 50),
            "duration_seconds": np.random.uniform(120, 300, 50),
            "total_score": np.random.uniform(200, 800, 50),
            "other_feature": np.random.uniform(0, 100, 50)
        })
        # ターゲットとの相関を作成
        self.y = (
            self.X["dance_genre_score"] * 0.5 +
            self.X["acoustic_genre_score"] * 0.3 +
            np.random.randn(50) * 10
        )

    def test_analyze_feature_importance_all(self):
        """全特徴量重要度分析テスト"""
        importance_df = analyze_feature_importance(self.X, self.y, "all")

        # 結果の基本構造確認
        assert len(importance_df) == len(self.X.columns)
        expected_columns = [
            "feature_name", "correlation", "f_score", "mutual_info", "rf_importance",
            "correlation_normalized", "f_score_normalized", "mutual_info_normalized",
            "rf_importance_normalized", "average_importance"
        ]
        assert all(col in importance_df.columns for col in expected_columns)

        # 重要度順にソートされている
        assert importance_df["average_importance"].is_monotonic_decreasing

        # 正規化値が0-1の範囲内
        for col in ["correlation_normalized", "f_score_normalized", "mutual_info_normalized", "rf_importance_normalized"]:
            assert importance_df[col].min() >= 0
            assert importance_df[col].max() <= 1

    def test_analyze_feature_importance_genre_category(self):
        """ジャンル特徴量カテゴリ分析テスト"""
        importance_df = analyze_feature_importance(self.X, self.y, "genre")

        # ジャンル特徴量のみが選択されている
        genre_features = [col for col in self.X.columns if "genre_score" in col]
        assert len(importance_df) == len(genre_features)
        assert all(feature in genre_features for feature in importance_df["feature_name"])

    def test_analyze_feature_importance_empty_category(self):
        """存在しないカテゴリでのテスト"""
        importance_df = analyze_feature_importance(self.X, self.y, "nonexistent")

        # 空のDataFrameが返される
        assert len(importance_df) == 0


class TestCompareGenreFeaturesToBpm:
    """ジャンル特徴量とBPM関係分析機能のテスト"""

    def setup_method(self):
        """テストデータの準備"""
        np.random.seed(42)
        # ジャンル特徴量とBPMに明確な関係を作成
        self.X = pd.DataFrame({
            "dance_genre_score": [10, 20, 80, 90, 50, 60, 30, 40, 70, 75],
            "acoustic_genre_score": [90, 80, 20, 10, 60, 50, 70, 65, 30, 25],
            "ballad_genre_score": [85, 75, 15, 25, 55, 45, 65, 70, 35, 30],
            "other_feature": np.random.uniform(0, 100, 10)
        })
        # ダンス系は高BPM、アコースティック系は低BPMに設定
        self.y = pd.Series([
            70, 75, 140, 150, 110, 115, 90, 95, 130, 135  # BPM values
        ])

    def test_compare_genre_features_to_bpm_basic(self):
        """基本的なジャンル特徴量とBPM関係分析テスト"""
        analysis_df = compare_genre_features_to_bpm(self.X, self.y)

        # ジャンル特徴量のみが分析されている
        genre_features = [col for col in self.X.columns if "genre_score" in col]
        assert len(analysis_df) == len(genre_features)
        assert all(feature in genre_features for feature in analysis_df["genre_feature"])

        # 必要な列が含まれている
        expected_columns = [
            "genre_feature", "high_group_mean_bpm", "high_group_std_bpm", "high_group_count",
            "mid_group_mean_bpm", "mid_group_std_bpm", "mid_group_count",
            "low_group_mean_bpm", "low_group_std_bpm", "low_group_count",
            "bpm_range", "correlation_with_bpm"
        ]
        assert all(col in analysis_df.columns for col in expected_columns)

        # グループごとのカウントが妥当
        for _, row in analysis_df.iterrows():
            total_count = row["high_group_count"] + row["mid_group_count"] + row["low_group_count"]
            assert total_count == len(self.y)

    def test_compare_genre_features_to_bpm_relationships(self):
        """ジャンル特徴量とBPMの関係性テスト"""
        analysis_df = compare_genre_features_to_bpm(self.X, self.y)

        # dance_genre_scoreは正の相関（高い値で高BPM）
        dance_row = analysis_df[analysis_df["genre_feature"] == "dance_genre_score"].iloc[0]
        assert dance_row["high_group_mean_bpm"] > dance_row["low_group_mean_bpm"]
        assert dance_row["correlation_with_bpm"] > 0

        # acoustic_genre_scoreは負の相関（高い値で低BPM）
        acoustic_row = analysis_df[analysis_df["genre_feature"] == "acoustic_genre_score"].iloc[0]
        assert acoustic_row["high_group_mean_bpm"] < acoustic_row["low_group_mean_bpm"]
        assert acoustic_row["correlation_with_bpm"] < 0

    def test_compare_genre_features_to_bpm_no_genre_features(self):
        """ジャンル特徴量がない場合のテスト"""
        X_no_genre = pd.DataFrame({
            "other_feature_1": np.random.uniform(0, 100, 10),
            "other_feature_2": np.random.uniform(0, 100, 10)
        })

        analysis_df = compare_genre_features_to_bpm(X_no_genre, self.y)

        # 空のDataFrameが返される
        assert len(analysis_df) == 0

    def test_compare_genre_features_to_bpm_sorting(self):
        """BPM範囲による結果ソートのテスト"""
        analysis_df = compare_genre_features_to_bpm(self.X, self.y)

        # bpm_rangeの絶対値で降順ソートされている
        bpm_ranges = analysis_df["bpm_range"].abs()
        assert bpm_ranges.is_monotonic_decreasing


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

    # 音楽ジャンル特徴量作成
    genre_features = create_music_genre_features(statistical_features)
    assert genre_features.shape[1] > statistical_features.shape[1]

    # ジャンル特徴量が正しく追加されている
    expected_genre_features = [
        "dance_genre_score", "acoustic_genre_score", "ballad_genre_score",
        "rock_genre_score", "electronic_genre_score", "ambient_genre_score"
    ]
    assert all(feature in genre_features.columns for feature in expected_genre_features)

    # 特徴量行列の準備
    feature_cols = [col for col in genre_features.columns if col not in ["id", "BeatsPerMinute"]]
    X = genre_features[feature_cols]
    y = genre_features["BeatsPerMinute"]

    # 特徴量選択
    X_selected, _ = select_features(X, y, method="kbest", k=10)
    assert X_selected.shape[1] == 10

    # スケーリング
    X_scaled, _, _, scaler = scale_features(X_selected)
    assert X_scaled.shape == X_selected.shape
    assert isinstance(scaler, StandardScaler)


class TestDetectMulticollinearity:
    """多重共線性検出機能のテスト"""

    def setup_method(self):
        """高相関特徴量を含むテストデータの準備"""
        np.random.seed(42)
        n_samples = 100

        # 基本特徴量
        feature_1 = np.random.normal(50, 10, n_samples)
        feature_2 = np.random.normal(30, 8, n_samples)
        independent_feature = np.random.normal(25, 5, n_samples)

        # 高相関特徴量を作成
        high_corr_feature = 0.8 * feature_1 + 0.2 * np.random.normal(0, 2, n_samples)  # feature_1と高相関
        genre_feature = 0.9 * feature_2 + 0.1 * np.random.normal(0, 1, n_samples)  # feature_2と高相関

        self.df = pd.DataFrame({
            "feature_1": feature_1,
            "feature_2": feature_2,
            "high_corr_feature": high_corr_feature,
            "genre_score_test": genre_feature,  # ジャンル特徴量
            "independent_feature": independent_feature
        })

    def test_detect_multicollinearity_basic(self):
        """基本的な多重共線性検出テスト"""
        result = detect_multicollinearity(self.df, threshold=0.7)

        # 高相関ペアが検出されることを確認
        assert not result.empty
        assert len(result) >= 2  # feature_1-high_corr_feature, feature_2-genre_score_testのペア

        # 必要な列が含まれていることを確認
        expected_columns = ["feature_1", "feature_2", "correlation", "priority_suggestion"]
        assert all(col in result.columns for col in expected_columns)

        # 相関値が閾値以上であることを確認
        assert all(result["correlation"] >= 0.7)

    def test_detect_multicollinearity_genre_prioritization(self):
        """ジャンル特徴量優先判定のテスト"""
        result = detect_multicollinearity(self.df, threshold=0.7)

        # ジャンル特徴量に関する推奨が正しいことを確認
        genre_pairs = result[
            (result["feature_1"].str.contains("genre_score") |
             result["feature_2"].str.contains("genre_score"))
        ]

        assert not genre_pairs.empty
        for _, row in genre_pairs.iterrows():
            if "genre_score" in row["feature_1"]:
                assert f"Keep {row['feature_1']} (genre feature)" in row["priority_suggestion"]
            elif "genre_score" in row["feature_2"]:
                assert f"Keep {row['feature_2']} (genre feature)" in row["priority_suggestion"]

    def test_detect_multicollinearity_no_pairs(self):
        """高相関ペアがない場合のテスト"""
        # 低相関のデータを作成
        independent_df = pd.DataFrame({
            "feature_1": np.random.normal(50, 10, 50),
            "feature_2": np.random.normal(30, 8, 50),
            "feature_3": np.random.normal(25, 5, 50)
        })

        result = detect_multicollinearity(independent_df, threshold=0.9)
        assert result.empty


class TestRemoveCorrelatedFeatures:
    """多重共線性特徴量除去機能のテスト"""

    def setup_method(self):
        """高相関特徴量を含むテストデータの準備"""
        np.random.seed(42)
        n_samples = 100

        # 基本特徴量
        feature_a = np.random.normal(50, 10, n_samples)
        feature_b = np.random.normal(30, 8, n_samples)
        independent = np.random.normal(25, 5, n_samples)

        # 高相関特徴量
        correlated_to_a = 0.85 * feature_a + 0.15 * np.random.normal(0, 2, n_samples)
        genre_correlated_to_b = 0.9 * feature_b + 0.1 * np.random.normal(0, 1, n_samples)

        self.df = pd.DataFrame({
            "feature_a": feature_a,
            "correlated_to_a": correlated_to_a,
            "feature_b": feature_b,
            "dance_genre_score": genre_correlated_to_b,  # ジャンル特徴量
            "independent": independent
        })

    def test_remove_correlated_features_genre_priority(self):
        """ジャンル特徴量優先での除去テスト"""
        original_cols = len(self.df.columns)

        cleaned_df, removal_info = remove_correlated_features(
            self.df, correlation_threshold=0.7, prioritize_genre_features=True
        )

        # 特徴量が除去されていることを確認
        assert len(cleaned_df.columns) < original_cols

        # ジャンル特徴量が保持されていることを確認
        assert "dance_genre_score" in cleaned_df.columns

        # 除去情報が正しく記録されていることを確認
        if not removal_info.empty:
            expected_columns = ["removed_feature", "kept_feature", "correlation", "removal_reason"]
            assert all(col in removal_info.columns for col in expected_columns)

            # ジャンル特徴量優先の理由が含まれている
            genre_priority_removals = removal_info[
                removal_info["removal_reason"].str.contains("genre feature")
            ]
            if not genre_priority_removals.empty:
                assert len(genre_priority_removals) > 0

    def test_remove_correlated_features_no_genre_priority(self):
        """ジャンル特徴量を優先しない場合のテスト"""
        cleaned_df, removal_info = remove_correlated_features(
            self.df, correlation_threshold=0.7, prioritize_genre_features=False
        )

        # 特徴量が除去されていることを確認
        assert len(cleaned_df.columns) <= len(self.df.columns)

        # 除去理由にジャンル特徴量優先が含まれていないことを確認
        if not removal_info.empty:
            genre_reasons = removal_info["removal_reason"].str.contains("genre feature").sum()
            # prioritize_genre_features=Falseなので、ジャンル特徴量優先の理由は0個であるべき
            assert genre_reasons == 0

    def test_remove_correlated_features_no_pairs(self):
        """多重共線性がない場合のテスト"""
        # 独立した特徴量のデータを作成
        independent_df = pd.DataFrame({
            "feature_1": np.random.normal(50, 10, 50),
            "feature_2": np.random.normal(30, 8, 50),
            "feature_3": np.random.normal(25, 5, 50)
        })

        cleaned_df, removal_info = remove_correlated_features(
            independent_df, correlation_threshold=0.9
        )

        # 特徴量が除去されていないことを確認
        assert len(cleaned_df.columns) == len(independent_df.columns)
        assert removal_info.empty


class TestEvaluateMulticollinearityImpact:
    """多重共線性除去効果評価機能のテスト"""

    def setup_method(self):
        """テストデータとモックの準備"""
        np.random.seed(42)
        n_samples = 100

        # 特徴量データ
        feature_1 = np.random.normal(50, 10, n_samples)
        feature_2 = np.random.normal(30, 8, n_samples)
        correlated_feature = 0.8 * feature_1 + 0.2 * np.random.normal(0, 2, n_samples)

        self.X_original = pd.DataFrame({
            "feature_1": feature_1,
            "feature_2": feature_2,
            "correlated_feature": correlated_feature
        })

        self.X_cleaned = self.X_original.drop("correlated_feature", axis=1)
        self.y = 2 * feature_1 + feature_2 + np.random.normal(0, 5, n_samples)

        self.removal_info = pd.DataFrame({
            "removed_feature": ["correlated_feature"],
            "kept_feature": ["feature_1"],
            "correlation": [0.8],
            "removal_reason": ["High correlation with feature_1"]
        })

    def test_evaluate_multicollinearity_impact_basic(self):
        """基本的な効果評価テスト（実際のLightGBMを使用）"""
        # 実際の評価を実行（軽量パラメータで）
        result = evaluate_multicollinearity_impact(
            self.X_original, self.X_cleaned, self.y, self.removal_info
        )

        # 結果の構造が正しいことを確認
        expected_keys = [
            "before_features_count", "after_features_count", "removed_features_count",
            "before_rmse", "after_rmse", "rmse_improvement", "improvement_percentage"
        ]
        assert all(key in result for key in expected_keys)

        # 数値が正しい範囲にあることを確認
        assert result["before_features_count"] == 3
        assert result["after_features_count"] == 2
        assert result["removed_features_count"] == 1
        assert isinstance(result["rmse_improvement"], (int, float))
        assert isinstance(result["improvement_percentage"], (int, float))

        # RMSEが正の値であることを確認
        assert result["before_rmse"] > 0
        assert result["after_rmse"] > 0