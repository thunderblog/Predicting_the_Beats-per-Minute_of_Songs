"""
Features.py - Refactored Feature Engineering Module

This module now serves as a backward-compatible interface to the new
modular feature engineering system located in src/features/ directory.

All original functions are preserved for backward compatibility while
internally utilizing the new refactored modules.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import explained_variance_score
from scipy.stats import zscore
from tqdm import tqdm
import typer
from itertools import combinations

from src.config import PROCESSED_DATA_DIR

# Import refactored modules
from src.features import (
    # Feature creators
    BasicInteractionCreator,
    ComprehensiveInteractionCreator,
    StatisticalFeatureCreator,
    MusicGenreFeatureCreator,
    DurationFeatureCreator,
    AdvancedFeatureCreator,
    LogTransformFeatureCreator,
    BinningFeatureCreator,
    FeaturePipeline,
    create_feature_pipeline,
    # Processing functions
    select_features as new_select_features,
    scale_features as new_scale_features,
    analyze_feature_importance as new_analyze_feature_importance,
    compare_genre_features_to_bpm as new_compare_genre_features_to_bpm,
    detect_multicollinearity as new_detect_multicollinearity,
)

app = typer.Typer()


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """既存の変数間の交互作用特徴量を作成する。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        交互作用特徴量が追加されたデータフレーム
    """
    # Use refactored module
    creator = BasicInteractionCreator()
    return creator.create_features(df)


def create_comprehensive_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggleサンプルコードの手法による包括的交互作用特徴量を作成する。

    全数値特徴量ペアの積、二乗項、比率特徴量を系統的に生成して
    特徴量空間を大幅に拡張し、非線形関係を捕捉する。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        包括的交互作用特徴量が追加されたデータフレーム
    """
    # Use refactored module
    creator = ComprehensiveInteractionCreator()
    return creator.create_features(df)


def create_music_genre_features(df: pd.DataFrame) -> pd.DataFrame:
    """音楽ジャンル推定に基づく特徴量を作成する。

    音楽理論に基づき、特徴量の組み合わせから暗黙的な
    ジャンル特徴量を推定してBPM予測精度を向上させる。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        音楽ジャンル推定特徴量が追加されたデータフレーム
    """
    # Use refactored module
    creator = MusicGenreFeatureCreator()
    return creator.create_features(df)


def create_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """TrackDurationMsから時間に基づく特徴量を作成する。

    Args:
        df: TrackDurationMs列を含む入力データフレーム

    Returns:
        時間特徴量が追加されたデータフレーム
    """
    # Use refactored module
    creator = DurationFeatureCreator()
    return creator.create_features(df)


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """既存の数値列から統計的特徴量を作成する。

    Args:
        df: 入力データフレーム

    Returns:
        統計的特徴量が追加されたデータフレーム
    """
    # Use refactored module
    creator = StatisticalFeatureCreator()
    return creator.create_features(df)


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """独立性の高い高次特徴量を作成する。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        高次特徴量が追加されたデータフレーム
    """
    # Use refactored module
    creator = AdvancedFeatureCreator()
    return creator.create_features(df)


def create_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """対数変換特徴量を作成する (TICKET-017-02)。

    全数値特徴量のlog1p変換と、変換特徴量同士の組み合わせ特徴量を生成して
    分布の歪み補正により予測精度向上を図る。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        対数変換特徴量が追加されたデータフレーム
    """
    logger.info("対数変換特徴量を作成中...")

    df_features = df.copy()

    # 対象特徴量（AudioLoudnessを除く8個）
    target_features = [
        'RhythmScore', 'VocalContent', 'AcousticQuality',
        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
        'TrackDurationMs', 'Energy'
    ]

    # 存在する特徴量のみ対象とする
    available_features = [col for col in target_features if col in df.columns]

    logger.info(f"対数変換対象: {len(available_features)}特徴量 - {available_features}")

    # 1. 基本log1p変換特徴量の作成
    log_feature_names = []
    for feature in available_features:
        log_feature_name = f"log1p_{feature}"

        # log1p変換（負値対応、1e-8でクリッピング）
        feature_values = df_features[feature].clip(lower=1e-8)
        df_features[log_feature_name] = np.log1p(feature_values)
        log_feature_names.append(log_feature_name)

    logger.info(f"基本log1p変換完了: {len(log_feature_names)}特徴量")

    # 2. log変換特徴量同士の組み合わせ特徴量
    if len(log_feature_names) >= 2:
        logger.info("log変換特徴量の組み合わせを作成中...")

        combination_count = 0

        # ペアワイズ積特徴量
        for i, feature1 in enumerate(log_feature_names):
            for j, feature2 in enumerate(log_feature_names[i+1:], i+1):
                # log(A) * log(B) = log(A^B) の近似
                combo_name = f"{feature1}_x_{feature2}"
                df_features[combo_name] = df_features[feature1] * df_features[feature2]
                combination_count += 1

        # 重要な比率特徴量
        if len(log_feature_names) >= 3:
            # log(TrackDurationMs)を基準とした比率
            if 'log1p_TrackDurationMs' in log_feature_names:
                base_log = 'log1p_TrackDurationMs'
                for other_log in log_feature_names:
                    if other_log != base_log:
                        ratio_name = f"{other_log}_div_{base_log}"
                        # ゼロ除算回避
                        df_features[ratio_name] = df_features[other_log] / (df_features[base_log] + 1e-8)
                        combination_count += 1

            # Energy - RhythmScore log space関係
            if 'log1p_Energy' in log_feature_names and 'log1p_RhythmScore' in log_feature_names:
                df_features['log_energy_rhythm_harmony'] = (
                    df_features['log1p_Energy'] + df_features['log1p_RhythmScore']
                ) / 2
                combination_count += 1

        logger.info(f"組み合わせ特徴量完了: {combination_count}特徴量")

    # 3. 対数空間での統計特徴量
    if len(log_feature_names) >= 2:
        logger.info("対数空間統計特徴量を作成中...")

        log_values = df_features[log_feature_names]

        # 対数空間での統計量
        df_features['log_features_mean'] = log_values.mean(axis=1)
        df_features['log_features_std'] = log_values.std(axis=1)
        df_features['log_features_range'] = log_values.max(axis=1) - log_values.min(axis=1)

        # 幾何平均（log空間での算術平均の逆変換）
        df_features['log_features_geometric_mean'] = np.expm1(df_features['log_features_mean'])

        logger.info("対数空間統計特徴量完了: 4特徴量")

    # 4. 分布正規化指標
    logger.info("分布正規化指標を作成中...")

    # 元特徴量の歪度改善指標
    skewness_improvements = []
    for original_feature in available_features:
        if original_feature in df.columns:
            original_skew = abs(df[original_feature].skew())
            log_feature = f"log1p_{original_feature}"
            if log_feature in df_features.columns:
                log_skew = abs(df_features[log_feature].skew())
                improvement = max(0, original_skew - log_skew)  # 改善度（正値のみ）
                skewness_improvements.append(improvement)

    if skewness_improvements:
        df_features['log_transformation_benefit'] = np.mean(skewness_improvements)
        logger.info(f"分布正規化指標完了: 平均改善度 {np.mean(skewness_improvements):.3f}")

    n_new_features = len(df_features.columns) - len(df.columns)
    logger.success(f"対数変換特徴量を作成完了: {n_new_features}個の新特徴量を追加")

    return df_features


def create_binning_features(df: pd.DataFrame) -> pd.DataFrame:
    """ビニング・カテゴリ特徴量を作成する (TICKET-017-03)。

    数値特徴量を分位数分割（septile、decile、quintile）でカテゴリ化し、
    各カテゴリの統計特徴量を生成して非線形関係を捕捉する。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        ビニング特徴量が追加されたデータフレーム
    """
    logger.info("ビニング・カテゴリ特徴量を作成中...")

    df_features = df.copy()

    # 対象特徴量（全基本数値特徴量）
    target_features = [
        'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
        'TrackDurationMs', 'Energy'
    ]

    # 存在する特徴量のみ対象とする
    available_features = [col for col in target_features if col in df.columns]

    logger.info(f"ビニング対象: {len(available_features)}特徴量 - {available_features}")

    # 1. 基本数値特徴量のビニング
    binning_configs = {
        'septile': 7,    # 7分位
        'decile': 10,    # 10分位
        'quintile': 5,   # 5分位
    }

    binning_count = 0

    for binning_type, n_bins in binning_configs.items():
        logger.info(f"{binning_type}分割（{n_bins}分位）を実行中...")

        for feature in available_features:
            feature_values = df_features[feature]

            # 分位数でビニング（等頻度分割）
            try:
                binned_feature_name = f"{feature}_{binning_type}_bin"

                binned_values, _ = pd.qcut(
                    feature_values,
                    q=n_bins,
                    retbins=True,
                    duplicates='drop',
                    labels=False  # 数値ラベル使用
                )

                df_features[binned_feature_name] = binned_values
                binning_count += 1

            except Exception as e:
                # 分位数分割が失敗した場合（値の種類が少ない等）
                logger.warning(f"  {feature}の{binning_type}分割スキップ: {e}")
                continue

    logger.info(f"基本ビニング完了: {binning_count}特徴量")

    # 2. log変換特徴量のビニング
    log_features = [col for col in df_features.columns if col.startswith('log1p_')]

    if log_features:
        logger.info(f"{len(log_features)}個のlog変換特徴量をquintile分割中...")

        log_binning_count = 0
        for log_feature in log_features:
            feature_values = df_features[log_feature]

            try:
                binned_feature_name = f"{log_feature}_quintile_bin"

                binned_values, _ = pd.qcut(
                    feature_values,
                    q=5,
                    retbins=True,
                    duplicates='drop',
                    labels=False
                )

                df_features[binned_feature_name] = binned_values
                log_binning_count += 1

            except Exception as e:
                logger.warning(f"  {log_feature}の5分位分割スキップ: {e}")
                continue

        logger.info(f"log変換ビニング完了: {log_binning_count}特徴量")

    # 3. ビン統計特徴量（BPM目的変数がある場合のみ）
    if 'BeatsPerMinute' in df_features.columns:
        logger.info("ビン統計特徴量を作成中...")

        bpm_values = df_features['BeatsPerMinute']
        binning_features = [col for col in df_features.columns if col.endswith('_bin')]
        stat_count = 0

        for binning_feature in binning_features:
            try:
                bin_values = df_features[binning_feature]

                # 欠損値を含むビンは除外
                valid_mask = ~(bin_values.isna() | bpm_values.isna())
                if valid_mask.sum() == 0:
                    continue

                valid_bins = bin_values[valid_mask]
                valid_bpm = bpm_values[valid_mask]

                # ビンごとの統計量計算
                bin_stats = valid_bpm.groupby(valid_bins).agg(['mean', 'std', 'count']).fillna(0)

                # 各サンプルに対応するビン統計量をマップ
                base_name = binning_feature.replace('_bin', '')

                # 平均BPM特徴量
                mean_feature_name = f"{base_name}_bin_mean_bpm"
                df_features[mean_feature_name] = bin_values.map(
                    dict(zip(bin_stats.index, bin_stats['mean']))
                ).fillna(valid_bpm.mean())
                stat_count += 1

                # 標準偏差特徴量
                std_feature_name = f"{base_name}_bin_std_bpm"
                df_features[std_feature_name] = bin_values.map(
                    dict(zip(bin_stats.index, bin_stats['std']))
                ).fillna(valid_bpm.std())
                stat_count += 1

            except Exception as e:
                logger.warning(f"  {binning_feature}の統計特徴量作成スキップ: {e}")
                continue

        logger.info(f"ビン統計特徴量完了: {stat_count}特徴量")

    n_new_features = len(df_features.columns) - len(df.columns)
    logger.success(f"ビニング・カテゴリ特徴量を作成完了: {n_new_features}個の新特徴量を追加")

    return df_features


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    method: str = "kbest",
    k: int = 20,
):
    """様々な選択手法を使用して最も重要な特徴量を選択する。

    Args:
        X_train: 訓練用特徴量行列
        y_train: 訓練用目的変数値
        X_val: 検証用特徴量行列（オプション）
        method: 特徴量選択手法 ('kbest', 'mutual_info', 'correlation')
        k: 選択する特徴量の数

    Returns:
        (selected_X_train, selected_X_val)のタプル
    """
    # Use refactored module
    return new_select_features(X_train, y_train, X_val, method, k)


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    scaler_type: str = "standard",
) -> tuple:
    """指定されたスケーリング手法で特徴量をスケールする。

    Args:
        X_train: 訓練用特徴量
        X_val: 検証用特徴量（オプション）
        X_test: テスト用特徴量（オプション）
        scaler_type: スケーラの種類 ('standard', 'robust', 'minmax')

    Returns:
        スケールされたデータセットとフィット済みスケーラのタプル
    """
    logger.info(f"{scaler_type}スケーリングを適用中...")

    # スケーラを選択
    scalers = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler()}

    scaler = scalers.get(scaler_type, StandardScaler())

    # 訓練データでフィット
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )

    results = [X_train_scaled]

    # 検証データが提供されている場合は変換
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), columns=X_val.columns, index=X_val.index
        )
        results.append(X_val_scaled)
    else:
        results.append(None)

    # テストデータが提供されている場合は変換
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        results.append(X_test_scaled)
    else:
        results.append(None)

    results.append(scaler)

    return tuple(results)


def analyze_feature_importance(
    X: pd.DataFrame, y: pd.Series, feature_category: str = "all"
) -> pd.DataFrame:
    """特徴量の重要度を複数の手法で分析する。

    Args:
        X: 特徴量行列
        y: 目的変数
        feature_category: 分析対象の特徴量カテゴリ ('genre', 'interaction', 'duration', 'statistical', 'all')

    Returns:
        特徴量重要度分析結果のDataFrame
    """
    logger.info(f"{feature_category}特徴量の重要度を分析中...")

    # 特徴量カテゴリによるフィルタリング
    if feature_category == "genre":
        target_features = [col for col in X.columns if "genre_score" in col]
    elif feature_category == "interaction":
        target_features = [
            col
            for col in X.columns
            if any(keyword in col for keyword in ["product", "ratio", "rhythm_mood_energy"])
        ]
    elif feature_category == "duration":
        target_features = [
            col
            for col in X.columns
            if any(keyword in col for keyword in ["duration", "track", "short", "long"])
        ]
    elif feature_category == "statistical":
        target_features = [
            col
            for col in X.columns
            if any(keyword in col for keyword in ["total", "mean", "std", "min", "max", "range"])
        ]
    else:
        if feature_category != "all":
            # 存在しないカテゴリの場合は空のリストを返す
            target_features = []
        else:
            target_features = X.columns.tolist()

    if not target_features:
        logger.warning(f"{feature_category}カテゴリの特徴量が見つかりません")
        return pd.DataFrame()

    X_filtered = X[target_features]

    # 1. 相関係数による重要度
    correlations = X_filtered.corrwith(y).abs()

    # 2. F統計量による重要度
    f_selector = SelectKBest(score_func=f_regression, k="all")
    f_selector.fit(X_filtered, y)
    f_scores = f_selector.scores_

    # 3. 相互情報量による重要度（サンプリングで高速化）
    if len(X_filtered) > 10000:
        # 大きなデータセットの場合はサンプリング
        sample_idx = np.random.choice(len(X_filtered), 10000, replace=False)
        X_sample = X_filtered.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        mi_selector = SelectKBest(score_func=mutual_info_regression, k="all")
        mi_selector.fit(X_sample, y_sample)
        logger.info(f"相互情報量計算: {len(X_sample)}サンプルでサンプリング実行")
    else:
        mi_selector = SelectKBest(score_func=mutual_info_regression, k="all")
        mi_selector.fit(X_filtered, y)
    mi_scores = mi_selector.scores_

    # 4. Random Forestによる重要度（軽量化）
    if len(X_filtered) > 10000:
        # 大きなデータセットの場合はサンプリング + 軽量パラメータ
        sample_idx = np.random.choice(len(X_filtered), 10000, replace=False)
        X_sample = X_filtered.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
        rf.fit(X_sample, y_sample)
        logger.info(f"Random Forest訓練: {len(X_sample)}サンプルでサンプリング実行")
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_filtered, y)
    rf_importances = rf.feature_importances_

    # 結果をDataFrameにまとめる
    importance_df = pd.DataFrame(
        {
            "feature_name": target_features,
            "correlation": correlations.values,
            "f_score": f_scores,
            "mutual_info": mi_scores,
            "rf_importance": rf_importances,
        }
    )

    # 各スコアを正規化（0-1の範囲）
    for col in ["correlation", "f_score", "mutual_info", "rf_importance"]:
        importance_df[f"{col}_normalized"] = (importance_df[col] - importance_df[col].min()) / (
            importance_df[col].max() - importance_df[col].min() + 1e-8
        )

    # 平均重要度スコアを計算
    importance_df["average_importance"] = importance_df[
        [
            "correlation_normalized",
            "f_score_normalized",
            "mutual_info_normalized",
            "rf_importance_normalized",
        ]
    ].mean(axis=1)

    # 重要度でソート
    importance_df = importance_df.sort_values("average_importance", ascending=False)

    logger.info(f"特徴量重要度分析完了: {len(target_features)}個の特徴量を分析")

    return importance_df


def compare_genre_features_to_bpm(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """ジャンル特徴量とBPMの関係を詳細分析する。

    Args:
        X: 特徴量行列（ジャンル特徴量を含む）
        y: BPM目的変数

    Returns:
        ジャンル特徴量とBPMの関係分析結果
    """
    logger.info("ジャンル特徴量とBPMの関係を分析中...")

    genre_features = [col for col in X.columns if "genre_score" in col]

    if not genre_features:
        logger.warning("ジャンル特徴量が見つかりません")
        return pd.DataFrame()

    results = []

    for feature in genre_features:
        # 特徴量値による分位数分割（高/中/低）
        feature_values = X[feature]
        high_threshold = feature_values.quantile(0.75)
        low_threshold = feature_values.quantile(0.25)

        high_mask = feature_values >= high_threshold
        mid_mask = (feature_values >= low_threshold) & (feature_values < high_threshold)
        low_mask = feature_values < low_threshold

        # 各グループのBPM統計
        high_bpm = y[high_mask]
        mid_bpm = y[mid_mask]
        low_bpm = y[low_mask]

        results.append(
            {
                "genre_feature": feature,
                "high_group_mean_bpm": high_bpm.mean(),
                "high_group_std_bpm": high_bpm.std(),
                "high_group_count": len(high_bpm),
                "mid_group_mean_bpm": mid_bpm.mean(),
                "mid_group_std_bpm": mid_bpm.std(),
                "mid_group_count": len(mid_bpm),
                "low_group_mean_bpm": low_bpm.mean(),
                "low_group_std_bpm": low_bpm.std(),
                "low_group_count": len(low_bpm),
                "bpm_range": high_bpm.mean() - low_bpm.mean(),
                "correlation_with_bpm": X[feature].corr(y),
            }
        )

    analysis_df = pd.DataFrame(results)
    analysis_df = analysis_df.sort_values("bpm_range", ascending=False, key=abs)

    logger.info(f"ジャンル特徴量分析完了: {len(genre_features)}個の特徴量を分析")

    return analysis_df


def detect_multicollinearity(X: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """高相関特徴量ペアを検出して多重共線性を分析する。

    Args:
        X: 特徴量行列
        threshold: 相関係数の閾値（この値以上で高相関とみなす）

    Returns:
        高相関ペア情報のDataFrame（feature_1, feature_2, correlation, priority_suggestion）
    """
    logger.info(f"多重共線性検出中（閾値: {threshold}）...")

    # 相関行列を計算
    corr_matrix = X.corr().abs()

    # 上三角行列から高相関ペアを抽出（対角成分は除外）
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            correlation = corr_matrix.iloc[i, j]

            if correlation >= threshold:
                feature_1 = corr_matrix.columns[i]
                feature_2 = corr_matrix.columns[j]

                # ジャンル特徴量の優先判定
                feature_1_is_genre = "genre_score" in feature_1
                feature_2_is_genre = "genre_score" in feature_2

                # どちらを保持すべきかの推奨を決定
                if feature_1_is_genre and feature_2_is_genre:
                    priority_suggestion = "Both genre features - consider combining"
                elif feature_1_is_genre:
                    priority_suggestion = f"Keep {feature_1} (genre feature)"
                elif feature_2_is_genre:
                    priority_suggestion = f"Keep {feature_2} (genre feature)"
                else:
                    priority_suggestion = "No genre feature - manual decision needed"

                high_corr_pairs.append({
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": correlation,
                    "feature_1_is_genre": feature_1_is_genre,
                    "feature_2_is_genre": feature_2_is_genre,
                    "priority_suggestion": priority_suggestion
                })

    if not high_corr_pairs:
        logger.info("高相関特徴量ペアは検出されませんでした")
        return pd.DataFrame()

    pairs_df = pd.DataFrame(high_corr_pairs)
    pairs_df = pairs_df.sort_values("correlation", ascending=False)

    logger.info(f"高相関ペア検出完了: {len(pairs_df)}組のペアを発見")

    return pairs_df


def remove_correlated_features(
    X: pd.DataFrame,
    correlation_threshold: float = 0.7,
    prioritize_genre_features: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """多重共線性のある特徴量を除去する。

    Args:
        X: 特徴量行列
        correlation_threshold: 相関係数の閾値
        prioritize_genre_features: ジャンル特徴量を優先して保持するかどうか

    Returns:
        (cleaned_X, removed_features_info)のタプル
        - cleaned_X: 多重共線性除去後の特徴量行列
        - removed_features_info: 除去された特徴量の情報
    """
    logger.info("多重共線性除去処理を開始...")

    # 高相関ペアを検出
    high_corr_pairs = detect_multicollinearity(X, correlation_threshold)

    if high_corr_pairs.empty:
        logger.info("除去すべき高相関特徴量は見つかりませんでした")
        return X, pd.DataFrame()

    # 除去する特徴量のリスト
    features_to_remove = set()
    removal_info = []

    for _, row in high_corr_pairs.iterrows():
        feature_1 = row["feature_1"]
        feature_2 = row["feature_2"]
        correlation = row["correlation"]

        # 既に除去対象となった特徴量はスキップ
        if feature_1 in features_to_remove or feature_2 in features_to_remove:
            continue

        # 除去する特徴量を決定
        if prioritize_genre_features:
            if row["feature_1_is_genre"] and not row["feature_2_is_genre"]:
                # feature_1がジャンル特徴量、feature_2が非ジャンル特徴量
                remove_feature = feature_2
                keep_feature = feature_1
                reason = "Non-genre feature removed in favor of genre feature"
            elif row["feature_2_is_genre"] and not row["feature_1_is_genre"]:
                # feature_2がジャンル特徴量、feature_1が非ジャンル特徴量
                remove_feature = feature_1
                keep_feature = feature_2
                reason = "Non-genre feature removed in favor of genre feature"
            elif row["feature_1_is_genre"] and row["feature_2_is_genre"]:
                # 両方がジャンル特徴量の場合は辞書順で後のものを除去
                if feature_1 < feature_2:
                    remove_feature = feature_2
                    keep_feature = feature_1
                else:
                    remove_feature = feature_1
                    keep_feature = feature_2
                reason = "Lexicographically later genre feature removed"
            else:
                # 両方とも非ジャンル特徴量の場合は辞書順で後のものを除去
                if feature_1 < feature_2:
                    remove_feature = feature_2
                    keep_feature = feature_1
                else:
                    remove_feature = feature_1
                    keep_feature = feature_2
                reason = "Lexicographically later non-genre feature removed"
        else:
            # ジャンル特徴量を優先しない場合は辞書順で決定
            if feature_1 < feature_2:
                remove_feature = feature_2
                keep_feature = feature_1
            else:
                remove_feature = feature_1
                keep_feature = feature_2
            reason = "Lexicographically later feature removed"

        features_to_remove.add(remove_feature)
        removal_info.append({
            "removed_feature": remove_feature,
            "kept_feature": keep_feature,
            "correlation": correlation,
            "removal_reason": reason
        })

    # 特徴量を除去
    cleaned_X = X.drop(columns=list(features_to_remove))
    removal_df = pd.DataFrame(removal_info)

    logger.info(f"多重共線性除去完了: {len(features_to_remove)}個の特徴量を除去")
    logger.info(f"残存特徴量数: {len(cleaned_X.columns)} (元: {len(X.columns)})")

    return cleaned_X, removal_df


def create_rhythm_periodicity_features(df: pd.DataFrame) -> pd.DataFrame:
    """ドラマー視点のリズム周期性特徴量を作成する。

    音楽理論に基づき、リズムパターン、周期性一貫性、疑似ドラム系特徴量、
    拍子・テンポ変動、周期性コヒーレンス指標を生成してBPM予測精度を向上させる。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        リズム周期性特徴量が追加されたデータフレーム
    """
    logger.info("ドラマー視点リズム周期性特徴量を作成中...")

    df_features = df.copy()

    # TODO(human): 4/4拍子・3/4拍子・シンコペーション検出のロジックを実装

    # 2. 周期性一貫性スコア（TrackDurationとBPM推定の整合性検証）
    logger.info("周期性一貫性スコアを作成中...")

    # 基本BPM推定（RhythmScore×Energyベース）
    estimated_bpm_base = 60 + (df["RhythmScore"] * df["Energy"] * 120)  # 60-180 BPM range

    # トラック長とBPMの理論的整合性
    # 一般的な楽曲では、BPM×楽曲長（分）×4/4拍子 = 総拍数が妥当な範囲にある
    track_minutes = df["TrackDurationMs"] / (1000 * 60)
    theoretical_beats = estimated_bpm_base * track_minutes * 4  # 4/4拍子仮定

    # 妥当な総拍数範囲（64-512拍）との整合性スコア
    ideal_beats_range = [64, 512]
    df_features["tempo_duration_consistency"] = 1 - np.abs(
        theoretical_beats.clip(ideal_beats_range[0], ideal_beats_range[1]) - theoretical_beats
    ) / (ideal_beats_range[1] - ideal_beats_range[0])

    # 3. 疑似ドラム系特徴量（キック・スネア・ハイハット推定密度）
    logger.info("疑似ドラム系特徴量を作成中...")

    # キック（低音域）推定：低RhythmScore + 高Energy = 重いキック
    df_features["pseudo_kick_density"] = (1 - df["RhythmScore"]) * df["Energy"] * df["AudioLoudness"]

    # スネア（中音域）推定：中RhythmScore + 中Energy = 安定したスネア
    rhythm_mid = 0.5 - np.abs(df["RhythmScore"] - 0.5)  # 0.5に近いほど高い
    energy_mid = 0.5 - np.abs(df["Energy"] - 0.5)
    df_features["pseudo_snare_density"] = rhythm_mid * energy_mid * df["InstrumentalScore"]

    # ハイハット（高音域）推定：高RhythmScore + 低Energy = 軽やかなハイハット
    df_features["pseudo_hihat_density"] = df["RhythmScore"] * (1 - df["Energy"]) * df["LivePerformanceLikelihood"]

    # ドラムセット全体の複雑性
    df_features["drum_complexity"] = (
        df_features["pseudo_kick_density"] +
        df_features["pseudo_snare_density"] +
        df_features["pseudo_hihat_density"]
    ) / 3

    # 4. 拍子・テンポ変動推定（ルバート、加速、減速パターン検出）
    logger.info("拍子・テンポ変動推定を作成中...")

    # ルバート（自由テンポ）推定：高AcousticQuality + 低RhythmScore
    df_features["rubato_likelihood"] = df["AcousticQuality"] * (1 - df["RhythmScore"]) * df["MoodScore"]

    # 加速パターン推定：高Energy×時間進行
    # 長い楽曲ほど加速の可能性が高い
    normalized_duration = (df["TrackDurationMs"] - df["TrackDurationMs"].min()) / (
        df["TrackDurationMs"].max() - df["TrackDurationMs"].min() + 1e-8
    )
    df_features["accelerando_likelihood"] = df["Energy"] * df["RhythmScore"] * normalized_duration

    # 減速パターン推定：高MoodScore + 中Energy
    df_features["ritardando_likelihood"] = df["MoodScore"] * (1 - df["Energy"]) * df["VocalContent"]

    # テンポ安定性指標
    df_features["tempo_stability"] = 1 - (
        df_features["rubato_likelihood"] +
        df_features["accelerando_likelihood"] +
        df_features["ritardando_likelihood"]
    ) / 3

    # 5. 周期性コヒーレンス指標（RhythmScore×Energy×時間整合性）
    logger.info("周期性コヒーレンス指標を作成中...")

    # 基本リズムコヒーレンス
    df_features["rhythm_energy_coherence"] = df["RhythmScore"] * df["Energy"]

    # 時間軸でのコヒーレンス（楽曲長との調和）
    log_duration = np.log1p(df["TrackDurationMs"])
    normalized_log_duration = (log_duration - log_duration.min()) / (log_duration.max() - log_duration.min() + 1e-8)

    df_features["temporal_coherence"] = df_features["rhythm_energy_coherence"] * normalized_log_duration

    # 全体的な周期性品質指標
    df_features["overall_periodicity_quality"] = (
        df_features["tempo_duration_consistency"] * 0.3 +
        df_features["drum_complexity"] * 0.25 +
        df_features["tempo_stability"] * 0.25 +
        df_features["temporal_coherence"] * 0.2
    )

    # 楽曲構造推定（イントロ・サビ・アウトロの推定）
    # イントロ推定：短めの楽曲で高RhythmScore
    intro_likelihood = (1 - normalized_duration) * df["RhythmScore"] * df["InstrumentalScore"]

    # サビ推定：中程度の楽曲長で高Energy + 高VocalContent
    chorus_likelihood = (1 - np.abs(normalized_duration - 0.5)) * df["Energy"] * df["VocalContent"]

    # アウトロ推定：長い楽曲で高MoodScore
    outro_likelihood = normalized_duration * df["MoodScore"] * df["AcousticQuality"]

    df_features["intro_section_likelihood"] = intro_likelihood
    df_features["chorus_section_likelihood"] = chorus_likelihood
    df_features["outro_section_likelihood"] = outro_likelihood

    # 楽曲構造の明確性
    df_features["song_structure_clarity"] = (
        intro_likelihood + chorus_likelihood + outro_likelihood
    ) / 3

    n_new_features = len(df_features.columns) - len(df.columns)
    logger.success(f"リズム周期性特徴量を作成完了: {n_new_features}個の新特徴量を追加")

    return df_features


def determine_optimal_components(X: pd.DataFrame, variance_threshold: float = 0.95, max_components: int = None) -> dict:
    """最適な主成分数を自動選択する。

    Args:
        X: 特徴量行列
        variance_threshold: 累積寄与率の閾値（この値以上になる主成分数を選択）
        max_components: 最大主成分数（指定されない場合は特徴量数の80%）

    Returns:
        {"n_components": int, "explained_variance_ratio": float, "individual_ratios": list}
    """
    logger.info(f"最適主成分数を決定中（累積寄与率閾値: {variance_threshold}）...")

    if max_components is None:
        max_components = min(len(X.columns), int(len(X.columns) * 0.8))

    # PCAを実行して寄与率を計算
    pca = PCA(n_components=max_components)
    pca.fit(X)

    # 累積寄与率を計算
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # 閾値を超える最初の主成分数を選択
    optimal_n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    if optimal_n_components == 1 and cumulative_variance[0] < variance_threshold:
        # 閾値に達しない場合は80%の主成分数を選択
        optimal_n_components = max_components

    result = {
        "n_components": optimal_n_components,
        "explained_variance_ratio": cumulative_variance[optimal_n_components - 1],
        "individual_ratios": pca.explained_variance_ratio_[:optimal_n_components].tolist()
    }

    logger.info(f"最適主成分数: {optimal_n_components}個（累積寄与率: {result['explained_variance_ratio']:.3f}）")

    return result


def create_dimensionality_reduction_features(
    df: pd.DataFrame,
    apply_pca: bool = True,
    apply_ica: bool = True,
    pca_variance_threshold: float = 0.95,
    ica_components: int = None
) -> pd.DataFrame:
    """PCA・ICAによる次元削減特徴量を作成する。

    Args:
        df: 元の特徴量を含むデータフレーム
        apply_pca: PCA変換を適用するかどうか
        apply_ica: ICA変換を適用するかどうか
        pca_variance_threshold: PCA主成分数決定の累積寄与率閾値
        ica_components: ICA成分数（指定されない場合は元特徴量数の50%）

    Returns:
        次元削減特徴量が追加されたデータフレーム
    """
    logger.info("次元削減特徴量を作成中...")

    df_features = df.copy()

    # 元特徴量を特定（ジャンル特徴量とリズム特徴量を除く）
    original_features = [
        "RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality",
        "InstrumentalScore", "LivePerformanceLikelihood", "MoodScore",
        "TrackDurationMs", "Energy"
    ]

    # ジャンル特徴量を特定
    genre_features = [col for col in df.columns if "genre_score" in col]

    # 元特徴量のPCA変換
    if apply_pca and len(original_features) > 1:
        # データフレームから元特徴量を抽出
        original_data = df_features[original_features].fillna(0)

        # データ標準化
        scaler = StandardScaler()
        original_scaled = scaler.fit_transform(original_data)

        # 最適主成分数を決定
        optimal_info = determine_optimal_components(
            pd.DataFrame(original_scaled, columns=original_features),
            variance_threshold=pca_variance_threshold
        )
        n_components = optimal_info["n_components"]

        # PCA変換実行
        pca = PCA(n_components=n_components)
        original_pca = pca.fit_transform(original_scaled)

        # 主成分特徴量を追加
        for i in range(n_components):
            df_features[f"pca_original_pc{i+1}"] = original_pca[:, i]

        logger.info(f"元特徴量PCA変換完了: {n_components}主成分（累積寄与率: {optimal_info['explained_variance_ratio']:.3f}）")

    # ジャンル特徴量のPCA変換
    if apply_pca and len(genre_features) > 2:
        # ジャンル特徴量を抽出
        genre_data = df_features[genre_features].fillna(0)

        # データ標準化
        scaler_genre = StandardScaler()
        genre_scaled = scaler_genre.fit_transform(genre_data)

        # 最適主成分数を決定（ジャンル特徴量用）
        optimal_genre_info = determine_optimal_components(
            pd.DataFrame(genre_scaled, columns=genre_features),
            variance_threshold=pca_variance_threshold
        )
        n_genre_components = optimal_genre_info["n_components"]

        # PCA変換実行
        pca_genre = PCA(n_components=n_genre_components)
        genre_pca = pca_genre.fit_transform(genre_scaled)

        # ジャンル主成分特徴量を追加
        for i in range(n_genre_components):
            df_features[f"pca_genre_pc{i+1}"] = genre_pca[:, i]

        logger.info(f"ジャンル特徴量PCA変換完了: {n_genre_components}主成分（累積寄与率: {optimal_genre_info['explained_variance_ratio']:.3f}）")

    # 元特徴量のICA変換
    if apply_ica and len(original_features) > 1:
        # ICA成分数を決定（元特徴量数の50%）
        if ica_components is None:
            ica_components = max(2, int(len(original_features) * 0.5))
        else:
            ica_components = min(ica_components, len(original_features))

        # データ標準化
        original_data = df_features[original_features].fillna(0)
        scaler_ica = StandardScaler()
        original_scaled_ica = scaler_ica.fit_transform(original_data)

        # ICA変換実行
        ica = FastICA(n_components=ica_components, random_state=42, max_iter=1000)
        original_ica = ica.fit_transform(original_scaled_ica)

        # 独立成分特徴量を追加
        for i in range(ica_components):
            df_features[f"ica_original_ic{i+1}"] = original_ica[:, i]

        logger.info(f"元特徴量ICA変換完了: {ica_components}独立成分")

    # ジャンル特徴量のICA変換
    if apply_ica and len(genre_features) > 2:
        # ICA成分数を決定（ジャンル特徴量数の50%）
        genre_ica_components = max(2, int(len(genre_features) * 0.5))

        # データ標準化
        genre_data = df_features[genre_features].fillna(0)
        scaler_ica_genre = StandardScaler()
        genre_scaled_ica = scaler_ica_genre.fit_transform(genre_data)

        # ICA変換実行
        ica_genre = FastICA(n_components=genre_ica_components, random_state=42, max_iter=1000)
        genre_ica = ica_genre.fit_transform(genre_scaled_ica)

        # ジャンル独立成分特徴量を追加
        for i in range(genre_ica_components):
            df_features[f"ica_genre_ic{i+1}"] = genre_ica[:, i]

        logger.info(f"ジャンル特徴量ICA変換完了: {genre_ica_components}独立成分")

    n_new_features = len(df_features.columns) - len(df.columns)
    logger.success(f"次元削減特徴量を作成完了: {n_new_features}個の新特徴量を追加")

    return df_features


def evaluate_multicollinearity_impact(
    X_original: pd.DataFrame,
    X_cleaned: pd.DataFrame,
    y: pd.Series,
    removal_info: pd.DataFrame
) -> dict:
    """多重共線性除去前後の性能比較を行う。

    Args:
        X_original: 除去前の特徴量行列
        X_cleaned: 除去後の特徴量行列
        y: 目的変数
        removal_info: 除去された特徴量の情報

    Returns:
        Before/After比較結果のdict
    """
    logger.info("多重共線性除去前後の影響を評価中...")

    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import mean_squared_error
    import lightgbm as lgb

    # クロスバリデーション設定
    cv_folds = 3  # 軽量化のため3フォールド
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    model_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 50,  # 軽量化
        "verbose": -1,
        "random_state": 42
    }

    # Before: 除去前の評価
    logger.info("除去前の性能を評価中...")
    model_original = lgb.LGBMRegressor(**model_params)
    cv_scores_original = cross_val_score(
        model_original, X_original, y, cv=kfold, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    rmse_original = -cv_scores_original.mean()
    std_original = cv_scores_original.std()

    # After: 除去後の評価
    logger.info("除去後の性能を評価中...")
    model_cleaned = lgb.LGBMRegressor(**model_params)
    cv_scores_cleaned = cross_val_score(
        model_cleaned, X_cleaned, y, cv=kfold, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    rmse_cleaned = -cv_scores_cleaned.mean()
    std_cleaned = cv_scores_cleaned.std()

    # 性能比較結果
    improvement = rmse_original - rmse_cleaned
    improvement_pct = (improvement / rmse_original) * 100

    comparison_result = {
        "before_features_count": len(X_original.columns),
        "after_features_count": len(X_cleaned.columns),
        "removed_features_count": len(removal_info),
        "before_rmse": rmse_original,
        "before_rmse_std": std_original,
        "after_rmse": rmse_cleaned,
        "after_rmse_std": std_cleaned,
        "rmse_improvement": improvement,
        "improvement_percentage": improvement_pct,
        "removed_features": removal_info["removed_feature"].tolist() if not removal_info.empty else []
    }

    # 結果ログ
    logger.info("=== 多重共線性除去の効果 ===")
    logger.info(f"除去前RMSE: {rmse_original:.4f} (±{std_original:.4f})")
    logger.info(f"除去後RMSE: {rmse_cleaned:.4f} (±{std_cleaned:.4f})")
    logger.info(f"改善: {improvement:.4f} ({improvement_pct:+.2f}%)")
    logger.info(f"特徴量数: {len(X_original.columns)} → {len(X_cleaned.columns)} (-{len(removal_info)})")

    return comparison_result


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    validation_path: Path = PROCESSED_DATA_DIR / "validation.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    create_interactions: bool = True,
    create_comprehensive_interactions: bool = False,
    create_duration: bool = True,
    create_statistical: bool = True,
    create_genre: bool = True,
    create_advanced: bool = False,
    create_rhythm: bool = False,
    create_log_features: bool = False,
    create_binning_features: bool = False,
    create_dimensionality_reduction: bool = False,
    apply_pca: bool = True,
    apply_ica: bool = True,
    pca_variance_threshold: float = 0.95,
    ica_components: int = None,
    remove_multicollinearity: bool = False,
    multicollinearity_threshold: float = 0.7,
    prioritize_genre_features: bool = True,
    select_features_flag: bool = False,
    feature_selection_method: str = "kbest",
    n_features: int = 20,
    apply_scaling: bool = True,
    scaler_type: str = "standard",
):
    """処理済みデータセットから拡張特徴量を生成する。

    Args:
        train_path: 訓練データCSVのパス
        validation_path: 検証データCSVのパス
        test_path: テストデータCSVのパス
        output_dir: 拡張特徴量を保存するディレクトリ
        create_interactions: 交互作用特徴量を作成するかどうか
        create_comprehensive_interactions: 包括的交互作用特徴量を作成するかどうか（Kaggleサンプルコード手法）
        create_duration: 時間ベースの特徴量を作成するかどうか
        create_statistical: 統計的特徴量を作成するかどうか
        create_genre: 音楽ジャンル推定特徴量を作成するかどうか
        create_advanced: 独立性の高い高次特徴量を作成するかどうか
        create_rhythm: ドラマー視点リズム周期性特徴量を作成するかどうか
        create_log_features: 対数変換特徴量を作成するかどうか（TICKET-017-02）
        create_binning_features: ビニング・カテゴリ特徴量を作成するかどうか（TICKET-017-03）
        create_dimensionality_reduction: PCA・ICA次元削減特徴量を作成するかどうか
        apply_pca: PCA変換を適用するかどうか（次元削減有効時）
        apply_ica: ICA変換を適用するかどうか（次元削減有効時）
        pca_variance_threshold: PCA主成分数決定の累積寄与率閾値
        ica_components: ICA成分数（指定されない場合は元特徴量数の50%）
        remove_multicollinearity: 多重共線性除去を行うかどうか
        multicollinearity_threshold: 多重共線性検出の相関閾値
        prioritize_genre_features: 多重共線性除去時にジャンル特徴量を優先するかどうか
        select_features_flag: 特徴量選択を適用するかどうか
        feature_selection_method: 特徴量選択の手法
        n_features: 選択する特徴量の数（選択有効時）
        apply_scaling: 特徴量スケーリングを適用するかどうか
        scaler_type: 適用する特徴量スケーリングの種類
    """
    logger.info("特徴量エンジニアリング処理を開始...")

    # データセットの読み込み
    logger.info("データセットを読み込み中...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(validation_path) if validation_path.exists() else None
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    datasets = {"train": train_df}
    if val_df is not None:
        datasets["validation"] = val_df
    if test_df is not None:
        datasets["test"] = test_df

    # 各データセットの処理
    enhanced_datasets = {}

    for name, df in tqdm(datasets.items(), desc="データセット処理中"):
        logger.info(f"{name}データセットを処理中...")
        enhanced_df = df.copy()

        # 交互作用特徴量の作成
        if create_interactions:
            enhanced_df = create_interaction_features(enhanced_df)

        # 包括的交互作用特徴量の作成（Kaggleサンプルコード手法）
        if create_comprehensive_interactions:
            enhanced_df = create_comprehensive_interaction_features(enhanced_df)

        # 時間特徴量の作成
        if create_duration:
            enhanced_df = create_duration_features(enhanced_df)

        # 統計的特徴量の作成
        if create_statistical:
            enhanced_df = create_statistical_features(enhanced_df)

        # 音楽ジャンル推定特徴量の作成
        if create_genre:
            enhanced_df = create_music_genre_features(enhanced_df)

        # 独立性の高い高次特徴量の作成
        if create_advanced:
            enhanced_df = create_advanced_features(enhanced_df)

        # ドラマー視点リズム周期性特徴量の作成
        if create_rhythm:
            enhanced_df = create_rhythm_periodicity_features(enhanced_df)

        # 対数変換特徴量の作成（TICKET-017-02）
        if create_log_features:
            enhanced_df = create_log_features(enhanced_df)

        # ビニング・カテゴリ特徴量の作成（TICKET-017-03）
        if create_binning_features:
            enhanced_df = create_binning_features(enhanced_df)

        # PCA・ICA次元削減特徴量の作成
        if create_dimensionality_reduction:
            enhanced_df = create_dimensionality_reduction_features(
                enhanced_df,
                apply_pca=apply_pca,
                apply_ica=apply_ica,
                pca_variance_threshold=pca_variance_threshold,
                ica_components=ica_components
            )

        enhanced_datasets[name] = enhanced_df
        logger.info(f"{name}データセット: {enhanced_df.shape[1]}特徴量を生成")

    # 特徴量行列の準備
    feature_cols = [
        col for col in enhanced_datasets["train"].columns if col not in ["id", "BeatsPerMinute"]
    ]

    X_train = enhanced_datasets["train"][feature_cols]
    y_train = (
        enhanced_datasets["train"]["BeatsPerMinute"]
        if "BeatsPerMinute" in enhanced_datasets["train"].columns
        else None
    )

    X_val = (
        enhanced_datasets.get("validation", {}).get(feature_cols)
        if "validation" in enhanced_datasets
        else None
    )
    X_test = (
        enhanced_datasets.get("test", {}).get(feature_cols)
        if "test" in enhanced_datasets
        else None
    )

    # 多重共線性除去処理
    multicollinearity_results = None
    if remove_multicollinearity:
        logger.info(f"多重共線性除去を実行中（閾値: {multicollinearity_threshold}）...")

        # 訓練データで多重共線性を検出・除去
        X_train_original = X_train.copy()
        X_train_cleaned, removal_info = remove_correlated_features(
            X_train,
            correlation_threshold=multicollinearity_threshold,
            prioritize_genre_features=prioritize_genre_features
        )

        # 検証・テストデータにも同じ除去を適用
        removed_features = removal_info["removed_feature"].tolist() if not removal_info.empty else []

        if X_val is not None:
            X_val = X_val.drop(columns=[col for col in removed_features if col in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(columns=[col for col in removed_features if col in X_test.columns])

        X_train = X_train_cleaned

        # Before/After性能比較（訓練データに目的変数がある場合のみ）
        if y_train is not None and not removal_info.empty:
            multicollinearity_results = evaluate_multicollinearity_impact(
                X_train_original, X_train_cleaned, y_train, removal_info
            )

        # 多重共線性除去結果を保存
        if not removal_info.empty:
            removal_info.to_csv(output_dir / "multicollinearity_removal_info.csv", index=False)

            # 高相関ペア情報も保存
            high_corr_pairs = detect_multicollinearity(X_train_original, multicollinearity_threshold)
            if not high_corr_pairs.empty:
                high_corr_pairs.to_csv(output_dir / "high_correlation_pairs.csv", index=False)

        logger.info(f"多重共線性除去完了: 最終特徴量数 {len(X_train.columns)}")

    # 特徴量選択
    if select_features_flag and y_train is not None:
        X_train, X_val = select_features(
            X_train, y_train, X_val, method=feature_selection_method, k=n_features
        )
        if X_test is not None:
            # テストセットに同じ特徴量選択を適用
            selected_features = X_train.columns
            X_test = X_test[selected_features]

    # 特徴量スケーリング
    if apply_scaling:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_val, X_test, scaler_type=scaler_type
        )

        # スケール済みバージョンで更新
        X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled

    # 拡張データセットの保存
    logger.info("拡張特徴量データセットを保存中...")

    # 拡張特徴量で完全なデータセットを再構築
    train_enhanced = enhanced_datasets["train"][["id"]].copy()
    if "BeatsPerMinute" in enhanced_datasets["train"].columns:
        train_enhanced["BeatsPerMinute"] = enhanced_datasets["train"]["BeatsPerMinute"]
    train_enhanced = pd.concat([train_enhanced, X_train], axis=1)
    train_enhanced.to_csv(output_dir / "train_features.csv", index=False)

    if X_val is not None:
        val_enhanced = enhanced_datasets["validation"][["id"]].copy()
        if "BeatsPerMinute" in enhanced_datasets["validation"].columns:
            val_enhanced["BeatsPerMinute"] = enhanced_datasets["validation"]["BeatsPerMinute"]
        val_enhanced = pd.concat([val_enhanced, X_val], axis=1)
        val_enhanced.to_csv(output_dir / "validation_features.csv", index=False)

    if X_test is not None:
        test_enhanced = enhanced_datasets["test"][["id"]].copy()
        test_enhanced = pd.concat([test_enhanced, X_test], axis=1)
        test_enhanced.to_csv(output_dir / "test_features.csv", index=False)

    # 特徴量情報の保存
    feature_info = pd.DataFrame(
        {"feature_name": X_train.columns, "feature_type": ["engineered"] * len(X_train.columns)}
    )
    feature_info.to_csv(output_dir / "feature_info.csv", index=False)

    # 特徴量重要度分析（訓練データにターゲットがある場合のみ）
    if y_train is not None:
        logger.info("特徴量重要度分析を実行中...")

        # 全特徴量の重要度分析
        all_importance = analyze_feature_importance(X_train, y_train, "all")
        all_importance.to_csv(output_dir / "feature_importance_all.csv", index=False)

        # ジャンル特徴量の重要度分析
        if create_genre:
            genre_importance = analyze_feature_importance(X_train, y_train, "genre")
            if not genre_importance.empty:
                genre_importance.to_csv(output_dir / "feature_importance_genre.csv", index=False)

                # ジャンル特徴量とBPMの関係分析
                genre_bpm_analysis = compare_genre_features_to_bpm(X_train, y_train)
                genre_bpm_analysis.to_csv(output_dir / "genre_bpm_analysis.csv", index=False)

                logger.info("ジャンル特徴量分析結果も保存しました")

        # 多重共線性除去結果も保存
        if multicollinearity_results is not None:
            import json
            multicollinearity_file = output_dir / "multicollinearity_impact_results.json"
            with open(multicollinearity_file, "w", encoding="utf-8") as f:
                json.dump(multicollinearity_results, f, indent=2, ensure_ascii=False)
            logger.info(f"多重共線性除去の効果分析結果を保存: {multicollinearity_file}")

    logger.success(f"特徴量エンジニアリング完了: {len(X_train.columns)}特徴量を生成")
    logger.info(f"出力ディレクトリ: {output_dir}")

    # 実行結果サマリー
    if remove_multicollinearity and multicollinearity_results:
        logger.info("=== 多重共線性除去サマリー ===")
        logger.info(f"除去された特徴量数: {multicollinearity_results['removed_features_count']}")
        logger.info(f"性能改善: {multicollinearity_results['rmse_improvement']:+.4f}")
        logger.info(f"改善率: {multicollinearity_results['improvement_percentage']:+.2f}%")

    # 実装テスト: --create-advancedオプションの動作確認
    if create_advanced and logger.level.name == "DEBUG":
        logger.info("=== TICKET-008-02実装テスト実行中 ===")

        # テスト用サンプルデータ作成
        test_data = pd.DataFrame({
            'RhythmScore': [0.7, 0.8, 0.6, 0.9, 0.5],
            'AudioLoudness': [0.6, 0.7, 0.5, 0.8, 0.4],
            'VocalContent': [0.8, 0.6, 0.9, 0.5, 0.7],
            'AcousticQuality': [0.5, 0.8, 0.7, 0.6, 0.9],
            'InstrumentalScore': [0.7, 0.5, 0.8, 0.9, 0.6],
            'LivePerformanceLikelihood': [0.4, 0.6, 0.5, 0.7, 0.8],
            'MoodScore': [0.6, 0.7, 0.8, 0.5, 0.9],
            'TrackDurationMs': [200000, 180000, 220000, 240000, 160000],
            'Energy': [0.8, 0.9, 0.7, 0.6, 0.5]
        })

        original_features = len(test_data.columns)
        logger.info(f"テストデータ: {test_data.shape[0]}サンプル, {original_features}特徴量")

        # 高次特徴量作成テスト
        try:
            enhanced_test_data = create_advanced_features(test_data)
            new_features = len(enhanced_test_data.columns) - original_features

            # 1. 特徴量数確認
            expected_features = 18  # 4+4+5+5=18個の新特徴量
            if new_features == expected_features:
                logger.success(f"✓ 特徴量数テスト: {new_features}個の新特徴量を正常に追加")
            else:
                logger.warning(f"⚠ 特徴量数不一致: 期待{expected_features}個, 実際{new_features}個")

            # 2. エラー検証（NaN, inf値の確認）
            nan_count = enhanced_test_data.isnull().sum().sum()
            inf_count = np.isinf(enhanced_test_data.select_dtypes(include=[np.number])).sum().sum()

            if nan_count == 0 and inf_count == 0:
                logger.success(f"✓ エラーテスト: NaN({nan_count}), inf({inf_count})値なし")
            else:
                logger.error(f"✗ エラー検出: NaN({nan_count}), inf({inf_count})値あり")

            # 3. 新特徴量の統計情報表示
            new_feature_names = enhanced_test_data.columns[original_features:]
            logger.info("=== 新特徴量統計情報 ===")

            for feature in new_feature_names:
                values = enhanced_test_data[feature]
                stats = f"{feature}: 平均={values.mean():.3f}, 標準偏差={values.std():.3f}, 範囲=[{values.min():.3f}, {values.max():.3f}]"
                logger.info(stats)

            # 4. 特徴量妥当性確認
            validity_checks = {
                "比率特徴量": [col for col in new_feature_names if 'ratio' in col],
                "対数変換特徴量": [col for col in new_feature_names if 'log_duration' in col or 'duration_category' in col],
                "標準化済み特徴量": [col for col in new_feature_names if 'standardized' in col],
                "音楽理論特徴量": [col for col in new_feature_names if any(x in col for x in ['tempo', 'performance', 'music', 'harmonic', 'song'])]
            }

            logger.info("=== 特徴量カテゴリ別確認 ===")
            for category, features in validity_checks.items():
                if features:
                    logger.info(f"{category}: {len(features)}個 - {', '.join(features[:2])}{'...' if len(features) > 2 else ''}")
                else:
                    logger.warning(f"{category}: 0個（期待値と異なる可能性）")

            # 5. 特定特徴量の値域チェック
            if 'duration_category' in enhanced_test_data.columns:
                cat_values = enhanced_test_data['duration_category'].unique()
                if set(cat_values).issubset({0, 1, 2}):
                    logger.success(f"✓ duration_categoryの値域確認: {sorted(cat_values)}")
                else:
                    logger.error(f"✗ duration_category値域エラー: {cat_values}")

            logger.success("=== TICKET-008-02実装テスト完了 ===")

        except Exception as e:
            logger.error(f"✗ 実装テスト失敗: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"詳細: {traceback.format_exc()}")

        logger.info("実装テスト詳細は DEBUG レベルでのみ表示されます")


if __name__ == "__main__":
    app()
