import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


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
        threshold: 相関係数の闾値（この値以上で高相関とみなす）

    Returns:
        高相関ペア情報のDataFrame（feature_1, feature_2, correlation, priority_suggestion）
    """
    logger.info(f"多重共線性検出中（闾値: {threshold}）...")

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