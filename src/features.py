from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR

app = typer.Typer()


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """既存の変数間の交互作用特徴量を作成する。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        交互作用特徴量が追加されたデータフレーム
    """
    logger.info("交互作用特徴量を作成中...")

    # 元のデータフレームをコピー
    df_features = df.copy()

    # リズムとエネルギーの交互作用
    df_features["rhythm_energy_product"] = df["RhythmScore"] * df["Energy"]
    df_features["rhythm_energy_ratio"] = df["RhythmScore"] / (df["Energy"] + 1e-8)

    # 音声特徴量の組み合わせ
    df_features["loudness_vocal_product"] = df["AudioLoudness"] * df["VocalContent"]
    df_features["acoustic_instrumental_ratio"] = df["AcousticQuality"] / (
        df["InstrumentalScore"] + 1e-8
    )

    # パフォーマンスとムード特徴量
    df_features["live_mood_product"] = df["LivePerformanceLikelihood"] * df["MoodScore"]
    df_features["energy_mood_product"] = df["Energy"] * df["MoodScore"]

    # 複雑な交互作用
    df_features["rhythm_mood_energy"] = df["RhythmScore"] * df["MoodScore"] * df["Energy"]

    return df_features


def create_music_genre_features(df: pd.DataFrame) -> pd.DataFrame:
    """音楽ジャンル推定に基づく特徴量を作成する。

    音楽理論に基づき、特徴量の組み合わせから暗黙的な
    ジャンル特徴量を推定してBPM予測精度を向上させる。

    Args:
        df: 元の特徴量を含むデータフレーム

    Returns:
        音楽ジャンル推定特徴量が追加されたデータフレーム
    """
    logger.info("音楽ジャンル推定特徴量を作成中...")

    df_features = df.copy()

    # ダンス系ジャンル特徴量: Energy×RhythmScore
    # 高エネルギー & 高リズムスコア = EDM/ダンス系楽曲の特徴 (通常120-140+ BPM)
    df_features["dance_genre_score"] = df["Energy"] * df["RhythmScore"]

    # アコースティック系ジャンル特徴量: AcousticQuality×InstrumentalScore
    # 高音響品質 & 高楽器演奏スコア = フォーク/クラシック系楽曲の特徴 (通常60-120 BPM)
    df_features["acoustic_genre_score"] = df["AcousticQuality"] * df["InstrumentalScore"]

    # バラード系ジャンル特徴量: VocalContent×MoodScore
    # 高ボーカル含有量 & 高ムードスコア = バラード/R&B系楽曲の特徴 (通常70-100 BPM)
    df_features["ballad_genre_score"] = df["VocalContent"] * df["MoodScore"]

    # 追加のジャンル関連特徴量
    # ロック/ポップ系: 中程度のエネルギー × ライブ演奏っぽさ (通常90-130 BPM)
    df_features["rock_genre_score"] = df["Energy"] * df["LivePerformanceLikelihood"]

    # エレクトロニック系: 低ボーカル × 高エネルギー (通常100-180 BPM)
    df_features["electronic_genre_score"] = (
        1 - df["VocalContent"] / (df["VocalContent"].max() + 1e-8)
    ) * df["Energy"]

    # アンビエント/チルアウト系: 低エネルギー × 高音響品質 (通常60-90 BPM)
    df_features["ambient_genre_score"] = (1 - df["Energy"] / (df["Energy"].max() + 1e-8)) * df[
        "AcousticQuality"
    ]

    return df_features


def create_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """TrackDurationMsから時間に基づく特徴量を作成する。

    Args:
        df: TrackDurationMs列を含む入力データフレーム

    Returns:
        時間特徴量が追加されたデータフレーム
    """
    logger.info("トラック時間特徴量を作成中...")

    df_features = df.copy()

    # ミリ秒を他の時間単位に変換
    df_features["track_duration_seconds"] = df["TrackDurationMs"] / 1000
    df_features["track_duration_minutes"] = df["TrackDurationMs"] / (1000 * 60)

    # 時間カテゴリ
    df_features["is_short_track"] = (df["TrackDurationMs"] < 180000).astype(int)  # 3分未満
    df_features["is_long_track"] = (df["TrackDurationMs"] > 300000).astype(int)  # 5分超

    # 時間区分
    duration_bins = [0, 120000, 180000, 240000, 300000, float("inf")]
    duration_labels = ["very_short", "short", "medium", "long", "very_long"]
    df_features["duration_category"] = pd.cut(
        df["TrackDurationMs"], bins=duration_bins, labels=duration_labels, include_lowest=True
    ).astype(str)

    # 時間カテゴリのワンホットエンコーディング
    duration_dummies = pd.get_dummies(df_features["duration_category"], prefix="duration")
    df_features = pd.concat([df_features, duration_dummies], axis=1)
    df_features.drop("duration_category", axis=1, inplace=True)

    return df_features


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """既存の数値列から統計的特徴量を作成する。

    Args:
        df: 入力データフレーム

    Returns:
        統計的特徴量が追加されたデータフレーム
    """
    logger.info("統計的特徴量を作成中...")

    df_features = df.copy()

    # 数値特徴量を選択（idとターゲットを除く）
    numerical_cols = [
        "RhythmScore",
        "AudioLoudness",
        "VocalContent",
        "AcousticQuality",
        "InstrumentalScore",
        "LivePerformanceLikelihood",
        "MoodScore",
        "Energy",
    ]

    # 全スコアの合計
    df_features["total_score"] = df[numerical_cols].sum(axis=1)

    # 全スコアの平均
    df_features["mean_score"] = df[numerical_cols].mean(axis=1)

    # スコアの標準偏差
    df_features["std_score"] = df[numerical_cols].std(axis=1)

    # 最小値と最大値
    df_features["min_score"] = df[numerical_cols].min(axis=1)
    df_features["max_score"] = df[numerical_cols].max(axis=1)
    df_features["range_score"] = df_features["max_score"] - df_features["min_score"]

    return df_features


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    method: str = "kbest",
    k: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    logger.info(f"{method}法で特徴量選択中 (k={k})...")

    if method == "kbest":
        # F統計量による特徴量選択
        selector = SelectKBest(score_func=f_regression, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)

        # 選択された特徴量のインデックスを取得
        selected_indices = selector.get_support(indices=True)
        selected_features = X_train.columns[selected_indices]

        # DataFrameとして再構築
        X_train_result = pd.DataFrame(
            X_train_selected, columns=selected_features, index=X_train.index
        )

        # 検証データがある場合は同じ特徴量で変換
        if X_val is not None:
            X_val_result = pd.DataFrame(
                selector.transform(X_val), columns=selected_features, index=X_val.index
            )
        else:
            X_val_result = None

    elif method == "mutual_info":
        # 相互情報量による特徴量選択
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)

        selected_indices = selector.get_support(indices=True)
        selected_features = X_train.columns[selected_indices]

        X_train_result = pd.DataFrame(
            X_train_selected, columns=selected_features, index=X_train.index
        )

        if X_val is not None:
            X_val_result = pd.DataFrame(
                selector.transform(X_val), columns=selected_features, index=X_val.index
            )
        else:
            X_val_result = None

    elif method == "correlation":
        # 相関係数による特徴量選択
        correlations = X_train.corrwith(y_train).abs()
        selected_features = correlations.nlargest(k).index.tolist()

        X_train_result = X_train[selected_features]
        X_val_result = X_val[selected_features] if X_val is not None else None

    elif method == "combined":
        # 組み合わせアプローチ：F統計量と相互情報量の平均スコア
        # F統計量による選択
        selector_f = SelectKBest(score_func=f_regression, k=len(X_train.columns))
        selector_f.fit(X_train, y_train)
        f_scores = selector_f.scores_

        # 相互情報量による選択
        selector_mi = SelectKBest(score_func=mutual_info_regression, k=len(X_train.columns))
        selector_mi.fit(X_train, y_train)
        mi_scores = selector_mi.scores_

        # スコアを正規化して平均を取る
        f_scores_normalized = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min())
        mi_scores_normalized = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
        combined_scores = (f_scores_normalized + mi_scores_normalized) / 2

        # 上位k個の特徴量を選択
        selected_indices = np.argsort(combined_scores)[-k:]
        selected_features = X_train.columns[selected_indices]

        X_train_result = X_train[selected_features]
        X_val_result = X_val[selected_features] if X_val is not None else None

    else:
        # 不明な手法の場合は元の特徴量を返す
        logger.warning(f"不明な特徴量選択手法: {method}。元の特徴量を使用します。")
        X_train_result = X_train
        X_val_result = X_val

    logger.success(f"{len(X_train_result.columns)}個の特徴量を選択しました")
    return X_train_result, X_val_result


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

    # 3. 相互情報量による重要度
    mi_selector = SelectKBest(score_func=mutual_info_regression, k="all")
    mi_selector.fit(X_filtered, y)
    mi_scores = mi_selector.scores_

    # 4. Random Forestによる重要度
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


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    validation_path: Path = PROCESSED_DATA_DIR / "validation.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    create_interactions: bool = True,
    create_duration: bool = True,
    create_statistical: bool = True,
    create_genre: bool = True,
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
        create_duration: 時間ベースの特徴量を作成するかどうか
        create_statistical: 統計的特徴量を作成するかどうか
        create_genre: 音楽ジャンル推定特徴量を作成するかどうか
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

        # 時間特徴量の作成
        if create_duration:
            enhanced_df = create_duration_features(enhanced_df)

        # 統計的特徴量の作成
        if create_statistical:
            enhanced_df = create_statistical_features(enhanced_df)

        # 音楽ジャンル推定特徴量の作成
        if create_genre:
            enhanced_df = create_music_genre_features(enhanced_df)

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

    logger.success(f"特徴量エンジニアリング完了: {len(X_train.columns)}特徴量を生成")
    logger.info(f"出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    app()
