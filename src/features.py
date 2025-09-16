from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
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


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    validation_path: Path = PROCESSED_DATA_DIR / "validation.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    create_interactions: bool = True,
    create_duration: bool = True,
    create_statistical: bool = True,
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

    logger.success(f"特徴量エンジニアリング完了: {len(X_train.columns)}特徴量を生成")
    logger.info(f"出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    app()
