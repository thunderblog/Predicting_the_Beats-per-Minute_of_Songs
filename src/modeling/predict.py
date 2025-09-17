from pathlib import Path
import pickle

import lightgbm as lgb
from loguru import logger
import numpy as np
import pandas as pd
from scripts.my_config import config
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_trained_models(exp_name: str, model_dir: Path) -> list[lgb.Booster]:
    """訓練済みLightGBMモデル（複数フォールド）を読み込む。

    Args:
        exp_name: 実験名
        model_dir: モデル保存ディレクトリ

    Returns:
        list[lgb.Booster]: 読み込まれたLightGBMモデルのリスト

    Raises:
        FileNotFoundError: モデルファイルが見つからない場合
    """
    models = []
    model_pattern = f"{exp_name}_fold_*_*.pkl"
    model_files = list(model_dir.glob(model_pattern))

    if not model_files:
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_dir}/{model_pattern}")

    logger.info(f"{len(model_files)}個のモデルファイルを読み込み中...")

    for model_file in sorted(model_files):
        logger.info(f"モデルを読み込み中: {model_file.name}")
        with open(model_file, "rb") as f:
            model = pickle.load(f)
            models.append(model)

    logger.success(f"{len(models)}個のモデルの読み込みが完了しました")
    return models


def make_ensemble_predictions(
    models: list[lgb.Booster], test_data: pd.DataFrame, feature_cols: list
) -> np.ndarray:
    """アンサンブル予測を実行する（複数モデルの平均）。

    Args:
        models: 訓練済みLightGBMモデルのリスト
        test_data: テストデータのDataFrame
        feature_cols: 特徴量列のリスト

    Returns:
        np.ndarray: アンサンブル予測値の配列
    """
    logger.info(
        f"アンサンブル予測を実行中... (データサイズ: {len(test_data)}, モデル数: {len(models)})"
    )

    # 特徴量データを抽出
    X_test = test_data[feature_cols]

    # 各モデルから予測を取得
    all_predictions = []
    for i, model in enumerate(models, 1):
        predictions = model.predict(X_test, num_iteration=model.best_iteration)
        all_predictions.append(predictions)
        logger.info(f"モデル{i}の予測完了 (平均予測値: {predictions.mean():.2f})")

    # アンサンブル（平均）
    ensemble_predictions = np.mean(all_predictions, axis=0)

    logger.success(
        f"アンサンブル予測が完了しました (平均予測値: {ensemble_predictions.mean():.2f})"
    )
    return ensemble_predictions


def save_submission(test_ids: pd.Series, predictions: np.ndarray, output_path: Path) -> None:
    """Kaggle提出形式でCSVファイルを保存する。

    Args:
        test_ids: テストデータのID列
        predictions: 予測値の配列
        output_path: 出力CSVファイルのパス
    """
    # 提出形式のDataFrameを作成
    submission_df = pd.DataFrame({"id": test_ids, config.target: predictions})

    # CSVファイルとして保存
    output_path.parent.mkdir(exist_ok=True, parents=True)
    submission_df.to_csv(output_path, index=False)

    logger.success(f"提出ファイルを保存しました: {output_path}")
    logger.info(f"提出データサイズ: {len(submission_df)} 行")
    logger.info(
        f"予測値の統計: 最小={predictions.min():.2f}, 最大={predictions.max():.2f}, 平均={predictions.mean():.2f}"
    )


def process_predictions(predictions: np.ndarray) -> np.ndarray:
    """予測値に対して後処理を実行する。

    Args:
        predictions: 生の予測値配列

    Returns:
        np.ndarray: 後処理済みの予測値配列
    """
    logger.info(
        f"予測値の後処理を実行中... (元の範囲: {predictions.min():.2f} - {predictions.max():.2f})"
    )

    # BPMの妥当な範囲でクリッピング（一般的に30-300 BPMが現実的）
    processed_predictions = np.clip(predictions, 30, 300)

    # 負の値や極端な値があった場合のログ出力
    clipped_count = np.sum((predictions < 30) | (predictions > 300))
    if clipped_count > 0:
        logger.warning(f"{clipped_count}個の予測値をクリッピングしました")

    logger.info(
        f"後処理完了 (新しい範囲: {processed_predictions.min():.2f} - {processed_predictions.max():.2f})"
    )
    return processed_predictions


@app.command()
def main(
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = None,
    output_path: Path = PROCESSED_DATA_DIR / "submission.csv",
    exp_name: str = config.exp_name,
):
    """訓練済みモデルを使用してテストデータの予測を実行する。

    Args:
        test_features_path: テストデータCSVのパス
        model_path: モデルファイルのパス（指定なしの場合は実験名から自動生成）
        output_path: 予測結果CSVの出力パス
        exp_name: 実験名（モデルファイル名の生成に使用）
    """
    logger.info(f"予測処理を開始 (実験名: {exp_name})...")

    # モデルパスが指定されていない場合は実験名から生成
    if model_path is None:
        model_path = MODELS_DIR / f"{exp_name}_lgb_model.pkl"

    # テストデータの読み込み
    logger.info(f"テストデータを読み込み中: {test_features_path}")
    if not test_features_path.exists():
        raise FileNotFoundError(f"テストデータファイルが見つかりません: {test_features_path}")

    test_df = pd.read_csv(test_features_path)
    logger.info(f"テストデータ形状: {test_df.shape}")

    # 特徴量列を取得（IDとターゲット以外）
    feature_cols = [col for col in test_df.columns if col not in ["id", config.target]]
    logger.info(f"特徴量数: {len(feature_cols)}")

    # 訓練済みモデルの読み込み
    models = load_trained_models(exp_name, model_path.parent)

    # アンサンブル予測実行
    predictions = make_ensemble_predictions(models, test_df, feature_cols)

    # 予測値の後処理
    processed_predictions = process_predictions(predictions)

    # Kaggle提出形式で保存
    save_submission(test_df["id"], processed_predictions, output_path)

    logger.success("予測処理が完了しました")


if __name__ == "__main__":
    app()
