"""
TICKET-001: データセット処理機能の実装（改善版）
KaggleのBPM予測コンペティション用データセット処理スクリプト - 参照透過性とコード品質を向上
"""

from pathlib import Path
import sys
from typing import NamedTuple, Tuple

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
import typer

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.my_config import config

app = typer.Typer()


class RawDataset(NamedTuple):
    """生データセットを格納するデータクラス."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def load_raw_data(train_path: Path, test_path: Path, sample_submission_path: Path) -> RawDataset:
    """生データを読み込む（参照透過性を保つため、パスを引数で受け取る).

    Args:
        train_path: 訓練データのパス
        test_path: テストデータのパス
        sample_submission_path: サンプルサブミッションのパス

    Returns:
        RawDataset containing train, test, and sample_submission DataFrames

    Raises:
        FileNotFoundError: 指定されたファイルが存在しない場合
        pd.errors.EmptyDataError: CSVファイルが空の場合
        pd.errors.ParserError: CSVファイルの形式が不正な場合
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        sample_submission_df = pd.read_csv(sample_submission_path)

        return RawDataset(train=train_df, test=test_df, sample_submission=sample_submission_df)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"データファイルが見つかりません: {e.filename}") from e
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError("空のCSVファイルです") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"CSVファイルの解析に失敗しました: {e}") from e


def log_dataset_info(dataset: RawDataset) -> None:
    """データセットの情報をログ出力する（副作用を分離）.

    Args:
        dataset: 読み込まれたデータセット
    """
    logger.info("生データを読み込み完了")
    logger.info(f"訓練データ: {dataset.train.shape}")
    logger.info(f"テストデータ: {dataset.test.shape}")
    logger.info(f"サンプルサブミッション: {dataset.sample_submission.shape}")


def validate_data(dataset: RawDataset) -> None:
    """データ品質と整合性をチェックする.

    Args:
        dataset: 検証対象のデータセット
    """
    logger.info("データ品質をチェック中...")

    train_missing = dataset.train.isnull().sum().sum()
    test_missing = dataset.test.isnull().sum().sum()

    if train_missing > 0:
        logger.warning(f"訓練データに {train_missing} 個の欠損値があります")
    else:
        logger.info("訓練データに欠損値はありません")

    if test_missing > 0:
        logger.warning(f"テストデータに {test_missing} 個の欠損値があります")
    else:
        logger.info("テストデータに欠損値はありません")

    # 特徴量の整合性チェック
    train_features = set(dataset.train.columns) - {"id", config.target}
    test_features = set(dataset.test.columns) - {"id"}

    if train_features != test_features:
        logger.error("特徴量の不整合:")
        logger.error(f"  訓練データのみ: {train_features - test_features}")
        logger.error(f"  テストデータのみ: {test_features - train_features}")
        raise ValueError("訓練データとテストデータの特徴量が一致しません")

    if train_features != set(config.features):
        logger.error("特徴量の不整合:")
        logger.error(f"  設定ファイルにない特徴量: {set(config.features) - train_features}")
        logger.error(f"  データにない特徴量: {set(train_features) - set(config.features)}")
        raise ValueError("Configと実際のデータの特徴量が一致しません")

    logger.success("データ品質チェック完了")


def analyze_target_distribution(train_df: pd.DataFrame) -> None:
    """ターゲット変数の分布を分析する.

    Args:
        train_df: 訓練データセット
    """
    logger.info("ターゲット変数の分析中...")

    target_col = config.target

    # 基本統計の計算
    stats = train_df[target_col].describe()
    logger.info(f"ターゲット変数{target_col}の基本統計量: {stats.to_string()}")

    # 外れ値の検出
    # IQRの計算
    Q1 = train_df[target_col].quantile(0.25)
    Q3 = train_df[target_col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = (
        (train_df[target_col] < lower_bound) | (train_df[target_col] > upper_bound)
    ).sum()

    logger.info(f"四分位数: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    logger.info(f"外れ値判定閾値: 下限={lower_bound:.2f}, 上限={upper_bound:.2f}")
    logger.info(f"外れ値の可能性のあるデータ: {outliers_count}件")

    logger.success("ターゲット分析完了")


def split_train_validation(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """訓練データを訓練セットと検証セットに分割する.

    Args:
        train_df: 訓練データセット

    Returns:
        Tuple containing (train_split, val_split)
    """
    logger.info(f"訓練データを分割中 (validation_size={config.test_size})...")

    train_split, val_split = train_test_split(
        train_df, test_size=config.test_size, random_state=config.random_state, stratify=None
    )

    logger.info(f"訓練セット: {train_split.shape}")
    logger.info(f"検証セット: {val_split.shape}")

    # ターゲット分布の確認
    logger.info(f"訓練セットのターゲット統計: {train_split[config.target].describe()}")
    logger.info(f"検証セットのターゲット統計: {val_split[config.target].describe()}")

    return train_split, val_split


def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量の要約統計を作成する.

    Args:
        df: データフレーム

    Returns:
        特徴量の要約統計データフレーム
    """
    logger.info("特徴量要約を作成中...")

    # 全ての列に対して統計情報を収集
    data_dtypes = df.dtypes
    missing_counts = df.isnull().sum()
    unique_counts = df.nunique()

    # 数値列のみに絞って統計を計算
    numeric_df = df.select_dtypes(include=["number"])
    numeric_stats = numeric_df.describe()

    logger.info(
        f"処理対象: {len(df.columns)}列 (数値型: {len(numeric_df.columns)}列, 非数値型: {len(df.columns) - len(numeric_df.columns)}列)"
    )

    # 各列の統計情報を構築
    summary_data = []
    for col in df.columns:
        col_info = {
            "feature": col,
            "data_type": str(data_dtypes[col]),
            "missing_count": missing_counts[col],
            "unique_count": unique_counts[col],
        }

        # 数値列の場合は統計値を追加
        if col in numeric_df.columns:
            col_info.update(
                {
                    "min_value": numeric_stats.loc["min", col],
                    "max_value": numeric_stats.loc["max", col],
                    "mean_value": numeric_stats.loc["mean", col],
                }
            )
        else:
            # 非数値列の場合はNaNで埋める
            col_info.update({"min_value": pd.NA, "max_value": pd.NA, "mean_value": pd.NA})

        summary_data.append(col_info)

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index("feature")

    logger.info(
        f"特徴量要約を作成完了: {summary_df.shape[0]}特徴量 × {summary_df.shape[1]}統計項目"
    )

    # 特徴量要約の詳細をログ表示
    logger.info("特徴量要約詳細:")
    logger.info(f"\n{summary_df.to_string()}")

    # 欠損値がある特徴量を特別に報告
    features_with_missing = summary_df[summary_df["missing_count"] > 0]
    if not features_with_missing.empty:
        logger.warning(f"欠損値を含む特徴量: {len(features_with_missing)}個")
        for feature, row in features_with_missing.iterrows():
            logger.warning(f"  {feature}: {row['missing_count']}個の欠損値")
    else:
        logger.info("すべての特徴量に欠損値はありません")

    logger.success("特徴量要約の作成が完了しました")

    return summary_df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
) -> None:
    """処理済みデータセットを保存する.

    Args:
        train_df: 訓練データセット
        val_df: 検証データセット
        test_df: テストデータセット
        sample_submission_df: サンプルサブミッション
    """
    logger.info("処理済みデータを保存中...")

    # 保存先ディレクトリを確認・作成
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)

    train_path = config.get_processed_path("train")
    val_path = config.get_processed_path("validation")
    test_path = config.get_processed_path("test")
    sample_path = config.get_processed_path("sample_submission")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    sample_submission_df.to_csv(sample_path, index=False)

    logger.success("処理済みデータを保存完了:")
    logger.success(f"  訓練データ: {train_path}")
    logger.success(f"  検証データ: {val_path}")
    logger.success(f"  テストデータ: {test_path}")
    logger.success(f"  サンプルサブミッション: {sample_path}")


@app.command()
def main():
    """データセット処理のメイン処理."""
    logger.info(f"データセット処理を開始 (実験名: {config.exp_name})")

    try:
        # 生データの読み込み
        train_path = config.get_raw_path("train")
        test_path = config.get_raw_path("test")
        sample_path = config.get_raw_path("sample_submission")

        dataset = load_raw_data(train_path, test_path, sample_path)
        log_dataset_info(dataset)

        # データ品質の検証
        validate_data(dataset)

        # ターゲット変数の分析
        analyze_target_distribution(dataset.train)

        # 特徴量要約の作成と保存
        feature_summary = create_feature_summary(dataset.train)

        # 特徴量要約の保存
        feature_summary_path = config.get_processed_path("feature_summary")
        feature_summary.to_csv(feature_summary_path)
        logger.info(f"特徴量要約を保存: {feature_summary_path}")

        # 訓練データの分割
        train_split, val_split = split_train_validation(dataset.train)

        # 処理済みデータの保存
        save_processed_data(train_split, val_split, dataset.test, dataset.sample_submission)

        logger.success("データセット処理が完了しました！")

    except Exception as e:
        logger.error(f"データセット処理中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    app()
