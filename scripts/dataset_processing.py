"""
TICKET-001: データセット処理機能の実装
KaggleのBPM予測コンペティション用データセット処理スクリプト
"""

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from bpm.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from script.my_config import config


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生データを読み込む.
    
    Returns:
        Tuple containing (train_df, test_df, sample_submission_df)
    """
    logger.info("生データを読み込み中...")
    
    # TODO(human): ファイルパスの設定
    # RAW_DATA_DIRを使用して、以下の3つのファイルパスを作成してください:
    # - train.csv
    # - test.csv  
    # - sample_submission.csv
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission_df = pd.read_csv(sample_path)
    
    logger.info(f"訓練データ: {train_df.shape}")
    logger.info(f"テストデータ: {test_df.shape}")
    logger.info(f"サンプルサブミッション: {sample_submission_df.shape}")
    
    return train_df, test_df, sample_submission_df


def validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """データ品質と整合性をチェックする.
    
    Args:
        train_df: 訓練データセット
        test_df: テストデータセット
    """
    logger.info("データ品質をチェック中...")
    
    # TODO(human): 欠損値チェックの実装
    # train_dfとtest_dfの欠損値の総数を計算してください
    # ヒント: .isnull().sum().sum() を使用
    
    if train_missing > 0:
        logger.warning(f"訓練データに {train_missing} 個の欠損値があります")
    else:
        logger.info("訓練データに欠損値はありません")
        
    if test_missing > 0:
        logger.warning(f"テストデータに {test_missing} 個の欠損値があります")
    else:
        logger.info("テストデータに欠損値はありません")
    
    # 特徴量の整合性チェック
    train_features = set(train_df.columns) - {'id', config.target}
    test_features = set(test_df.columns) - {'id'}
    
    if train_features != test_features:
        logger.error(f"特徴量の不整合:")
        logger.error(f"  訓練データのみ: {train_features - test_features}")
        logger.error(f"  テストデータのみ: {test_features - train_features}")
        raise ValueError("訓練データとテストデータの特徴量が一致しません")
    
    # TODO(human): 設定ファイルとの整合性チェック
    # config.featuresと実際のデータの特徴量を比較してください
    # データの特徴量はtrain_featuresとして既に計算済みです
    
    logger.success("データ品質チェック完了")


def analyze_target_distribution(train_df: pd.DataFrame) -> None:
    """ターゲット変数の分布を分析する.
    
    Args:
        train_df: 訓練データセット
    """
    logger.info("ターゲット変数の分析中...")
    
    target_col = config.target
    
    # TODO(human): 基本統計の計算
    # train_df[target_col]の基本統計量を計算して表示してください
    # ヒント: .describe() を使用
    
    # TODO(human): 外れ値の検出
    # IQR法を使用して外れ値を検出してください
    # Q1, Q3を計算し、IQR = Q3 - Q1として外れ値の閾値を設定
    
    logger.info(f"外れ値の可能性のあるデータ: {outliers_count}件")
    
    logger.success("ターゲット分析完了")


def split_train_validation(
    train_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """訓練データを訓練セットと検証セットに分割する.
    
    Args:
        train_df: 訓練データセット
        
    Returns:
        Tuple containing (train_split, val_split)
    """
    logger.info(f"訓練データを分割中 (validation_size={config.test_size})...")
    
    # TODO(human): データ分割の実装
    # train_test_splitを使用してデータを分割してください
    # パラメータ:
    # - test_size: config.test_size
    # - random_state: config.random_state
    # - stratify: None (回帰タスクのため)
    
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
    # TODO(human): 特徴量要約の作成
    # 各特徴量について以下の統計を計算してください:
    # - データ型
    # - 欠損値の数
    # - ユニーク値の数
    # - 最小値、最大値、平均値
    
    return summary_df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame
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
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # TODO(human): 保存パスの定義
    # PROCESSED_DATA_DIRを使用して以下のファイルパスを作成してください:
    # - train.csv
    # - validation.csv
    # - test.csv
    # - sample_submission.csv
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    sample_submission_df.to_csv(sample_path, index=False)
    
    logger.success(f"処理済みデータを保存完了:")
    logger.success(f"  訓練データ: {train_path}")
    logger.success(f"  検証データ: {val_path}")
    logger.success(f"  テストデータ: {test_path}")
    logger.success(f"  サンプルサブミッション: {sample_path}")


def main():
    """データセット処理のメイン処理."""
    logger.info(f"データセット処理を開始 (実験名: {config.exp_name})")
    
    try:
        # 生データの読み込み
        train_df, test_df, sample_submission_df = load_raw_data()
        
        # データ品質の検証
        validate_data(train_df, test_df)
        
        # ターゲット変数の分析
        analyze_target_distribution(train_df)
        
        # 特徴量要約の作成と保存
        feature_summary = create_feature_summary(train_df)
        
        # TODO(human): 特徴量要約の保存
        # feature_summaryをPROCESSED_DATA_DIR/feature_summary.csvに保存してください
        
        # 訓練データの分割
        train_split, val_split = split_train_validation(train_df)
        
        # 処理済みデータの保存
        save_processed_data(train_split, val_split, test_df, sample_submission_df)
        
        logger.success("データセット処理が完了しました！")
        
    except Exception as e:
        logger.error(f"データセット処理中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()