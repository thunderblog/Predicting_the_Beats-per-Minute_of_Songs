#!/usr/bin/env python3
"""
TICKET-017正則化版75特徴量データセット生成スクリプト

最高LB性能26.38534を記録したTICKET-017正則化版実験の
75特徴量データセットを再現します。

生成手順:
1. 基本データ読み込み (11特徴量)
2. 包括的交互作用特徴量生成 (11→137)
3. 対数変換特徴量生成 (137→186)
4. SelectKBest特徴量選択 (186→75)
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from loguru import logger
import typer

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))
from scripts.my_config import config
from src.features import create_comprehensive_interaction_features, create_log_features

app = typer.Typer()

@app.command()
def main(
    train_path: Path = config.processed_data_dir / "train.csv",
    validation_path: Path = config.processed_data_dir / "validation.csv",
    test_path: Path = config.processed_data_dir / "test.csv",
    output_dir: Path = config.processed_data_dir,
    n_features: int = 75
):
    """TICKET-017正則化版75特徴量データセットを生成"""

    logger.info("TICKET-017正則化版75特徴量データセット生成開始...")

    # TODO(human): 基本データ読み込みと結合
    # train.csvとvalidation.csvを読み込み、結合して完全な訓練データを作成

    # TODO(human): 包括的交互作用特徴量生成 (11→137特徴量)
    # create_comprehensive_interaction_features関数を使用

    # TODO(human): 対数変換特徴量生成 (137→186特徴量)
    # create_log_features関数を使用

    # TODO(human): SelectKBest特徴量選択 (186→75特徴量)
    # SelectKBest(score_func=f_regression, k=75)を使用してF統計量ベースで選択

    # TODO(human): テストデータにも同じ特徴量変換を適用
    # 訓練データで学習したselectorを使用してテストデータを変換

    # TODO(human): 結果データ保存
    # train_ticket017_75_features.csv, test_ticket017_75_features.csvとして保存

    logger.success("75特徴量データセット生成完了")

if __name__ == "__main__":
    app()