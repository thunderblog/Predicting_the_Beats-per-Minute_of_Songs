from typing import Tuple, Optional

import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    scaler_type: str = "standard",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], object]:
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
    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler()
    }

    scaler = scalers.get(scaler_type, StandardScaler())

    # 訓練データでフィット
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )

    # 検証データが提供されている場合は変換
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), columns=X_val.columns, index=X_val.index
        )

    # テストデータが提供されている場合は変換
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler