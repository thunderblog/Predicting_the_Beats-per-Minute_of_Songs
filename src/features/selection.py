from typing import Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    method: str = "kbest",
    k: int = 20,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """様々な選択手法を使用して最も重要な特徴量を選択する。

    Args:
        X_train: 訓練用特徴量行列
        y_train: 訓練用目的変数値
        X_val: 検証用特徴量行列（オプション）
        method: 特徴量選択手法 ('kbest', 'mutual_info', 'correlation', 'combined')
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