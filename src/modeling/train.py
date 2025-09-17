from datetime import datetime
import json
from pathlib import Path
import pickle

import lightgbm as lgb
from loguru import logger
import numpy as np
import pandas as pd
from scripts.my_config import config
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    val_path: Path = PROCESSED_DATA_DIR / "validation_features.csv",
    model_dir: Path = MODELS_DIR,
    use_cross_validation: bool = True,
    n_folds: int = 5,
    exp_name: str = config.exp_name,
):
    """LightGBM回帰モデルの訓練を実行する。

    Args:
        train_path: 訓練データCSVのパス
        val_path: 検証データCSVのパス（クロスバリデーション無効時のみ使用）
        model_dir: モデル保存ディレクトリ
        use_cross_validation: クロスバリデーションを使用するかどうか
        n_folds: クロスバリデーションのフォールド数
        exp_name: 実験名（モデル保存時の識別用）
    """
    logger.info(f"LightGBM回帰モデルの訓練を開始 (実験名: {exp_name})...")

    # モデル保存ディレクトリを作成
    model_dir.mkdir(exist_ok=True, parents=True)

    # 訓練データの読み込み
    logger.info(f"訓練データを読み込み中: {train_path}")
    train_df = pd.read_csv(train_path)

    # 特徴量とターゲットを分離
    feature_cols = [col for col in train_df.columns if col not in ["id", config.target]]
    X_train = train_df[feature_cols]
    y_train = train_df[config.target]

    logger.info(f"特徴量数: {len(feature_cols)}, サンプル数: {len(X_train)}")

    if use_cross_validation:
        # クロスバリデーション実行
        cv_scores, models = train_with_cross_validation(X_train, y_train, n_folds=n_folds)

        # 結果の保存
        save_cv_results(cv_scores, models, model_dir, exp_name, feature_cols)

    else:
        # 単一モデル訓練（検証データ使用）
        logger.info(f"検証データを読み込み中: {val_path}")
        val_df = pd.read_csv(val_path)
        X_val = val_df[feature_cols]
        y_val = val_df[config.target]

        model, train_score, val_score = train_single_model(X_train, y_train, X_val, y_val)

        # モデルと結果の保存
        save_single_model(model, train_score, val_score, model_dir, exp_name, feature_cols)

    logger.success("LightGBM回帰モデルの訓練が完了しました。")


def train_with_cross_validation(X: pd.DataFrame, y: pd.Series, n_folds: int = 5):
    """クロスバリデーションでモデルを訓練する。

    Args:
        X: 特徴量データフレーム
        y: ターゲット変数
        n_folds: フォールド数

    Returns:
        cv_scores: 各フォールドのスコア
        models: 訓練済みモデルのリスト
    """
    logger.info(f"{n_folds}フォールドクロスバリデーションを実行中...")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=config.random_state)
    cv_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(X), total=n_folds)):
        logger.info(f"フォールド {fold + 1}/{n_folds} を訓練中...")

        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        # LightGBMデータセット作成
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

        # LightGBMパラメータ
        params = {
            "objective": config.objective,
            "metric": config.metric,
            "boosting_type": "gbdt",
            "num_leaves": config.num_leaves,
            "learning_rate": config.learning_rate,
            "feature_fraction": config.feature_fraction,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": config.random_state,
        }

        # モデル訓練
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "eval"],
            num_boost_round=config.n_estimators,
            callbacks=[
                lgb.early_stopping(config.stopping_rounds),
                lgb.log_evaluation(config.log_evaluation),
            ],
        )

        # 予測とスコア計算
        y_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))

        cv_scores.append(fold_rmse)
        models.append(model)

        logger.info(f"フォールド {fold + 1} RMSE: {fold_rmse:.4f}")

    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)

    logger.success("クロスバリデーション完了")
    logger.info(f"平均RMSE: {mean_cv_score:.4f} ± {std_cv_score:.4f}")

    return cv_scores, models


def train_single_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
):
    """単一モデルを訓練する。

    Args:
        X_train: 訓練用特徴量
        y_train: 訓練用ターゲット
        X_val: 検証用特徴量
        y_val: 検証用ターゲット

    Returns:
        model: 訓練済みモデル
        train_score: 訓練スコア
        val_score: 検証スコア
    """
    logger.info("単一モデルを訓練中...")

    # LightGBMデータセット作成
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # LightGBMパラメータ
    params = {
        "objective": config.objective,
        "metric": config.metric,
        "boosting_type": "gbdt",
        "num_leaves": config.num_leaves,
        "learning_rate": config.learning_rate,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": config.random_state,
    }

    # モデル訓練
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "eval"],
        num_boost_round=config.n_estimators,
        callbacks=[
            lgb.early_stopping(config.stopping_rounds),
            lgb.log_evaluation(config.log_evaluation),
        ],
    )

    # スコア計算
    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)

    train_score = np.sqrt(mean_squared_error(y_train, train_pred))
    val_score = np.sqrt(mean_squared_error(y_val, val_pred))

    logger.info(f"訓練RMSE: {train_score:.4f}")
    logger.info(f"検証RMSE: {val_score:.4f}")

    return model, train_score, val_score


def save_cv_results(
    cv_scores: list, models: list, model_dir: Path, exp_name: str, feature_cols: list
):
    """クロスバリデーション結果を保存する。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # スコア情報を保存
    results = {
        "experiment_name": exp_name,
        "timestamp": timestamp,
        "cv_scores": cv_scores,
        "mean_cv_score": np.mean(cv_scores),
        "std_cv_score": np.std(cv_scores),
        "n_folds": len(cv_scores),
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "config": {
            "objective": config.objective,
            "metric": config.metric,
            "num_leaves": config.num_leaves,
            "learning_rate": config.learning_rate,
            "feature_fraction": config.feature_fraction,
            "n_estimators": config.n_estimators,
            "stopping_rounds": config.stopping_rounds,
        },
    }

    # 結果をJSONで保存
    results_path = model_dir / f"{exp_name}_cv_results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # モデルを保存
    for i, model in enumerate(models):
        model_path = model_dir / f"{exp_name}_fold_{i + 1}_{timestamp}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    logger.success(f"クロスバリデーション結果を保存: {results_path}")
    logger.info(f"モデル保存完了: {len(models)}個のモデル")


def save_single_model(
    model, train_score: float, val_score: float, model_dir: Path, exp_name: str, feature_cols: list
):
    """単一モデル結果を保存する。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # スコア情報を保存
    results = {
        "experiment_name": exp_name,
        "timestamp": timestamp,
        "train_rmse": train_score,
        "val_rmse": val_score,
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "config": {
            "objective": config.objective,
            "metric": config.metric,
            "num_leaves": config.num_leaves,
            "learning_rate": config.learning_rate,
            "feature_fraction": config.feature_fraction,
            "n_estimators": config.n_estimators,
            "stopping_rounds": config.stopping_rounds,
        },
    }

    # 結果をJSONで保存
    results_path = model_dir / f"{exp_name}_results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # モデルを保存
    model_path = model_dir / f"{exp_name}_model_{timestamp}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.success(f"モデル結果を保存: {results_path}")
    logger.info(f"モデル保存完了: {model_path}")


if __name__ == "__main__":
    app()
