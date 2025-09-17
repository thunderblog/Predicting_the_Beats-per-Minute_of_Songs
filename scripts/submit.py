"""
Kaggle BPM予測コンペティション用提出スクリプト

データ処理、モデル訓練（必要に応じて）、予測生成、
提出ファイル作成を行うエンドツーエンドパイプライン
"""

import sys
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from loguru import logger
import typer
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from scripts.my_config import config

app = typer.Typer()


def ensure_directories() -> None:
    """必要なディレクトリが存在しない場合は作成する"""
    directories = [MODELS_DIR, PROCESSED_DATA_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
        logger.info(f"ディレクトリを確認/作成: {directory}")


def load_experiment_models(exp_name: str, model_dir: Path) -> List[Any]:
    """指定された実験の訓練済みモデルを読み込む

    Args:
        exp_name: 実験名 (例: 'exp01')
        model_dir: モデルファイルが保存されているディレクトリ

    Returns:
        読み込まれたモデルのリスト

    Raises:
        FileNotFoundError: 実験のモデルファイルが見つからない場合
    """
    models = []
    model_pattern = f"{exp_name}_fold_*_*.pkl"
    model_files = list(model_dir.glob(model_pattern))

    if not model_files:
        raise FileNotFoundError(f"実験 {exp_name} のモデルファイルが見つかりません: {model_pattern}")

    logger.info(f"実験 {exp_name}: {len(model_files)}個のモデルファイルを読み込み中...")

    for model_file in sorted(model_files):
        logger.info(f"モデルを読み込み中: {model_file.name}")
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                models.append(model)
        except Exception as e:
            logger.error(f"モデルの読み込みに失敗: {model_file.name} - {e}")
            continue

    if not models:
        raise RuntimeError(f"実験 {exp_name} の有効なモデルが読み込めませんでした")

    logger.success(f"実験 {exp_name}: {len(models)}個のモデルの読み込みが完了")
    return models


def ensemble_predictions(predictions_list: List[np.ndarray], method: str = "mean") -> np.ndarray:
    """複数のモデル予測結果をアンサンブルする

    Args:
        predictions_list: 異なるモデル/実験からの予測結果のリスト
        method: アンサンブル方法 ('mean', 'median', 'weighted_mean')

    Returns:
        アンサンブル予測結果
    """
    if not predictions_list:
        raise ValueError("予測結果のリストが空です")

    predictions_array = np.column_stack(predictions_list)

    if method == "mean":
        ensemble_pred = np.mean(predictions_array, axis=1)
    elif method == "median":
        ensemble_pred = np.median(predictions_array, axis=1)
    elif method == "weighted_mean":
        # TODO(human): 重み付きアンサンブルロジックを実装
        # 現在は単純な平均にフォールバック
        logger.warning("重み付きアンサンブルは未実装のため、平均を使用します")
        ensemble_pred = np.mean(predictions_array, axis=1)
    else:
        raise ValueError(f"サポートされていないアンサンブル方法: {method}")

    logger.info(f"アンサンブル方法: {method}, モデル数: {len(predictions_list)}")
    return ensemble_pred


def create_submission_file(
    test_predictions: np.ndarray,
    test_ids: np.ndarray,
    output_path: Path,
    exp_names: List[str],
    ensemble_method: str
) -> None:
    """Kaggle提出用CSVファイルを作成する

    Args:
        test_predictions: テストセットの最終アンサンブル予測結果
        test_ids: テストサンプルのID
        output_path: 提出ファイルの保存パス
        exp_names: 使用した実験名のリスト
        ensemble_method: 使用したアンサンブル方法
    """
    submission_df = pd.DataFrame({
        'id': test_ids,
        config.target: test_predictions
    })

    # 提出ファイルを保存
    submission_df.to_csv(output_path, index=False)

    # 提出情報をログ出力
    logger.success(f"提出ファイルを作成: {output_path}")
    logger.info(f"使用実験: {', '.join(exp_names)}")
    logger.info(f"アンサンブル方法: {ensemble_method}")
    logger.info(f"予測統計: mean={test_predictions.mean():.2f}, "
               f"std={test_predictions.std():.2f}, "
               f"min={test_predictions.min():.2f}, "
               f"max={test_predictions.max():.2f}")


@app.command()
def main(
    exp_names: str = config.exp_name,
    ensemble_method: str = "mean",
    output_dir: Optional[Path] = None,
    test_data_path: Optional[Path] = None,
    run_full_pipeline: bool = False,
) -> None:
    """Kaggle提出用エンドツーエンドパイプラインを実行する

    Args:
        exp_names: カンマ区切りの実験名 (例: 'exp01,exp02')
        ensemble_method: アンサンブル方法 ('mean', 'median', 'weighted_mean')
        output_dir: 提出ファイルの出力ディレクトリ (デフォルト: models/)
        test_data_path: テストデータCSVのパス (デフォルト: processed/test_features.csv)
        run_full_pipeline: 予測前にデータ処理と訓練を実行するかどうか
    """
    logger.info("Kaggle BPM予測コンペティション - 提出パイプライン開始")

    # ディレクトリ設定
    ensure_directories()

    # 実験名を解析
    exp_list = [name.strip() for name in exp_names.split(',')]
    logger.info(f"使用する実験: {exp_list}")

    # デフォルトパスを設定
    if output_dir is None:
        output_dir = MODELS_DIR
    if test_data_path is None:
        test_data_path = PROCESSED_DATA_DIR / "test_features.csv"

    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        # フルパイプライン実行機能
        if run_full_pipeline:
            logger.info("フルパイプライン実行を開始します...")

            try:
                # ステップ1: データ処理
                logger.info("ステップ1: データ処理を実行中...")
                from src.dataset import app as dataset_app
                import sys
                from io import StringIO

                # dataset.pyを実行（引数なしでデフォルト動作）
                original_argv = sys.argv
                try:
                    sys.argv = ['dataset.py']  # CLIアプリケーションとして実行
                    dataset_app()
                    logger.success("データ処理が完了しました")
                except Exception as e:
                    logger.error(f"データ処理中にエラー: {e}")
                    raise
                finally:
                    sys.argv = original_argv

                # ステップ2: 特徴量エンジニアリング
                logger.info("ステップ2: 特徴量エンジニアリングを実行中...")
                from src.features import app as features_app

                try:
                    sys.argv = ['features.py']
                    features_app()
                    logger.success("特徴量エンジニアリングが完了しました")
                except Exception as e:
                    logger.error(f"特徴量エンジニアリング中にエラー: {e}")
                    raise
                finally:
                    sys.argv = original_argv

                # ステップ3: モデル訓練
                logger.info("ステップ3: モデル訓練を実行中...")
                from src.modeling.train import app as train_app

                try:
                    sys.argv = ['train.py']
                    train_app()
                    logger.success("モデル訓練が完了しました")
                except Exception as e:
                    logger.error(f"モデル訓練中にエラー: {e}")
                    raise
                finally:
                    sys.argv = original_argv

                logger.success("フルパイプライン実行が完了しました")

            except Exception as e:
                logger.error(f"フルパイプライン実行中に致命的エラー: {e}")
                logger.info("手動でステップを実行してください:")
                logger.info("1. python src/dataset.py")
                logger.info("2. python src/features.py")
                logger.info("3. python src/modeling/train.py")
                raise

        # テストデータを読み込み
        logger.info(f"テストデータを読み込み中: {test_data_path}")
        test_df = pd.read_csv(test_data_path)

        # 特徴量とIDを抽出
        test_ids = test_df['id'].values
        feature_cols = [col for col in test_df.columns if col not in ['id', config.target]]
        X_test = test_df[feature_cols].values

        logger.info(f"テストサンプル数: {len(test_df)}, 特徴量数: {len(feature_cols)}")

        # 各実験のモデルを読み込んで予測を生成
        all_predictions = []
        experiment_results = {}

        for exp_name in exp_list:
            try:
                models = load_experiment_models(exp_name, MODELS_DIR)

                # この実験の予測を生成
                exp_predictions = []
                for i, model in enumerate(models):
                    pred = model.predict(X_test)
                    exp_predictions.append(pred)
                    logger.info(f"実験 {exp_name} - Fold {i+1}: 予測完了")

                # この実験のフォールド間で予測を平均化
                exp_ensemble = np.mean(exp_predictions, axis=0)
                all_predictions.append(exp_ensemble)

                experiment_results[exp_name] = {
                    'n_models': len(models),
                    'mean_prediction': exp_ensemble.mean(),
                    'std_prediction': exp_ensemble.std()
                }

                logger.success(f"実験 {exp_name}: {len(models)}モデルの予測完了")

            except Exception as e:
                logger.error(f"実験 {exp_name} の処理中にエラー: {e}")
                continue

        if not all_predictions:
            raise RuntimeError("有効な予測結果が得られませんでした")

        # 実験間で予測をアンサンブル
        final_predictions = ensemble_predictions(all_predictions, ensemble_method)

        # 提出ファイルを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_suffix = "_".join(exp_list)
        submission_filename = f"submission_{exp_suffix}_{ensemble_method}_{timestamp}.csv"
        submission_path = output_dir / submission_filename

        create_submission_file(
            final_predictions,
            test_ids,
            submission_path,
            exp_list,
            ensemble_method
        )

        # 実験結果を保存
        results_data = {
            'timestamp': timestamp,
            'experiments_used': exp_list,
            'ensemble_method': ensemble_method,
            'experiment_results': experiment_results,
            'final_predictions': {
                'mean': float(final_predictions.mean()),
                'std': float(final_predictions.std()),
                'min': float(final_predictions.min()),
                'max': float(final_predictions.max())
            },
            'submission_file': submission_filename
        }

        results_path = output_dir / f"submission_results_{exp_suffix}_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.success(f"結果ファイルを保存: {results_path}")
        logger.success("提出パイプライン完了")

    except Exception as e:
        logger.error(f"提出パイプライン実行中にエラー: {e}")
        raise


if __name__ == "__main__":
    app()