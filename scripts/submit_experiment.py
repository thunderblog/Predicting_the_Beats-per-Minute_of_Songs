#!/usr/bin/env python3
"""
Kaggle自動提出スクリプト

実験ディレクトリから自動的にsubmissionファイルを検出して提出するスクリプト。
実験メタデータに基づいて適切なメッセージを生成します。

Usage:
    python scripts/submit_experiment.py --experiment-name exp005_ticket008_03_dimensionality_reduction
    python scripts/submit_experiment.py --submission-file data/processed/submission_ticket008_03_dimensionality.csv
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# プロジェクトルートをPythonパスに追加
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import typer
from loguru import logger

# PROJ_ROOTを直接定義（src.configへの依存を回避）
PROJ_ROOT = project_root


def find_submission_file(experiment_dir: Path) -> Optional[Path]:
    """実験ディレクトリからsubmissionファイルを検索"""
    patterns = ["submission*.csv", "*.csv"]

    for pattern in patterns:
        files = list(experiment_dir.glob(pattern))
        if files:
            # 最新のファイルを返す
            return max(files, key=lambda f: f.stat().st_mtime)

    return None


def load_experiment_metadata(experiment_dir: Path) -> Dict[str, Any]:
    """実験メタデータを読み込み"""
    results_file = experiment_dir / "results.json"

    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # フォールバック：CV結果ファイルから情報を取得
    cv_files = list(experiment_dir.glob("*cv_results*.json"))
    if cv_files:
        with open(cv_files[0], 'r', encoding='utf-8') as f:
            cv_data = json.load(f)
            return {
                "experiment_name": cv_data.get("experiment_name", "unknown"),
                "cross_validation": {
                    "mean_rmse": cv_data.get("mean_cv_score", 0.0),
                    "std_rmse": cv_data.get("std_cv_score", 0.0)
                },
                "data_config": {
                    "n_features": cv_data.get("feature_count", 0)
                }
            }

    return {}


def generate_submit_message(metadata: Dict[str, Any], experiment_name: str) -> str:
    """実験メタデータからsubmitメッセージを生成"""
    base_message = f"{experiment_name.upper()}"

    # CV情報を追加
    if "cross_validation" in metadata:
        cv_score = metadata["cross_validation"].get("mean_rmse", 0.0)
        base_message += f" | CV: {cv_score:.4f}"

    # 特徴量数を追加
    if "data_config" in metadata:
        n_features = metadata["data_config"].get("n_features", 0)
        if n_features > 0:
            base_message += f" | {n_features}特徴量"

    # LB情報を追加
    if "leaderboard_results" in metadata:
        lb_score = metadata["leaderboard_results"].get("public_lb_score")
        if lb_score:
            base_message += f" | LB: {lb_score:.5f}"

    # 実験の説明を追加
    description = metadata.get("description", "")
    if description:
        base_message += f" | {description}"

    return base_message


def submit_to_kaggle(
    submission_file: Path,
    message: str,
    competition_id: str = "playground-series-s5e9",
    dry_run: bool = False
) -> bool:
    """Kaggleに提出"""

    if not submission_file.exists():
        logger.error(f"提出ファイルが見つかりません: {submission_file}")
        return False

    cmd = [
        "kaggle", "competitions", "submit",
        "-c", competition_id,
        "-f", str(submission_file),
        "-m", message
    ]

    logger.info(f"提出コマンド: {' '.join(cmd)}")

    if dry_run:
        logger.info("ドライランモード：実際の提出は行いません")
        return True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.success("提出完了！")
        logger.info(f"出力: {result.stdout}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"提出失敗: {e}")
        logger.error(f"エラー出力: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("kaggle CLIが見つかりません。kaggleがインストールされているか確認してください。")
        return False


def main(
    experiment_name: Optional[str] = typer.Option(None, "--experiment-name", "-e", help="実験名（例：exp005_ticket008_03_dimensionality_reduction）"),
    submission_file: Optional[str] = typer.Option(None, "--submission-file", "-f", help="提出ファイルのパス"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="カスタム提出メッセージ"),
    competition_id: str = typer.Option("playground-series-s5e9", "--competition", "-c", help="コンペティションID"),
    dry_run: bool = typer.Option(False, "--dry-run", help="ドライランモード（実際の提出は行わない）")
):
    """実験結果を自動的にKaggleに提出"""

    if experiment_name:
        # 実験ディレクトリから提出
        experiment_dir = PROJ_ROOT / "experiments" / experiment_name

        if not experiment_dir.exists():
            logger.error(f"実験ディレクトリが見つかりません: {experiment_dir}")
            sys.exit(1)

        # 提出ファイルを検索
        submission_path = find_submission_file(experiment_dir)
        if not submission_path:
            logger.error(f"提出ファイルが見つかりません: {experiment_dir}")
            sys.exit(1)

        # メタデータを読み込み
        metadata = load_experiment_metadata(experiment_dir)

        # メッセージを生成
        if not message:
            message = generate_submit_message(metadata, experiment_name)

        logger.info(f"実験: {experiment_name}")
        logger.info(f"提出ファイル: {submission_path}")
        logger.info(f"メッセージ: {message}")

    elif submission_file:
        # 直接ファイルを指定
        submission_path = Path(submission_file)

        if not message:
            message = f"Manual submission: {submission_path.name}"

        logger.info(f"提出ファイル: {submission_path}")
        logger.info(f"メッセージ: {message}")

    else:
        logger.error("--experiment-name または --submission-file のいずれかを指定してください")
        sys.exit(1)

    # 提出実行
    success = submit_to_kaggle(submission_path, message, competition_id, dry_run)

    if not success:
        sys.exit(1)

    logger.success("提出処理完了")


if __name__ == "__main__":
    typer.run(main)