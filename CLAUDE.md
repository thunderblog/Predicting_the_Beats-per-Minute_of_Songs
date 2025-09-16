# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 言語とコミュニケーション
- **すべてのレスポンスは日本語で行う**
- コメントや変数名は英語を使用
- ログメッセージやユーザー向けメッセージは日本語を推奨

## Common Development Commands

This project uses Make for common tasks:

- `make requirements` - Install Python dependencies
- `make format` - Format code with ruff (auto-fix and format)
- `make lint` - Check code formatting and linting with ruff
- `make test` - Run tests with pytest
- `make data` - Process dataset using src/dataset.py
- `make clean` - Remove compiled Python files and __pycache__ directories
- `make help` - Show all available Make targets

Alternative direct commands:
- `python -m pytest tests` - Run tests directly
- `ruff format` - Format code 
- `ruff check --fix` - Fix linting issues
- `ruff format --check && ruff check` - Check formatting and linting

## Project Architecture

This is a Cookiecutter Data Science project following standard ML/data science structure:

### Core Module Structure (`src/`)
- `config.py` - Central configuration with project paths (DATA_DIR, MODELS_DIR, etc.) and logging setup
- `dataset.py` - Data processing pipeline (CLI with typer)
- `features.py` - Feature engineering utilities
- `plots.py` - Visualization utilities
- `modeling/` - Machine learning components
  - `train.py` - Model training pipeline (CLI with typer)
  - `predict.py` - Model inference pipeline (CLI with typer)

### Data Organization
- `data/raw/` - Original, immutable data
- `data/interim/` - Intermediate processed data
- `data/processed/` - Final datasets for modeling
- `data/external/` - Third-party data sources
- `models/` - Trained models and predictions
- `reports/figures/` - Generated visualizations

### Key Dependencies
- **loguru** - Structured logging (configured in config.py with tqdm integration)
- **typer** - CLI framework for all main modules
- **tqdm** - Progress bars (integrated with loguru)
- **ruff** - Linting and formatting (configured in pyproject.toml)
- **pytest** - Testing framework

### Configuration Notes
- Line length: 99 characters (pyproject.toml)
- Python version: ~3.10.0
- Ruff includes import sorting (isort) with src as known first-party
- Environment variables loaded via python-dotenv in config.py
- All main modules (dataset.py, train.py, predict.py) are CLI applications using typer

### Running Main Modules
Each main module can be run directly or via Make:
- `python src/dataset.py` or `make data`
- `python src/modeling/train.py`
- `python src/modeling/predict.py`

## コーディング規則とベストプラクティス

### PEP 8準拠
- **行の長さ**: 99文字（pyproject.tomlで設定済み）
- **インデント**: スペース4つ
- **命名規則**:
  - 変数・関数: snake_case
  - クラス: PascalCase
  - 定数: UPPER_SNAKE_CASE
  - プライベート属性: 先頭に単一アンダースコア（_variable）

### インポート管理
- ruffによる自動ソート設定済み（isort統合）
- インポート順序: 標準ライブラリ → サードパーティ → ローカルモジュール
- `bpm`は第一パーティライブラリとして設定済み

### 型ヒント
- Python 3.10の機能を活用
- 関数の引数と戻り値に型ヒントを記述
- `from pathlib import Path`を使用してパスを型安全に扱う

### エラーハンドリング
- 具体的な例外クラスをキャッチ
- ログ出力にはloguru使用（config.pyで設定済み）
- 適切なログレベル使用（info, warning, error, success）

### ドキュメント
- 関数・クラスにはdocstringを記述（Google Style推奨）
- 複雑な処理には適切なコメント
- 設定値や定数には説明コメント

### テスト
- `tests/`ディレクトリにpytestを使用
- テスト関数は`test_`で開始
- アサーション前に適切なセットアップ

### データ処理のベストプラクティス
- `config.py`で定義されたパス定数を使用
- データ処理にはtqdmでプログレスバー表示
- 中間データは`data/interim/`に保存
- 最終データは`data/processed/`に保存

### CLI設計
- typerを使用した型安全なCLI
- デフォルト値は`config.py`の定数を使用
- コマンドライン引数には適切な説明を付与

### ログ設定
- loguruがtqdmと統合済み（config.pyで設定）
- 処理開始時: `logger.info()`
- 処理完了時: `logger.success()`
- エラー時: `logger.error()`

## Git ブランチ命名規則

### 基本パターン
- `feature/ticket-XXX/機能名` - 新機能開発用ブランチ
- `bugfix/ticket-XXX/修正内容` - バグ修正用ブランチ
- `hotfix/緊急修正内容` - 緊急修正用ブランチ

### 命名例
```
feature/ticket-001/dataset-processing
feature/ticket-001/data-validation
feature/ticket-002/lightgbm-implementation
feature/ticket-002/cross-validation
feature/ticket-003/model-loading
feature/ticket-003/prediction-pipeline
```

### 利点
- **階層的管理**: チケット単位での機能管理が可能
- **拡張性**: 同一チケット内で複数ブランチに分割可能
- **追跡性**: チケット番号から要件が明確に追跡可能
- **一貫性**: 全チケットで統一された命名規則

## プロジェクト情報

### Kaggleコンペティション: "Predicting the Beats-per-Minute of Songs"
- **コンペ種別**: 2025 Kaggle Playground Series (September 2025)
- **問題設定**: 楽曲のBeats-per-Minute (BPM)を予測する回帰問題
- **評価指標**: Root Mean Squared Error (RMSE)
- **データセット**: 合成データ（実世界データから生成）

### データセット概要
- **ターゲット変数**: `BeatsPerMinute` - 楽曲のBPM（連続値）
- **特徴量** (9個):
  - `RhythmScore` - リズムスコア
  - `AudioLoudness` - 音声の音量レベル
  - `VocalContent` - ボーカル含有量
  - `AcousticQuality` - 音響品質
  - `InstrumentalScore` - 楽器演奏スコア
  - `LivePerformanceLikelihood` - ライブ演奏っぽさ
  - `MoodScore` - ムードスコア
  - `TrackDurationMs` - トラック長（ミリ秒）
  - `Energy` - エネルギーレベル

### モデリング方針
- **基本アプローチ**: 回帰問題として扱う
- **使用予定モデル**: LightGBM、その他の機械学習モデルも検討
- **実験管理**: script/my_config.pyのCFGクラスで設定を管理

## 開発タスクチケット（データ分析の適切な順序）

### 第1段階: データ理解とEDA
1. **[TICKET-001] データセット処理機能の実装** ✅ **完了**
   - ファイル: `src/dataset.py`
   - 現状: 実装完了
   - 要件:
     - CSVデータの読み込み・前処理
     - 訓練・テストデータの分割
     - データ品質チェック
     - 前処理済みデータの保存

2. **[TICKET-005] データ可視化機能** ✅ **完了**
   - ファイル: `src/plots.py`
   - 現状: 実装完了
   - 要件:
     - EDAプロット（ターゲット分布、特徴量分布）
     - 特徴量間相関分析
     - 外れ値可視化
     - 予測結果可視化

### 第2段階: 特徴量改善
3. **[TICKET-004] 特徴量エンジニアリング機能** ✅ **完了**
   - ファイル: `src/features.py`
   - 現状: 実装完了
   - 要件:
     - EDA結果を基にした新特徴量作成（交互作用・時間・統計的特徴量）
     - 特徴量選択機能（F統計量・相互情報量・相関・組み合わせ）
     - スケーリング機能（Standard・Robust・MinMaxスケーラ対応）

### 第3段階: モデル開発
4. **[TICKET-002] LightGBM回帰モデルの訓練機能** ✅ **完了**
   - ファイル: `src/modeling/train.py`
   - 現状: 実装完了
   - 要件:
     - LightGBMRegressor実装（scripts/my_config.py設定利用）
     - クロスバリデーション機能（KFold対応）
     - モデル保存機能（pickle + JSON結果）
     - RMSEメトリクス（平均RMSE: 26.47）
     - Early Stopping & ログ出力機能

5. **[TICKET-003] モデル推論機能の実装**
   - ファイル: `src/modeling/predict.py`
   - 現状: プレースホルダーコード
   - 要件:
     - 訓練済みモデルの読み込み
     - テストデータでの予測
     - Kaggle提出形式での出力

### 第4段階: 品質保証
6. **[TICKET-006] テストケースの拡充** ✅ **完了**
   - ディレクトリ: `tests/`
   - 現状: 包括的なテストスイート完成
   - 実装完了内容:
     - 各モジュールのユニットテスト（dataset、features、train、predict）
     - データパイプライン統合テスト
     - モデル性能テスト（モック使用）
     - 76テストケース作成、66テスト成功（87%成功率）

### 第5段階: 運用準備
7. **[TICKET-007] Kaggleサブミッション用スクリプト**
   - 新規ファイル: `scripts/submit.py`
   - 要件:
     - エンドツーエンドパイプライン
     - サブミッション形式での出力
     - 複数モデル実行機能

## 実験管理システム

### 実験結果の記録構造
各実験は `experiments/` ディレクトリ配下に独立したディレクトリを作成して管理する。

```
experiments/
├── exp01_baseline_lgb/          # 実験ディレクトリ（命名規則: exp{番号}_{概要}）
│   ├── config.json              # 実験設定・ハイパーパラメータ
│   ├── results.json             # CV・LB結果・性能指標
│   ├── submission.csv           # Kaggle提出ファイル
│   ├── models/                  # 訓練済みモデルファイル
│   │   ├── {exp_name}_fold_1_*.pkl
│   │   └── {exp_name}_fold_2_*.pkl
│   └── README.md                # 実験メモ・考察・Next Steps
├── exp02_feature_engineering/   # 次の実験例
└── exp03_model_ensemble/        # 次の実験例
```

### ファイル構成の詳細

#### config.json - 実験設定
```json
{
  "experiment_name": "exp01_baseline_lgb",
  "description": "実験の目的・概要",
  "date_created": "YYYY-MM-DD",
  "model_config": {
    "model_type": "LightGBM",
    "ハイパーパラメータ": "値"
  },
  "data_config": {
    "train_samples": 数値,
    "test_samples": 数値,
    "n_features": 数値,
    "cv_folds": 数値
  },
  "features": {
    "original_features": ["特徴量リスト"],
    "engineered_features": ["特徴量リスト"],
    "feature_selection": "手法説明",
    "scaling": "スケーリング手法"
  },
  "preprocessing": {
    "missing_values": "欠損値処理",
    "outlier_handling": "外れ値処理",
    "feature_engineering": "特徴量エンジニアリング概要"
  }
}
```

#### results.json - 実験結果
```json
{
  "experiment_name": "実験名",
  "timestamp": "実行日時",
  "cross_validation": {
    "cv_strategy": "KFold",
    "n_folds": 数値,
    "mean_rmse": 数値,
    "fold_results": {
      "fold_1": {"rmse": 数値, "model_file": "ファイル名"}
    }
  },
  "leaderboard_results": {
    "submission_date": "提出日",
    "public_lb_score": 数値,
    "public_lb_rank": 数値,
    "private_lb_score": 数値,
    "private_lb_rank": 数値
  },
  "prediction_results": {
    "ensemble_method": "アンサンブル手法",
    "n_models": 数値,
    "test_predictions": {
      "mean_prediction": 数値,
      "min_prediction": 数値,
      "max_prediction": 数値
    }
  },
  "performance_metrics": {
    "cv_vs_lb_consistency": "CV-LB差",
    "overfitting_indicator": "過学習判定"
  },
  "notes": ["実験で得られた知見・気づき"]
}
```

#### README.md - 実験メモ
各実験ディレクトリに以下の構成でREADME.mdを作成：

```markdown
# Experiment XX: 実験名

## 概要
- 実験目的・仮説
- 実施日・所要時間

## モデル性能
### Cross Validation Results
- CV Score、戦略、フォールド数

### Leaderboard Results
- Public/Private LB Score
- 順位・改善幅

## モデル設定
- アルゴリズム・ハイパーパラメータ
- 特徴量リスト・エンジニアリング内容

## 技術実装
- 予測パイプライン概要
- ファイル構成

## 考察・気づき
### 成功要因
### 改善の余地

## Next Steps
- 次回実験のアイデア・改善案
```

### 実験管理のベストプラクティス

#### 命名規則
- **実験ディレクトリ**: `exp{番号2桁}_{概要}`（例：`exp01_baseline_lgb`）
- **日付形式**: `YYYY-MM-DD`で統一
- **モデルファイル**: 既存の命名規則`{exp_name}_fold_{N}_{timestamp}.pkl`を継承

#### 記録すべき情報
1. **再現性確保**: 設定・パラメータ・データ分割・乱数シード
2. **性能追跡**: CV・LB・Private LB・順位変動
3. **技術詳細**: アンサンブル手法・後処理・特徴量
4. **知見蓄積**: 成功要因・失敗要因・改善案

#### 実験後のルーチン
1. 結果確認後、即座に`experiments/{exp_name}/`ディレクトリを作成
2. config.json・results.json・README.mdを作成
3. submission.csvとmodelファイルをコピー
4. 次回実験のためのNext Stepsを記録