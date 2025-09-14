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

2. **[TICKET-005] データ可視化機能** 🔄 **実装中**
   - ファイル: `src/plots.py`
   - 現状: プレースホルダーコード
   - 要件:
     - EDAプロット（ターゲット分布、特徴量分布）
     - 特徴量間相関分析
     - 外れ値可視化
     - 予測結果可視化

### 第2段階: 特徴量改善
3. **[TICKET-004] 特徴量エンジニアリング機能**
   - ファイル: `src/features.py`
   - 現状: 空ファイル
   - 要件:
     - EDA結果を基にした新特徴量作成
     - 特徴量選択機能
     - スケーリング機能

### 第3段階: モデル開発
4. **[TICKET-002] LightGBM回帰モデルの訓練機能**
   - ファイル: `src/modeling/train.py`
   - 現状: プレースホルダーコード
   - 要件:
     - LightGBMRegressor実装
     - script/my_config.pyの設定を利用
     - クロスバリデーション
     - モデル保存機能
     - RMSEメトリクス

5. **[TICKET-003] モデル推論機能の実装**
   - ファイル: `src/modeling/predict.py`
   - 現状: プレースホルダーコード
   - 要件:
     - 訓練済みモデルの読み込み
     - テストデータでの予測
     - Kaggle提出形式での出力

### 第4段階: 品質保証
6. **[TICKET-006] テストケースの拡充**
   - ディレクトリ: `tests/`
   - 現状: 基本テストのみ
   - 要件:
     - 各モジュールのユニットテスト
     - データパイプラインテスト
     - モデル性能テスト

### 第5段階: 運用準備
7. **[TICKET-007] Kaggleサブミッション用スクリプト**
   - 新規ファイル: `scripts/submit.py`
   - 要件:
     - エンドツーエンドパイプライン
     - サブミッション形式での出力
     - 複数モデル実行機能