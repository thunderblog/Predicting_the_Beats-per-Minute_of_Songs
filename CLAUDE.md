# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 開発環境
- **OS**: Windows 11
- **プラットフォーム**: win32
- **Python**: ~3.10.0
- **シェル**: PowerShell/Command Prompt
- **パス区切り**: バックスラッシュ（\）使用

### 環境固有の考慮事項
- **ファイルコピー**: `copy` (Windows) の代わりに `cp` (Unix) を使用
- **パス指定**: Windows形式 `C:\Users\...` とUnix形式 `/c/Users/...` の混在に注意
- **改行コード**: CRLF (Windows) vs LF (Unix) の違いを考慮
- **コマンドライン**: PowerShellとCommand Promptの両方で動作するよう配慮

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

### Windows環境でのコマンド実行
```bash
# ディレクトリ移動
cd experiments\exp005_ticket008_03_dimensionality_reduction

# ファイルコピー（Bashでcp、Windows Commandでcopy）
cp models\*.json experiments\exp005_ticket008_03_dimensionality_reduction\
copy models\*.json experiments\exp005_ticket008_03_dimensionality_reduction\

# ファイル一覧確認
ls data\processed\*.csv          # PowerShell/Bash
dir data\processed\*.csv         # Command Prompt

# Kaggle提出
kaggle competitions submit -c playground-series-s5e9 -f "data\processed\submission.csv" -m "提出メッセージ"

# 自動提出スクリプト
python scripts\submit_experiment.py --experiment-name exp005_ticket008_03_dimensionality_reduction
```

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
7. **[TICKET-007] Kaggleサブミッション用スクリプト** ✅ **完了**
   - 新規ファイル: `scripts/submit.py`
   - 現状: 実装完了
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

## 精度向上タスクチケット

### 第6段階: 高度な特徴量エンジニアリング
8. **[TICKET-008] 音楽ジャンル推定特徴量の作成** ✅ **完了**
   - ファイル: `src/features.py` (機能拡張)
   - 現状: 実装完了、評価結果でballad_genre_scoreが最重要特徴量に選出
   - 要件:
     - Energy×RhythmScore → ダンス系ジャンル特徴量
     - AcousticQuality×InstrumentalScore → アコースティック系特徴量
     - VocalContent×MoodScore → バラード系特徴量
     - 各ジャンル特徴量のBPM予測への寄与度分析
   - 課題: 多重共線性によりRMSE改善が微小（-0.0010）

   **8.1 [TICKET-008-01] 多重共線性除去による特徴量最適化**
   - ファイル: `src/features.py` (機能拡張)
   - 親チケット: TICKET-008（音楽ジャンル推定特徴量）
   - 背景:
     - ジャンル特徴量と元特徴量の高相関確認（ballad_genre_score vs VocalContent: 0.803）
     - dance_genre_score vs Energy: 0.871の強い相関
   - 要件:
     - 高相関ペア（>0.7）の元特徴量自動除去機能
     - ジャンル特徴量優先の特徴量選択ロジック
     - 相関行列による自動検出・除去システム
     - Before/After性能比較機能

   **8.2 [TICKET-008-02] 独立性の高い高次特徴量の開発**
   - ファイル: `src/features.py` (新規関数追加)
   - 親チケット: TICKET-008（音楽ジャンル推定特徴量）
   - 要件:
     - 比率ベース特徴量（VocalContent/Energy、AcousticQuality/Loudness）
     - 対数変換時間特徴量（log(TrackDurationMs) × RhythmScore）
     - 標準化済み交互作用（Z-score正規化後の積）
     - 音楽理論ベースの複雑指標（テンポ×音響品質の複合指標）

   **8.3 [TICKET-008-03] 次元削減とPCA特徴量の実装**
   - 新規ファイル: `src/features/dimensionality_reduction.py`
   - 親チケット: TICKET-008（音楽ジャンル推定特徴量）
   - 要件:
     - PCA変換による主成分特徴量作成
     - ジャンル特徴量群の主成分分析
     - 元特徴量群の独立成分分析（ICA）
     - 最適な主成分数の自動選択（分散寄与率）

9. **[TICKET-009] テンポカテゴリ特徴量の実装**
   - ファイル: `src/features.py` (機能拡張)
   - 要件:
     - BPM範囲による楽曲カテゴリ推定（遅・中・速テンポ）
     - 各特徴量とテンポカテゴリの関係性分析
     - テンポ予測用の確率的特徴量作成

10. **[TICKET-010] 音楽的複雑性・時間的特徴量の開発**
    - ファイル: `src/features.py` (機能拡張)
    - 要件:
      - RhythmScore/Energy → リズム複雑性指標
      - TrackDurationMs ベースの長さカテゴリ特徴量
      - LivePerformanceLikelihood×Energy → パフォーマンス特徴量
      - 楽曲構造推定特徴量（イントロ・サビ・アウトロ推定）

### 第7段階: モデル多様化とアンサンブル
11. **[TICKET-011] アルゴリズム多様化による性能向上**
    - 背景: LightGBMベースラインの限界突破を目指す
    - 方針: GBDT系とニューラルネットワーク系の組み合わせで多様性確保

    **11.1 [TICKET-011-01] XGBoost回帰モデルの実装**
    - ファイル: `src/modeling/train.py` (アルゴリズム選択機能拡張)
    - 優先度: 中（GBDT比較ベースライン確立）
    - 要件:
      - XGBoostRegressor実装とハイパーパラメータ調整
      - LightGBMとの性能・特徴量重要度比較
      - 既存CVパイプライン流用による効率実装

    **11.2 [TICKET-011-02] CatBoost回帰モデルの実装**
    - ファイル: `src/modeling/train.py` (アルゴリズム選択機能拡張)
    - 優先度: 低（GBDT系類似性能予想）
    - 要件:
      - CatBoostRegressor実装
      - カテゴリ特徴量自動処理活用
      - GBDT系3種比較分析

    **11.3 [TICKET-011-03] Random Forest回帰モデルの実装**
    - ファイル: `src/modeling/train.py` (アルゴリズム選択機能拡張)
    - 優先度: 低（アンサンブルベース確立）
    - 要件:
      - RandomForestRegressor実装
      - 木の多様性確保（アンサンブル下地）
      - 特徴量重要度の安定性分析

    **11.4 [TICKET-011-04] Multi-Layer Perceptron (MLP) 実装** 🧠 **高優先度**
    - 新規ファイル: `src/modeling/neural_models.py`
    - 優先度: 最高（根本的な学習メカニズム差による性能向上期待）
    - 要件:
      - PyTorch/TensorFlow基盤のMLP回帰モデル
      - 表形式データ向け最適化（BatchNorm、Dropout）
      - 特徴量スケーリング統合パイプライン
      - Early Stopping・学習率スケジューリング

    **11.5 [TICKET-011-05] TabNet実装** 🧠 **高優先度**
    - 新規ファイル: `src/modeling/tabnet_model.py`
    - 優先度: 高（表形式データ特化の革新的アーキテクチャ）
    - 要件:
      - Google TabNet実装（pytorch-tabnet使用）
      - 特徴量選択機能内蔵活用
      - 解釈可能性分析（Attention weights）
      - GPU/CPU両対応の訓練パイプライン

    **11.6 [TICKET-011-06] Neural Oblivious Decision Trees (NODE) 実装**
    - 新規ファイル: `src/modeling/node_model.py`
    - 優先度: 中（実験的手法、決定木×NN融合）
    - 要件:
      - NODE実装（決定木構造のニューラルネット）
      - 勾配ブースティングとNNの利点融合
      - 解釈性とパフォーマンスの両立検証

12. **[TICKET-012] アンサンブル手法の実装**
    - 新規ファイル: `src/modeling/ensemble.py`
    - 要件:
      - 加重平均アンサンブル
      - スタッキング（2層スタッキング）
      - ブレンディング手法
      - アンサンブル重み最適化

### 第8段階: ハイパーパラメータ最適化
13. **[TICKET-013] Optuna最適化システム実装**
    - 新規ファイル: `src/modeling/optimization.py`
    - 要件:
      - 全モデル対応の統一的最適化フレームワーク
      - ベイジアン最適化によるハイパーパラメータ探索
      - 早期停止とトライアル履歴管理
      - CV性能向上の追跡・可視化

### 第9段階: 高度なクロスバリデーション
14. **[TICKET-014] CV戦略改善とデータ分析**
    - ファイル: `src/modeling/train.py` (機能拡張)
    - 要件:
      - StratifiedKFold（BPMレンジ別分割）
      - GroupKFold（楽曲類似性グループ別分割）
      - TimeSeriesSplit（時系列分割の検討）
      - CV-LB一貫性分析とリーク検出

### 第10段階: 実験の体系化
15. **[TICKET-015] 実験管理システムの自動化**
    - 新規ファイル: `scripts/experiment_manager.py`
    - 要件:
      - 実験設定の自動記録（config.json）
      - 結果の自動集約（results.json）
      - 実験比較・可視化ダッシュボード
      - A/Bテスト機能とベンチマーク追跡

### 第11段階: リズム周期性特徴量（高優先度）
16. **[TICKET-016] ドラマー視点リズム周期性特徴量の実装** 🎵 **高優先度**
    - ファイル: `src/features.py` (新規関数追加)
    - 背景: ドラマーの経験知からのBPM予測アプローチ
    - 音楽的根拠: スネア・キック・ハイハットの周期パターンがBPM決定の核心
    - 要件:
      - **リズムパターン推定特徴量**（4/4拍子、3/4拍子、シンコペーション検出）
      - **周期性一貫性スコア**（TrackDurationとBPM推定の整合性検証）
      - **疑似ドラム系特徴量**（キック・スネア・ハイハット推定密度）
      - **拍子・テンポ変動推定**（ルバート、加速、減速パターン検出）
      - **周期性コヒーレンス指標**（RhythmScore×Energy×時間整合性）
    - 実装アプローチ:
      - 既存特徴量の組み合わせで疑似周波数解析
      - 音楽理論の拍子パターンを数値化
      - 時間軸整合性検証（楽曲長とBPMの妥当性）
      - BPMレンジ別典型パターンのマッチング
    - 期待効果: ドラマー直感に基づく高精度BPM予測、実用的音楽知識の数値化