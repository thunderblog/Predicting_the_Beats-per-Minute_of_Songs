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
- **エモジ禁止**: コード内、ファイル名、ログメッセージにはエモジを使用しない（エラーの原因）

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
- `features.py` - Feature engineering utilities (backward-compatible interface)
- `features/` - **リファクタリング済み特徴量エンジニアリングモジュール**
  - `base.py` - 基底クラス `BaseFeatureCreator` と共通処理
  - `interaction.py` - 交互作用特徴量作成器
  - `statistical.py` - 統計的特徴量作成器
  - `genre.py` - 音楽ジャンル特徴量作成器
  - `duration.py` - 時間特徴量作成器
  - `advanced.py` - 高次特徴量作成器
  - `selection.py` - 特徴量選択機能
  - `scaling.py` - スケーリング機能
  - `analysis.py` - 特徴量重要度分析
  - `__init__.py` - 公開APIとパイプライン管理
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
- `python src/features.py` - Feature engineering CLI (CLI with typer)

### 特徴量エンジニアリングの使用方法

#### **従来の方法（後方互換性保持）**
```python
# 従来通りの関数インターフェース
from src.features import (
    create_interaction_features,
    create_statistical_features,
    create_music_genre_features,
    select_features,
    scale_features
)

# 既存コードはそのまま動作
df_with_interactions = create_interaction_features(df)
df_with_stats = create_statistical_features(df_with_interactions)
```

#### **新しいクラスベース方法（推奨）**
```python
# 個別の特徴量作成器を使用
from src.features import (
    BasicInteractionCreator,
    StatisticalFeatureCreator,
    MusicGenreFeatureCreator
)

creator = BasicInteractionCreator()
result = creator.create_features(df)
print(f"Created features: {creator.created_features}")
```

#### **パイプライン方法（最も推奨）**
```python
# デフォルトパイプラインの使用
from src.features import create_feature_pipeline

pipeline = create_feature_pipeline()
result = pipeline.execute(df)

# 実行サマリー確認
summary = pipeline.get_execution_summary()
print(summary)
```

#### **カスタムパイプライン構築**
```python
# 特定の特徴量作成器のみ使用
from src.features import FeaturePipeline, BasicInteractionCreator, StatisticalFeatureCreator

pipeline = FeaturePipeline()
pipeline.add_creator(BasicInteractionCreator())
pipeline.add_creator(StatisticalFeatureCreator())

result = pipeline.execute(df)

# 条件分岐実行
result = pipeline.execute(df, creators_to_run=["BasicInteraction"])
```

#### **統一特徴量生成システム（TICKET-016実装）**

trainとtestデータで完全に同一の特徴量セットを生成する標準化手順：

```python
# 統一特徴量生成の実行
python scripts/unified_feature_generation.py

# 出力ファイル確認
data/processed/train_unified_75_features.csv  # 訓練データ（67特徴量）
data/processed/test_unified_75_features.csv   # テストデータ（67特徴量）
```

**特徴量構成（67特徴量）**:
1. **基本特徴量（9個）**: 元データの全特徴量
2. **交互作用特徴量（22個）**:
   - 乗算交互作用（RhythmScore², AudioLoudness×VocalContent等）
   - 除算交互作用（除零対策: +1e-8）
3. **対数変換特徴量（36個）**:
   - 基本対数変換（log1p）
   - 対数特徴量間の交互作用・除算
   - 対数特徴量統計量（平均・標準偏差・範囲・幾何平均）

**使用上の注意**:
- 特徴量参照順序: 対数特徴量作成前に元特徴量の存在確認必須
- 特徴量一致検証: 生成後にtrain/test間での完全一致を確認
- 命名規則: `{feat1}_x_{feat2}` (交互作用), `{feat1}_div_{feat2}` (除算), `log1p_{feat}` (対数)

**性能実績**:
- CV性能: 26.463984 ± 0.006159 (BPM Stratified KFold)
- LB性能: 26.38823 (40特徴量版から-0.0003改善)
- 安定性: 高い再現性とCV-LB一貫性を確保

## 引継ぎ資料作成ルール

### セッション終了時の必須作業
- **引継ぎ資料作成**: 毎回セッション終了前に `docs/handovers/HANDOVER_YYYYMMDD.md` を作成
- **最新版更新**: `docs/handovers/HANDOVER_latest.md` を最新内容で更新
- **配置場所**: `docs/handovers/` ディレクトリに日付順で管理
- **内容**: 実装完了項目、現在の最高性能、次のステップ、技術資産、即座に実行可能コマンド
- **更新**: CLAUDE.mdの現状サマリーを最新情報に更新
- **目的**: 次回セッション開始時の効率化、実装継続性確保

### 引継ぎ資料の構成要件
1. **実装完了項目**: チケット単位の成果まとめ
2. **性能ベンチマーク**: 最高LB性能と最新結果の比較
3. **次のステップ**: 優先順位付きアクションプラン
4. **技術資産**: 利用可能システム・データセット・スクリプト
5. **実行コマンド**: 即座に継続可能なコマンド一覧

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
3. **[TICKET-004] 特徴量エンジニアリング機能** ✅ **完了** 🔄 **リファクタリング済み**
   - メインファイル: `src/features.py` (後方互換インターフェース)
   - モジュール構成: `src/features/` ディレクトリに機能分離
     - `base.py` - 基底クラス・共通処理
     - `interaction.py` - 交互作用特徴量
     - `statistical.py` - 統計的特徴量
     - `genre.py` - ジャンル特徴量
     - `duration.py` - 時間特徴量
     - `advanced.py` - 高次特徴量
     - `selection.py` - 特徴量選択
     - `scaling.py` - スケーリング
     - `analysis.py` - 分析機能
     - `__init__.py` - 公開API定義
   - アーキテクチャ:
     - 基底クラス `BaseFeatureCreator` による統一インターフェース
     - `FeaturePipeline` によるワークフロー管理
     - 単一責任の原則に基づく機能分離
   - 機能:
     - EDA結果を基にした新特徴量作成（交互作用・時間・統計的特徴量）
     - 特徴量選択機能（F統計量・相互情報量・相関・組み合わせ）
     - スケーリング機能（Standard・Robust・MinMaxスケーラ対応）
     - **後方互換性**: 既存の関数インターフェースを完全保持

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

### 🏆 **完了済み: 完全データセット活用** ✅ **TICKET-029実装完了**

**29. [TICKET-029] 完全data/rawデータセット活用システム** 🎯 **実装完了**
   - **実装ファイル**: `scripts/raw_data_feature_generation.py`
   - **ブランチ**: `feature/ticket-029/complete-raw-dataset-system`
   - **背景**: data/processedで104,833行（25%）のデータ欠落を発見
   - **解決策**: data/rawの完全なデータセット（524,164サンプル）を活用
   - **成果**: 新記録LB 26.38713達成（-0.00821改善）

   **29.1 データ品質問題の特定と解決**
   - 問題: data/processed 419,332行 vs data/raw 524,164行
   - 原因: 前処理過程での大量データ欠落
   - 解決: 完全なdata/rawから直接特徴量生成

   **29.2 統一特徴量生成システム再構築**
   - 生成特徴量: 76特徴量（9基本 + 67エンジニアリング）
   - 構成: 交互作用30個、除算19個、対数5個、音楽理論4個、統計9個
   - 品質: 欠損値0、完全な特徴量一致

   **29.3 性能向上結果**
   - CV性能: 26.4588 (LGB 26.4593 ± 0.0254, CAT 26.4591 ± 0.0258)
   - LB性能: 26.38713（新記録、-0.00821改善）
   - アンサンブル重み: LGB 37.5% + CAT 62.5%（最適化済み）
   - CV-LB格差: -0.072（良好な汎化性能）

### 🚨 **緊急優先段階: データ品質根本問題解決** ⚡ **戦略転換**

**25. [TICKET-025] 境界値集中問題解決システム** 🏆 **最優先**
   - 新規ファイル: `src/data/boundary_value_transformer.py`
   - 背景: **基本9特徴量の7/9で境界値集中という致命的データ品質問題発見**
   - 根本原因: 合成データ生成の測定上限・アルゴリズム制約による情報量欠失
   - 目標: CV-LB格差+0.076→+0.030以下の大幅改善

   **25.1 [TICKET-025-01] 0値集中特徴量の対数変換**
   - 対象: InstrumentalScore(33.17%が0.000), AcousticQuality(16.95%が0.000)
   - 手法: log1p(x + ε)変換による情報量復活
   - 実装:
     ```python
     def log_transform_zero_concentrated(data, epsilon=1e-8):
         return np.log1p(data + epsilon)
     ```

   **25.2 [TICKET-025-02] 最小値集中特徴量のランク変換**
   - 対象: VocalContent(30.33%が0.024), LivePerformanceLikelihood(16.08%が0.024)
   - 手法: ランク正規化による分散化
   - 実装:
     ```python
     def rank_normalize_concentrated(data):
         return stats.rankdata(data) / len(data)
     ```

   **25.3 [TICKET-025-03] 境界値集中特徴量の変換**
   - 対象: RhythmScore(3.43%が0.975), AudioLoudness(10.97%が-1.357)
   - 手法: 逆変換・shifted log変換
   - 実装: Box-Cox変換やYeo-Johnson変換の適用

   **25.4 [TICKET-025-04] TrackDurationMs不連続性修正**
   - 発見: 190-200秒区間で99.6%欠損(12,281サンプル消失)
   - 問題: 3分10秒楽曲の異常な不在
   - 手法: 補間・平滑化による連続性復元

**26. [TICKET-026] 合成データ特性解析システム** 🔍 **高優先**
   - 新規ファイル: `scripts/synthetic_data_pattern_analysis.py`
   - 目標: Kaggle合成データ生成制約の逆工学
   - 要件:
     - 境界値制約パターンの体系的分析
     - 相関構造の人工的パターン検出
     - 分布形状の理論分布からの逸脱
     - データ生成アルゴリズム制約の特定

**27. [TICKET-027] データ品質統合実験システム** 🧪 **高優先**
   - 新規ファイル: `scripts/data_quality_experiment.py`
   - 目標: 全変換手法の統合検証
   - 要件:
     - 変換前後のCV-LB一貫性比較
     - 特徴量重要度変化の分析
     - アンサンブル性能への影響評価
     - 最適変換パラメータの探索

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

18. **[TICKET-018] 高性能アンサンブルシステム実装** 🏆 **最高優先度**
    - 新規ファイル: `src/modeling/ensemble_models.py`
    - 背景: サンプルコードの3モデルアンサンブル（XGB+LGBM+CatBoost）の統合
    - 目標: 単一モデル限界突破による性能向上

    **18.1 [TICKET-018-01] XGBoost・CatBoost回帰モデルの実装**
    - 要件:
      - XGBRegressor実装（CUDA対応、最適化済みパラメータ）
      - CatBoostRegressor実装（カテゴリ特徴量自動処理）
      - 既存LightGBMとの統一クロスバリデーション
    - 実装詳細:
      - サンプルコードのハイパーパラメータ移植
      - カテゴリ特徴量の自動検出・処理
      - 5フォールドCV統一による公平比較

    **18.2 [TICKET-018-02] Optuna重み最適化システム**
    - 機能: `optimize_ensemble_weights()` 関数追加
    - 要件:
      - 3モデル出力の最適重み探索（ベイジアン最適化）
      - Out-of-Fold予測によるアンサンブル最適化
      - 重み制約（合計=1）下での最適化
    - 実装詳細:
      - Optuna Studyによる500トライアル最適化
      - `suggest_float()`による連続パラメータ探索
      - RMSE最小化目的関数

19. **[TICKET-019] 実験管理システムの実装**
    - 新規ファイル: `scripts/kaggle_experiment_runner.py`
    - 要件:
      - TICKET-017・018統合実験の自動実行
      - 性能比較レポート自動生成
      - Kaggle提出自動化
    - 期待効果: ベースライン26.47→目標26.38（-0.09改善）の検証

20. **[TICKET-020] 高次特徴量版アンサンブルシステム** 🏆 **最高優先度**
    - ファイル: `src/modeling/ensemble.py` (機能拡張)
    - 背景: TICKET-018基本アンサンブル（184特徴量、LB 26.38922）の改良版
    - 目標: TICKET-017-02の75高次特徴量によるアンサンブル性能向上

    **20.1 [TICKET-020-01] XGBoost除去とLightGBM+CatBoost二元アンサンブル**
    - 理由: TICKET-018でXGBoost重み2.1%と最小寄与、計算効率向上
    - 要件:
      - XGBoost関連コード除去（train_xgboost_fold、predict処理）
      - LightGBM+CatBoost二元システムへの簡素化
      - Optuna重み最適化の二元対応（制約: w_lgb + w_cat = 1）
    - 実装詳細:
      - `EnsembleRegressor`クラスの二元モデル対応
      - Out-of-Fold予測生成の効率化
      - 重み最適化の高速化（パラメータ空間削減）

    **20.2 [TICKET-020-02] TICKET-017-02高次特徴量統合**
    - データファイル: 75特徴量版（TICKET-017-01+02統合版）
    - 期待効果: 高次特徴量による表現力向上
    - 要件:
      - 75特徴量データセットの特定・確認
      - 既存184特徴量版からの特徴量セット変更
      - CV-LB一貫性の高次特徴量版での検証
    - 実装詳細:
      - `train_ticket017_01_02_combined.csv` 等の適切なデータセット選択
      - 特徴量数による計算時間・メモリ使用量の最適化
      - 高次交互作用特徴量のモデル解釈性分析

    **20.3 [TICKET-020-03] 性能ベンチマーク・比較分析**
    - 目標: 最高性能TICKET-017正則化版（LB 26.38534）の超越
    - 要件:
      - アンサンブル vs 単一モデル性能比較
      - 特徴量数（75 vs 184）の効果分析
      - CV安定性とLB一貫性の定量評価
    - 実装詳細:
      - 実験結果の体系的記録（experiment_results.csv更新）
      - モデル性能の統計的有意性検定
      - 特徴量重要度のアンサンブル間比較

## 次期開発戦略（TICKET-020完了後）

### 実験結果分析に基づく改善方向性

**現状サマリー（2025-09-29戦略転換・重大発見）**:
- **最高LB性能**: exp09_1（TICKET-017正則化版、26.38534）
- **モデル基盤完成**: TICKET-021完全検証、アンサンブル重み確定（変更不要）
- **🚨重大発見**: **基本9特徴量の7/9で境界値集中という致命的データ品質問題**
  - InstrumentalScore: 33.17%が0.000（楽器演奏なし）
  - VocalContent: 30.33%が0.024（ボーカル最小値）
  - AcousticQuality: 16.95%が0.000（音響品質なし）
  - AudioLoudness: 10.97%が-1.357（全サンプル負値）+ 測定上限問題
  - TrackDurationMs: 190-200秒区間で99.6%欠損（12,281サンプル消失）
- **根本原因**: 合成データ生成の測定上限・アルゴリズム制約による情報量欠失
- **戦略転換**: 境界値変換による情報量復活が最優先（外れ値除去より根本的）
- **新優先度**: 境界値変換システム(TICKET-025) → 合成データ解析(TICKET-026) → 統合実験(TICKET-027)
- **次の目標**: 境界値変換によるCV-LB格差+0.076→+0.030以下の大幅改善

### 第11段階: 最高性能追求戦略
21. **[TICKET-021] 正則化二元アンサンブル** ✅ **実装完了**
    - ファイル: `src/modeling/ensemble.py` ✅ **統合完了**
    - 実装状況: exp18で検証済み（LB 26.38814、微劣化+0.0001）
    - 結果: CV 26.464、重み LightGBM 60.1% + CatBoost 39.9%
    - 効果: 異なる正則化手法統合による多様性確保、精密最適化の余地あり
    - 実行スクリプト: `run_ticket021_ensemble.py`
    - 次の改善: trials数増加（10→50）で精密最適化実行推奨

22. **[TICKET-022] CV戦略改善システム** ✅ **実装済み（効果限定的）**
    - ファイル: `src/modeling/cross_validation.py` ✅ **実装完了**
    - 背景: CV-LB格差（-0.077）の解消による汎化性能向上
    - 結果: CV安定性改善も LB性能向上には至らず

    **22.1 [TICKET-022-01] StratifiedKFold実装** ✅ **実装完了**
    - 実装状況: exp15で検証済み（CV標準偏差10倍改善、LB改善なし）
    - 結果: CV 26.464760±0.005038、LB 26.38853
    - 効果: 安定性向上（標準偏差0.06→0.005）、性能向上効果なし

    **22.2 [TICKET-022-02] GroupKFold検討** ✅ **実装完了**
    - 実装状況: exp17で検証済み（効果限定的）
    - 結果: CV 26.470073±0.099227、LB 26.38804（微改善-0.00019）
    - 効果: データリーク防止確認、CV-LB格差拡大傾向

    **22.3 [TICKET-022-03] CatBoost単体最適化** 🏆 **新規追加・最高優先度**
    - 新規ファイル: `src/modeling/optimization.py` (CatBoostOptimizer追加)
    - 背景: CatBoostが基本パラメータのみでチューニング未実施
    - 目標: 単体性能最大化によるアンサンブル品質向上
    - 要件:
      - Optuna CatBoost専用ハイパーパラメータ最適化
      - depth、l2_leaf_reg、border_count等の重要パラメータ探索
      - BPM Stratified KFold + 67特徴量での最適化
      - LightGBM単体性能（26.464前後）の超越
    - 期待効果: TICKET-021二元アンサンブルの前提条件整備

23. **[TICKET-023] Deep Learning統合システム** 🧠 **高優先度**
    - 新規ファイル: `src/modeling/tabnet_ensemble.py`
    - 背景: 表形式データ特化ニューラルネットワークによる非線形パターン捕捉
    - 目標: GBDT系とDeep Learning系の異質モデル統合によるアンサンブル多様性向上

    **22.1 [TICKET-022-01] TabNet実装と統合**
    - 要件:
      - PyTorch TabNet実装（75特徴量対応）
      - 既存二元アンサンブル（LGB+CAT）への第3軸追加
      - GPU/CPU両対応の統一訓練パイプライン
    - 実装詳細:
      - 特徴量重要度のAttention weights分析
      - 三元アンサンブル重み最適化（Optuna拡張）
      - ニューラルネット特有の前処理パイプライン統合

23. **[TICKET-023] 高次特徴量エンジニアリング** 📊 **中優先度**
    - ファイル: `src/features/advanced.py` (機能拡張)
    - 背景: 音楽ドメイン知識とデータサイエンス手法の融合
    - 目標: 楽曲構造・音楽理論に基づく新次元特徴量創出

    **23.1 [TICKET-023-01] 音楽理論ベース特徴量**
    - 要件:
      - ハーモニー解析（コード進行パターン、調性分析）
      - リズム周期性（ビート間隔変動、シンコペーション検出）
      - 楽曲構造（イントロ・サビ・アウトロ長比率）
    - 実装詳細:
      - 既存特徴量との相関分析による差別化確保
      - ドメインエキスパート知識の数理モデル化
      - 特徴量重要度による効果検証

24. **[TICKET-024] メタ学習・スタッキング** 🎯 **低優先度**
    - 新規ファイル: `src/modeling/meta_learning.py`
    - 背景: 既存高性能モデル群の予測を高次特徴量化
    - 目標: Level-2メタモデルによる予測精度向上

    **24.1 [TICKET-024-01] Level-2スタッキング実装**
    - 要件:
      - 複数ベースモデル（LGB, CAT, TabNet）のOut-of-Fold予測統合
      - メタ特徴量（予測値統計量、信頼度、一貫性指標）生成
      - Linear/Ridge/XGBoost等の軽量メタモデル選択最適化
    - 実装詳細:
      - Cross-validation内でのリーク防止機構
      - メタモデル過学習抑制（正則化・Early Stopping）
      - 階層的予測システムの安定性確保

### 開発優先順位と期待効果
1. **TICKET-021** → LB 26.385未満達成（最高優先）
2. **TICKET-022** → 異質モデル統合による安定性向上
3. **TICKET-023** → ドメイン知識活用による差別化
4. **TICKET-024** → メタ学習による最終性能向上

### 成功指標
- **短期目標**: LB 26.38未満（exp09_1超越）
- **中期目標**: CV-LB一貫性向上（格差0.05以内）
- **長期目標**: 複数手法統合による26.37台達成