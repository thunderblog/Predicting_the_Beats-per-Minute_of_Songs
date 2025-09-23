# 🎵 Predicting the Beats-per-Minute of Songs

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Kaggle Playground Series競技: 楽曲のBeats-per-Minute (BPM)を予測するMLプロジェクト
🏆 **実験管理システム** | 🎼 **音楽理論ベース特徴量** | 🤖 **自動提出システム**

## About This Competition

このプロジェクトは **Kaggle Playground Series (September 2025)** の "Predicting the Beats-per-Minute of Songs" 競技用のソリューションです。

### 🎯 問題設定
- **タスク**: 楽曲の特徴量から Beats-per-Minute (BPM) を予測する回帰問題
- **評価指標**: Root Mean Squared Error (RMSE)
- **データセット**: 実世界データから生成された合成データセット

### 📊 特徴量
- `RhythmScore` - リズムスコア
- `AudioLoudness` - 音声の音量レベル
- `VocalContent` - ボーカル含有量
- `AcousticQuality` - 音響品質
- `InstrumentalScore` - 楽器演奏スコア
- `LivePerformanceLikelihood` - ライブ演奏っぽさ
- `MoodScore` - ムードスコア
- `TrackDurationMs` - トラック長（ミリ秒）
- `Energy` - エネルギーレベル

## 🚀 Quick Start

### 環境設定
```bash
make requirements  # 依存関係をインストール
```

### 基本的なワークフロー
```bash
# 1. データ処理
make data          # データセット処理を実行

# 2. モデル訓練
python src/modeling/train.py

# 3. 予測・提出
python src/modeling/predict.py
```

### 🧪 実験管理システム（推奨）
```bash
# 実験の実行と自動提出
python scripts/submit_experiment.py --experiment-name "my_experiment"

# 結果確認
ls experiments/  # 実験結果を確認
```

### 開発者用コマンド
```bash
make format        # コードフォーマット
make lint          # リンターチェック
make test          # テスト実行
make clean         # 一時ファイル削除
```

## 📁 Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project
├── CLAUDE.md          <- Claude Code assistant project instructions
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── docs               <- Documentation files and usage guides
│   ├── KAGGLE_SUBMIT_GUIDE.md     <- Kaggle submission guide
│   ├── TICKET-008_USAGE_GUIDE.md  <- Genre features usage guide
│   └── TICKET-008-01_USAGE_GUIDE.md <- Multicollinearity removal guide
│
├── experiments        <- 🧪 Experiment management system
│   ├── exp01_baseline_lgb/         <- Baseline LightGBM experiment
│   ├── exp02_multicollinearity_removal/ <- Feature optimization experiments
│   ├── exp03_advanced_features/    <- Advanced feature engineering
│   ├── exp004_ticket016_rhythm_periodicity/ <- Rhythm-based features
│   ├── exp005_ticket008_03_dimensionality_reduction/ <- PCA features
│   └── experiment_results.csv     <- Consolidated experiment results
│
├── models             <- Trained and serialized models, model predictions
│
├── notebooks          <- Jupyter notebooks for exploratory analysis
│
├── pyproject.toml     <- Project configuration file with package metadata
│
├── scripts            <- 🤖 Automation and utility scripts
│   ├── my_config.py           <- Project configuration settings
│   ├── submit.py              <- Kaggle submission script
│   ├── submit_experiment.py   <- Automated experiment runner
│   └── evaluate_genre_features.py <- Feature evaluation utilities
│
├── references         <- Data dictionaries, manuals, and explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── tests              <- Test files for code quality assurance
│
└── src                <- 🎼 Source code for use in this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to process and split data (CLI with typer)
    │
    ├── features.py             <- Feature engineering and selection (CLI with typer)
    │                              [包含: ジャンル特徴量, 次元削減, リズム周期性]
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference (CLI with typer)
    │   └── train.py            <- Code to train models (CLI with typer)
    │
    └── plots.py                <- Code to create visualizations (CLI with typer)
```

## 🔬 Development Workflow

### 実験管理システム
各実験は `experiments/` ディレクトリで体系的に管理されています：

```bash
# 新しい実験の実行
python scripts/submit_experiment.py --experiment-name "exp006_new_feature"

# 実験結果の確認
cat experiments/experiment_results.csv
```

### データ分析の順序
1. **データ理解とEDA** (`src/dataset.py`, `src/plots.py`)
2. **特徴量エンジニアリング** (`src/features.py`)
3. **モデル開発・訓練** (`src/modeling/train.py`)
4. **予測・提出** (`src/modeling/predict.py` または `scripts/submit_experiment.py`)
5. **結果分析・次回実験計画**

### Code Quality

このプロジェクトは以下の品質基準を採用しています：
- **Ruff**: リンティングとコードフォーマット (99文字制限)
- **Type Hints**: Python 3.10+ の型ヒントを使用
- **Loguru**: 構造化ログ出力 (tqdm統合済み)
- **Typer**: 型安全なCLIフレームワーク
- **Pytest**: 包括的テストスイート (76テストケース)

### チケットベース開発

```
feature/ticket-XXX/機能名
```

実装済み: TICKET-001〜008 (データ処理, 可視化, 特徴量, モデル訓練, テスト, サブミッション, ジャンル特徴量)

### 設定ファイル

- **プロジェクト設定**: `scripts/my_config.py`
- **開発者設定**: `CLAUDE.md`

## 🎵 Advanced Features

このプロジェクトには音楽理論に基づく高度な特徴量エンジニアリング機能が実装されています。

### 🎼 TICKET-008シリーズ: 音楽理論ベース特徴量

#### 8.1 ジャンル推定特徴量 ✅
- **ballad_genre_score**: VocalContent×MoodScore（バラード系楽曲推定）
- **dance_genre_score**: Energy×RhythmScore（ダンス系楽曲推定）
- **ambient_genre_score**: AcousticQuality×InstrumentalScore（アンビエント系推定）
- **成果**: ballad_genre_scoreが最重要特徴量に選出

#### 8.2 多重共線性除去 ✅
- 高相関ペア（>0.7）の自動検出・除去システム
- ジャンル特徴量優先の最適化

#### 8.3 次元削減特徴量 ✅
- PCA変換による主成分特徴量
- 最適主成分数の自動選択（分散寄与率ベース）

### 🥁 TICKET-016: リズム周期性特徴量

ドラマー視点からのBPM予測アプローチ：

```python
# リズムパターン推定特徴量の生成例
from src.features import create_rhythm_periodicity_features

features = create_rhythm_periodicity_features(data)
# -> rhythm_consistency_score, tempo_stability_index など
```

**特徴**:
- **周期性一貫性スコア**: TrackDurationとBPM推定の整合性検証
- **疑似ドラム系特徴量**: キック・スネア・ハイハット密度推定
- **拍子変動推定**: ルバート、加速、減速パターン検出

### 🚀 実験結果

| 実験名 | RMSE | 改善度 | 主要特徴 |
|--------|------|--------|----------|
| exp01_baseline_lgb | 26.47 | - | LightGBM Baseline |
| exp02_multicollinearity_removal | 26.46 | -0.01 | 多重共線性除去 |
| exp03_advanced_features | 26.45 | -0.02 | ジャンル特徴量 |
| exp004_ticket016 | TBD | TBD | リズム周期性 |
| exp005_dimensionality_reduction | TBD | TBD | PCA特徴量 |

### 📖 詳細ガイド

- [`docs/TICKET-008_USAGE_GUIDE.md`](docs/TICKET-008_USAGE_GUIDE.md) - ジャンル特徴量使用ガイド
- [`docs/TICKET-008-01_USAGE_GUIDE.md`](docs/TICKET-008-01_USAGE_GUIDE.md) - 多重共線性除去ガイド
- [`docs/KAGGLE_SUBMIT_GUIDE.md`](docs/KAGGLE_SUBMIT_GUIDE.md) - Kaggle提出ガイド

--------

