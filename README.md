# Predicting the Beats-per-Minute of Songs

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Kaggle Playground Series競技: 楽曲のBeats-per-Minute (BPM)を予測するプロジェクト

## About This Competition

このプロジェクトは **Kaggle Playground Series (September 2025)** の "Predicting the Beats-per-Minute of Songs" 競技用のソリューションです。

### 問題設定
- **タスク**: 楽曲の特徴量から Beats-per-Minute (BPM) を予測する回帰問題
- **評価指標**: Root Mean Squared Error (RMSE)
- **データセット**: 実世界データから生成された合成データセット

### 特徴量
- `RhythmScore` - リズムスコア
- `AudioLoudness` - 音声の音量レベル
- `VocalContent` - ボーカル含有量
- `AcousticQuality` - 音響品質
- `InstrumentalScore` - 楽器演奏スコア
- `LivePerformanceLikelihood` - ライブ演奏っぽさ
- `MoodScore` - ムードスコア
- `TrackDurationMs` - トラック長（ミリ秒）
- `Energy` - エネルギーレベル

## Quick Start

### 環境設定
```bash
make requirements  # 依存関係をインストール
```

### データ処理
```bash
make data          # データセット処理を実行
# または
python src/dataset.py
```

### モデル訓練
```bash
python src/modeling/train.py
```

### 予測
```bash
python src/modeling/predict.py
```

### その他のコマンド
```bash
make format        # コードフォーマット
make lint          # リンターチェック
make test          # テスト実行
make clean         # 一時ファイル削除
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         src and configuration for tools like ruff
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to process and split data (CLI with typer)
    │
    ├── features.py             <- Code to create features for modeling (CLI with typer)
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models (CLI with typer)
    │   └── train.py            <- Code to train models (CLI with typer)
    │
    └── plots.py                <- Code to create visualizations (CLI with typer)
```

## Development Workflow

### データ分析の順序
1. **データ理解とEDA** (`src/dataset.py`, `src/plots.py`)
2. **特徴量エンジニアリング** (`src/features.py`)
3. **モデル開発** (`src/modeling/train.py`, `src/modeling/predict.py`)
4. **テストと品質保証**
5. **Kaggleサブミッション**

### Code Quality

このプロジェクトは以下の品質基準を採用しています：
- **Ruff**: リンティングとコードフォーマット (99文字制限)
- **Type Hints**: Python 3.10+ の型ヒントを使用
- **Loguru**: 構造化ログ出力
- **Typer**: 型安全なCLIフレームワーク

### ブランチ戦略

```
feature/ticket-XXX/機能名
```

### 設定ファイル

プロジェクト設定は `scripts/my_config.py` で管理されています。

--------

