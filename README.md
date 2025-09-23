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

## 📁 プロジェクト構成

```
├── LICENSE            <- オープンソースライセンス
├── Makefile           <- make data や make train などの便利コマンド
├── README.md          <- 開発者向けトップレベルREADME
├── CLAUDE.md          <- Claude Code アシスタント用プロジェクト指示書
├── data
│   ├── external       <- サードパーティからのデータ
│   ├── interim        <- 変換済み中間データ
│   ├── processed      <- モデリング用最終データセット
│   └── raw            <- 元の不変データダンプ
│
├── docs               <- ドキュメントファイルと使用ガイド
│   ├── KAGGLE_SUBMIT_GUIDE.md     <- Kaggle提出ガイド
│   ├── TICKET-008_USAGE_GUIDE.md  <- ジャンル特徴量使用ガイド
│   └── TICKET-008-01_USAGE_GUIDE.md <- 多重共線性除去ガイド
│
├── experiments        <- 🧪 実験管理システム
│   ├── exp01_baseline_lgb/         <- ベースラインLightGBM実験
│   ├── exp02_multicollinearity_removal/ <- 特徴量最適化実験
│   ├── exp03_advanced_features/    <- 高度な特徴量エンジニアリング
│   ├── exp004_ticket016_rhythm_periodicity/ <- リズムベース特徴量
│   ├── exp005_ticket008_03_dimensionality_reduction/ <- PCA特徴量
│   └── experiment_results.csv     <- 統合実験結果
│
├── models             <- 訓練済みモデル、予測結果
│
├── notebooks          <- 探索的分析用Jupyterノートブック
│
├── pyproject.toml     <- パッケージメタデータ付きプロジェクト設定ファイル
│
├── scripts            <- 🤖 自動化・ユーティリティスクリプト
│   ├── my_config.py           <- プロジェクト設定
│   ├── submit.py              <- Kaggle提出スクリプト
│   ├── submit_experiment.py   <- 自動実験実行ツール
│   └── evaluate_genre_features.py <- 特徴量評価ユーティリティ
│
├── references         <- データ辞書、マニュアル、説明資料
│
├── reports            <- HTML、PDF、LaTeX等の生成分析結果
│   └── figures        <- レポート用生成グラフィック・図表
│
├── requirements.txt   <- 分析環境再現用要件ファイル
│
├── tests              <- コード品質保証用テストファイル
│
└── src                <- 🎼 このプロジェクト用ソースコード
    │
    ├── __init__.py             <- srcをPythonモジュール化
    │
    ├── config.py               <- 有用な変数と設定の保存
    │
    ├── dataset.py              <- データ処理・分割スクリプト (typer CLI)
    │
    ├── features.py             <- 特徴量エンジニアリング・選択 (typer CLI)
    │                              [包含: ジャンル特徴量, 次元削減, リズム周期性]
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- モデル推論実行コード (typer CLI)
    │   └── train.py            <- モデル訓練コード (typer CLI)
    │
    └── plots.py                <- 可視化作成コード (typer CLI)
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

