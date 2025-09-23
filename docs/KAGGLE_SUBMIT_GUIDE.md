# Kaggle Submit 手順書

## 概要
このプロジェクトでのKaggle提出の手順とツールの使用方法について説明します。

## 前提条件

### 1. Kaggle CLI セットアップ
```bash
# Kaggle CLIのインストール
pip install kaggle

# API認証設定
# 1. Kaggle > Account > API > Create New API Token
# 2. ダウンロードしたkaggle.jsonを以下に配置
# Windows: C:\Users\{ユーザー名}\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json

# 権限設定（Linux/Macのみ）
chmod 600 ~/.kaggle/kaggle.json
```

### 2. 環境確認
```bash
# Kaggle CLI動作確認
kaggle competitions list

# コンペティション参加確認
kaggle competitions list -s playground-series-s5e9
```

## 提出方法

### 🚀 **方法1: 自動Submitスクリプト（推奨）**

#### 実験名指定での提出
```bash
# Windows PowerShell / Command Prompt
python scripts/submit_experiment.py --experiment-name exp005_ticket008_03_dimensionality_reduction

# メッセージ例: "EXP005_TICKET008_03_DIMENSIONALITY_REDUCTION | CV: 26.4668 | 26特徴量 | PCA/ICA次元削減特徴量と多重共線性除去による性能改善実験"
```

#### ファイル直接指定での提出
```bash
python scripts/submit_experiment.py --submission-file data/processed/submission_ticket008_03_dimensionality.csv --message "手動提出テスト"
```

#### ドライラン（テスト実行）
```bash
# 実際の提出は行わず、コマンドだけを確認
python scripts/submit_experiment.py --experiment-name exp005_ticket008_03_dimensionality_reduction --dry-run
```

### 📋 **方法2: 手動CLI Submit**

#### 基本コマンド
```bash
# Windows PowerShell / Command Prompt
kaggle competitions submit -c playground-series-s5e9 -f submission_ticket008_03_dimensionality.csv -m "TICKET008-003: PCA/ICA次元削減特徴量"
```

#### 詳細メッセージ付き
```bash
kaggle competitions submit -c playground-series-s5e9 -f submission_ticket008_03_dimensionality.csv -m "TICKET008-003: PCA/ICA次元削減特徴量 | CV: 26.4668 | 26特徴量 | 多重共線性除去済み"
```

#### フルパス指定版
```bash
# 任意のディレクトリから実行可能
kaggle competitions submit -c playground-series-s5e9 -f "C:\Users\AshigayaH\Documents\Code\Python\kaggle\Predicting_the_Beats-per-Minute_of_Songs\data\processed\submission_ticket008_03_dimensionality.csv" -m "TICKET008-003: PCA/ICA次元削減特徴量"
```

#### 実験ディレクトリからの実行
```bash
# Windows
cd experiments\exp005_ticket008_03_dimensionality_reduction
kaggle competitions submit -c playground-series-s5e9 -f submission_ticket008_03_dimensionality.csv -m "TICKET008-003: PCA/ICA次元削減特徴量"
```

## Windows環境固有の注意点

### パス指定
```bash
# ❌ 誤った書き方（Linuxスタイル）
kaggle competitions submit -c playground-series-s5e9 -f data/processed/submission.csv

# ✅ 正しい書き方（Windowsスタイル）
kaggle competitions submit -c playground-series-s5e9 -f data\processed\submission.csv

# ✅ または引用符で囲む
kaggle competitions submit -c playground-series-s5e9 -f "data/processed/submission.csv"
```

### ディレクトリ移動
```bash
# Windows Command Prompt / PowerShell
cd experiments\exp005_ticket008_03_dimensionality_reduction

# Git Bash (Windows)
cd experiments/exp005_ticket008_03_dimensionality_reduction
```

### ファイルの確認
```bash
# Windows
dir data\processed\submission*.csv
dir experiments\exp005_ticket008_03_dimensionality_reduction

# PowerShell
ls data\processed\submission*.csv
ls experiments\exp005_ticket008_03_dimensionality_reduction
```

## 提出ファイルの確認

### 1. ファイル形式チェック
```bash
# ヘッダー確認
head -5 data\processed\submission_ticket008_03_dimensionality.csv

# PowerShellの場合
Get-Content data\processed\submission_ticket008_03_dimensionality.csv -Head 5
```

### 2. ファイルサイズ・行数確認
```bash
# Windows
wc -l data\processed\submission_ticket008_03_dimensionality.csv

# PowerShell
(Get-Content data\processed\submission_ticket008_03_dimensionality.csv | Measure-Object -Line).Lines
```

### 3. 予測値の範囲確認
```python
# Python での確認
import pandas as pd
df = pd.read_csv('data/processed/submission_ticket008_03_dimensionality.csv')
print(f"Shape: {df.shape}")
print(f"BeatsPerMinute range: {df['BeatsPerMinute'].min():.2f} - {df['BeatsPerMinute'].max():.2f}")
print(f"Mean: {df['BeatsPerMinute'].mean():.2f}")
```

## 提出後の確認

### 1. 提出履歴確認
```bash
kaggle competitions submissions -c playground-series-s5e9
```

### 2. Leaderboard確認
```bash
kaggle competitions leaderboard -c playground-series-s5e9 --path leaderboard.csv
```

## トラブルシューティング

### よくあるエラーと対処法

#### 1. API認証エラー
```
401 - Unauthorized
```
**対処法**: `kaggle.json`ファイルの配置と内容を確認

#### 2. ファイルが見つからない
```
FileNotFoundError: submission.csv
```
**対処法**: ファイルパスとカレントディレクトリを確認

#### 3. コンペティション参加未完了
```
403 - Forbidden
```
**対処法**: Kaggleサイトでコンペティションに参加済みか確認

#### 4. ファイル形式エラー
```
Submission format is incorrect
```
**対処法**: CSVヘッダー（`id,BeatsPerMinute`）と行数を確認

### Windows特有のトラブル

#### パス区切り文字エラー
```bash
# エラーの原因
cd experiments/exp005_ticket008_03_dimensionality_reduction  # Unix style

# 正しい書き方
cd experiments\exp005_ticket008_03_dimensionality_reduction  # Windows style
```

#### PowerShell実行ポリシーエラー
```bash
# エラーが出る場合
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 自動化のベストプラクティス

### 1. 実験ディレクトリの構成
```
experiments/exp005_ticket008_03_dimensionality_reduction/
├── submission_*.csv           # 提出ファイル
├── results.json              # 実験メタデータ
├── README.md                 # 実験レポート
└── submit_commands.txt       # 手動submitコマンド
```

### 2. メッセージ命名規則
```
{TICKET名}: {手法概要} | CV: {CVスコア} | {特徴量数}特徴量 | {LBスコア}
```

**例**:
```
TICKET008-003: PCA/ICA次元削減特徴量 | CV: 26.4668 | 26特徴量 | LB: 26.38834
```

### 3. バッチ処理スクリプト例
```bash
# Windows Batch (.bat)
@echo off
echo 実験結果を提出中...
python scripts/submit_experiment.py --experiment-name %1
if %errorlevel% equ 0 (
    echo 提出完了！
) else (
    echo 提出失敗
    exit /b 1
)
```

## 使用例

### TICKET008-003の提出例
```bash
# 方法1: 自動スクリプト（推奨）
python scripts/submit_experiment.py --experiment-name exp005_ticket008_03_dimensionality_reduction

# 方法2: 手動コマンド
kaggle competitions submit -c playground-series-s5e9 -f "experiments\exp005_ticket008_03_dimensionality_reduction\submission_ticket008_03_dimensionality.csv" -m "TICKET008-003: PCA/ICA次元削減特徴量 | CV: 26.4668 | 26特徴量"

# 方法3: ディレクトリ移動して実行
cd experiments\exp005_ticket008_03_dimensionality_reduction
kaggle competitions submit -c playground-series-s5e9 -f submission_ticket008_03_dimensionality.csv -m "TICKET008-003: PCA/ICA次元削減特徴量"
```

## 関連ファイル

- `scripts/submit_experiment.py` - 自動提出スクリプト
- `experiments/*/submit_commands.txt` - 手動コマンド集
- `experiments/*/results.json` - 実験メタデータ
- `CLAUDE.md` - Windows環境情報とプロジェクト設定