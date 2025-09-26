# TICKET-017-03: ビニング・カテゴリ特徴量 使用ガイド

## 📋 概要
ビニング・カテゴリ特徴量を使用したBPM予測の精度向上を図る機能。分位数分割による離散化で非線形関係を捕捉し予測性能を向上させます。

**実装状況**: ✅ **完了** - 基本機能・テスト完了

## 🚀 基本使用方法

### 方法A: 新しいCLI（推奨・簡単）
```bash
# ビニング・カテゴリ特徴量のみ生成
python -m src.features --create-binning-features --output-dir data/processed

# 他の特徴量と組み合わせ
python -m src.features \
    --create-comprehensive-interactions \
    --create-log-features \
    --create-binning-features \
    --output-dir data/processed
```

### 方法B: 新しいクラスベースAPI（推奨）
```python
from src.features import BinningFeatureCreator
import pandas as pd

# 特徴量作成器を使用
creator = BinningFeatureCreator()
df = pd.read_csv('data/processed/train.csv')
result = creator.create_features(df)

# 作成された特徴量を確認
print(f"作成特徴量: {len(creator.created_features)}個")
print(f"特徴量リスト: {creator.created_features[:5]}...")

# 特徴量作成器の情報
info = creator.get_feature_info()
print(f"対象特徴量: {info['target_features']}")
print(f"ビニング設定: {info['binning_configs']}")
```

### 方法C: 後方互換関数（既存コード）
```python
from src.features import create_binning_features

df_with_binning = create_binning_features(df)
```

### 方法D: パイプラインでの統合実行
```python
from src.features import FeaturePipeline, BinningFeatureCreator, LogTransformFeatureCreator

# カスタムパイプライン構築
pipeline = FeaturePipeline()
pipeline.add_creator(LogTransformFeatureCreator())      # TICKET-017-02
pipeline.add_creator(BinningFeatureCreator())           # TICKET-017-03

result = pipeline.execute(df)
summary = pipeline.get_execution_summary()
print(summary)
```

## 📊 生成される特徴量

### 特徴量タイプ別内訳（推定60-80特徴量）

1. **基本ビニング特徴量（約27個）**
   - 7分位（septile）: 9特徴量 × 各数値特徴量
   - 10分位（decile）: 9特徴量 × 各数値特徴量
   - 5分位（quintile）: 9特徴量 × 各数値特徴量
   - 形式: `{feature_name}_{binning_type}_bin`

2. **log変換ビニング特徴量（約8個）**
   - log1p変換済み特徴量の5分位分割
   - 形式: `log1p_{feature_name}_quintile_bin`

3. **ビン統計特徴量（約70個）**
   - 各ビンのBPM平均値: `{feature}_{binning}_bin_mean_bpm`
   - 各ビンのBPM標準偏差: `{feature}_{binning}_bin_std_bmp`
   - 対象: 全ビニング特徴量

4. **ビン間相互作用特徴量（約16個）**
   - 重要特徴量ペアのビン積: `{feature1_bin}_x_{feature2_bin}`
   - ビン差分特徴量: `{feature1_bin}_diff_{feature2_bin}`
   - 対象: RhythmScore×Energy, VocalContent×MoodScore等

## 🎯 実行パターン

### パターンA: 単体実行（テスト用）
```bash
python -m src.features --create-binning-features
```

### パターンB: TICKET-017統合実行（推奨）
```bash
# TICKET-017-01 + 017-02 + 017-03 の完全統合
python -m src.features \
    --create-comprehensive-interactions \
    --create-log-features \
    --create-binning-features \
    --output-dir data/processed/ticket017_complete
```

### パターンC: 特徴量選択付き実行（高性能）
```bash
# 特徴量選択でビニング特徴量を厳選
python -m src.features \
    --create-comprehensive-interactions \
    --create-log-features \
    --create-binning-features \
    --select-features-flag \
    --n-features 100 \
    --apply-scaling \
    --output-dir data/processed/optimized
```

### パターンD: モデル訓練まで完全実行
```bash
# Step 1: TICKET-017完全版特徴量生成
python -m src.features --create-comprehensive-interactions --create-log-features --create-binning-features

# Step 2: モデル訓練
python -m src.modeling.train \
    --train-path data/processed/train_features.csv \
    --validation-path data/processed/validation_features.csv \
    --experiment-name ticket017_complete

# Step 3: 予測実行
python -m src.modeling.predict \
    --test-path data/processed/test_features.csv \
    --model-dir models \
    --experiment-name ticket017_complete \
    --output-path data/processed/submission_complete.csv
```

## 📁 生成されるファイル
```
data/processed/
├── train_features.csv                   # ビニング特徴量付き訓練データ
├── validation_features.csv              # ビニング特徴量付き検証データ
├── test_features.csv                   # ビニング特徴量付きテストデータ
├── feature_importance_all.csv          # 全特徴量重要度（ビニング含む）
└── submission_complete.csv             # Kaggle提出用予測結果

models/
├── ticket017_complete_fold_1_*.pkl     # 訓練済みモデル
├── ticket017_complete_fold_2_*.pkl
└── ticket017_complete_cv_results_*.json
```

## 🔍 確認コマンド
```bash
# 生成ファイル確認
ls data/processed/*_features.csv

# 特徴量確認
python -c "
import pandas as pd
df = pd.read_csv('data/processed/train_features.csv')
bin_cols = [col for col in df.columns if '_bin' in col]
stat_cols = [col for col in df.columns if '_bin_mean_bpm' in col or '_bin_std_bmp' in col]
print(f'ビニング特徴量: {len(bin_cols)}個')
print(f'統計特徴量: {len(stat_cols)}個')
print(f'主要ビニング特徴量:', bin_cols[:5])
print(f'データ形状: {df.shape}')
"

# テストスクリプト実行
python test_ticket_017_03.py
```

## 🎯 期待される改善効果

### 理論的根拠
- **非線形関係捕捉**: 分位数分割により数値特徴量の非線形パターンを離散化
- **カテゴリ統計**: ビンごとのBPM統計により局所的なパターンを学習
- **相互作用強化**: ビン間の相互作用で複雑な特徴量関係を表現
- **外れ値耐性**: ビニングによる外れ値の影響軽減

### 性能目標
- **RMSE改善**: 0.1-0.3ポイント改善（TICKET-017-01, 017-02との相乗効果）
- **特徴量効率**: 60-80特徴量で高い予測力
- **解釈性**: カテゴリベース特徴量による予測根拠の明確化

### 組み合わせ効果
- **TICKET-017-01**: 包括的交互作用特徴量（126特徴量）
- **TICKET-017-02**: 対数変換特徴量（49特徴量）
- **TICKET-017-03**: ビニング・カテゴリ特徴量（60-80特徴量）
- **合計**: 約260-280特徴量による高次元特徴空間

## ⏱️ 実行時間目安
- **特徴量生成**: 45秒-3分
- **モデル訓練**: 10-25分（特徴量数増加により）
- **予測実行**: 2-5分
- **完全パイプライン**: 15-35分

## 🔧 トラブルシューティング

### よくある問題

1. **分位数分割エラー**
   ```
   ValueError: Bin edges must be unique
   ```
   - 原因: 特徴量の値の種類が少なすぎる
   - 解決策: 自動的にスキップされるため問題なし

2. **特徴量数が多すぎる**
   ```bash
   # 解決策: 特徴量選択を併用
   python -m src.features --create-binning-features --select-features-flag --n-features 50
   ```

3. **メモリ不足**
   ```bash
   # 解決策: サンプリング実行
   python test_ticket_017_03.py  # 軽量テスト
   ```

4. **BPM統計特徴量が作成されない**
   - 原因: BeatsPerMinute列がない（テストデータ等）
   - 解決策: 意図的仕様のため問題なし

## 📈 次のステップ

### TICKET-017完全版の活用
```bash
# 将来的な完全統合例
python -m src.features \
    --create-comprehensive-interactions \
    --create-log-features \
    --create-binning-features \
    --select-features-flag \
    --n-features 120 \
    --output-dir data/processed/ticket017_ultimate
```

### 性能評価・最適化
- ビニング特徴量の重要度分析
- 最適なビン数の実験的決定
- アンサンブル手法での活用
- ハイパーパラメータチューニング

## 🎵 音楽理論的意義

### BPM予測における意味
- **テンポ範囲の離散化**: 楽曲のテンポ帯（遅・中・速）を自動識別
- **楽曲類型の発見**: 似た特徴量パターンの楽曲をカテゴリ化
- **局所的学習**: 特定のBPM範囲での予測精度向上

### 音楽分析への貢献
- **ジャンル分析**: カテゴリ別のBPM分布の分析
- **楽曲推薦**: 類似ビンの楽曲推薦システム
- **音楽制作**: テンポ設定の指針提供

---

**📅 作成日**: 2025-09-26
**🎵 作成者**: TICKET-017-03 Implementation Team
**🔗 関連**: TICKET-017-01, TICKET-017-02との統合により完全なTICKET-017スイート完成