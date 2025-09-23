# TICKET-017-01 包括的交互作用特徴量 実行手順書

## 📋 概要

TICKET-017-01で実装された**包括的交互作用特徴量**（Kaggleサンプルコード手法）の使用方法を説明します。

### 🎯 実装内容
- **126個の新特徴量**: 積特徴量45個 + 二乗特徴量9個 + 比率特徴量72個
- **性能向上**: +0.53%のRMSE改善確認済み
- **基本特徴量**: 9個 → 135個（約15倍に拡張）

## 🚀 実行手順

### 1. 基本的な使用方法

#### コマンドライン実行
```bash
# 基本実行（包括的交互作用特徴量を生成）
python src/features.py --create-comprehensive-interactions

# 出力先を指定
python src/features.py --create-comprehensive-interactions --output-dir data/processed/enhanced

# 他の特徴量と組み合わせ
python src/features.py \
    --create-comprehensive-interactions \
    --create-genre \
    --create-statistical \
    --output-dir data/processed/full_features
```

#### Pythonスクリプトでの使用
```python
import sys
sys.path.append('src')
from features import create_comprehensive_interaction_features
import pandas as pd

# データ読み込み
df = pd.read_csv('data/processed/train.csv')

# 包括的交互作用特徴量を生成
enhanced_df = create_comprehensive_interaction_features(df)

print(f"元特徴量: {len(df.columns)}個")
print(f"拡張後: {len(enhanced_df.columns)}個")
print(f"新特徴量: {len(enhanced_df.columns) - len(df.columns)}個")
```

### 2. モデル訓練での使用

#### 基本的なモデル訓練
```bash
# 1. 拡張特徴量を生成
python src/features.py --create-comprehensive-interactions --output-dir data/processed/comprehensive

# 2. モデル訓練（拡張特徴量を使用）
python src/modeling/train.py \
    --train-path data/processed/comprehensive/train_features.csv \
    --validation-path data/processed/comprehensive/validation_features.csv \
    --save-model-path models/lgbm_comprehensive_features.pkl
```

#### 性能テスト実行
```bash
# 軽量性能テスト（1000サンプル、3-fold CV）
python scripts/test_ticket017_01.py

# 結果確認
cat results/ticket017_01_performance_test.json
```

### 3. 実験管理での使用

#### 実験ディレクトリ作成
```bash
# 実験用ディレクトリ作成
mkdir -p experiments/exp_ticket017_01_comprehensive_features

# 特徴量生成
python src/features.py \
    --create-comprehensive-interactions \
    --output-dir experiments/exp_ticket017_01_comprehensive_features/features

# モデル訓練と結果保存
python src/modeling/train.py \
    --train-path experiments/exp_ticket017_01_comprehensive_features/features/train_features.csv \
    --save-model-path experiments/exp_ticket017_01_comprehensive_features/models/
```

## 📊 生成される特徴量の詳細

### 特徴量タイプ別内訳

1. **積特徴量（45個）**
   - 形式: `{feature1}_x_{feature2}`
   - 例: `RhythmScore_x_Energy`, `VocalContent_x_MoodScore`
   - 全ペア組み合わせの積（9C2 = 36個 + 自分同士9個）

2. **二乗特徴量（9個）**
   - 形式: `{feature}_squared`
   - 例: `RhythmScore_squared`, `Energy_squared`
   - 各基本特徴量の二乗

3. **比率特徴量（72個）**
   - 形式: `{feature1}_div_{feature2}`
   - 例: `VocalContent_div_Energy`, `RhythmScore_div_TrackDurationMs`
   - ゼロ除算対策済み（分母に1e-6加算）

### 基本特徴量（9個）
```
RhythmScore, AudioLoudness, VocalContent, AcousticQuality,
InstrumentalScore, LivePerformanceLikelihood, MoodScore,
TrackDurationMs, Energy
```

## ⚠️ 注意事項とベストプラクティス

### 計算性能
- **処理時間**: 大型データセット（10万件以上）では数分かかる場合あり
- **メモリ使用量**: 元データの約15倍のメモリが必要
- **推奨**: 段階的処理やサンプリングを検討

### 過学習対策
```bash
# 特徴量選択と組み合わせ
python src/features.py \
    --create-comprehensive-interactions \
    --select-features-flag \
    --feature-selection-method combined \
    --n-features 50

# 正則化と組み合わせ
python src/features.py \
    --create-comprehensive-interactions \
    --apply-scaling \
    --scaler-type robust
```

### 品質チェック
```python
# NaN/inf値のチェック
enhanced_df = create_comprehensive_interaction_features(df)
nan_count = enhanced_df.isnull().sum().sum()
inf_count = np.isinf(enhanced_df.select_dtypes(include=[np.number])).sum().sum()
print(f"NaN: {nan_count}, inf: {inf_count}")

# 新特徴量の統計確認
new_features = [col for col in enhanced_df.columns if col not in df.columns]
print(enhanced_df[new_features].describe())
```

## 📈 期待される性能向上

### ベンチマーク結果
- **軽量テスト**: +0.53%改善（27.6187 → 27.4710 RMSE）
- **特徴量数**: 9個 → 135個
- **サンプルコード目標**: 26.38 RMSE（部分的達成）

### 組み合わせ推奨
```bash
# TICKET-017-02, 017-03と組み合わせ
python src/features.py \
    --create-comprehensive-interactions \
    --create-log-features \
    --create-binning-features \
    --output-dir data/processed/ticket017_full
```

## 🔧 トラブルシューティング

### よくある問題

1. **メモリエラー**
   ```bash
   # 解決策: サンプリング実行
   python -c "
   import pandas as pd
   df = pd.read_csv('data/processed/train.csv').sample(n=5000)
   df.to_csv('data/processed/train_sample.csv', index=False)
   "
   ```

2. **処理時間過長**
   ```bash
   # 解決策: 軽量テスト実行
   python scripts/test_ticket017_01.py
   ```

3. **ModuleNotFoundError**
   ```bash
   # 解決策: PYTHONPATH設定
   PYTHONPATH=. python src/features.py --create-comprehensive-interactions
   ```

## 📁 関連ファイル

- **実装**: `src/features.py` (`create_comprehensive_interaction_features`)
- **テストスクリプト**: `scripts/test_ticket017_01.py`
- **性能レポート**: `docs/TICKET-017-01_Performance_Test_Report.md`
- **チケット仕様**: `CLAUDE.md` (TICKET-017-01セクション)

---

**次のステップ**: TICKET-017-02（対数変換特徴量）とTICKET-017-03（ビニング特徴量）を実装して、さらなる性能向上を目指す。