# TICKET-017-02: 対数変換特徴量 使用ガイド

## 📋 概要
対数変換特徴量を使用したBPM予測の精度向上を図る機能。分布の歪み補正により予測性能を向上させます。

**実装状況**: ✅ **完了** - 基本機能・テスト完了

## 🚀 基本使用方法

### 方法A: 新しいCLI（推奨・簡単）
```bash
# 対数変換特徴量のみ生成
python -m src.features --create-log-features --output-dir data/processed

# 他の特徴量と組み合わせ
python -m src.features \
    --create-comprehensive-interactions \
    --create-log-features \
    --create-genre \
    --output-dir data/processed
```

### 方法B: 新しいクラスベースAPI（推奨）
```python
from src.features import LogTransformFeatureCreator
import pandas as pd

# 特徴量作成器を使用
creator = LogTransformFeatureCreator()
df = pd.read_csv('data/processed/train.csv')
result = creator.create_features(df)

# 作成された特徴量を確認
print(f"作成特徴量: {len(creator.created_features)}個")
print(f"特徴量リスト: {creator.created_features[:5]}...")

# 特徴量作成器の情報
info = creator.get_feature_info()
print(f"対象特徴量: {info['target_features']}")
print(f"除外特徴量: {info['exclude_features']}")
```

### 方法C: 後方互換関数（既存コード）
```python
from src.features import create_log_features

df_with_log = create_log_features(df)
```

### 方法D: パイプラインでの統合実行
```python
from src.features import FeaturePipeline, LogTransformFeatureCreator, ComprehensiveInteractionCreator

# カスタムパイプライン構築
pipeline = FeaturePipeline()
pipeline.add_creator(ComprehensiveInteractionCreator())  # TICKET-017-01
pipeline.add_creator(LogTransformFeatureCreator())      # TICKET-017-02

result = pipeline.execute(df)
summary = pipeline.get_execution_summary()
print(summary)
```

## 📊 生成される特徴量

### 特徴量タイプ別内訳（49特徴量）

1. **基本log1p変換特徴量（8個）**
   - 対象: RhythmScore, VocalContent, AcousticQuality, InstrumentalScore, LivePerformanceLikelihood, MoodScore, TrackDurationMs, Energy
   - 除外: AudioLoudness（設計仕様）
   - 形式: `log1p_{feature_name}`

2. **組み合わせ特徴量（36個）**
   - ペアワイズ積: `log1p_{feature1}_x_log1p_{feature2}`
   - 比率特徴量: `{log_feature}_div_log1p_TrackDurationMs`
   - 調和平均: `log_energy_rhythm_harmony`

3. **統計特徴量（4個）**
   - `log_features_mean`: 対数空間平均
   - `log_features_std`: 対数空間標準偏差
   - `log_features_range`: 対数空間範囲
   - `log_features_geometric_mean`: 幾何平均

4. **分布正規化指標（1個）**
   - `log_transformation_benefit`: 分布改善度

## 🎯 実行パターン

### パターンA: 単体実行（テスト用）
```bash
python -m src.features --create-log-features
```

### パターンB: TICKET-017統合実行（推奨）
```bash
# TICKET-017-01 + 017-02 の組み合わせ
python -m src.features \
    --create-comprehensive-interactions \
    --create-log-features \
    --output-dir data/processed/ticket017_combined
```

### パターンC: 完全パイプライン（高性能）
```bash
# 特徴量選択付き実行
python -m src.features \
    --create-comprehensive-interactions \
    --create-log-features \
    --create-genre \
    --select-features-flag \
    --n-features 75 \
    --apply-scaling \
    --output-dir data/processed/optimized
```

### パターンD: モデル訓練まで完全実行
```bash
# Step 1: 対数変換特徴量生成
python -m src.features --create-comprehensive-interactions --create-log-features

# Step 2: モデル訓練
python -m src.modeling.train \
    --train-path data/processed/train_features.csv \
    --validation-path data/processed/validation_features.csv \
    --experiment-name ticket017_02_log_features

# Step 3: 予測実行
python -m src.modeling.predict \
    --test-path data/processed/test_features.csv \
    --model-dir models \
    --experiment-name ticket017_02_log_features \
    --output-path data/processed/submission_log_features.csv
```

## 📁 生成されるファイル
```
data/processed/
├── train_features.csv                   # 対数変換特徴量付き訓練データ
├── validation_features.csv              # 対数変換特徴量付き検証データ
├── test_features.csv                   # 対数変換特徴量付きテストデータ
├── feature_importance_all.csv          # 全特徴量重要度（対数変換含む）
└── submission_log_features.csv         # Kaggle提出用予測結果

models/
├── ticket017_02_log_features_fold_1_*.pkl    # 訓練済みモデル
├── ticket017_02_log_features_fold_2_*.pkl
└── ticket017_02_log_features_cv_results_*.json
```

## 🔍 確認コマンド
```bash
# 生成ファイル確認
ls data/processed/*_features.csv

# 特徴量確認
python -c "
import pandas as pd
df = pd.read_csv('data/processed/train_features.csv')
log_cols = [col for col in df.columns if 'log1p_' in col or 'log_features_' in col]
print(f'対数変換特徴量: {len(log_cols)}個')
print('主要特徴量:', log_cols[:5])
print(f'データ形状: {df.shape}')
"

# テストスクリプト実行
python test_ticket_017_02.py
```

## 🎯 期待される改善効果

### 理論的根拠
- **分布正規化**: log1p変換により歪んだ分布を正規分布に近似
- **非線形関係捕捉**: 対数空間での組み合わせ特徴量で複雑なパターンを学習
- **統計的安定性**: 幾何平均等の安定した統計量で予測精度向上

### 性能目標
- **RMSE改善**: 0.05-0.15ポイント（TICKET-017-01との相乗効果）
- **特徴量効率**: 49特徴量で高い情報密度
- **汎化性能**: 対数変換による外れ値耐性向上

### 組み合わせ効果
- **TICKET-017-01**: 包括的交互作用特徴量（164特徴量）
- **TICKET-017-02**: 対数変換特徴量（49特徴量）
- **合計**: 約213特徴量による高次元特徴空間

## ⏱️ 実行時間目安
- **特徴量生成**: 30秒-2分
- **モデル訓練**: 5-15分
- **予測実行**: 1-3分
- **完全パイプライン**: 10-20分

## 🔧 トラブルシューティング

### よくある問題

1. **AudioLoudnessが含まれない**
   - 仕様: AudioLoudnessは設計上除外されています
   - 変更方法: `LogTransformFeatureCreator(exclude_features=[])`

2. **特徴量数が多すぎる**
   ```bash
   # 解決策: 特徴量選択を併用
   python -m src.features --create-log-features --select-features-flag --n-features 30
   ```

3. **メモリ不足**
   ```bash
   # 解決策: サンプリング実行
   python test_ticket_017_02.py  # 軽量テスト
   ```

## 📈 次のステップ

### TICKET-017-03（ビニング特徴量）との連携
```bash
# 将来的な完全統合例（017-03実装後）
python -m src.features \
    --create-comprehensive-interactions \
    --create-log-features \
    --create-binning-features \
    --select-features-flag \
    --n-features 100 \
    --output-dir data/processed/ticket017_complete
```

### 性能評価・最適化
- 特徴量重要度分析による最適化
- ハイパーパラメータチューニング
- アンサンブル手法での活用

---

**📅 作成日**: 2025-09-26
**🎵 作成者**: TICKET-017-02 Implementation Team
**🔗 関連**: TICKET-017-01（包括的交互作用特徴量）との組み合わせ推奨