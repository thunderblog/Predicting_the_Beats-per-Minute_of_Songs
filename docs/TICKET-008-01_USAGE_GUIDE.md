# TICKET-008-01: 多重共線性除去機能 使用ガイド

## 🎯 概要
TICKET-008で発見された多重共線性問題を自動解決する機能。ジャンル特徴量と元特徴量間の高相関を検出し、音楽理論に基づく優先判定で最適化を実現。

## 📋 基本コマンド

### 1. 多重共線性除去付き特徴量生成
```bash
# 基本実行（推奨）
python -m src.features \
  --create-genre \
  --remove-multicollinearity \
  --multicollinearity-threshold 0.7 \
  --prioritize-genre-features \
  --output-dir data/processed

# 簡略形（デフォルト値使用）
python -m src.features --create-genre --remove-multicollinearity
```

**新規オプション:**
- `--remove-multicollinearity`: 多重共線性除去を有効化
- `--multicollinearity-threshold 0.7`: 相関検出閾値（デフォルト0.7）
- `--prioritize-genre-features`: ジャンル特徴量優先モード（デフォルトtrue）
- `--no-prioritize-genre-features`: ジャンル特徴量優先を無効化

### 2. 閾値調整での実行例
```bash
# 厳格な閾値（高相関のみ除去）
python -m src.features --create-genre --remove-multicollinearity --multicollinearity-threshold 0.8

# 緩い閾値（より多くの相関を除去）
python -m src.features --create-genre --remove-multicollinearity --multicollinearity-threshold 0.6

# ジャンル特徴量を優先しない場合
python -m src.features --create-genre --remove-multicollinearity --no-prioritize-genre-features
```

## 🔍 機能詳細

### 検出される高相関ペア例（TICKET-008問題）
- **ballad_genre_score ↔ VocalContent**: 0.803
- **dance_genre_score ↔ Energy**: 0.871
- **acoustic_genre_score ↔ InstrumentalScore**: 0.655

### ジャンル特徴量優先ロジック
1. **ジャンル vs 元特徴量**: ジャンル特徴量を保持
2. **ジャンル vs ジャンル**: 辞書順で早いものを保持
3. **元特徴量 vs 元特徴量**: 辞書順で早いものを保持

## 📊 出力ファイル

### 生成されるファイル
```
data/processed/
├── train_features.csv                    # 多重共線性除去後の訓練データ
├── validation_features.csv               # 多重共線性除去後の検証データ
├── test_features.csv                     # 多重共線性除去後のテストデータ
├── multicollinearity_removal_info.csv    # 除去された特徴量の詳細
├── high_correlation_pairs.csv            # 検出された高相関ペア一覧
└── multicollinearity_impact_results.json # Before/After性能比較結果
```

### 除去情報ファイルの内容
```csv
removed_feature,kept_feature,correlation,removal_reason
VocalContent,ballad_genre_score,0.803,Non-genre feature removed in favor of genre feature
Energy,dance_genre_score,0.871,Non-genre feature removed in favor of genre feature
```

## 🎯 実行パターン

### パターンA: 標準実行（推奨）
```bash
python -m src.features \
  --create-genre \
  --remove-multicollinearity \
  --output-dir data/processed
```
**効果**: 閾値0.7で多重共線性除去、ジャンル特徴量優先

### パターンB: 性能重視（厳格）
```bash
python -m src.features \
  --create-genre \
  --remove-multicollinearity \
  --multicollinearity-threshold=0.8 \
  --output-dir data/processed
```
**効果**: 高相関ペアのみ除去、特徴量数維持

### パターンC: モデル軽量化重視
```bash
python -m src.features \
  --create-genre \
  --remove-multicollinearity \
  --multicollinearity-threshold=0.6 \
  --output-dir data/processed
```
**効果**: より多くの特徴量除去、モデル簡素化

### パターンD: 完全パイプライン（訓練→予測）
```bash
# Step 1: 多重共線性除去付き特徴量生成
python -m src.features --create-genre --remove-multicollinearity --output-dir data/processed

# Step 2: 最適化されたデータセットで訓練
python -m src.modeling.train \
  --train-path=data/processed/train_features.csv \
  --validation-path=data/processed/validation_features.csv \
  --experiment-name=optimized_multicollinearity_lgb

# Step 3: 予測実行
python -m src.modeling.predict \
  --test-path=data/processed/test_features.csv \
  --model-dir=models \
  --experiment-name=optimized_multicollinearity_lgb \
  --output-path=data/processed/submission_optimized.csv
```

## 🔬 性能評価結果

### 実測値（1000サンプル、閾値0.6での検証）
- **除去前RMSE**: 28.4568 (±0.8711)
- **除去後RMSE**: 28.3524 (±0.5855)
- **改善**: +0.1044 (**+0.37%**)
- **特徴量数**: 21 → 15 (-6個除去)
- **安定性向上**: 標準偏差改善

### 除去された特徴量例
1. `VocalContent` → `ballad_genre_score`保持
2. `Energy` → `dance_genre_score`保持
3. `InstrumentalScore` → `acoustic_genre_score`保持
4. `LivePerformanceLikelihood` → `rock_genre_score`保持

## 🛠️ トラブルシューティング

### 問題1: 高相関ペアが検出されない
```bash
# 解決策: 閾値を下げる
python -m src.features --create-genre --remove-multicollinearity --multicollinearity-threshold 0.6
```

### 問題2: 重要な特徴量が除去される
```bash
# 解決策: ジャンル特徴量優先を無効化
python -m src.features --create-genre --remove-multicollinearity --no-prioritize-genre-features
```

### 問題3: 性能改善が見られない
- 元々多重共線性が少ない可能性
- より厳格な閾値（0.8-0.9）を試す
- 他の特徴量エンジニアリング手法との組み合わせを検討

## 📈 最適化のヒント

### 1. 閾値選択指針
- **0.8以上**: 厳格（明確な冗長性のみ除去）
- **0.7**: 標準（バランス型、推奨）
- **0.6以下**: 積極的（モデル軽量化重視）

### 2. ジャンル特徴量の価値
- 音楽理論に基づく解釈可能性
- ドメイン知識の活用
- モデルの説明性向上

### 3. パフォーマンス最適化
- Before/After比較で効果測定
- クロスバリデーションでの安定性確認
- 特徴量重要度の変化監視

## 🔍 確認コマンド

### 生成ファイル確認
```bash
# 多重共線性除去結果確認
ls data/processed/multicollinearity_*

# 除去された特徴量確認
cat data/processed/multicollinearity_removal_info.csv

# 性能改善結果確認
cat data/processed/multicollinearity_impact_results.json
```

### データ形状確認
```bash
python -c "
import pandas as pd
original = pd.read_csv('data/processed/train.csv')
optimized = pd.read_csv('data/processed/train_features.csv')
print(f'元データ: {original.shape}')
print(f'最適化後: {optimized.shape}')
print(f'特徴量削減: {original.shape[1] - optimized.shape[1]}個')
"
```

## ⏱️ 実行時間目安
- **特徴量生成**: 2-5分
- **多重共線性検出**: 30秒-2分
- **Before/After比較**: 3-10分（データサイズ依存）
- **完全パイプライン**: 15-45分

## 🎯 期待される効果
- **RMSE改善**: 0.1-0.5ポイント
- **特徴量効率化**: 5-15%の削減
- **モデル安定性**: 標準偏差改善
- **解釈性向上**: ジャンル特徴量の活用

---

**📅 作成日**: 2025-09-17
**🔧 作成者**: TICKET-008-01 Implementation Team
**🎵 関連**: TICKET-008 音楽ジャンル推定特徴量の多重共線性問題解決