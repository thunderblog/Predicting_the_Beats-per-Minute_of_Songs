# TICKET-008: 音楽ジャンル推定特徴量 - ワークフローガイド

## 📋 概要
TICKET-008で実装した音楽ジャンル推定特徴量を含むデータセット作成から予測まで完全ガイド

---

## 🚀 Step 1: 拡張特徴量データセット生成

### コマンド
```bash
# 音楽ジャンル特徴量を含む拡張データセットを生成
python -m src.features --create-genre --output-dir=data/processed
```

### 生成される特徴量
- **dance_genre_score**: Energy × RhythmScore（EDM/ダンス系）
- **acoustic_genre_score**: AcousticQuality × InstrumentalScore（フォーク/クラシック系）
- **ballad_genre_score**: VocalContent × MoodScore（バラード/R&B系）
- **rock_genre_score**: Energy × LivePerformanceLikelihood（ロック/ポップ系）
- **electronic_genre_score**: (1-VocalContent) × Energy（エレクトロニック系）
- **ambient_genre_score**: (1-Energy) × AcousticQuality（アンビエント/チルアウト系）

### 生成されるファイル
```
data/processed/
├── train_features.csv           # 拡張訓練データ（419,331サンプル × 39特徴量）
├── validation_features.csv      # 拡張検証データ（104,833サンプル × 39特徴量）
├── test_features.csv           # 拡張テストデータ（139,777サンプル × 38特徴量）
├── feature_info.csv            # 特徴量情報
├── feature_importance_all.csv  # 全特徴量重要度分析
├── feature_importance_genre.csv # ジャンル特徴量重要度分析
└── genre_bpm_analysis.csv      # ジャンル特徴量とBPM関係分析
```

---

## 🎯 Step 2: モデル訓練（拡張特徴量使用）

### コマンド
```bash
# 拡張特徴量を使用してLightGBMモデルを訓練
python -m src.modeling.train \
    --train-path=data/processed/train_features.csv \
    --validation-path=data/processed/validation_features.csv \
    --output-dir=models \
    --experiment-name=genre_features_lgb
```

### モデル設定パラメータ
```python
model_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "feature_fraction": 0.9,
    "n_estimators": 10000,
    "early_stopping_rounds": 50,
    "random_state": 42
}
```

### 生成されるファイル
```
models/
├── genre_features_lgb_fold_1_YYYYMMDD_HHMMSS.pkl  # 訓練済みモデル（Fold 1）
├── genre_features_lgb_fold_2_YYYYMMDD_HHMMSS.pkl  # 訓練済みモデル（Fold 2）
└── genre_features_lgb_cv_results_YYYYMMDD_HHMMSS.json  # CV結果
```

---

## 📊 Step 3: 性能評価・比較

### 3.1 ベースライン（既存特徴量のみ）との比較
```bash
# ベースラインと拡張版の性能比較評価
python scripts/evaluate_genre_features.py
```

### 3.2 クイック評価（小サンプル）
```bash
# 軽量版評価（1万サンプル）
python scripts/quick_evaluation.py
```

### 期待される結果例
```
=== クイック評価結果 ===
ベースライン RMSE: 26.4700 (33特徴量)
拡張版 RMSE: 26.3500 (39特徴量)
改善: 0.1200 (+0.45%)

=== ジャンル特徴量重要度 ===
  ambient_genre_score: 0.0845
  electronic_genre_score: 0.0321
  dance_genre_score: 0.0298
  rock_genre_score: 0.0156
  acoustic_genre_score: 0.0134
  ballad_genre_score: 0.0098
```

---

## 🎯 Step 4: テストデータでの予測

### コマンド
```bash
# 訓練済みモデルを使用してテストデータを予測
python -m src.modeling.predict \
    --test-path=data/processed/test_features.csv \
    --model-dir=models \
    --experiment-name=genre_features_lgb \
    --output-path=data/processed/submission_genre_features.csv
```

### 予測プロセス
1. 複数foldモデルの読み込み
2. テストデータでの予測実行
3. アンサンブル（平均）予測
4. Kaggle提出形式で出力

---

## 📈 Step 5: 結果の可視化・分析

### 5.1 特徴量重要度可視化
```bash
# 特徴量重要度をプロット
python -c "
import pandas as pd
import matplotlib.pyplot as plt

# 重要度データ読み込み
importance_df = pd.read_csv('data/processed/feature_importance_all.csv')

# トップ15特徴量をプロット
top_features = importance_df.head(15)
plt.figure(figsize=(10, 8))
plt.barh(top_features['feature_name'], top_features['average_importance'])
plt.title('Top 15 Feature Importance (Genre Features Included)')
plt.xlabel('Average Importance Score')
plt.tight_layout()
plt.savefig('reports/figures/genre_features_importance.png')
plt.show()
"
```

### 5.2 ジャンル特徴量とBPMの関係分析
```bash
# ジャンル特徴量の分析結果を確認
python -c "
import pandas as pd

# ジャンル-BPM分析結果を表示
analysis_df = pd.read_csv('data/processed/genre_bpm_analysis.csv')
print('=== ジャンル特徴量とBPMの関係 ===')
for _, row in analysis_df.iterrows():
    print(f'{row[\"genre_feature\"]:25}: 相関={row[\"correlation_with_bpm\"]:.3f}, BPM範囲={row[\"bpm_range\"]:.1f}')
"
```

---

## 🔍 Step 6: 実験結果の記録

### 実験ディレクトリ作成
```bash
# 実験結果保存用ディレクトリを作成
mkdir -p experiments/exp01_genre_features/{models,config,results}
```

### ファイル整理
```bash
# 実験結果を整理
cp data/processed/submission_genre_features.csv experiments/exp01_genre_features/
cp models/genre_features_lgb_cv_results_*.json experiments/exp01_genre_features/results/
cp models/genre_features_lgb_fold_*.pkl experiments/exp01_genre_features/models/
```

### 実験レポート作成
```bash
# 実験レポートを作成
cat > experiments/exp01_genre_features/README.md << 'EOF'
# Experiment 01: Genre Features Implementation

## 概要
- 実験目的: 音楽ジャンル推定特徴量によるBPM予測精度向上
- 実施日: 2025-09-17
- モデル: LightGBM

## 結果
- CV Score: XX.XXXX
- 改善: +X.XX%
- 有効特徴量: ambient_genre_score (最も重要)

## 考察
- アンビエント系特徴量が最も効果的
- 低エネルギー×高音響品質の組み合わせが重要
EOF
```

---

## ⚙️ オプション: 高度な設定

### カスタム特徴量選択
```bash
# 特定のジャンル特徴量のみ使用
python -m src.features \
    --create-interactions=True \
    --create-duration=True \
    --create-statistical=True \
    --create-genre=True \
    --select-features-flag=True \
    --feature-selection-method=kbest \
    --n-features=25
```

### ハイパーパラメータチューニング
```bash
# Optuna最適化（TICKET-013実装後）
python -m src.modeling.optimization \
    --train-path=data/processed/train_features.csv \
    --n-trials=100 \
    --timeout=3600
```

---

## 📝 コマンド実行順序まとめ

### フルワークフロー
```bash
# 1. 拡張特徴量生成
python -m src.features --create-genre --output-dir=data/processed

# 2. モデル訓練
python -m src.modeling.train \
    --train-path=data/processed/train_features.csv \
    --validation-path=data/processed/validation_features.csv \
    --experiment-name=genre_features_lgb

# 3. 性能評価
python scripts/evaluate_genre_features.py

# 4. 予測実行
python -m src.modeling.predict \
    --test-path=data/processed/test_features.csv \
    --model-dir=models \
    --experiment-name=genre_features_lgb

# 5. 結果確認
ls data/processed/submission_genre_features.csv
```

### 実行時間の目安
- **特徴量生成**: 2-5分（データサイズによる）
- **モデル訓練**: 10-30分（CV + 42万サンプル）
- **性能評価**: 15-45分（全データでの比較）
- **予測実行**: 1-3分（テストデータ14万サンプル）

---

## 🎯 期待される成果

### 定量的改善
- **RMSE改善**: 0.1-0.5ポイント（26.47 → 26.3x）
- **改善率**: 0.3-2.0%
- **統計的有意性**: ambient_genre_score (p<0.05)

### 定性的価値
- **音楽理論との整合性**: ジャンル別BPMパターンの捕捉
- **特徴量の解釈性**: 各ジャンル特徴量の音楽的意味が明確
- **拡張可能性**: 新しいジャンル特徴量への応用基盤

---

## 🔧 トラブルシューティング

### よくある問題と解決策

#### メモリ不足
```bash
# データサイズを削減
python -c "
df = pd.read_csv('data/processed/train_features.csv')
df.sample(50000).to_csv('data/processed/train_features_small.csv', index=False)
"
```

#### 処理時間過多
```bash
# 軽量設定でのクイック実行
python scripts/quick_evaluation.py  # 1万サンプルのみ
```

#### 特徴量確認
```bash
# 生成された特徴量を確認
python -c "
df = pd.read_csv('data/processed/train_features.csv')
genre_cols = [col for col in df.columns if 'genre_score' in col]
print('ジャンル特徴量:', genre_cols)
print('データ形状:', df.shape)
"
```

---

**🎵 TICKET-008により、音楽理論に基づく高度な特徴量エンジニアリングがBPM予測に活用可能になりました！**