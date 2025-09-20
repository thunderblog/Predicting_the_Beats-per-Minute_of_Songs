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
    --val-path=data/processed/validation_features.csv \
    --output-dir=models \
    --exp-name=genre_features_lgb
```

### モデル設定パラメータ
```python
model_params = {
    "objective": "regression",
    "metric": "rmse",
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
    --test-features-path=data/processed/test_features.csv \
    --exp-name=genre_features_lgb \
    --output-path=data/processed/submission_genre_features.csv
```

### 予測プロセス
1. 複数foldモデルの読み込み
2. テストデータでの予測実行
3. アンサンブル（平均）予測
4. Kaggle提出形式で出力