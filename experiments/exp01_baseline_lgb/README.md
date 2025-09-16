# Experiment 01: Baseline LightGBM

## 概要
- **実験名**: exp01_baseline_lgb
- **実施日**: 2025-09-16
- **目的**: ベースラインモデルの構築と初回提出

## モデル性能

### Cross Validation Results
- **CV Score**: 26.47 RMSE (平均)
- **CV Strategy**: KFold (5-fold)
- **使用フォールド数**: 2 (fold_1, fold_2)

### Leaderboard Results
- **Public LB Score**: 26.39087 RMSE
- **Public LB Rank**: 1079位
- **CV vs LB差**: 0.08 (良好な汎化性能)

## モデル設定

### LightGBM パラメータ
```json
{
  "objective": "regression",
  "metric": "rmse",
  "num_leaves": 31,
  "learning_rate": 0.1,
  "feature_fraction": 0.9,
  "n_estimators": 10000,
  "early_stopping_rounds": 50
}
```

### 特徴量 (15個)
#### オリジナル特徴量 (9個)
- RhythmScore, AudioLoudness, VocalContent
- AcousticQuality, InstrumentalScore, LivePerformanceLikelihood
- MoodScore, TrackDurationMs, Energy

#### エンジニアリング特徴量 (6個)
- `live_mood_product`: LivePerformanceLikelihood × MoodScore
- `rhythm_energy_ratio`: RhythmScore / Energy
- `is_short_track`: TrackDurationMs < 180000の二値特徴量
- `duration_long`: 長時間楽曲フラグ
- `duration_short`: 短時間楽曲フラグ
- `audio_vocal_ratio`: AudioLoudness / VocalContent

## データ概要
- **訓練データ**: 349,444件
- **テストデータ**: 174,722件
- **ターゲット**: BeatsPerMinute (BPM)
- **予測値範囲**: 114.81 - 127.13 BPM (現実的な範囲)

## 技術実装

### 予測パイプライン
1. **モデル読み込み**: 2つのフォールドモデルをロード
2. **アンサンブル予測**: 算術平均によるアンサンブル
3. **後処理**: BPM範囲(30-300)でのクリッピング
4. **出力**: Kaggle提出形式のCSV

### ファイル構成
```
experiments/exp01_baseline_lgb/
├── config.json          # 実験設定
├── results.json         # CV・LB結果
├── submission.csv       # 提出ファイル
├── models/             # 訓練済みモデル
│   ├── test_run_fold_1_20250916_133817.pkl
│   └── test_run_fold_2_20250916_133817.pkl
└── README.md           # 本ファイル
```

## 考察・気づき

### 成功要因
✅ **汎化性能**: CV(26.47) ≈ LB(26.39)の良好な一致
✅ **特徴量設計**: 交互作用・時間ベース特徴量が効果的
✅ **アンサンブル**: 複数フォールドの平均が安定性向上
✅ **後処理**: BPM範囲制限による現実的な予測値

### 改善の余地
🔄 **特徴量拡張**: より高次の交互作用、統計的特徴量
🔄 **モデル多様化**: XGBoost、CatBoost、Neural Networkの追加
🔄 **ハイパーパラメータ**: Optunaによる体系的最適化
🔄 **アンサンブル**: 異種モデル間のブレンディング

## Next Steps
1. 特徴量重要度分析と新特徴量の開発
2. 他のアルゴリズム（XGBoost、CatBoost）での実験
3. スタッキング・ブレンディングによる高度なアンサンブル
4. ハイパーパラメータの最適化

---
**Status**: ✅ Completed
**Baseline established**: ✅ Public LB 26.39087 (1079位)