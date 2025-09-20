# TICKET-008-02: 独立性の高い高次特徴量 - ワークフローガイド

## 📋 概要
TICKET-008-02で実装した18個の高次特徴量を活用し、BPM予測精度の更なる向上を図る実験

---

## 🚀 Step 1: 高次特徴量データセット生成

### 推奨コマンド（フル機能版）
```bash
# ジャンル特徴量 + 多重共線性除去 + 高次特徴量のフルパイプライン
python -m src.features --create-genre --remove-multicollinearity --create-advanced --output-dir=data/processed
```

### 生成される高次特徴量（18個）

#### 1. 比率ベース特徴量（4個）
- `vocal_energy_ratio`: VocalContent / Energy（ボーカル重視度）
- `acoustic_loudness_ratio`: AcousticQuality / AudioLoudness（音響品質対音量比）
- `rhythm_duration_ratio`: RhythmScore / log(TrackDurationMs)（時間補正リズム）
- `instrumental_live_ratio`: InstrumentalScore / LivePerformanceLikelihood（楽器性対ライブ性）

#### 2. 対数変換時間特徴量（4個）
- `log_duration_rhythm`: log(TrackDurationMs) × RhythmScore
- `log_duration_energy`: log(TrackDurationMs) × Energy
- `log_duration_mood`: log(TrackDurationMs) × MoodScore
- `duration_category`: 時間の3段階カテゴリ（0=短, 1=中, 2=長）

#### 3. 標準化済み交互作用特徴量（5個）
- `standardized_vocal_mood`: zscore(VocalContent) × zscore(MoodScore)
- `standardized_energy_rhythm`: zscore(Energy) × zscore(RhythmScore)
- `standardized_acoustic_loudness`: zscore(AcousticQuality) × zscore(AudioLoudness)
- `standardized_vocal_energy`: zscore(VocalContent) × zscore(Energy)
- `standardized_rhythm_mood`: zscore(RhythmScore) × zscore(MoodScore)

#### 4. 音楽理論ベース複雑指標（5個）
- `tempo_complexity`: (RhythmScore × AcousticQuality) / Energy
- `performance_dynamics`: LivePerformanceLikelihood × InstrumentalScore
- `music_density`: (AudioLoudness × VocalContent × InstrumentalScore) / log(TrackDurationMs)
- `harmonic_complexity`: (AcousticQuality × MoodScore) / Energy
- `song_structure_indicator`: RhythmScore × log(TrackDurationMs) × LivePerformanceLikelihood

### 技術的特徴
- **ゼロ除算対策**: すべての除算で1e-8を加算
- **独立性保証**: Z-score正規化による多重共線性回避
- **音楽理論ベース**: BPM予測に特化した音楽的複雑性指標

---

## 🎯 Step 2: モデル訓練（高次特徴量使用）

### コマンド
```bash
# 高次特徴量を含むデータセットでLightGBM訓練
python -m src.modeling.train \
    --train-path=data/processed/train_features.csv \
    --val-path=data/processed/validation_features.csv \
    --exp-name=advanced_features_lgb
```

### 期待される特徴量構成
- **元特徴量**: 9個
- **ジャンル特徴量**: 3個（ballad, dance, acoustic_genre_score）
- **基本エンジニアリング**: 交互作用、時間、統計特徴量
- **高次特徴量**: 18個（新規追加）
- **多重共線性除去後**: 最適化された特徴量セット

---

## 📊 Step 3: 性能評価・分析

### 3.1 ベースライン比較
```bash
# 現在のベースライン（TICKET-008-01結果）
echo "ベースライン LB: 26.3879（多重共線性除去版）"

# 高次特徴量版の結果確認
python -c "
import pandas as pd
submission = pd.read_csv('data/processed/submission_advanced_features.csv')
print(f'予測数: {len(submission)}')
print(f'予測範囲: [{submission[\"BeatsPerMinute\"].min():.2f}, {submission[\"BeatsPerMinute\"].max():.2f}]')
print(f'予測平均: {submission[\"BeatsPerMinute\"].mean():.2f}')
"
```

### 3.2 特徴量重要度分析
```bash
# 高次特徴量の重要度確認
python -c "
import pandas as pd
importance_df = pd.read_csv('data/processed/feature_importance_all.csv')
advanced_features = importance_df[importance_df['feature_name'].str.contains('ratio|log_duration|standardized|tempo|performance|music|harmonic|song')]
print('=== 高次特徴量重要度 Top 10 ===')
print(advanced_features.head(10)[['feature_name', 'average_importance']])
"
```

### 3.3 多重共線性効果確認
```bash
# 多重共線性除去の効果分析
cat data/processed/multicollinearity_impact_results.json
```

---

## 🎯 Step 4: テストデータでの予測

### コマンド
```bash
# 高次特徴量版での予測実行
python -m src.modeling.predict \
    --test-features-path=data/processed/test_features.csv \
    --exp-name=advanced_features_lgb \
    --output-path=data/processed/submission_advanced_features.csv
```

### 予測結果の品質確認
```bash
# 予測値の分布確認
python -c "
import pandas as pd
import numpy as np
submission = pd.read_csv('data/processed/submission_advanced_features.csv')
bpm = submission['BeatsPerMinute']
print(f'予測統計:')
print(f'  平均: {bpm.mean():.2f}')
print(f'  標準偏差: {bpm.std():.2f}')
print(f'  範囲: [{bpm.min():.2f}, {bpm.max():.2f}]')
print(f'  異常値(<50 or >200): {((bpm < 50) | (bpm > 200)).sum()}')
"
```

---

## 📈 Step 5: 実験結果の記録と分析

### 5.1 experiment_results.csv更新
```bash
# 実験結果CSVに新しい行を追加
python -c "
import pandas as pd
results_df = pd.read_csv('experiments/experiment_results.csv')

# 新しい実験行を追加（結果確認後に値を更新）
new_row = {
    'exp_id': 'exp03',
    'exp_name': 'advanced_features',
    'description': '独立性の高い高次特徴量（18個）追加',
    'date': '2025-09-20',
    'cv_score': 'TBD',
    'cv_std': 'TBD',
    'lb_score': 'TBD',
    'cv_lb_diff': 'TBD',
    'improvement_from_baseline': 'TBD',
    'improvement_from_previous': 'TBD',
    'model_type': 'LightGBM',
    'n_features': 'TBD',
    'n_samples': 419331,
    'cv_folds': 5,
    'training_time_min': 'TBD',
    'feature_engineering': '比率・対数・標準化済み交互作用・音楽理論',
    'hyperparameters': '{\"n_estimators\": 1000, \"learning_rate\": 0.1}',
    'preprocessing': '多重共線性除去済み',
    'ensemble_method': '5-fold平均',
    'status': 'running',
    'submission_file': 'submission_advanced_features.csv',
    'notes': '18個の高次特徴量追加。重要特徴量分析要確認'
}

# CSVに追加
results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
results_df.to_csv('experiments/experiment_results.csv', index=False)
print('experiment_results.csv を更新しました')
"
```

### 5.2 exp03実験ディレクトリ整備
```bash
# 実験結果をexp03ディレクトリに整理
mkdir -p experiments/exp03_advanced_features/{models,results}

# モデルファイルのコピー（実行完了後）
cp models/advanced_features_lgb_*.pkl experiments/exp03_advanced_features/models/
cp models/advanced_features_lgb_cv_results_*.json experiments/exp03_advanced_features/results/

# 提出ファイルのコピー
cp data/processed/submission_advanced_features.csv experiments/exp03_advanced_features/
```

---

## 🔍 Step 6: 詳細分析と考察

### 6.1 特徴量カテゴリ別効果分析
```bash
# カテゴリ別重要度分析
python -c "
import pandas as pd
importance_df = pd.read_csv('data/processed/feature_importance_all.csv')

categories = {
    '比率ベース': 'ratio',
    '対数変換時間': 'log_duration',
    '標準化済み交互作用': 'standardized',
    '音楽理論ベース': 'tempo|performance|music|harmonic|song'
}

print('=== カテゴリ別特徴量重要度分析 ===')
for cat_name, pattern in categories.items():
    cat_features = importance_df[importance_df['feature_name'].str.contains(pattern)]
    if not cat_features.empty:
        avg_importance = cat_features['average_importance'].mean()
        top_feature = cat_features.iloc[0]['feature_name']
        print(f'{cat_name}: 平均重要度={avg_importance:.4f}, トップ={top_feature}')
"
```

### 6.2 CV-LB一貫性分析
```bash
# CV結果の確認（実行完了後）
python -c "
import json
with open('models/advanced_features_lgb_cv_results_*.json', 'r') as f:
    cv_results = json.load(f)
cv_score = cv_results['mean_rmse']
# LBスコアと比較
lb_score = # [実際のLBスコア]
consistency = abs(cv_score - lb_score)
print(f'CV-LB一貫性: CV={cv_score:.4f}, LB={lb_score:.4f}, 差={consistency:.4f}')
"
```

---

## 🎯 期待される成果

### 定量的改善目標
- **LB改善**: 26.3879 → 26.2x～26.3x（0.1-0.2ポイント改善）
- **改善率**: 0.2-0.5%
- **特徴量増加**: +18個の高次特徴量

### 重要特徴量予測
1. **tempo_complexity**: BPMに直結する複雑性指標
2. **performance_dynamics**: ライブ性×楽器性の組み合わせ
3. **音楽密度系特徴量**: 楽曲の密度とBPMの関係

### 技術的価値
- **独立性確保**: Z-score正規化による多重共線性回避
- **音楽理論統合**: ドメイン知識を活用した特徴量設計
- **スケーラビリティ**: 他の音楽予測タスクへの応用可能性

---

## 🔧 トラブルシューティング

### メモリ不足対策
```bash
# 小サンプルでのテスト実行
python -m src.features --create-advanced --output-dir=data/processed/test
```

### 特徴量検証
```bash
# 生成された特徴量の確認
python -c "
df = pd.read_csv('data/processed/train_features.csv')
advanced_cols = [col for col in df.columns if any(x in col for x in ['ratio', 'log_duration', 'standardized', 'tempo', 'performance', 'music', 'harmonic', 'song'])]
print(f'高次特徴量数: {len(advanced_cols)}個（期待値: 18個）')
if len(advanced_cols) != 18:
    print('⚠ 特徴量数が期待値と異なります')
"
```

---

**🎵 TICKET-008-02により、音楽理論と統計学を融合した高度な特徴量エンジニアリングでBPM予測精度の向上を実現！**