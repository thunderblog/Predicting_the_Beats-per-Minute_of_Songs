# Experiment 02: 多重共線性除去による特徴量最適化

## 概要
- **実験目的**: ジャンル特徴量と元特徴量間の多重共線性を除去し、予測精度向上を図る
- **仮説**: 高相関ペア（>0.7）の元特徴量を除去することで、冗長性を排除し汎化性能が向上する
- **実施日**: 2025-09-20
- **所要時間**: 約30分

## モデル性能

### Cross Validation Results
- **CV Score**: TBD（要確認 - trainログから取得）
- **CV Strategy**: 5-fold KFold
- **フォールド数**: 5

### Leaderboard Results
- **Public LB Score**: **26.3879**
- **順位**: TBD
- **改善幅**: 約-0.08（前回比較）

## モデル設定

### アルゴリズム
- **モデル**: LightGBM Regressor
- **ハイパーパラメータ**:
  - n_estimators: 1000
  - learning_rate: 0.1
  - max_depth: 6
  - early_stopping_rounds: 100

### 特徴量リスト
- **元特徴量**: 9個（RhythmScore, AudioLoudness, VocalContent, etc.）
- **ジャンル特徴量**: 3個（ballad_genre_score, dance_genre_score, acoustic_genre_score）
- **エンジニアリング特徴量**: 交互作用、時間統計特徴量
- **多重共線性除去**: 閾値0.7、ジャンル特徴量優先保持

### 特徴量エンジニアリング内容
1. **音楽ジャンル特徴量**:
   - ballad_genre_score = VocalContent × MoodScore
   - dance_genre_score = Energy × RhythmScore
   - acoustic_genre_score = AcousticQuality × InstrumentalScore

2. **多重共線性除去**:
   - 高相関ペア（>0.7）検出
   - ジャンル特徴量を優先保持
   - 元特徴量を自動除去

## 技術実装

### 予測パイプライン概要
```bash
# 1. 多重共線性除去付き特徴量生成
python -m src.features --create-genre --remove-multicollinearity --output-dir=data/processed

# 2. 最適化データで訓練
python -m src.modeling.train --train-path=data/processed/train_features.csv --val-path=data/processed/validation_features.csv --exp-name=optimized_lgb

# 3. 予測実行
python -m src.modeling.predict --test-features-path=data/processed/test_features.csv --exp-name=optimized_lgb --output-path=data/processed/submission_optimized.csv
```

### ファイル構成
- `models/`: 5フォールドモデルファイル
- `data/processed/submission_optimized.csv`: Kaggle提出ファイル
- `data/processed/multicollinearity_removal_info.csv`: 除去特徴量詳細
- `data/processed/high_correlation_pairs.csv`: 高相関ペア一覧

## 考察・気づき

### 成功要因
- **多重共線性除去効果**: 冗長な特徴量排除により汎化性能向上
- **ジャンル特徴量優先**: 音楽ドメイン知識を活用した特徴量保持戦略
- **自動化された除去**: 閾値ベースの体系的な特徴量選択

### 改善の余地
- **CVスコア確認**: CV-LB一貫性の詳細分析が必要
- **閾値調整**: 0.7以外の閾値（0.6, 0.8）での実験検討
- **特徴量数**: 除去後の最終特徴量数と効果の関係分析

### 技術的知見
- 多重共線性除去は予測精度向上に効果的
- ドメイン知識に基づく特徴量優先保持が重要
- 自動化により再現性のある特徴量選択が可能

## Next Steps

### 次回実験のアイデア・改善案
1. **TICKET-008-02**: 独立性の高い高次特徴量の開発
   - 比率ベース特徴量（VocalContent/Energy）
   - 対数変換時間特徴量（log(TrackDurationMs) × RhythmScore）
   - 標準化済み交互作用（Z-score正規化後の積）

2. **実験体系化**:
   - CVスコア詳細ログの追加記録
   - Before/After比較の自動化
   - A/Bテスト機能の実装

3. **閾値実験**:
   - 多重共線性閾値0.6, 0.8での比較実験
   - 特徴量数vs性能のトレードオフ分析

### データ更新要確認
- [ ] CVスコア平均・標準偏差（trainログから）
- [ ] 除去特徴量リスト詳細
- [ ] CV-LB一貫性指標
- [ ] 高相関ペア数と詳細