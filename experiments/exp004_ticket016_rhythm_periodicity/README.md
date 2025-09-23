# Experiment 004: TICKET-016 ドラマー視点リズム周期性特徴量

## 実験概要
- **実験ID**: exp004
- **チケット**: TICKET-016
- **実験名**: ドラマー視点リズム周期性特徴量の実装と評価
- **実施日**: 2025-09-23
- **ブランチ**: feature/ticket-016/rhythm-periodicity
- **所要時間**: 約45分（予定）

## 背景と目的
### 課題認識
従来の数値的特徴量では捉えきれない「時間軸での一貫性」や「拍子パターン」がBPM予測において重要な要素となっている。

### 実験仮説
ドラマーの経験知に基づくリズム周期性特徴量により、スネア・キック・ハイハットの周期パターンがBPM決定の核心となる音楽的直感を数値化することで、予測精度が向上する。

### 期待効果
- 音楽理論に基づく高精度BPM予測
- 従来特徴量では見落とされがちな時間軸一貫性の導入
- 実用的音楽知識の機械学習への組み込み

## 実装内容

### 新特徴量群（5カテゴリ、19個の特徴量）

#### 1. リズムパターン推定特徴量
- **4/4拍子推定**: 一般的で安定したパターン（TODO: 実装予定）
- **3/4拍子推定**: ワルツ系の特徴（TODO: 実装予定）
- **シンコペーション検出**: 複雑なリズムパターン（TODO: 実装予定）

#### 2. 周期性一貫性スコア（TrackDurationとBPM推定の整合性検証）
- `tempo_duration_consistency`: BPM×楽曲長の理論的整合性

#### 3. 疑似ドラム系特徴量（キック・スネア・ハイハット推定密度）
- `pseudo_kick_density`: 低音域（低RhythmScore + 高Energy）
- `pseudo_snare_density`: 中音域（中RhythmScore + 中Energy）
- `pseudo_hihat_density`: 高音域（高RhythmScore + 低Energy）
- `drum_complexity`: ドラムセット全体の複雑性

#### 4. 拍子・テンポ変動推定（ルバート、加速、減速パターン検出）
- `rubato_likelihood`: 自由テンポ推定
- `accelerando_likelihood`: 加速パターン推定
- `ritardando_likelihood`: 減速パターン推定
- `tempo_stability`: テンポ安定性指標

#### 5. 周期性コヒーレンス指標（RhythmScore×Energy×時間整合性）
- `rhythm_energy_coherence`: 基本リズムコヒーレンス
- `temporal_coherence`: 時間軸でのコヒーレンス
- `overall_periodicity_quality`: 全体的な周期性品質指標

#### ボーナス: 楽曲構造推定特徴量
- `intro_section_likelihood`: イントロ推定
- `chorus_section_likelihood`: サビ推定
- `outro_section_likelihood`: アウトロ推定
- `song_structure_clarity`: 楽曲構造の明確性

## 技術実装

### 新関数の追加
```python
def create_rhythm_periodicity_features(df: pd.DataFrame) -> pd.DataFrame:
    """ドラマー視点のリズム周期性特徴量を作成する。"""
```

### 特徴量生成パイプライン
```bash
python -m src.features \
  --create-interactions \
  --create-duration \
  --create-statistical \
  --create-genre \
  --create-advanced \
  --create-rhythm \
  --remove-multicollinearity \
  --multicollinearity-threshold=0.7 \
  --prioritize-genre-features \
  --apply-scaling \
  --scaler-type=standard
```

### モデル訓練
```bash
python -m src.modeling.train \
  --train-features-path=data/processed/train_features.csv \
  --validation-features-path=data/processed/validation_features.csv \
  --exp-name=ticket016_rhythm_features \
  --n-estimators=1000 \
  --learning-rate=0.1 \
  --early-stopping-rounds=100
```

## 実験結果

### モデル性能
#### Cross Validation Results
- **CV Score**: TBD
- **CV Strategy**: 5-fold KFold
- **CV Standard Deviation**: TBD

#### Leaderboard Results
- **Public LB Score**: TBD
- **LB Improvement**: TBD (vs TICKET008-01: 26.39006)
- **Public LB Rank**: TBD

### 特徴量重要度分析
リズム周期性特徴量の重要度TOP5：
1. TBD
2. TBD
3. TBD
4. TBD
5. TBD

### 技術的成果
- **新特徴量数**: 19個
- **総特徴量数**: TBD（既存+新規）
- **訓練時間**: TBD分
- **多重共線性除去**: TBD個の特徴量除去

## 考察・気づき

### 成功要因
- TBD（実験完了後に記載）

### 改善の余地
- TBD（実験完了後に記載）

### 音楽理論的観点
- ドラマー視点の特徴量が従来の数値的特徴量を補完
- 時間軸一貫性の数値化による新しいアプローチ
- 楽曲構造推定の可能性を示唆

## Next Steps

### 短期的改善案
1. **リズムパターン推定の完成**: 4/4拍子・3/4拍子・シンコペーション検出ロジック
2. **個別特徴量効果分析**: 各リズム特徴量の個別寄与度測定
3. **音楽ジャンル別分析**: ジャンル特徴量とリズム特徴量の相互作用

### 中長期的発展
1. **TICKET-017: 高次リズム特徴量**: より複雑な拍子パターン（5/4、7/8拍子など）
2. **TICKET-018: ハーモニー系特徴量**: 和音進行・調性推定
3. **TICKET-019: メロディー系特徴量**: 音程・スケール推定

### 他の実験との組み合わせ
- **アンサンブル**: 従来特徴量（TICKET008）とリズム特徴量（TICKET016）の重み付き組み合わせ
- **特徴量選択**: リズム特徴量に特化した選択アルゴリズム開発

## ファイル構成
```
experiments/exp004_ticket016_rhythm_periodicity/
├── README.md                 # 本ファイル
├── execution_guide.md        # 実行手順書
├── config.json              # 実験設定（実行後作成）
├── results.json             # 実験結果（実行後作成）
├── submission.csv           # Kaggle提出ファイル（実行後作成）
├── feature_importance.csv   # 特徴量重要度（実行後作成）
├── commands.txt             # 実行コマンド履歴
└── models/                  # 訓練済みモデル（実行後作成）
```

## 実験履歴
- **作成日**: 2025-09-23
- **最終更新**: 2025-09-23
- **ステータス**: 実装完了、実行準備中

---

**注記**: この実験はドラマーの音楽的直感を機械学習に組み込む革新的なアプローチです。音楽理論とデータサイエンスの融合による新しい特徴量エンジニアリング手法として位置づけられます。