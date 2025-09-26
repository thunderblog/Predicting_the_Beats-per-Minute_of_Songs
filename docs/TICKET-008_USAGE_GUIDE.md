# TICKET-008: 音楽ジャンル推定特徴量 使用ガイド 🔄 **リファクタリング対応版**

## 🎵 概要
音楽理論に基づく6つのジャンル推定特徴量を使用したBPM予測の実行手順

**⚠️ 重要**: src/features.py がリファクタリングされ、新しいモジュラー構造に変更されました。
- **新構造**: `src/features/` ディレクトリで機能分離
- **後方互換**: 既存の関数インターフェースを完全保持
- **新機能**: クラスベースとパイプライン管理機能追加

## 📋 実行コマンド

### 実行方法の選択

#### **方法A: 新しいCLI（推奨・簡単）**
```bash
# ジャンル特徴量のみ生成
python -m src.features --create-genre --output-dir=data/processed

# 他の特徴量を無効化してジャンル特徴量のみ
python -m src.features \
    --no-create-interactions \
    --no-create-statistical \
    --no-create-duration \
    --create-genre \
    --output-dir=data/processed
```

#### **方法B: 新しいクラスベースAPI（高度）**
```python
# 個別特徴量作成器の使用
from src.features import MusicGenreFeatureCreator
import pandas as pd

creator = MusicGenreFeatureCreator()
df = pd.read_csv('data/processed/train.csv')
result = creator.create_features(df)
print(f"Created features: {creator.created_features}")

# 統合パイプラインの使用
from src.features import create_feature_pipeline
pipeline = create_feature_pipeline()  # ジャンル特徴量含む
result = pipeline.execute(df)
summary = pipeline.get_execution_summary()
print(summary)
```

#### **方法C: カスタムパイプライン構築**
```python
from src.features import FeaturePipeline, MusicGenreFeatureCreator, StatisticalFeatureCreator

# カスタムパイプライン作成
pipeline = FeaturePipeline()
pipeline.add_creator(MusicGenreFeatureCreator())
pipeline.add_creator(StatisticalFeatureCreator())

# 実行
result = pipeline.execute(df)
summary = pipeline.get_execution_summary()
print(summary)

# 条件付き実行
result = pipeline.execute(df, creators_to_run=["MusicGenre"])
```

#### **方法D: 従来の関数インターフェース（後方互換）**
```python
# 既存コードはそのまま動作
from src.features import create_music_genre_features

df_with_genre = create_music_genre_features(df)
```
**生成される特徴量:**
- `dance_genre_score`: Energy × RhythmScore
- `acoustic_genre_score`: AcousticQuality × InstrumentalScore
- `ballad_genre_score`: VocalContent × MoodScore
- `rock_genre_score`: Energy × LivePerformanceLikelihood
- `electronic_genre_score`: (1-VocalContent) × Energy
- `ambient_genre_score`: (1-Energy) × AcousticQuality ⭐️ **最も効果的**

### 2. モデル訓練
```bash
python -m src.modeling.train \
    --train-path=data/processed/train_features.csv \
    --validation-path=data/processed/validation_features.csv \
    --experiment-name=genre_features_lgb
```

### 3. 予測実行
```bash
python -m src.modeling.predict \
    --test-path=data/processed/test_features.csv \
    --model-dir=models \
    --experiment-name=genre_features_lgb \
    --output-path=data/processed/submission_genre_features.csv
```

## 📊 評価・分析

### 性能比較（ベースライン vs 拡張版）
```bash
python scripts/evaluate_genre_features.py
```

### クイック評価（軽量版）
```bash
python scripts/quick_evaluation.py
```

## 🎯 実行パターン

### パターンA: 予測まで完全実行
```bash
# Step 1-3を順次実行
python -m src.features --create-genre --output-dir=data/processed
python -m src.modeling.train --train-path=data/processed/train_features.csv --validation-path=data/processed/validation_features.csv --experiment-name=genre_features_lgb
python -m src.modeling.predict --test-path=data/processed/test_features.csv --model-dir=models --experiment-name=genre_features_lgb --output-path=data/processed/submission_genre_features.csv
```

### パターンB: 評価のみ
```bash
# Step 1 + 評価
python -m src.features --create-genre --output-dir=data/processed
python scripts/evaluate_genre_features.py
```

## 📁 生成されるファイル
```
data/processed/
├── train_features.csv              # 拡張訓練データ（39特徴量）
├── validation_features.csv         # 拡張検証データ
├── test_features.csv              # 拡張テストデータ
├── submission_genre_features.csv   # Kaggle提出用予測結果
├── feature_importance_all.csv      # 全特徴量重要度
├── feature_importance_genre.csv    # ジャンル特徴量重要度
└── genre_bpm_analysis.csv         # ジャンル-BPM関係分析

models/
├── genre_features_lgb_fold_1_*.pkl    # 訓練済みモデル
├── genre_features_lgb_fold_2_*.pkl
└── genre_features_lgb_cv_results_*.json
```

## 🔍 確認コマンド
```bash
# 生成ファイル確認
ls data/processed/train_features.csv
ls data/processed/submission_genre_features.csv
ls models/*genre_features_lgb*

# 特徴量確認
python -c "
import pandas as pd
df = pd.read_csv('data/processed/train_features.csv')
genre_cols = [col for col in df.columns if 'genre_score' in col]
print('ジャンル特徴量:', genre_cols)
print('データ形状:', df.shape)
"
```

## ⏱️ 実行時間目安
- **特徴量生成**: 2-5分
- **モデル訓練**: 10-30分（42万サンプル）
- **予測実行**: 1-3分
- **評価**: 15-45分（全データ比較）

## 🎯 期待される改善
- **RMSE改善**: 0.1-0.5ポイント
- **統計的有意性**: ambient_genre_score (p<0.05)
- **音楽理論との整合性**: ジャンル別BPMパターンの捕捉

---
**📅 作成日**: 2025-09-17
**🎼 作成者**: TICKET-008 Implementation Team