# TICKET-008-03 実行手順書

## Phase 1: 環境確認

### 1.1 ブランチ確認
```bash
git status
git branch --show-current
# feature/ticket-008-03/dimensionality-reduction であることを確認
```

### 1.2 実装確認
```bash
# 次元削減機能がsrc/features.pyに実装されていることを確認
findstr /n "create_dimensionality_reduction_features" src\features.py
findstr /n "determine_optimal_components" src\features.py
```

## Phase 2: 基本動作テスト

### 2.1 軽量テストデータでの動作確認
```python
# experiments/exp005_ticket008_03_dimensionality_reduction/ から実行
python -c "
import sys
sys.path.append('../../..')
from src.features import create_dimensionality_reduction_features
import pandas as pd
import numpy as np

# テスト用データ作成
test_data = pd.DataFrame({
    'RhythmScore': np.random.rand(100),
    'AudioLoudness': np.random.rand(100),
    'VocalContent': np.random.rand(100),
    'AcousticQuality': np.random.rand(100),
    'InstrumentalScore': np.random.rand(100),
    'LivePerformanceLikelihood': np.random.rand(100),
    'MoodScore': np.random.rand(100),
    'TrackDurationMs': np.random.randint(120000, 300000, 100),
    'Energy': np.random.rand(100),
    'dance_genre_score': np.random.rand(100),
    'acoustic_genre_score': np.random.rand(100),
    'ballad_genre_score': np.random.rand(100)
})

print('テストデータ:', test_data.shape)

# 次元削減実行
result = create_dimensionality_reduction_features(test_data)
print('処理完了:', result.shape)
print('PCA特徴量:', [col for col in result.columns if 'pca_' in col])
print('ICA特徴量:', [col for col in result.columns if 'ica_' in col])
"
```

### 2.2 CLIオプション確認
```bash
# ヘルプでオプション確認
python -m src.features --help | findstr /i dimensionality

# または、全ヘルプを表示して確認
python -m src.features --help
```

## Phase 3: パラメータ最適化テスト

### 3.1 PCA分散閾値の調整テスト
```python
# 異なる分散閾値での比較
python -c "
import sys
sys.path.append('../../..')
from src.features import create_dimensionality_reduction_features
import pandas as pd
import numpy as np

# テストデータ準備
test_data = pd.DataFrame({
    'RhythmScore': np.random.rand(500),
    'AudioLoudness': np.random.rand(500),
    'VocalContent': np.random.rand(500),
    'AcousticQuality': np.random.rand(500),
    'InstrumentalScore': np.random.rand(500),
    'LivePerformanceLikelihood': np.random.rand(500),
    'MoodScore': np.random.rand(500),
    'TrackDurationMs': np.random.randint(120000, 300000, 500),
    'Energy': np.random.rand(500),
    'dance_genre_score': np.random.rand(500),
    'acoustic_genre_score': np.random.rand(500),
    'ballad_genre_score': np.random.rand(500)
})

# 異なる閾値でテスト
for threshold in [0.85, 0.90, 0.95, 0.99]:
    result = create_dimensionality_reduction_features(
        test_data,
        pca_variance_threshold=threshold,
        apply_ica=False
    )
    pca_features = [col for col in result.columns if 'pca_' in col]
    print(f'閾値{threshold}: {len(pca_features)}個のPCA特徴量')
"
```

### 3.2 ICA成分数の調整テスト
```python
# 異なるICA成分数での比較
python -c "
import sys
sys.path.append('../../..')
from src.features import create_dimensionality_reduction_features
import pandas as pd
import numpy as np

# テストデータ準備（前回と同じ）
test_data = pd.DataFrame({
    'RhythmScore': np.random.rand(300),
    'AudioLoudness': np.random.rand(300),
    'VocalContent': np.random.rand(300),
    'AcousticQuality': np.random.rand(300),
    'InstrumentalScore': np.random.rand(300),
    'LivePerformanceLikelihood': np.random.rand(300),
    'MoodScore': np.random.rand(300),
    'TrackDurationMs': np.random.randint(120000, 300000, 300),
    'Energy': np.random.rand(300),
    'dance_genre_score': np.random.rand(300),
    'acoustic_genre_score': np.random.rand(300),
    'ballad_genre_score': np.random.rand(300)
})

# 異なるICA成分数でテスト
for ica_comp in [2, 3, 4, 5]:
    result = create_dimensionality_reduction_features(
        test_data,
        apply_pca=False,
        ica_components=ica_comp
    )
    ica_features = [col for col in result.columns if 'ica_' in col]
    print(f'ICA成分数{ica_comp}: {len(ica_features)}個のICA特徴量')
"
```

## Phase 4: 実データ軽量テスト

### 4.1 サンプリングベース高速テスト
```bash
# 元データの10%サンプルで実行（高速化）
python -c "
import pandas as pd
import numpy as np

# 訓練データを読み込んでサンプリング
train_df = pd.read_csv('data/processed/train.csv')
sample_size = min(1000, len(train_df) // 10)
sample_df = train_df.sample(n=sample_size, random_state=42)
sample_df.to_csv('data/processed/train_sample.csv', index=False)

# 検証・テストデータも同様にサンプリング
for dataset in ['validation', 'test']:
    try:
        df = pd.read_csv(f'data/processed/{dataset}.csv')
        sample_size = min(500, len(df) // 10)
        sample_df = df.sample(n=sample_size, random_state=42)
        sample_df.to_csv(f'data/processed/{dataset}_sample.csv', index=False)
        print(f'{dataset}サンプル作成完了: {sample_size}行')
    except:
        print(f'{dataset}データが見つかりません')
"
```

### 4.2 サンプルデータでの特徴量生成
```bash
# サンプルデータで次元削減特徴量生成（高速）
python -m src.features --train-path=data/processed/train_sample.csv --validation-path=data/processed/validation_sample.csv --test-path=data/processed/test_sample.csv --create-interactions --create-duration --create-statistical --create-genre --create-dimensionality-reduction --apply-pca --apply-ica --pca-variance-threshold=0.90 --ica-components=4 --remove-multicollinearity --multicollinearity-threshold=0.7 --prioritize-genre-features --apply-scaling --scaler-type=standard
```

## Phase 5: 性能評価（軽量版）

### 5.1 軽量クロスバリデーション
```bash
# サンプルデータでモデル訓練（3フォールド・軽量設定）
python -m src.modeling.train --train-path=data/processed/train_features_sample.csv --val-path=data/processed/validation_features_sample.csv --exp-name=ticket008_03_dimensionality_sample --use-cross-validation --n-folds=3 --n-estimators=100 --learning-rate=0.2 --early-stopping-rounds=20
```

### 5.2 特徴量重要度分析
```python
# 次元削減特徴量の重要度を確認
python -c "
import pandas as pd
import json

# 結果ファイルから特徴量重要度を確認
try:
    with open('models/ticket008_03_dimensionality_sample_cv_results*.json', 'r') as f:
        results = json.load(f)

    # 次元削減特徴量の重要度を抽出
    features = results.get('feature_names', [])
    pca_features = [f for f in features if 'pca_' in f]
    ica_features = [f for f in features if 'ica_' in f]

    print(f'PCA特徴量数: {len(pca_features)}')
    print(f'ICA特徴量数: {len(ica_features)}')
    print(f'CV RMSE: {results.get(\"mean_cv_score\", \"N/A\")}')
except Exception as e:
    print(f'結果ファイル読み込みエラー: {e}')
"
```

## Phase 6: 結果保存・分析

### 6.1 実験結果の保存
```bash
# 結果ファイルを実験ディレクトリにコピー
cp models/ticket008_03_dimensionality_sample_cv_results_*.json experiments/exp005_ticket008_03_dimensionality_reduction/
cp data/processed/feature_importance_all.csv experiments/exp005_ticket008_03_dimensionality_reduction/feature_importance.csv

# 実行コマンド履歴を保存
cat > experiments/exp005_ticket008_03_dimensionality_reduction/commands.txt << 'EOF'
# TICKET-008-03 実行コマンド履歴

# Phase 2: 基本動作テスト
python -c "テスト用データでの動作確認..."

# Phase 4: サンプルデータ特徴量生成
python -m src.features \
  --train-path=data/processed/train_sample.csv \
  --validation-path=data/processed/validation_sample.csv \
  --test-path=data/processed/test_sample.csv \
  --create-interactions \
  --create-duration \
  --create-statistical \
  --create-genre \
  --create-dimensionality-reduction \
  --apply-pca \
  --apply-ica \
  --pca-variance-threshold=0.90 \
  --ica-components=4 \
  --remove-multicollinearity \
  --multicollinearity-threshold=0.7 \
  --prioritize-genre-features \
  --apply-scaling \
  --scaler-type=standard

# Phase 5: 軽量モデル訓練
python -m src.modeling.train \
  --train-path=data/processed/train_features_sample.csv \
  --val-path=data/processed/validation_features_sample.csv \
  --exp-name=ticket008_03_dimensionality_sample \
  --use-cross-validation \
  --n-folds=3 \
  --n-estimators=100 \
  --learning-rate=0.2 \
  --early-stopping-rounds=20

# 実行日時: 2025-09-23
# 実行者: Claude Code
# ブランチ: feature/ticket-008-03/dimensionality-reduction
EOF
```

## Phase 7: 本格実行（オプション）

### 7.1 フルデータでの実行（時間要注意）
```bash
# 【注意】フルデータでの実行は長時間（10-30分）を要する可能性があります

# フルデータ特徴量生成
python -m src.features --create-interactions --create-duration --create-statistical --create-genre --create-dimensionality-reduction --apply-pca --apply-ica --pca-variance-threshold=0.90 --ica-components=4 --remove-multicollinearity --multicollinearity-threshold=0.7 --prioritize-genre-features --apply-scaling --scaler-type=standard

# フルデータモデル訓練
python -m src.modeling.train --train-path=data/processed/train_features.csv --val-path=data/processed/validation_features.csv --exp-name=ticket008_03_dimensionality_full --use-cross-validation --n-folds=5

# 結果保存
copy models\ticket008_03_dimensionality_full_cv_results_*.json experiments\exp005_ticket008_03_dimensionality_reduction\
```

## トラブルシューティング

### メモリ不足の場合
```bash
# データサイズを更に縮小
python -c "
import pandas as pd
train_df = pd.read_csv('data/processed/train.csv')
mini_sample = train_df.sample(n=500, random_state=42)
mini_sample.to_csv('data/processed/train_mini.csv', index=False)
print('ミニサンプル作成完了: 500行')
"
```

### 処理時間短縮
```bash
# PCAのみ実行（ICAをスキップ）
python -m src.features --create-dimensionality-reduction --apply-pca --no-apply-ica --pca-variance-threshold=0.85
```

### 実行時間の目安
- **Phase 2-3**: 1-2分
- **Phase 4-5**: 5-10分
- **Phase 7**: 15-45分（データサイズ依存）