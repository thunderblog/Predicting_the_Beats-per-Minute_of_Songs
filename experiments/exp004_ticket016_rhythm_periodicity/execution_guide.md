# TICKET-016 実行手順書

## 📋 実行概要
- **実験ID**: exp004
- **実装内容**: 5つのリズム周期性特徴量群（19個の新特徴量）
- **期待効果**: ドラマー直感に基づくBPM予測精度向上
- **実行時間**: 約30-45分（テスト込み）

---

## 🔄 Phase 1: 機能完成とメインモジュール統合

### Step 1.1: リズムパターン推定特徴量の完成
**現在のステータス**: TODO(human)部分の実装が必要

**必要な作業**:
```python
# src/features.py の TODO(human) 部分を実装
# 1. リズムパターン推定特徴量（4/4拍子、3/4拍子、シンコペーション検出）
logger.info("リズムパターン推定特徴量を作成中...")

# 4/4拍子推定: 一般的で安定したパターン
# RhythmScore×Energyが中程度で安定 = 4/4拍子的
df_features["beat_4_4_likelihood"] = ...

# 3/4拍子推定: ワルツ系の特徴
# 高AcousticQuality×中RhythmScore = 3/4拍子的
df_features["beat_3_4_likelihood"] = ...

# シンコペーション検出: 複雑なリズムパターン
# 高RhythmScore×高Energy×不規則性 = シンコペーション
df_features["syncopation_likelihood"] = ...
```

### Step 1.2: メインモジュールに統合
```python
# src/features.py のmain関数を更新
# create_rhythm引数を追加し、create_rhythm_periodicity_features関数を呼び出し
```

---

## 🧪 Phase 2: 機能テストと動作検証

### Step 2.1: 基本動作テスト
```bash
# 作業ディレクトリ: experiments/exp004_ticket016_rhythm_periodicity/

# 小さなサンプルデータでリズム特徴量生成テスト
python -c "
import sys
sys.path.append('../../..')
import pandas as pd
import numpy as np
from src.features import create_rhythm_periodicity_features

# テストデータ作成
test_df = pd.DataFrame({
    'RhythmScore': [0.7, 0.8, 0.6],
    'Energy': [0.8, 0.6, 0.9],
    'TrackDurationMs': [200000, 180000, 220000],
    'AudioLoudness': [0.6, 0.7, 0.5],
    'InstrumentalScore': [0.7, 0.5, 0.8],
    'LivePerformanceLikelihood': [0.4, 0.6, 0.5],
    'MoodScore': [0.6, 0.7, 0.8],
    'VocalContent': [0.8, 0.6, 0.9],
    'AcousticQuality': [0.5, 0.8, 0.7]
})

print('=== TICKET-016 基本動作テスト ===')
print(f'入力データ形状: {test_df.shape}')

# リズム特徴量生成
try:
    result = create_rhythm_periodicity_features(test_df)
    new_features_count = len(result.columns) - len(test_df.columns)
    print(f'✓ 成功: {new_features_count}個の新特徴量を生成')

    # 生成された特徴量リスト
    new_features = [col for col in result.columns if col not in test_df.columns]
    print('\\n生成された特徴量:')
    for i, feature in enumerate(new_features, 1):
        print(f'  {i:2d}. {feature}')

    print(f'\\n期待値: 19個, 実際: {new_features_count}個')
    if new_features_count == 19:
        print('✓ 特徴量数テスト: 合格')
    else:
        print('⚠ 特徴量数テスト: 確認が必要')

except Exception as e:
    print(f'✗ エラー: {type(e).__name__}: {e}')
    import traceback
    print(traceback.format_exc())
"
```

### Step 2.2: エラー検証
```bash
# NaN値、無限値チェック
python -c "
import sys
sys.path.append('../../..')
import pandas as pd
import numpy as np
from src.features import create_rhythm_periodicity_features

print('=== TICKET-016 エラー検証テスト ===')

# 極端値テストデータ
extreme_df = pd.DataFrame({
    'RhythmScore': [0.0, 1.0, 0.5],
    'Energy': [0.0, 1.0, 0.5],
    'TrackDurationMs': [30000, 600000, 200000],  # 0.5分〜10分
    'AudioLoudness': [0.0, 1.0, 0.5],
    'InstrumentalScore': [0.0, 1.0, 0.5],
    'LivePerformanceLikelihood': [0.0, 1.0, 0.5],
    'MoodScore': [0.0, 1.0, 0.5],
    'VocalContent': [0.0, 1.0, 0.5],
    'AcousticQuality': [0.0, 1.0, 0.5]
})

print('極端値テストデータ:')
print(extreme_df)

try:
    result = create_rhythm_periodicity_features(extreme_df)

    # エラー値チェック
    nan_count = result.isnull().sum().sum()
    inf_count = np.isinf(result.select_dtypes(include=[np.number])).sum().sum()

    print(f'\\nエラー値検査結果:')
    print(f'  NaN値: {nan_count}個')
    print(f'  無限値: {inf_count}個')

    if nan_count == 0 and inf_count == 0:
        print('✓ エラー値テスト: 合格')
    else:
        print('⚠ エラー値テスト: 問題あり')

    # 値域チェック（0-1の範囲を期待）
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[-10:]:  # 最後の10個の特徴量をチェック
        values = result[col]
        print(f'  {col}: [{values.min():.3f}, {values.max():.3f}]')

except Exception as e:
    print(f'✗ エラー: {type(e).__name__}: {e}')
"
```

---

## 🎵 Phase 3: 実データでの特徴量生成

### Step 3.1: 既存データセットで特徴量生成
```bash
cd ../../../  # プロジェクトルートに戻る

# リズム特徴量を含む拡張特徴量セット生成
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

# コマンド履歴を保存
echo "python -m src.features --create-interactions --create-duration --create-statistical --create-genre --create-advanced --create-rhythm --remove-multicollinearity --multicollinearity-threshold=0.7 --prioritize-genre-features --apply-scaling --scaler-type=standard" > experiments/exp004_ticket016_rhythm_periodicity/commands.txt
```

### Step 3.2: 特徴量情報確認
```bash
# 生成された特徴量リストと統計情報確認
python -c "
import pandas as pd

print('=== TICKET-016 特徴量生成結果 ===')

# 特徴量情報読み込み
try:
    feature_info = pd.read_csv('data/processed/feature_info.csv')
    print(f'総特徴量数: {len(feature_info)}')
except FileNotFoundError:
    print('⚠ feature_info.csvが見つかりません')

# リズム特徴量フィルタ
try:
    train_features = pd.read_csv('data/processed/train_features.csv')
    all_features = train_features.columns.tolist()

    rhythm_keywords = [
        'tempo_duration', 'pseudo_', 'drum_', 'rubato', 'accelerando', 'ritardando',
        'tempo_stability', 'rhythm_energy_coherence', 'temporal_coherence',
        'periodicity_quality', 'section_likelihood', 'structure_clarity',
        'beat_4_4', 'beat_3_4', 'syncopation'
    ]

    rhythm_features = [f for f in all_features if any(keyword in f for keyword in rhythm_keywords)]

    print(f'\\nリズム周期性特徴量: {len(rhythm_features)}個')
    print('生成されたリズム特徴量:')
    for i, feature in enumerate(sorted(rhythm_features), 1):
        print(f'  {i:2d}. {feature}')

    # データセット形状確認
    print(f'\\nデータセット形状:')
    print(f'  train_features.csv: {train_features.shape}')

    try:
        test_features = pd.read_csv('data/processed/test_features.csv')
        print(f'  test_features.csv: {test_features.shape}')
    except FileNotFoundError:
        print('  test_features.csv: 見つかりません')

except FileNotFoundError:
    print('⚠ train_features.csvが見つかりません')
    print('特徴量生成が完了していない可能性があります')
"
```

---

## 🏃 Phase 4: モデル訓練と性能評価

### Step 4.1: リズム特徴量を含むモデル訓練
```bash
# 新特徴量でLightGBM訓練
python -m src.modeling.train \
  --train-features-path=data/processed/train_features.csv \
  --validation-features-path=data/processed/validation_features.csv \
  --exp-name=ticket016_rhythm_features \
  --n-estimators=1000 \
  --learning-rate=0.1 \
  --early-stopping-rounds=100

# コマンド履歴に追加
echo "python -m src.modeling.train --train-features-path=data/processed/train_features.csv --validation-features-path=data/processed/validation_features.csv --exp-name=ticket016_rhythm_features --n-estimators=1000 --learning-rate=0.1 --early-stopping-rounds=100" >> experiments/exp004_ticket016_rhythm_periodicity/commands.txt
```

### Step 4.2: 予測とサブミッション生成
```bash
# テストデータで予測実行
python -m src.modeling.predict \
  --test-features-path=data/processed/test_features.csv \
  --exp-name=ticket016_rhythm_features \
  --output-path=data/processed/submission_ticket016_rhythm.csv

# コマンド履歴に追加
echo "python -m src.modeling.predict --test-features-path=data/processed/test_features.csv --exp-name=ticket016_rhythm_features --output-path=data/processed/submission_ticket016_rhythm.csv" >> experiments/exp004_ticket016_rhythm_periodicity/commands.txt

# サブミッションファイルを実験フォルダにコピー
cp data/processed/submission_ticket016_rhythm.csv experiments/exp004_ticket016_rhythm_periodicity/submission.csv
```

---

## 📊 Phase 5: 結果分析と比較

### Step 5.1: 特徴量重要度分析
```bash
# リズム特徴量の重要度確認
python -c "
import pandas as pd

print('=== TICKET-016 特徴量重要度分析 ===')

try:
    # 特徴量重要度読み込み
    importance_df = pd.read_csv('data/processed/feature_importance_all.csv')

    # リズム特徴量のみフィルタ
    rhythm_pattern = '|'.join([
        'tempo_duration', 'pseudo_', 'drum_', 'rubato', 'accelerando', 'ritardando',
        'tempo_stability', 'rhythm_energy_coherence', 'temporal_coherence',
        'periodicity_quality', 'section_likelihood', 'structure_clarity',
        'beat_4_4', 'beat_3_4', 'syncopation'
    ])

    rhythm_importance = importance_df[importance_df['feature_name'].str.contains(rhythm_pattern)]

    print('リズム周期性特徴量 重要度TOP10:')
    if len(rhythm_importance) > 0:
        rhythm_top10 = rhythm_importance.nlargest(10, 'average_importance')
        for i, (_, row) in enumerate(rhythm_top10.iterrows(), 1):
            print(f'  {i:2d}. {row[\"feature_name\"]}: {row[\"average_importance\"]:.4f}')
    else:
        print('  リズム特徴量が見つかりませんでした')

    # 全特徴量での順位確認
    print('\\n全特徴量でのリズム特徴量の位置:')
    all_sorted = importance_df.sort_values('average_importance', ascending=False)
    for i, (_, row) in enumerate(all_sorted.iterrows(), 1):
        if any(keyword in row['feature_name'] for keyword in rhythm_pattern.split('|')):
            print(f'  {i:3d}位: {row[\"feature_name\"]} ({row[\"average_importance\"]:.4f})')
            if i <= 20:  # TOP20以内のリズム特徴量をマーク
                print('       ★ TOP20入り')

except FileNotFoundError:
    print('⚠ feature_importance_all.csvが見つかりません')
    print('特徴量重要度分析がまだ完了していない可能性があります')
"

# 特徴量重要度を実験フォルダにコピー
cp data/processed/feature_importance_all.csv experiments/exp004_ticket016_rhythm_periodicity/feature_importance.csv
```

### Step 5.2: 性能比較分析
```bash
# 前実験（TICKET008-01）との性能比較
python -c "
import pandas as pd

print('=== TICKET-016 性能比較分析 ===')

# 実験結果読み込み
try:
    results_df = pd.read_csv('experiments/experiment_results.csv')

    # 最新の実験結果表示
    print('最近の実験結果:')
    latest_experiments = results_df.tail(3)
    for _, row in latest_experiments.iterrows():
        lb_score = row['lb_score']
        if pd.notna(lb_score) and lb_score != 'TBD':
            print(f'  {row[\"exp_name\"]}: LB {float(lb_score):.5f}')
        else:
            print(f'  {row[\"exp_name\"]}: LB {lb_score}')

    # ベースライン比較
    baseline_experiments = results_df[results_df['exp_name'].str.contains('baseline|exp02')]
    if len(baseline_experiments) > 0:
        print('\\nベースライン実験:')
        for _, row in baseline_experiments.iterrows():
            lb_score = row['lb_score']
            if pd.notna(lb_score) and lb_score != 'TBD':
                print(f'  {row[\"exp_name\"]}: LB {float(lb_score):.5f}')

    print('\\n次回追加予定: exp004 (ticket016_rhythm_features)')
    print('※ Kaggle提出後にLB結果を記録してください')

except FileNotFoundError:
    print('⚠ experiments/experiment_results.csvが見つかりません')
"
```

---

## 🎯 Phase 6: 実験結果記録

### Step 6.1: experiment_results.csv更新
```bash
# LB結果が判明したら以下を実行
# 例: LB 26.38500の場合

python -c "
import pandas as pd
from datetime import datetime

print('=== TICKET-016 実験結果記録 ===')

# ユーザーにLB結果を入力してもらう
print('Kaggle Leaderboard結果を入力してください:')
lb_score = input('LB Score: ')

try:
    lb_score_float = float(lb_score)

    # CSV読み込み
    df = pd.read_csv('experiments/experiment_results.csv')

    # 前回のLB結果取得（改善計算用）
    last_lb = df[df['lb_score'] != 'TBD']['lb_score'].iloc[-1]
    improvement_from_previous = lb_score_float - float(last_lb)

    # 新しい実験行を追加
    new_row = {
        'exp_id': 'exp004',
        'exp_name': 'ticket016_rhythm_features',
        'description': 'ドラマー視点リズム周期性特徴量',
        'date': '2025-09-23',
        'cv_score': 'TBD',
        'cv_std': 'TBD',
        'lb_score': lb_score_float,
        'cv_lb_diff': 'TBD',
        'improvement_from_baseline': 'TBD',
        'improvement_from_previous': f'{improvement_from_previous:+.5f}',
        'model_type': 'LightGBM',
        'n_features': 'TBD',
        'n_samples': 419331,
        'cv_folds': 5,
        'training_time_min': 'TBD',
        'feature_engineering': 'リズム周期性+ジャンル+多重共線性除去',
        'hyperparameters': '{\"n_estimators\": 1000, \"learning_rate\": 0.1, \"early_stopping\": 100}',
        'preprocessing': 'ドラマー視点特徴量追加',
        'ensemble_method': '5-fold平均',
        'status': 'completed',
        'submission_file': 'submission_ticket016_rhythm.csv',
        'notes': '音楽理論ベース革新的アプローチ'
    }

    # 行を追加してCSV保存
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv('experiments/experiment_results.csv', index=False)

    print(f'✓ 実験結果を記録しました: LB {lb_score_float:.5f}')
    print(f'  前回からの改善: {improvement_from_previous:+.5f}')

except ValueError:
    print('⚠ 無効なLB Scoreです。数値を入力してください。')
except Exception as e:
    print(f'✗ エラー: {e}')
"
```

### Step 6.2: 実験設定ファイル作成
```bash
cd experiments/exp004_ticket016_rhythm_periodicity/

# config.json作成
python -c "
import json
from datetime import datetime

config = {
    'experiment_name': 'exp004_ticket016_rhythm_features',
    'description': 'ドラマー視点のリズム周期性特徴量による BPM予測精度向上実験',
    'date_created': '2025-09-23',
    'ticket_number': 'TICKET-016',
    'branch': 'feature/ticket-016/rhythm-periodicity',
    'model_config': {
        'model_type': 'LightGBM',
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'early_stopping_rounds': 100,
        'objective': 'regression',
        'metric': 'rmse'
    },
    'data_config': {
        'train_samples': 419331,
        'test_samples': 'TBD',
        'n_features': 'TBD',
        'cv_folds': 5
    },
    'features': {
        'original_features': [
            'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
            'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
            'TrackDurationMs', 'Energy'
        ],
        'engineered_features': [
            'interaction_features', 'duration_features', 'statistical_features',
            'genre_features', 'advanced_features', 'rhythm_periodicity_features'
        ],
        'new_rhythm_features': [
            'tempo_duration_consistency', 'pseudo_kick_density', 'pseudo_snare_density',
            'pseudo_hihat_density', 'drum_complexity', 'rubato_likelihood',
            'accelerando_likelihood', 'ritardando_likelihood', 'tempo_stability',
            'rhythm_energy_coherence', 'temporal_coherence', 'overall_periodicity_quality',
            'intro_section_likelihood', 'chorus_section_likelihood', 'outro_section_likelihood',
            'song_structure_clarity'
        ],
        'feature_selection': '多重共線性除去（閾値0.7）',
        'scaling': 'StandardScaler'
    },
    'preprocessing': {
        'missing_values': 'なし',
        'outlier_handling': 'なし',
        'feature_engineering': 'ドラマー視点リズム周期性特徴量19個追加',
        'multicollinearity_removal': True,
        'multicollinearity_threshold': 0.7
    },
    'innovation': {
        'approach': 'ドラマー視点の音楽的直感を数値化',
        'key_concepts': [
            '時間軸一貫性の導入',
            '疑似ドラム系特徴量による周期性捕捉',
            '楽曲構造推定による全体的理解',
            'テンポ変動パターンの数値化'
        ],
        'musical_theory_basis': 'スネア・キック・ハイハットの周期パターン分析'
    }
}

with open('config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print('✓ config.jsonを作成しました')
"

# results.jsonのテンプレート作成
python -c "
import json

results_template = {
    'experiment_name': 'exp004_ticket016_rhythm_features',
    'timestamp': '2025-09-23T00:00:00',
    'cross_validation': {
        'cv_strategy': 'KFold',
        'n_folds': 5,
        'mean_rmse': 'TBD',
        'rmse_std': 'TBD',
        'fold_results': {
            'fold_1': {'rmse': 'TBD', 'model_file': 'ticket016_rhythm_features_fold_1_*.pkl'},
            'fold_2': {'rmse': 'TBD', 'model_file': 'ticket016_rhythm_features_fold_2_*.pkl'},
            'fold_3': {'rmse': 'TBD', 'model_file': 'ticket016_rhythm_features_fold_3_*.pkl'},
            'fold_4': {'rmse': 'TBD', 'model_file': 'ticket016_rhythm_features_fold_4_*.pkl'},
            'fold_5': {'rmse': 'TBD', 'model_file': 'ticket016_rhythm_features_fold_5_*.pkl'}
        }
    },
    'leaderboard_results': {
        'submission_date': '2025-09-23',
        'public_lb_score': 'TBD',
        'public_lb_rank': 'TBD',
        'private_lb_score': 'TBD',
        'private_lb_rank': 'TBD'
    },
    'feature_analysis': {
        'total_features': 'TBD',
        'rhythm_features_count': 19,
        'top_rhythm_features': 'TBD',
        'rhythm_features_in_top20': 'TBD'
    },
    'performance_metrics': {
        'cv_vs_lb_consistency': 'TBD',
        'improvement_from_previous': 'TBD',
        'overfitting_indicator': 'TBD'
    },
    'notes': [
        'ドラマー視点の音楽理論を機械学習に初導入',
        'リズム周期性特徴量19個を新規開発',
        '時間軸一貫性という新しい観点を数値化',
        '従来の数値的特徴量では捉えきれない音楽的直感を実装'
    ]
}

with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results_template, f, indent=2, ensure_ascii=False)

print('✓ results.jsonテンプレートを作成しました')
"
```

---

## ✅ 実行チェックリスト

### Phase 1: 機能完成
- [ ] リズムパターン推定特徴量の完成（TODO(human)部分）
- [ ] メインモジュールへの統合

### Phase 2: テストと検証
- [ ] 基本動作テスト実行
- [ ] エラー検証（NaN/無限値チェック）

### Phase 3: 実データ処理
- [ ] 特徴量生成実行
- [ ] 特徴量情報確認

### Phase 4: モデル訓練
- [ ] LightGBMモデル訓練
- [ ] 予測・サブミッション生成

### Phase 5: 分析
- [ ] 特徴量重要度分析
- [ ] 性能比較分析

### Phase 6: 記録
- [ ] experiment_results.csv更新
- [ ] 実験設定ファイル作成

---

## 🎵 期待される成果

### 技術的成果
- **新特徴量**: 19個のリズム周期性特徴量
- **革新性**: 音楽理論とデータサイエンスの融合
- **実用性**: ドラマー直感の機械学習への応用

### 予測性能
- **期待改善**: 時間軸一貫性による予測精度向上
- **特徴量寄与**: リズム特徴量のTOP20入り
- **モデル解釈性**: 音楽的に解釈可能な特徴量

### 将来展開
- **TICKET-017**: 高次リズム特徴量（複雑な拍子）
- **TICKET-018**: ハーモニー系特徴量
- **TICKET-019**: メロディー系特徴量

この手順書に従って、TICKET-016の革新的なリズム周期性特徴量を段階的に実装・評価してください。