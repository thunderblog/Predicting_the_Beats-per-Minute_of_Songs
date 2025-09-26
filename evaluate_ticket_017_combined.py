#!/usr/bin/env python3
"""
TICKET-017 組み合わせ評価スクリプト

TICKET-017-01（包括的交互作用）+ TICKET-017-02（対数変換）の
組み合わせ効果を評価します。
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import time

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

def load_sample_data():
    """サンプルデータを読み込む"""
    try:
        # 実データがある場合は使用
        data_path = Path("data/processed/train.csv")
        if data_path.exists():
            print(f"実データを読み込み: {data_path}")
            df = pd.read_csv(data_path)
            # 大きすぎる場合はサンプリング
            if len(df) > 5000:
                df = df.sample(n=5000, random_state=42)
                print(f"サンプリング実行: {len(df)}サンプル")
            return df
    except Exception as e:
        print(f"実データ読み込みエラー: {e}")

    # フォールバック: 合成データ作成
    print("合成データを作成中...")
    np.random.seed(42)
    n_samples = 1000

    data = {
        'id': range(n_samples),
        'RhythmScore': np.random.uniform(0.1, 1.0, n_samples),
        'AudioLoudness': np.random.uniform(0.1, 1.0, n_samples),
        'VocalContent': np.random.uniform(0.0, 1.0, n_samples),
        'AcousticQuality': np.random.uniform(0.1, 1.0, n_samples),
        'InstrumentalScore': np.random.uniform(0.0, 1.0, n_samples),
        'LivePerformanceLikelihood': np.random.uniform(0.0, 1.0, n_samples),
        'MoodScore': np.random.uniform(0.1, 1.0, n_samples),
        'TrackDurationMs': np.random.uniform(120000, 360000, n_samples),  # 2-6分
        'Energy': np.random.uniform(0.1, 1.0, n_samples),
    }

    # BPMをある程度現実的に生成（特徴量から予測可能な関係を作る）
    rhythm_factor = data['RhythmScore'] * data['Energy']
    duration_factor = np.log1p(data['TrackDurationMs']) / 20
    mood_factor = data['MoodScore'] * data['VocalContent']

    # 基本BPM（60-200の範囲）
    base_bpm = 80 + rhythm_factor * 60 + duration_factor * 20 + mood_factor * 40
    # ノイズ追加
    noise = np.random.normal(0, 5, n_samples)
    data['BeatsPerMinute'] = np.clip(base_bpm + noise, 60, 200)

    df = pd.DataFrame(data)
    print(f"合成データ作成完了: {len(df)}サンプル, BPM範囲: {df['BeatsPerMinute'].min():.1f}-{df['BeatsPerMinute'].max():.1f}")

    return df

def evaluate_features(df, feature_creation_func, name):
    """特徴量作成関数を評価する"""
    print(f"\n=== {name} 評価開始 ===")

    # 特徴量作成
    start_time = time.time()
    try:
        enhanced_df = feature_creation_func(df)
        creation_time = time.time() - start_time

        print(f"特徴量作成時間: {creation_time:.2f}秒")
        print(f"元特徴量数: {len(df.columns)} -> 拡張後: {len(enhanced_df.columns)} (+{len(enhanced_df.columns) - len(df.columns)})")

        # データ品質チェック
        nan_count = enhanced_df.isnull().sum().sum()
        inf_count = np.isinf(enhanced_df.select_dtypes(include=[np.number])).sum().sum()

        if nan_count > 0 or inf_count > 0:
            print(f"警告: NaN({nan_count}), inf({inf_count})値が検出されました")
            enhanced_df = enhanced_df.fillna(0)
            enhanced_df = enhanced_df.replace([np.inf, -np.inf], 0)

        # 特徴量とターゲットを分離
        feature_cols = [col for col in enhanced_df.columns if col not in ['id', 'BeatsPerMinute']]
        X = enhanced_df[feature_cols]
        y = enhanced_df['BeatsPerMinute']

        print(f"評価用特徴量数: {len(feature_cols)}")

        # クロスバリデーション評価
        print("クロスバリデーション実行中...")
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)  # 軽量化のため3分割

        # LightGBMモデル
        model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=100,  # 軽量化
            verbose=-1,
            random_state=42,
            force_col_wise=True
        )

        cv_start = time.time()
        cv_scores = cross_val_score(
            model, X, y,
            cv=kfold,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        cv_time = time.time() - cv_start

        mean_rmse = -cv_scores.mean()
        std_rmse = cv_scores.std()

        print(f"CV RMSE: {mean_rmse:.4f} (±{std_rmse:.4f})")
        print(f"CV評価時間: {cv_time:.2f}秒")

        return {
            'name': name,
            'features_count': len(feature_cols),
            'creation_time': creation_time,
            'cv_rmse': mean_rmse,
            'cv_std': std_rmse,
            'cv_time': cv_time,
            'data_quality': {'nan': nan_count, 'inf': inf_count}
        }

    except Exception as e:
        print(f"エラー: {type(e).__name__}: {e}")
        return {
            'name': name,
            'error': str(e)
        }

def main():
    """メイン評価処理"""
    print("TICKET-017 組み合わせ評価開始")
    print("=" * 50)

    # データ読み込み
    df = load_sample_data()

    results = []

    # 1. ベースライン（元特徴量のみ）
    def baseline(df):
        feature_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
                       'TrackDurationMs', 'Energy']
        return df[['id', 'BeatsPerMinute'] + feature_cols].copy()

    result = evaluate_features(df, baseline, "ベースライン（元特徴量のみ）")
    results.append(result)

    # 2. TICKET-017-01のみ
    try:
        from src.features import create_comprehensive_interaction_features

        def ticket_017_01_only(df):
            return create_comprehensive_interaction_features(df)

        result = evaluate_features(df, ticket_017_01_only, "TICKET-017-01（包括的交互作用のみ）")
        results.append(result)
    except Exception as e:
        print(f"TICKET-017-01評価スキップ: {e}")

    # 3. TICKET-017-02のみ
    try:
        from src.features import create_log_features

        def ticket_017_02_only(df):
            return create_log_features(df)

        result = evaluate_features(df, ticket_017_02_only, "TICKET-017-02（対数変換のみ）")
        results.append(result)
    except Exception as e:
        print(f"TICKET-017-02評価スキップ: {e}")

    # 4. TICKET-017-01 + 017-02 組み合わせ
    try:
        def ticket_017_combined(df):
            # Step 1: 包括的交互作用特徴量
            enhanced_df = create_comprehensive_interaction_features(df)
            # Step 2: 対数変換特徴量
            enhanced_df = create_log_features(enhanced_df)
            return enhanced_df

        result = evaluate_features(df, ticket_017_combined, "TICKET-017 組み合わせ（017-01 + 017-02）")
        results.append(result)
    except Exception as e:
        print(f"TICKET-017組み合わせ評価スキップ: {e}")

    # 結果サマリー
    print("\n" + "=" * 60)
    print("評価結果サマリー")
    print("=" * 60)

    valid_results = [r for r in results if 'error' not in r]

    if valid_results:
        # 結果テーブル
        print(f"{'手法':<30} {'特徴量数':<8} {'RMSE':<10} {'作成時間':<8} {'CV時間':<8}")
        print("-" * 70)

        for result in valid_results:
            name = result['name'][:28]
            features = result['features_count']
            rmse = f"{result['cv_rmse']:.4f}"
            creation = f"{result['creation_time']:.1f}s"
            cv_time = f"{result['cv_time']:.1f}s"

            print(f"{name:<30} {features:<8} {rmse:<10} {creation:<8} {cv_time:<8}")

        # 改善効果分析
        if len(valid_results) > 1:
            baseline_rmse = valid_results[0]['cv_rmse']
            print(f"\nベースラインからの改善:")

            for result in valid_results[1:]:
                improvement = baseline_rmse - result['cv_rmse']
                improvement_pct = (improvement / baseline_rmse) * 100

                print(f"  {result['name']}: {improvement:+.4f} ({improvement_pct:+.2f}%)")

        # 最良結果
        best_result = min(valid_results, key=lambda x: x['cv_rmse'])
        print(f"\n最良結果: {best_result['name']}")
        print(f"   RMSE: {best_result['cv_rmse']:.4f}")
        print(f"   特徴量数: {best_result['features_count']}")

    else:
        print("有効な評価結果がありません")

    print("\n組み合わせ評価完了")

if __name__ == "__main__":
    main()