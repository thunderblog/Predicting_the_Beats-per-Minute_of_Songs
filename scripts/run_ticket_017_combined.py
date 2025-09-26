#!/usr/bin/env python3
"""
TICKET-017組み合わせ実行スクリプト

TICKET-017-01（包括的交互作用）+ TICKET-017-02（対数変換）の
組み合わせ特徴量を生成し、モデル訓練まで実行する。
"""

import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
import lightgbm as lgb
import time

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.features import create_comprehensive_interaction_features, create_log_features

def run_ticket_017_combined():
    """TICKET-017組み合わせの完全実行"""
    print("TICKET-017組み合わせ実行開始")
    print("=" * 50)

    # データ読み込み
    data_path = Path("data/processed/train.csv")
    if not data_path.exists():
        print(f"エラー: {data_path} が見つかりません")
        return False

    print(f"データ読み込み: {data_path}")
    df = pd.read_csv(data_path)
    print(f"元データ: {df.shape}")

    # Step 1: TICKET-017-01（包括的交互作用特徴量）
    print("\nStep 1: TICKET-017-01 包括的交互作用特徴量生成")
    start_time = time.time()
    df_step1 = create_comprehensive_interaction_features(df)
    step1_time = time.time() - start_time
    print(f"TICKET-017-01完了: {df.shape} -> {df_step1.shape} (+{len(df_step1.columns) - len(df.columns)})")
    print(f"処理時間: {step1_time:.2f}秒")

    # Step 2: TICKET-017-02（対数変換特徴量）
    print("\nStep 2: TICKET-017-02 対数変換特徴量生成")
    start_time = time.time()
    df_combined = create_log_features(df_step1)
    step2_time = time.time() - start_time
    print(f"TICKET-017-02完了: {df_step1.shape} -> {df_combined.shape} (+{len(df_combined.columns) - len(df_step1.columns)})")
    print(f"処理時間: {step2_time:.2f}秒")

    # 最終結果
    total_new_features = len(df_combined.columns) - len(df.columns)
    print(f"\n最終結果: 元{len(df.columns)}特徴量 -> {len(df_combined.columns)}特徴量 (+{total_new_features})")

    # 特徴量とターゲットを分離
    feature_cols = [col for col in df_combined.columns if col not in ['id', 'BeatsPerMinute']]
    X = df_combined[feature_cols]
    y = df_combined['BeatsPerMinute']

    print(f"評価用特徴量数: {len(feature_cols)}")

    # クロスバリデーション評価
    print("\nStep 3: クロスバリデーション評価")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.1,
        n_estimators=100,
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

    # 結果保存
    print("\nStep 4: 結果保存")
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "train_ticket017_combined.csv"
    df_combined.to_csv(output_path, index=False)
    print(f"組み合わせ特徴量を保存: {output_path}")

    # サマリー出力
    print("\n" + "=" * 50)
    print("TICKET-017組み合わせ実行完了")
    print("=" * 50)
    print(f"最終特徴量数: {len(feature_cols)}")
    print(f"CV RMSE: {mean_rmse:.4f}")
    print(f"総処理時間: {step1_time + step2_time + cv_time:.2f}秒")
    print(f"保存先: {output_path}")

    return True

def main():
    """メイン実行"""
    success = run_ticket_017_combined()
    if success:
        print("\n✅ TICKET-017組み合わせ実行成功")
    else:
        print("\n❌ TICKET-017組み合わせ実行失敗")
        sys.exit(1)

if __name__ == "__main__":
    main()