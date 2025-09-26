#!/usr/bin/env python3
"""
TICKET-017組み合わせ Kaggle提出スクリプト

TICKET-017-01（包括的交互作用）+ TICKET-017-02（対数変換）の
組み合わせ特徴量でモデル訓練・予測・提出まで実行する。
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import lightgbm as lgb
import time

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.features import create_comprehensive_interaction_features, create_log_features

def create_ticket_017_features(df):
    """TICKET-017組み合わせ特徴量を作成"""
    print("TICKET-017組み合わせ特徴量生成中...")

    # Step 1: TICKET-017-01（包括的交互作用特徴量）
    print("  Step 1: 包括的交互作用特徴量")
    df_step1 = create_comprehensive_interaction_features(df)
    step1_features = len(df_step1.columns) - len(df.columns)
    print(f"    +{step1_features}特徴量")

    # Step 2: TICKET-017-02（対数変換特徴量）
    print("  Step 2: 対数変換特徴量")
    df_combined = create_log_features(df_step1)
    step2_features = len(df_combined.columns) - len(df_step1.columns)
    print(f"    +{step2_features}特徴量")

    total_features = len(df_combined.columns) - len(df.columns)
    print(f"  合計: +{total_features}特徴量 (元{len(df.columns)} -> {len(df_combined.columns)})")

    return df_combined

def submit_ticket_017_combined():
    """TICKET-017組み合わせでのKaggle提出実行"""
    print("TICKET-017組み合わせ Kaggle提出開始")
    print("=" * 60)

    # データ読み込み
    print("Step 1: データ読み込み")
    try:
        train_path = Path("data/processed/train.csv")
        validation_path = Path("data/processed/validation.csv")
        test_path = Path("data/processed/test.csv")

        train_df = pd.read_csv(train_path)
        validation_df = pd.read_csv(validation_path)
        test_df = pd.read_csv(test_path)

        print(f"  訓練データ: {train_df.shape}")
        print(f"  検証データ: {validation_df.shape}")
        print(f"  テストデータ: {test_df.shape}")

    except Exception as e:
        print(f"エラー: データ読み込み失敗 - {e}")
        return False

    # 訓練・検証データ結合
    full_train_df = pd.concat([train_df, validation_df], ignore_index=True)
    print(f"  結合後訓練データ: {full_train_df.shape}")

    # Step 2: TICKET-017組み合わせ特徴量生成
    print("\nStep 2: TICKET-017組み合わせ特徴量生成")
    start_time = time.time()

    # 訓練データ
    print("  訓練データ処理中...")
    train_features = create_ticket_017_features(full_train_df)

    # テストデータ
    print("  テストデータ処理中...")
    test_features = create_ticket_017_features(test_df)

    feature_time = time.time() - start_time
    print(f"  特徴量生成時間: {feature_time:.2f}秒")

    # 特徴量とターゲット分離
    feature_cols = [col for col in train_features.columns if col not in ['id', 'BeatsPerMinute']]
    X_train = train_features[feature_cols]
    y_train = train_features['BeatsPerMinute']
    X_test = test_features[feature_cols]

    print(f"  最終特徴量数: {len(feature_cols)}")

    # Step 3: モデル訓練（5-Fold CV）
    print("\nStep 3: LightGBMモデル訓練")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    models = []
    cv_scores = []
    predictions = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        print(f"  Fold {fold}/5 訓練中...")

        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # LightGBMモデル
        model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=1000,
            early_stopping_rounds=100,
            verbose=-1,
            random_state=42,
            force_col_wise=True
        )

        # 訓練
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            callbacks=[lgb.log_evaluation(0)]
        )

        # 検証スコア
        val_pred = model.predict(X_fold_val)
        fold_rmse = np.sqrt(np.mean((y_fold_val - val_pred) ** 2))
        cv_scores.append(fold_rmse)

        # テスト予測
        test_pred = model.predict(X_test)
        predictions += test_pred / 5

        models.append(model)
        print(f"    Fold {fold} RMSE: {fold_rmse:.4f}")

    mean_cv_rmse = np.mean(cv_scores)
    std_cv_rmse = np.std(cv_scores)
    print(f"  CV RMSE: {mean_cv_rmse:.4f} (±{std_cv_rmse:.4f})")

    # Step 4: 提出ファイル作成
    print("\nStep 4: Kaggle提出ファイル作成")
    submission_df = pd.DataFrame({
        'id': test_features['id'],
        'BeatsPerMinute': predictions
    })

    # ファイル保存
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    submission_path = f"data/processed/submission_ticket017_combined_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"  提出ファイル保存: {submission_path}")
    print(f"  予測範囲: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"  予測平均: {predictions.mean():.2f}")

    # サマリー
    print("\n" + "=" * 60)
    print("TICKET-017組み合わせ Kaggle提出完了")
    print("=" * 60)
    print(f"特徴量数: {len(feature_cols)}")
    print(f"CV RMSE: {mean_cv_rmse:.4f} (±{std_cv_rmse:.4f})")
    print(f"提出ファイル: {submission_path}")
    print(f"提出コマンド:")
    print(f'kaggle competitions submit -c playground-series-s5e9 -f "{submission_path}" -m "TICKET-017 Combined Features (CV: {mean_cv_rmse:.4f})"')

    return True

def main():
    """メイン実行"""
    success = submit_ticket_017_combined()
    if success:
        print("\n✅ TICKET-017組み合わせ提出準備完了")
    else:
        print("\n❌ TICKET-017組み合わせ提出失敗")
        sys.exit(1)

if __name__ == "__main__":
    main()