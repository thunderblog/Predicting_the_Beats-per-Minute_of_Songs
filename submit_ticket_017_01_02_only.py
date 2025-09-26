#!/usr/bin/env python3
"""
TICKET-017-01+02版 Kaggle提出スクリプト

最高性能(26.38534)を達成したTICKET-017-01（包括的交互作用）+ TICKET-017-02（対数変換）のみで
再現実験を行う。ビニング特徴量は含めない。
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import time
import gc

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.features import (
    create_comprehensive_interaction_features,
    create_log_features
)

def create_ticket_017_01_02_features(df):
    """TICKET-017-01+02特徴量を作成（ビニング特徴量なし）"""
    print("TICKET-017-01+02特徴量生成中...")

    # Step 1: TICKET-017-01（包括的交互作用特徴量）
    print("  Step 1: 包括的交互作用特徴量")
    df_step1 = create_comprehensive_interaction_features(df)
    step1_features = len(df_step1.columns) - len(df.columns)
    print(f"    +{step1_features}特徴量")

    # メモリ最適化
    gc.collect()

    # Step 2: TICKET-017-02（対数変換特徴量）
    print("  Step 2: 対数変換特徴量")
    df_step2 = create_log_features(df_step1)
    step2_features = len(df_step2.columns) - len(df_step1.columns)
    print(f"    +{step2_features}特徴量")

    # 中間データクリーンアップ
    del df_step1
    gc.collect()

    total_features = len(df_step2.columns) - len(df.columns)
    print(f"  合計: +{total_features}特徴量 (元{len(df.columns)} -> {len(df_step2.columns)})")

    return df_step2

def submit_ticket_017_01_02_only():
    """TICKET-017-01+02版でのKaggle提出実行（最高性能再現）"""
    print("TICKET-017-01+02版 Kaggle提出開始")
    print("=== 最高性能26.38534の再現実験 ===")
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

    # メモリクリーンアップ
    del train_df, validation_df
    gc.collect()

    # Step 2: TICKET-017-01+02特徴量生成
    print("\\nStep 2: TICKET-017-01+02特徴量生成")
    start_time = time.time()

    # 訓練データ
    print("  訓練データ処理中...")
    train_features = create_ticket_017_01_02_features(full_train_df)

    # テストデータ
    print("  テストデータ処理中...")
    test_features = create_ticket_017_01_02_features(test_df)

    feature_time = time.time() - start_time
    print(f"  特徴量生成時間: {feature_time:.2f}秒")

    # メモリクリーンアップ
    del full_train_df, test_df
    gc.collect()

    # 特徴量とターゲット分離
    feature_cols = [col for col in train_features.columns if col not in ['id', 'BeatsPerMinute']]
    X_train_full = train_features[feature_cols]
    y_train = train_features['BeatsPerMinute']

    # テストデータの特徴量を訓練データと合わせる
    common_features = [col for col in feature_cols if col in test_features.columns]
    X_train_full = X_train_full[common_features]
    X_test_full = test_features[common_features]

    print(f"  共通特徴量数: {len(common_features)}")

    # メモリクリーンアップ
    del train_features, test_features
    gc.collect()

    # Step 3: 特徴量選択（最高性能時の設定に合わせて）
    print("\\nStep 3: 特徴量選択")
    n_features_select = min(75, len(common_features))  # 最高性能時は約75特徴量

    print(f"  特徴量選択中: {len(common_features)} -> {n_features_select}")

    selector = SelectKBest(score_func=f_regression, k=n_features_select)
    X_train = selector.fit_transform(X_train_full, y_train)
    X_test = selector.transform(X_test_full)

    selected_features = selector.get_support()
    selected_feature_names = [name for name, selected in zip(common_features, selected_features) if selected]

    print(f"  選択特徴量数: {n_features_select}")
    print(f"  上位特徴量: {selected_feature_names[:5]}")

    # メモリクリーンアップ
    del X_train_full, X_test_full
    gc.collect()

    # Step 4: 最高性能時の正則化設定でモデル訓練（5-Fold CV）
    print("\\nStep 4: 最高性能正則化LightGBMモデル訓練")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    models = []
    cv_scores = []
    predictions = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        print(f"  Fold {fold}/5 訓練中...")

        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # 最高性能時の強化正則化LightGBMモデル（26.38534達成時の設定）
        model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            num_leaves=20,           # 高次元対応
            learning_rate=0.03,     # 保守的学習率
            n_estimators=2000,      # 十分な学習回数
            early_stopping_rounds=200,  # 保守的早期停止
            reg_alpha=2.0,          # 強いL1正則化
            reg_lambda=2.0,         # 強いL2正則化
            feature_fraction=0.7,   # 強い特徴量サンプリング
            bagging_fraction=0.7,   # 強いデータサンプリング
            bagging_freq=5,
            min_child_samples=50,   # 過学習抑制強化
            subsample_for_bin=200000,
            max_depth=6,            # 深度制限
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

    # Step 5: 提出ファイル作成
    print("\\nStep 5: Kaggle提出ファイル作成")

    # テストデータのIDを取得（元のtest_dfから）
    test_df_original = pd.read_csv("data/processed/test.csv")

    submission_df = pd.DataFrame({
        'id': test_df_original['id'],
        'BeatsPerMinute': predictions
    })

    # ファイル保存
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    submission_path = f"data/processed/submission_ticket017_01_02_only_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"  提出ファイル保存: {submission_path}")
    print(f"  予測範囲: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"  予測平均: {predictions.mean():.2f}")

    # 特徴量重要度トップ10
    print("\\n  特徴量重要度 Top 10:")
    feature_importance = models[0].feature_importances_
    importance_pairs = list(zip(selected_feature_names, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)

    for i, (feature_name, importance) in enumerate(importance_pairs[:10], 1):
        feature_type = "original"
        if any(x in feature_name for x in ['_x_', '_squared', '_div_']):
            feature_type = "comprehensive"
        elif 'log1p_' in feature_name:
            feature_type = "log_transform"

        print(f"    {i:2d}. {feature_name[:40]:<40} {importance:8.4f} ({feature_type})")

    # サマリー
    print("\\n" + "=" * 60)
    print("TICKET-017-01+02版 Kaggle提出完了")
    print("=== 最高性能26.38534の再現実験 ===")
    print("=" * 60)
    print(f"共通特徴量数: {len(common_features)}")
    print(f"選択特徴量数: {n_features_select}")
    print(f"CV RMSE: {mean_cv_rmse:.4f} (±{std_cv_rmse:.4f})")
    print(f"提出ファイル: {submission_path}")
    print(f"再現設定:")
    print(f"  - TICKET-017-01: 包括的交互作用特徴量")
    print(f"  - TICKET-017-02: 対数変換特徴量")
    print(f"  - ビニング特徴量: なし（性能劣化要因として除外）")
    print(f"  - 強化正則化: reg_alpha=2.0, reg_lambda=2.0")
    print(f"  - サンプリング強化: 0.7")
    print(f"提出コマンド:")
    print(f'kaggle competitions submit -c playground-series-s5e9 -f "{submission_path}" -m "TICKET-017-01+02 Only (CV: {mean_cv_rmse:.4f}, Features: {n_features_select}, Reproduce 26.38534)"')

    return True

def main():
    """メイン実行"""
    success = submit_ticket_017_01_02_only()
    if success:
        print("\\n成功: TICKET-017-01+02版提出準備完了")
        print("最高性能26.38534の再現実験完了")
    else:
        print("\\n失敗: TICKET-017-01+02版提出失敗")
        sys.exit(1)

if __name__ == "__main__":
    main()