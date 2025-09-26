#!/usr/bin/env python3
"""
TICKET-017完全版 Kaggle提出スクリプト

TICKET-017-01（包括的交互作用）+ TICKET-017-02（対数変換）+ TICKET-017-03（ビニング）の
完全統合版でモデル訓練・予測・提出まで実行する。
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import time

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.features import (
    create_comprehensive_interaction_features,
    create_log_features,
    create_binning_features
)

def create_ticket_017_complete_features(df):
    """TICKET-017完全版特徴量を作成"""
    print("TICKET-017完全版特徴量生成中...")

    # Step 1: TICKET-017-01（包括的交互作用特徴量）
    print("  Step 1: 包括的交互作用特徴量")
    df_step1 = create_comprehensive_interaction_features(df)
    step1_features = len(df_step1.columns) - len(df.columns)
    print(f"    +{step1_features}特徴量")

    # Step 2: TICKET-017-02（対数変換特徴量）
    print("  Step 2: 対数変換特徴量")
    df_step2 = create_log_features(df_step1)
    step2_features = len(df_step2.columns) - len(df_step1.columns)
    print(f"    +{step2_features}特徴量")

    # Step 3: TICKET-017-03（ビニング・カテゴリ特徴量）
    print("  Step 3: ビニング・カテゴリ特徴量")
    df_complete = create_binning_features(df_step2)
    step3_features = len(df_complete.columns) - len(df_step2.columns)
    print(f"    +{step3_features}特徴量")

    total_features = len(df_complete.columns) - len(df.columns)
    print(f"  合計: +{total_features}特徴量 (元{len(df.columns)} -> {len(df_complete.columns)})")

    return df_complete

def submit_ticket_017_complete():
    """TICKET-017完全版でのKaggle提出実行"""
    print("TICKET-017完全版 Kaggle提出開始")
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

    # Step 2: TICKET-017完全版特徴量生成
    print("\nStep 2: TICKET-017完全版特徴量生成")
    start_time = time.time()

    # 訓練データ
    print("  訓練データ処理中...")
    train_features = create_ticket_017_complete_features(full_train_df)

    # テストデータ
    print("  テストデータ処理中...")
    test_features = create_ticket_017_complete_features(test_df)

    feature_time = time.time() - start_time
    print(f"  特徴量生成時間: {feature_time:.2f}秒")

    # 特徴量とターゲット分離
    feature_cols = [col for col in train_features.columns if col not in ['id', 'BeatsPerMinute']]
    X_train_full = train_features[feature_cols]
    y_train = train_features['BeatsPerMinute']
    X_test_full = test_features[feature_cols]

    print(f"  全特徴量数: {len(feature_cols)}")

    # Step 3: 特徴量選択（高次元対応）
    print("\nStep 3: 特徴量選択（高次元対応）")
    n_features_select = 150  # 完全版では150特徴量に

    selector = SelectKBest(score_func=f_regression, k=n_features_select)
    X_train = selector.fit_transform(X_train_full, y_train)
    X_test = selector.transform(X_test_full)

    selected_features = selector.get_support()
    selected_feature_names = [name for name, selected in zip(feature_cols, selected_features) if selected]

    print(f"  選択特徴量数: {n_features_select}")
    print(f"  上位特徴量: {selected_feature_names[:5]}")

    # 特徴量タイプ別の選択状況分析
    feature_types = {
        'comprehensive': [f for f in selected_feature_names if any(x in f for x in ['_x_', '_squared', '_div_'])],
        'log_transform': [f for f in selected_feature_names if 'log1p_' in f],
        'binning': [f for f in selected_feature_names if '_bin' in f],
        'original': [f for f in selected_feature_names if not any(x in f for x in ['_x_', '_squared', '_div_', 'log1p_', '_bin'])]
    }

    print("  選択特徴量タイプ別:")
    for ftype, features in feature_types.items():
        print(f"    {ftype}: {len(features)}特徴量")

    # Step 4: 高度な正則化モデル訓練（5-Fold CV）
    print("\nStep 4: 高度な正則化LightGBMモデル訓練")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    models = []
    cv_scores = []
    predictions = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        print(f"  Fold {fold}/5 訓練中...")

        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # 高次元対応の強化LightGBMモデル
        model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            num_leaves=20,           # 31→20 (高次元対応)
            learning_rate=0.03,     # 0.05→0.03 (より保守的)
            n_estimators=2000,      # 1500→2000 (早期停止期待)
            early_stopping_rounds=200,  # 150→200 (より保守的)
            reg_alpha=2.0,          # 1.0→2.0 (L1正則化強化)
            reg_lambda=2.0,         # 1.0→2.0 (L2正則化強化)
            feature_fraction=0.7,   # 0.8→0.7 (特徴量サンプリング強化)
            bagging_fraction=0.7,   # 0.8→0.7 (データサンプリング強化)
            bagging_freq=5,
            min_child_samples=50,   # 30→50 (過学習抑制強化)
            subsample_for_bin=200000,  # 高次元対応
            max_depth=6,            # 深度制限追加
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
    print("\nStep 5: Kaggle提出ファイル作成")
    submission_df = pd.DataFrame({
        'id': test_features['id'],
        'BeatsPerMinute': predictions
    })

    # ファイル保存
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    submission_path = f"data/processed/submission_ticket017_complete_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"  提出ファイル保存: {submission_path}")
    print(f"  予測範囲: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"  予測平均: {predictions.mean():.2f}")

    # 特徴量重要度トップ10
    print("\n  特徴量重要度 Top 10:")
    feature_importance = models[0].feature_importances_
    importance_pairs = list(zip(selected_feature_names, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)

    for i, (feature_name, importance) in enumerate(importance_pairs[:10], 1):
        feature_type = "original"
        if any(x in feature_name for x in ['_x_', '_squared', '_div_']):
            feature_type = "comprehensive"
        elif 'log1p_' in feature_name:
            feature_type = "log_transform"
        elif '_bin' in feature_name:
            feature_type = "binning"

        print(f"    {i:2d}. {feature_name[:40]:<40} {importance:8.4f} ({feature_type})")

    # サマリー
    print("\n" + "=" * 60)
    print("TICKET-017完全版 Kaggle提出完了")
    print("=" * 60)
    print(f"全特徴量数: {len(feature_cols)}")
    print(f"選択特徴量数: {n_features_select}")
    print(f"CV RMSE: {mean_cv_rmse:.4f} (±{std_cv_rmse:.4f})")
    print(f"提出ファイル: {submission_path}")
    print(f"改善点:")
    print(f"  - TICKET-017完全統合: 017-01 + 017-02 + 017-03")
    print(f"  - 高次元特徴量選択: {len(feature_cols)}→{n_features_select}")
    print(f"  - 強化された正則化")
    print(f"  - 保守的学習率: 0.03")
    print(f"提出コマンド:")
    print(f'kaggle competitions submit -c playground-series-s5e9 -f "{submission_path}" -m "TICKET-017 Complete (CV: {mean_cv_rmse:.4f}, Features: {n_features_select})"')

    return True

def main():
    """メイン実行"""
    success = submit_ticket_017_complete()
    if success:
        print("\n成功: TICKET-017完全版提出準備完了")
    else:
        print("\n失敗: TICKET-017完全版提出失敗")
        sys.exit(1)

if __name__ == "__main__":
    main()