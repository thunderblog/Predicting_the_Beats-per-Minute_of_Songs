"""
TICKET-022 BPM Stratified戦略で訓練したモデルでの推論・提出
CV安定性10.16倍改善効果のLB検証
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import warnings
import json
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

from scripts.my_config import config
import lightgbm as lgb

def load_stratified_models() -> list:
    """BPM Stratified戦略で訓練したモデルを読み込み"""
    logger.info("BPM Stratified訓練済みモデル読み込み開始...")

    # 先ほどの実験で訓練されたモデル（ここでは直接使用するため、モデルパスを特定する必要がある）
    # 実際には実験スクリプトでモデルを保存していないため、再訓練が必要
    models = []

    logger.warning("訓練済みモデルが保存されていません。再訓練を実行します...")
    return None

def retrain_and_predict():
    """BPM Stratified戦略で再訓練して推論実行"""
    logger.info("TICKET-022 BPM Stratified 再訓練・推論実行...")

    # データ読み込み
    train_data_path = config.processed_data_dir / "train_ticket017_75_features.csv"
    test_data_path = config.processed_data_dir / "test_ticket017_75_features_full.csv"

    logger.info(f"訓練データ読み込み: {train_data_path}")
    train_df = pd.read_csv(train_data_path)

    logger.info(f"テストデータ読み込み: {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    # 特徴量とターゲットを分離（共通特徴量のみ使用）
    train_feature_cols = [col for col in train_df.columns if col not in ["id", "BeatsPerMinute"]]
    test_feature_cols = [col for col in test_df.columns if col != "id"]

    # 共通特徴量を特定
    common_features = list(set(train_feature_cols) & set(test_feature_cols))
    logger.info(f"共通特徴量数: {len(common_features)}")
    logger.info(f"訓練特徴量数: {len(train_feature_cols)}, テスト特徴量数: {len(test_feature_cols)}")

    if len(common_features) == 0:
        logger.error("共通特徴量が見つかりません")
        return

    X_train = train_df[common_features]
    y_train = train_df["BeatsPerMinute"]
    X_test = test_df[common_features]

    logger.info(f"特徴量数: {len(common_features)}, 訓練サンプル: {len(X_train)}, テストサンプル: {len(X_test)}")

    # BPM層化分割
    def create_bpm_bins(y: pd.Series, n_bins: int = 10) -> np.ndarray:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
        return bins

    def train_lightgbm_fold(X_train, y_train, X_val, y_val):
        """LightGBM訓練（TICKET017正則化版パラメータ）"""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 20,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'reg_alpha': 2.0,
            'reg_lambda': 2.0,
            'min_child_samples': 20,
            'random_state': config.random_state,
            'verbosity': -1
        }

        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(0)
            ]
        )

        return model

    # BPM Stratified KFold訓練
    from sklearn.model_selection import StratifiedKFold
    from tqdm import tqdm

    n_folds = 5
    stratify_labels = create_bpm_bins(y_train)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_state)

    models = []
    cv_scores = []
    test_predictions = np.zeros(len(X_test))

    logger.info("BPM Stratified KFold訓練・推論開始...")

    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X_train, stratify_labels), total=n_folds)):
        logger.info(f"フォールド {fold + 1}/{n_folds} 訓練・推論中...")

        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # 訓練
        model = train_lightgbm_fold(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
        models.append(model)

        # 検証スコア
        val_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
        fold_rmse = np.sqrt(np.mean((y_fold_val - val_pred) ** 2))
        cv_scores.append(fold_rmse)

        # テスト予測
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        test_predictions += test_pred / n_folds

        logger.info(f"フォールド {fold + 1} RMSE: {fold_rmse:.6f}")

    # CV結果
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    logger.success(f"BPM Stratified KFold完了")
    logger.info(f"平均CV RMSE: {mean_cv:.6f} ± {std_cv:.6f}")
    logger.info(f"フォールド別スコア: {[round(s, 6) for s in cv_scores]}")

    # 提出ファイル作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'BeatsPerMinute': test_predictions
    })

    submission_path = config.processed_data_dir / f"submission_ticket022_stratified_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    logger.success(f"提出ファイル作成完了: {submission_path}")
    logger.info(f"予測統計:")
    logger.info(f"- 最小値: {test_predictions.min():.4f}")
    logger.info(f"- 最大値: {test_predictions.max():.4f}")
    logger.info(f"- 平均値: {test_predictions.mean():.4f}")
    logger.info(f"- 標準偏差: {test_predictions.std():.4f}")

    # 実験記録
    experiment_record = {
        "experiment_name": "exp15_ticket022_stratified_lgb_baseline",
        "timestamp": timestamp,
        "cv_strategy": "bpm_stratified",
        "model_type": "lightgbm",
        "cv_results": {
            "mean_cv_rmse": mean_cv,
            "std_cv_rmse": std_cv,
            "fold_scores": cv_scores,
            "stability_improvement": "10.16x vs standard_kfold"
        },
        "prediction_stats": {
            "min_prediction": test_predictions.min(),
            "max_prediction": test_predictions.max(),
            "mean_prediction": test_predictions.mean(),
            "std_prediction": test_predictions.std()
        },
        "submission_file": str(submission_path),
        "expected_lb_improvement": "CV-LB格差 -0.077 → 大幅改善期待"
    }

    record_path = config.processed_data_dir / f"experiment_record_ticket022_{timestamp}.json"
    with open(record_path, 'w') as f:
        json.dump(experiment_record, f, indent=2, default=str)

    logger.info(f"実験記録保存: {record_path}")

    return submission_path, mean_cv, std_cv

def submit_to_kaggle(submission_path: Path, cv_score: float):
    """Kaggle提出（オプション）"""
    logger.info("Kaggle提出準備...")

    message = f"TICKET-022 BPM Stratified Strategy - CV: {cv_score:.6f} (10.16x stability improvement)"

    logger.info("手動でKaggle提出してください:")
    logger.info(f"ファイル: {submission_path}")
    logger.info(f"メッセージ: {message}")
    logger.info("期待効果: CV-LB格差の大幅改善（従来-0.077 → 改善）")

if __name__ == "__main__":
    logger.info("TICKET-022 BPM Stratified推論・提出スクリプト開始")

    submission_path, cv_score, cv_std = retrain_and_predict()

    logger.success("推論完了！")
    logger.info(f"CV RMSE: {cv_score:.6f} ± {cv_std:.6f}")
    logger.info("CV安定性改善: 10.16倍")

    submit_to_kaggle(submission_path, cv_score)

    logger.info("次のステップ:")
    logger.info("1. Kaggleに提出してLBスコアを確認")
    logger.info("2. CV-LB格差の改善効果を検証")
    logger.info("3. 効果が確認できればアンサンブル実験に進行")