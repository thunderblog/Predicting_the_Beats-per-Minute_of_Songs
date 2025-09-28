"""
TICKET-022 CV戦略比較実験スクリプト

複数のCV戦略（Standard KFold, BPM Stratified, Music Similarity Group）を比較し、
CV-LB一貫性を改善する最適な戦略を特定する。

実行方法:
    python scripts/cv_strategy_comparison.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from src.modeling.cross_validation import CVStrategyManager
from src.modeling.train import train_with_cross_validation
from scripts.my_config import config

def run_cv_strategy_comparison():
    """CV戦略比較実験を実行"""
    logger.info("TICKET-022 CV戦略比較実験開始...")

    # データ読み込み (75特徴量版を使用)
    data_path = config.processed_data_dir / "train_ticket017_75_features.csv"
    logger.info(f"データ読み込み: {data_path}")

    if not data_path.exists():
        logger.error(f"データファイルが見つかりません: {data_path}")
        return

    df = pd.read_csv(data_path)

    # 特徴量とターゲットを分離
    feature_cols = [col for col in df.columns if col not in ["id", "BeatsPerMinute"]]
    X = df[feature_cols]
    y = df["BeatsPerMinute"]

    logger.info(f"データ形状: {X.shape}, ターゲット: {y.shape}")
    logger.info(f"特徴量数: {len(feature_cols)}, サンプル数: {len(X)}")

    # CV戦略管理者を作成
    cv_manager = CVStrategyManager()

    # 1. CV戦略の分割品質比較
    logger.info("\n=== CV戦略分割品質比較 ===")
    strategy_results = cv_manager.compare_strategies(X, y)

    # 2. 実際のモデル性能比較 (LightGBMで統一)
    logger.info("\n=== 実際のモデル性能比較 ===")

    strategies = ['standard_kfold', 'bpm_stratified', 'music_similarity_group']
    performance_results = []

    for strategy_name in strategies:
        logger.info(f"\n--- {strategy_name} でLightGBM訓練 ---")

        try:
            # LightGBMでクロスバリデーション実行
            cv_scores, models = train_with_cross_validation(
                X, y,
                n_folds=5,
                model_type="lightgbm",
                cv_strategy=strategy_name
            )

            mean_cv = np.mean(cv_scores)
            std_cv = np.std(cv_scores)

            performance_results.append({
                'strategy': strategy_name,
                'mean_cv_rmse': mean_cv,
                'std_cv_rmse': std_cv,
                'cv_stability': std_cv,  # Lower is better
                'fold_scores': cv_scores
            })

            logger.info(f"{strategy_name} 結果:")
            logger.info(f"- 平均CV RMSE: {mean_cv:.6f}")
            logger.info(f"- CV標準偏差: {std_cv:.6f}")
            logger.info(f"- フォールド別スコア: {[round(s, 6) for s in cv_scores]}")

        except Exception as e:
            logger.error(f"{strategy_name} でエラー発生: {e}")
            continue

    # 3. 結果分析とレポート生成
    logger.info("\n=== CV戦略比較結果分析 ===")

    performance_df = pd.DataFrame(performance_results)
    logger.info("モデル性能比較:")
    logger.info(performance_df[['strategy', 'mean_cv_rmse', 'std_cv_rmse', 'cv_stability']].round(6))

    # 最適戦略の推奨
    if len(performance_df) > 0:
        # CV安定性（標準偏差）が最も低い戦略を推奨
        best_stability_idx = performance_df['cv_stability'].idxmin()
        best_stability_strategy = performance_df.loc[best_stability_idx]

        # CV性能が最も良い戦略
        best_performance_idx = performance_df['mean_cv_rmse'].idxmin()
        best_performance_strategy = performance_df.loc[best_performance_idx]

        logger.info(f"\n推奨CV戦略（安定性重視）: {best_stability_strategy['strategy']}")
        logger.info(f"- CV RMSE: {best_stability_strategy['mean_cv_rmse']:.6f}")
        logger.info(f"- CV安定性: {best_stability_strategy['cv_stability']:.6f}")

        logger.info(f"\n最高性能CV戦略: {best_performance_strategy['strategy']}")
        logger.info(f"- CV RMSE: {best_performance_strategy['mean_cv_rmse']:.6f}")
        logger.info(f"- CV安定性: {best_performance_strategy['cv_stability']:.6f}")

        # 結果をCSVに保存
        results_path = config.processed_data_dir / "cv_strategy_comparison_results.csv"
        performance_df.to_csv(results_path, index=False)
        logger.info(f"結果を保存: {results_path}")

        return best_stability_strategy['strategy'], performance_df

    else:
        logger.error("有効な結果が得られませんでした")
        return None, None

def analyze_cv_lb_consistency():
    """CV-LB一貫性分析"""
    logger.info("\n=== CV-LB一貫性分析 ===")

    # 過去実験結果からCV-LB格差を分析
    experiment_results_path = Path("experiments/experiment_results.csv")

    if experiment_results_path.exists():
        exp_df = pd.read_csv(experiment_results_path)

        # CV-LB差のあるデータのみ抽出
        valid_data = exp_df.dropna(subset=['cv_lb_diff'])

        if len(valid_data) > 0:
            logger.info("過去実験のCV-LB格差分析:")
            logger.info(f"- 平均CV-LB差: {valid_data['cv_lb_diff'].mean():.6f}")
            logger.info(f"- CV-LB差標準偏差: {valid_data['cv_lb_diff'].std():.6f}")
            logger.info(f"- CV-LB差範囲: {valid_data['cv_lb_diff'].min():.6f} ～ {valid_data['cv_lb_diff'].max():.6f}")

            # 一貫性の良い実験を特定
            consistency_threshold = -0.05  # CV-LB差が-0.05以上の実験
            consistent_experiments = valid_data[valid_data['cv_lb_diff'] >= consistency_threshold]

            if len(consistent_experiments) > 0:
                logger.info(f"\nCV-LB一貫性の良い実験 (差≥{consistency_threshold}):")
                for _, exp in consistent_experiments.iterrows():
                    logger.info(f"- {exp['exp_name']}: CV-LB差 {exp['cv_lb_diff']:.6f}")
        else:
            logger.warning("CV-LB差データが不足しています")
    else:
        logger.warning("実験結果ファイルが見つかりません")

if __name__ == "__main__":
    logger.info("TICKET-022 CV戦略改善実験開始")

    # CV戦略比較実験（直接実行）
    best_strategy, results_df = run_cv_strategy_comparison()

    if best_strategy:
        logger.success(f"最適CV戦略特定完了: {best_strategy}")
        logger.info("次のステップ: この戦略でアンサンブルモデルを再訓練してください")
    else:
        logger.error("CV戦略比較実験が失敗しました")