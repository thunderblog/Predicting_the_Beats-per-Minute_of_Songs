#!/usr/bin/env python3
"""
特徴量選択手法の包括的比較スクリプト

目的:
- 包括的交互作用特徴量から最重要特徴量を特定
- 複数の特徴量選択手法を比較
- 特徴量数による性能変化を分析
- 最適な特徴量組み合わせを発見
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
import warnings
import time
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
proj_root = Path(__file__).parent.parent
sys.path.append(str(proj_root))

from src.features import create_comprehensive_interaction_features
from src.config import PROCESSED_DATA_DIR

def load_test_sample(sample_size: int = 1000):
    """最適サンプルサイズ（1000件）でデータを読み込む"""
    logger.info(f"テスト用サンプルデータを読み込み中（サイズ: {sample_size}）...")

    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    sample_df = train_df.sample(n=sample_size, random_state=42)

    logger.info(f"サンプルデータ準備完了: {len(sample_df)}件")
    return sample_df

def quick_evaluate_features(X, y, feature_name="Default"):
    """軽量3-fold CVで特徴量性能を評価"""
    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 50,
        'verbose': -1,
        'random_state': 42
    }

    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kfold.split(X):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        model = lgb.LGBMRegressor(**model_params)
        model.fit(X_train_fold, y_train_fold)

        y_pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        cv_scores.append(rmse)

    avg_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)

    return {
        'feature_combination': feature_name,
        'n_features': X.shape[1],
        'avg_rmse': avg_rmse,
        'std_rmse': std_rmse,
        'cv_scores': cv_scores
    }

def analyze_feature_importance(X, y, method='all'):
    """複数手法で特徴量重要度を分析"""
    logger.info(f"特徴量重要度分析中（手法: {method}）...")

    importance_scores = {}

    if method in ['all', 'f_regression']:
        # F統計量による重要度
        f_selector = SelectKBest(score_func=f_regression, k='all')
        f_selector.fit(X, y)
        importance_scores['f_regression'] = f_selector.scores_

    if method in ['all', 'mutual_info']:
        # 相互情報量による重要度
        mi_selector = SelectKBest(score_func=mutual_info_regression, k='all')
        mi_selector.fit(X, y)
        importance_scores['mutual_info'] = mi_selector.scores_

    if method in ['all', 'correlation']:
        # 相関係数による重要度
        correlations = X.corrwith(y).abs()
        importance_scores['correlation'] = correlations.values

    if method in ['all', 'random_forest']:
        # Random Forestによる重要度
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importance_scores['random_forest'] = rf.feature_importances_

    if method in ['all', 'lightgbm']:
        # LightGBMによる重要度
        lgb_model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        lgb_model.fit(X, y)
        importance_scores['lightgbm'] = lgb_model.feature_importances_

    # 正規化
    normalized_scores = {}
    for key, scores in importance_scores.items():
        normalized_scores[key] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # 平均重要度計算
    if len(normalized_scores) > 1:
        avg_importance = np.mean(list(normalized_scores.values()), axis=0)
    else:
        avg_importance = list(normalized_scores.values())[0]

    # 結果DataFrame作成
    results_df = pd.DataFrame({
        'feature_name': X.columns,
        'avg_importance': avg_importance,
        **normalized_scores
    })

    results_df = results_df.sort_values('avg_importance', ascending=False)

    logger.info(f"特徴量重要度分析完了: 上位5特徴量")
    for i, (_, row) in enumerate(results_df.head().iterrows()):
        logger.info(f"  {i+1}. {row['feature_name']}: {row['avg_importance']:.4f}")

    return results_df

def select_features_by_method(X, y, method='avg_importance', k=20):
    """指定手法で特徴量を選択"""
    logger.info(f"特徴量選択中（手法: {method}, k={k}）...")

    if method == 'avg_importance':
        # 平均重要度による選択
        importance_df = analyze_feature_importance(X, y)
        selected_features = importance_df.head(k)['feature_name'].tolist()

    elif method == 'f_regression':
        # F統計量による選択
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()

    elif method == 'mutual_info':
        # 相互情報量による選択
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()

    elif method == 'correlation':
        # 相関係数による選択
        correlations = X.corrwith(y).abs()
        selected_features = correlations.nlargest(k).index.tolist()

    elif method == 'rfe_lgb':
        # RFE + LightGBMによる選択
        lgb_estimator = lgb.LGBMRegressor(n_estimators=30, random_state=42, verbose=-1)
        selector = RFE(estimator=lgb_estimator, n_features_to_select=k)
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()

    else:
        logger.error(f"不明な特徴量選択手法: {method}")
        return X.columns.tolist()[:k]

    logger.info(f"選択された特徴量: {selected_features[:5]}... (計{len(selected_features)}個)")
    return selected_features

def test_feature_selection_methods(X_all, y, baseline_result):
    """複数の特徴量選択手法を比較テスト"""
    logger.info("=== 特徴量選択手法比較開始 ===")

    # テストする特徴量選択手法
    selection_methods = {
        'avg_importance': '平均重要度',
        'f_regression': 'F統計量',
        'mutual_info': '相互情報量',
        'correlation': '相関係数',
        'rfe_lgb': 'RFE+LightGBM'
    }

    # テストする特徴量数
    feature_counts = [10, 20, 30, 50]

    results = []

    for method_key, method_name in selection_methods.items():
        logger.info(f"\n--- {method_name}による特徴量選択 ---")

        for k in feature_counts:
            if k > len(X_all.columns):
                continue

            try:
                start_time = time.time()

                # 特徴量選択
                selected_features = select_features_by_method(X_all, y, method_key, k)
                X_selected = X_all[selected_features]

                # 性能評価
                result = quick_evaluate_features(X_selected, y, f"{method_name}(k={k})")
                result['selection_method'] = method_key
                result['selection_method_name'] = method_name
                result['k_features'] = k
                result['processing_time'] = time.time() - start_time

                # ベースラインとの比較
                improvement = baseline_result['avg_rmse'] - result['avg_rmse']
                improvement_pct = (improvement / baseline_result['avg_rmse']) * 100
                result['improvement'] = improvement
                result['improvement_pct'] = improvement_pct

                results.append(result)

                logger.info(f"  k={k}: RMSE={result['avg_rmse']:.4f}, 改善={improvement:+.4f} ({improvement_pct:+.2f}%)")

            except Exception as e:
                logger.error(f"  k={k}: エラー - {e}")
                continue

    return results

def test_feature_number_scaling(X_all, y, baseline_result, method='avg_importance'):
    """特徴量数による性能変化を詳細分析"""
    logger.info(f"=== 特徴量数スケーリング分析（{method}）===")

    # より細かい特徴量数でテスト
    feature_counts = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, len(X_all.columns)]

    results = []

    for k in feature_counts:
        if k > len(X_all.columns):
            k = len(X_all.columns)

        try:
            start_time = time.time()

            if k == len(X_all.columns):
                # 全特徴量使用
                X_selected = X_all
                feature_name = f"全特徴量(k={k})"
            else:
                # 特徴量選択
                selected_features = select_features_by_method(X_all, y, method, k)
                X_selected = X_all[selected_features]
                feature_name = f"選択特徴量(k={k})"

            # 性能評価
            result = quick_evaluate_features(X_selected, y, feature_name)
            result['k_features'] = k
            result['processing_time'] = time.time() - start_time

            # ベースラインとの比較
            improvement = baseline_result['avg_rmse'] - result['avg_rmse']
            improvement_pct = (improvement / baseline_result['avg_rmse']) * 100
            result['improvement'] = improvement
            result['improvement_pct'] = improvement_pct

            results.append(result)

            logger.info(f"k={k:3d}: RMSE={result['avg_rmse']:.4f}, 改善={improvement:+.4f} ({improvement_pct:+.2f}%), 時間={result['processing_time']:.1f}秒")

        except Exception as e:
            logger.error(f"k={k}: エラー - {e}")
            continue

    return results

def analyze_feature_categories(X_all, y, baseline_result):
    """特徴量カテゴリ別の効果を分析"""
    logger.info("=== 特徴量カテゴリ別分析 ===")

    # 基本特徴量
    basic_features = [
        'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
        'TrackDurationMs', 'Energy'
    ]

    # 特徴量カテゴリ分類
    categories = {
        'basic': [f for f in basic_features if f in X_all.columns],
        'interaction': [f for f in X_all.columns if '_x_' in f],
        'squared': [f for f in X_all.columns if '_squared' in f],
        'ratio': [f for f in X_all.columns if '_div_' in f]
    }

    category_results = []

    for cat_name, features in categories.items():
        if not features:
            continue

        logger.info(f"\n--- {cat_name}特徴量（{len(features)}個）---")

        try:
            X_category = X_all[features]
            result = quick_evaluate_features(X_category, y, f"{cat_name}特徴量")

            improvement = baseline_result['avg_rmse'] - result['avg_rmse']
            improvement_pct = (improvement / baseline_result['avg_rmse']) * 100
            result['improvement'] = improvement
            result['improvement_pct'] = improvement_pct
            result['category'] = cat_name

            category_results.append(result)

            logger.info(f"  RMSE: {result['avg_rmse']:.4f}, 改善: {improvement:+.4f} ({improvement_pct:+.2f}%)")

        except Exception as e:
            logger.error(f"  エラー: {e}")
            continue

    return category_results

def main():
    """メイン実行関数"""
    logger.info("=== 特徴量選択手法包括比較テスト ===")

    # 1. データ準備
    sample_data = load_test_sample(sample_size=1000)

    # 基本特徴量でベースライン測定
    basic_features = [
        'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
        'TrackDurationMs', 'Energy'
    ]

    X_basic = sample_data[basic_features]
    y = sample_data['BeatsPerMinute']

    logger.info("--- ベースライン測定 ---")
    baseline_result = quick_evaluate_features(X_basic, y, "ベースライン（基本特徴量）")
    logger.info(f"ベースライン RMSE: {baseline_result['avg_rmse']:.4f}")

    # 2. 包括的交互作用特徴量生成
    logger.info("--- 包括的交互作用特徴量生成 ---")
    enhanced_data = create_comprehensive_interaction_features(sample_data)
    feature_cols = [col for col in enhanced_data.columns if col not in ['id', 'BeatsPerMinute']]
    X_all = enhanced_data[feature_cols]

    logger.info(f"全特徴量数: {len(X_all.columns)}個")

    # 3. 特徴量選択手法比較
    selection_results = test_feature_selection_methods(X_all, y, baseline_result)

    # 4. 特徴量数スケーリング分析
    scaling_results = test_feature_number_scaling(X_all, y, baseline_result)

    # 5. 特徴量カテゴリ別分析
    category_results = analyze_feature_categories(X_all, y, baseline_result)

    # 6. 結果サマリー
    logger.info("\n=== 最良結果サマリー ===")

    # 選択手法別最良結果
    best_by_method = {}
    for result in selection_results:
        method = result['selection_method_name']
        if method not in best_by_method or result['improvement'] > best_by_method[method]['improvement']:
            best_by_method[method] = result

    logger.info("特徴量選択手法別最良結果:")
    for method, result in best_by_method.items():
        logger.info(f"  {method}: k={result['k_features']}, RMSE={result['avg_rmse']:.4f}, 改善={result['improvement']:+.4f}")

    # 特徴量数別最良結果
    best_overall = max(scaling_results, key=lambda x: x['improvement'])
    logger.info(f"\n全体最良結果:")
    logger.info(f"  {best_overall['feature_combination']}: RMSE={best_overall['avg_rmse']:.4f}, 改善={best_overall['improvement']:+.4f}")

    # 7. 結果保存
    import json
    results_summary = {
        'baseline': baseline_result,
        'selection_method_comparison': selection_results,
        'feature_number_scaling': scaling_results,
        'category_analysis': category_results,
        'best_results': {
            'best_by_method': best_by_method,
            'best_overall': best_overall
        }
    }

    results_file = Path("results/feature_selection_comparison.json")
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)

    logger.success(f"包括比較結果を保存: {results_file}")

    return results_summary

if __name__ == "__main__":
    main()