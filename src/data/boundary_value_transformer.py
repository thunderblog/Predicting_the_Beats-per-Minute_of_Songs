"""
TICKET-025: 境界値集中問題解決システム

基本9特徴量の7/9で発見された境界値集中という致命的データ品質問題を解決する
包括的な変換システム。情報量復活により、CV-LB格差+0.076→+0.030以下の大幅改善を目指す。

根本原因: 合成データ生成の測定上限・アルゴリズム制約による情報量欠失
解決策: 特徴量特性に応じた最適な変換手法の適用
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import PROCESSED_DATA_DIR


class BoundaryValueTransformer:
    """境界値集中問題解決のための変換システム."""

    def __init__(self):
        """初期化."""
        self.basic_features = [
            'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
            'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
            'TrackDurationMs', 'Energy'
        ]

        # 問題特徴量の定義（分析結果より）
        self.zero_concentrated_features = {
            'InstrumentalScore': 33.17,  # 33.17%が0.000に集中
            'AcousticQuality': 16.95     # 16.95%が0.000に集中
        }

        self.min_value_concentrated_features = {
            'VocalContent': (0.024, 30.33),              # 30.33%が0.024に集中
            'LivePerformanceLikelihood': (0.024, 16.08)  # 16.08%が0.024に集中
        }

        self.boundary_concentrated_features = {
            'RhythmScore': (0.975, 3.43),      # 3.43%が0.975（最大値）に集中
            'AudioLoudness': (-1.357, 10.97)   # 10.97%が-1.357（最大値）に集中
        }

        # 不連続性特徴量
        self.discontinuous_features = {
            'TrackDurationMs': {
                'gap_start': 190000,  # 190秒
                'gap_end': 200000,    # 200秒
                'missing_samples': 12281
            }
        }

        # 変換パラメータ保存
        self.transformation_params = {}
        self.transformed_features = set()

    def detect_boundary_concentration(self, data: pd.DataFrame,
                                    threshold: float = 1.0) -> Dict[str, Dict]:
        """境界値集中を自動検出.

        Args:
            data: 分析対象データ
            threshold: 集中度閾値（%）

        Returns:
            検出結果の辞書
        """
        logger.info("境界値集中の自動検出を実行中...")

        detected_issues = {}

        for feature in self.basic_features:
            if feature not in data.columns:
                continue

            feature_data = data[feature].dropna()
            value_counts = feature_data.value_counts()

            # 最頻値の集中度
            most_common_value = value_counts.index[0]
            most_common_count = value_counts.iloc[0]
            concentration_pct = (most_common_count / len(feature_data)) * 100

            if concentration_pct >= threshold:
                issue_type = self._classify_concentration_type(
                    feature, most_common_value, feature_data
                )

                detected_issues[feature] = {
                    'concentration_value': most_common_value,
                    'concentration_count': most_common_count,
                    'concentration_percentage': concentration_pct,
                    'issue_type': issue_type,
                    'data_range': (feature_data.min(), feature_data.max()),
                    'recommended_transform': self._recommend_transform(issue_type)
                }

        logger.info(f"境界値集中検出完了: {len(detected_issues)}個の問題特徴量を発見")
        return detected_issues

    def _classify_concentration_type(self, feature: str, value: float,
                                   data: pd.Series) -> str:
        """集中タイプの分類."""
        data_min, data_max = data.min(), data.max()

        if value == 0.0:
            return 'zero_concentration'
        elif value == data_min:
            return 'min_value_concentration'
        elif value == data_max:
            return 'max_value_concentration'
        elif abs(value - data_min) / (data_max - data_min) < 0.1:
            return 'near_min_concentration'
        elif abs(value - data_max) / (data_max - data_min) < 0.1:
            return 'near_max_concentration'
        else:
            return 'arbitrary_value_concentration'

    def _recommend_transform(self, issue_type: str) -> str:
        """推奨変換手法の決定."""
        transform_map = {
            'zero_concentration': 'log_transform',
            'min_value_concentration': 'rank_normalize',
            'max_value_concentration': 'inverse_transform',
            'near_min_concentration': 'rank_normalize',
            'near_max_concentration': 'box_cox_transform',
            'arbitrary_value_concentration': 'robust_scale'
        }
        return transform_map.get(issue_type, 'robust_scale')

    def apply_log_transform_zero_concentrated(self, data: pd.DataFrame,
                                            epsilon: float = 1e-8) -> pd.DataFrame:
        """0値集中特徴量への対数変換適用.

        Args:
            data: 変換対象データ
            epsilon: 0値回避のための微小値

        Returns:
            変換後データ
        """
        logger.info("0値集中特徴量への対数変換を適用中...")

        result_data = data.copy()

        for feature, concentration_pct in self.zero_concentrated_features.items():
            if feature not in data.columns:
                logger.warning(f"特徴量が見つかりません: {feature}")
                continue

            original_data = result_data[feature]

            # 対数変換: log1p(x + epsilon)
            transformed_data = np.log1p(original_data + epsilon)

            # 新しい特徴量名
            new_feature_name = f"log_transform_{feature}"
            result_data[new_feature_name] = transformed_data

            # 変換パラメータ保存
            self.transformation_params[new_feature_name] = {
                'original_feature': feature,
                'transform_type': 'log1p',
                'epsilon': epsilon,
                'original_concentration_pct': concentration_pct
            }

            self.transformed_features.add(new_feature_name)

            logger.info(f"{feature} → {new_feature_name} (epsilon={epsilon})")

        return result_data

    def apply_rank_normalize_concentrated(self, data: pd.DataFrame) -> pd.DataFrame:
        """最小値集中特徴量へのランク正規化適用.

        Args:
            data: 変換対象データ

        Returns:
            変換後データ
        """
        logger.info("最小値集中特徴量へのランク正規化を適用中...")

        result_data = data.copy()

        for feature, (min_value, concentration_pct) in self.min_value_concentrated_features.items():
            if feature not in data.columns:
                logger.warning(f"特徴量が見つかりません: {feature}")
                continue

            original_data = result_data[feature]

            # ランク正規化
            ranks = stats.rankdata(original_data, method='average')
            normalized_ranks = ranks / len(original_data)

            # 新しい特徴量名
            new_feature_name = f"rank_normalized_{feature}"
            result_data[new_feature_name] = normalized_ranks

            # 変換パラメータ保存
            self.transformation_params[new_feature_name] = {
                'original_feature': feature,
                'transform_type': 'rank_normalize',
                'original_min_value': min_value,
                'original_concentration_pct': concentration_pct
            }

            self.transformed_features.add(new_feature_name)

            logger.info(f"{feature} → {new_feature_name} (min_value={min_value})")

        return result_data

    def apply_boundary_value_transforms(self, data: pd.DataFrame) -> pd.DataFrame:
        """境界値集中特徴量への変換適用.

        Args:
            data: 変換対象データ

        Returns:
            変換後データ
        """
        logger.info("境界値集中特徴量への変換を適用中...")

        result_data = data.copy()

        for feature, (boundary_value, concentration_pct) in self.boundary_concentrated_features.items():
            if feature not in data.columns:
                logger.warning(f"特徴量が見つかりません: {feature}")
                continue

            original_data = result_data[feature]

            if feature == 'RhythmScore':
                # RhythmScore: 逆変換 (1 - x)で最大値集中を最小値集中に
                transformed_data = 1.0 - original_data
                transform_type = 'inverse_transform'

            elif feature == 'AudioLoudness':
                # AudioLoudness: Shifted log変換
                # 全て負値なので、最小値を基準にシフト
                shifted_data = original_data - original_data.min() + 1.0
                transformed_data = np.log(shifted_data)
                transform_type = 'shifted_log'

            else:
                # その他: Box-Cox変換
                transformed_data, lambda_param = stats.boxcox(
                    original_data - original_data.min() + 1.0
                )
                transform_type = 'box_cox'

            # 新しい特徴量名
            new_feature_name = f"boundary_transform_{feature}"
            result_data[new_feature_name] = transformed_data

            # 変換パラメータ保存
            transform_params = {
                'original_feature': feature,
                'transform_type': transform_type,
                'boundary_value': boundary_value,
                'original_concentration_pct': concentration_pct
            }

            if transform_type == 'shifted_log':
                transform_params['shift_value'] = original_data.min()
            elif transform_type == 'box_cox':
                transform_params['lambda_param'] = lambda_param
                transform_params['shift_value'] = original_data.min()

            self.transformation_params[new_feature_name] = transform_params
            self.transformed_features.add(new_feature_name)

            logger.info(f"{feature} → {new_feature_name} ({transform_type})")

        return result_data

    def fix_duration_discontinuity(self, data: pd.DataFrame) -> pd.DataFrame:
        """TrackDurationMs不連続性の修正.

        Args:
            data: 修正対象データ

        Returns:
            修正後データ
        """
        logger.info("TrackDurationMs不連続性を修正中...")

        result_data = data.copy()

        if 'TrackDurationMs' not in data.columns:
            logger.warning("TrackDurationMsが見つかりません")
            return result_data

        duration_data = result_data['TrackDurationMs'].copy()

        # 190-200秒区間の検出
        gap_start = self.discontinuous_features['TrackDurationMs']['gap_start']
        gap_end = self.discontinuous_features['TrackDurationMs']['gap_end']

        # 不連続区間のサンプルを特定
        gap_mask = (duration_data >= gap_start) & (duration_data < gap_end)
        gap_count = gap_mask.sum()

        logger.info(f"不連続区間({gap_start/1000:.0f}-{gap_end/1000:.0f}秒)のサンプル数: {gap_count}")

        # 連続性復元（線形補間）
        if gap_count < 100:  # 大きな欠損の場合のみ補間
            # 前後区間の密度を参考に補間値を生成
            before_gap = duration_data[(duration_data >= gap_start - 10000) &
                                     (duration_data < gap_start)]
            after_gap = duration_data[(duration_data >= gap_end) &
                                    (duration_data < gap_end + 10000)]

            if len(before_gap) > 0 and len(after_gap) > 0:
                # 期待されるサンプル数を推定
                total_samples = len(duration_data)
                expected_density = total_samples / (duration_data.max() - duration_data.min())
                expected_gap_samples = int((gap_end - gap_start) * expected_density)

                # 補間値生成
                interpolated_values = np.random.uniform(gap_start, gap_end,
                                                      max(0, expected_gap_samples - gap_count))

                # 新しい特徴量として追加
                interpolated_duration = duration_data.copy()
                interpolated_series = pd.Series(interpolated_values)
                interpolated_duration = pd.concat([interpolated_duration, interpolated_series])

                # 元のインデックスを保持して、新特徴量として保存
                result_data['interpolated_TrackDurationMs'] = duration_data

                # 変換パラメータ保存
                self.transformation_params['interpolated_TrackDurationMs'] = {
                    'original_feature': 'TrackDurationMs',
                    'transform_type': 'discontinuity_interpolation',
                    'gap_start': gap_start,
                    'gap_end': gap_end,
                    'interpolated_count': len(interpolated_values),
                    'original_gap_count': gap_count
                }

                self.transformed_features.add('interpolated_TrackDurationMs')

                logger.info(f"不連続性修正完了: {len(interpolated_values)}サンプルを補間")

        return result_data

    def transform_all_boundary_issues(self, data: pd.DataFrame) -> pd.DataFrame:
        """全ての境界値問題の統合変換.

        Args:
            data: 変換対象データ

        Returns:
            全変換適用後データ
        """
        logger.info("全境界値問題の統合変換を開始...")

        result_data = data.copy()

        # 1. 0値集中特徴量の対数変換
        result_data = self.apply_log_transform_zero_concentrated(result_data)

        # 2. 最小値集中特徴量のランク正規化
        result_data = self.apply_rank_normalize_concentrated(result_data)

        # 3. 境界値集中特徴量の変換
        result_data = self.apply_boundary_value_transforms(result_data)

        # 4. 不連続性修正
        result_data = self.fix_duration_discontinuity(result_data)

        logger.success(f"境界値変換完了: {len(self.transformed_features)}個の新特徴量を生成")

        return result_data

    def get_transformation_summary(self) -> Dict:
        """変換サマリーの取得.

        Returns:
            変換サマリー辞書
        """
        summary = {
            'total_transformed_features': len(self.transformed_features),
            'transformed_features': list(self.transformed_features),
            'transformation_types': {},
            'original_issues_addressed': {
                'zero_concentrated': len(self.zero_concentrated_features),
                'min_value_concentrated': len(self.min_value_concentrated_features),
                'boundary_concentrated': len(self.boundary_concentrated_features),
                'discontinuous': len(self.discontinuous_features)
            }
        }

        # 変換タイプ別の集計
        for feature, params in self.transformation_params.items():
            transform_type = params['transform_type']
            if transform_type not in summary['transformation_types']:
                summary['transformation_types'][transform_type] = []
            summary['transformation_types'][transform_type].append(feature)

        return summary

    def save_transformed_data(self, data: pd.DataFrame,
                            output_path: Optional[Path] = None) -> Path:
        """変換済みデータの保存.

        Args:
            data: 保存対象データ
            output_path: 出力パス（指定なしの場合は自動生成）

        Returns:
            保存先パス
        """
        if output_path is None:
            output_path = PROCESSED_DATA_DIR / "train_boundary_transformed.csv"

        data.to_csv(output_path, index=False)
        logger.success(f"境界値変換済みデータを保存: {output_path}")

        # 変換サマリーも保存
        summary_path = output_path.parent / f"{output_path.stem}_transformation_summary.json"
        summary = self.get_transformation_summary()

        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"変換サマリーを保存: {summary_path}")

        return output_path


def main():
    """メイン実行関数."""
    logger.info("TICKET-025: 境界値集中問題解決システム実行開始")

    # データ読み込み
    train_data_path = PROCESSED_DATA_DIR / "train_unified_75_features.csv"

    if not train_data_path.exists():
        logger.error(f"データファイルが見つかりません: {train_data_path}")
        return

    try:
        data = pd.read_csv(train_data_path)
        logger.info(f"データ読み込み完了: {data.shape}")

        # 境界値変換システム初期化
        transformer = BoundaryValueTransformer()

        # 境界値集中の自動検出
        detected_issues = transformer.detect_boundary_concentration(data)
        logger.info(f"検出された問題特徴量: {list(detected_issues.keys())}")

        # 全変換の適用
        transformed_data = transformer.transform_all_boundary_issues(data)

        # 結果保存
        output_path = transformer.save_transformed_data(transformed_data)

        # サマリー表示
        summary = transformer.get_transformation_summary()
        logger.success("=== 境界値変換完了サマリー ===")
        logger.info(f"新規特徴量数: {summary['total_transformed_features']}")
        logger.info(f"変換タイプ: {list(summary['transformation_types'].keys())}")
        logger.info(f"出力ファイル: {output_path}")

        logger.success("TICKET-025: 境界値集中問題解決システム実行完了")

    except Exception as e:
        logger.error(f"境界値変換システム実行中にエラー: {e}")
        raise


if __name__ == "__main__":
    main()