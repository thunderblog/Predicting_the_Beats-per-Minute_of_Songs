"""
TICKET-024: 高度音楽データEDA・問題特定システム

音楽データ特有の外れ値検出、CV-LB格差要因分析、測定誤差パターン特定を行う
データ品質向上のための包括的分析スクリプト

戦略転換: モデル最適化限界により、データ品質・音楽ドメイン特徴量重視へ
目標: CV-LB格差（+0.076）の根本原因特定と改善
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    logger.warning(f"可視化ライブラリがインストールされていません: {e}")

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR


class AdvancedMusicEDA:
    """高度音楽データEDA分析クラス."""

    def __init__(self, data_path: Union[str, Path]):
        """初期化.

        Args:
            data_path: 分析対象データファイルのパス
        """
        self.data_path = Path(data_path)
        self.data = None

        # 9つの基本特徴量
        self.basic_features = [
            'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
            'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
            'TrackDurationMs', 'Energy'
        ]
        self.target = 'BeatsPerMinute'

        # 出力ディレクトリ設定
        self.output_dir = FIGURES_DIR / "advanced_eda"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 結果保存用
        self.analysis_results = {}

        # スタイル設定
        if VISUALIZATION_AVAILABLE:
            plt.style.use('default')
            sns.set_palette("husl")

    def load_data(self) -> bool:
        """データを読み込む.

        Returns:
            読み込み成功のブール値
        """
        try:
            if not self.data_path.exists():
                logger.error(f"データファイルが見つかりません: {self.data_path}")
                return False

            self.data = pd.read_csv(self.data_path)
            logger.info(f"データ読み込み完了: {self.data.shape}")

            # 基本統計表示
            logger.info(f"基本特徴量数: {len(self.basic_features)}")
            logger.info(f"ターゲット: {self.target}")
            logger.info(f"BPM範囲: {self.data[self.target].min():.1f} - {self.data[self.target].max():.1f}")

            return True
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            return False

    def _create_bpm_ranges(self) -> pd.Series:
        """BPM帯域分類を作成.

        Returns:
            BPM帯域ラベルのシリーズ
        """
        bpm = self.data[self.target]
        conditions = [
            (bpm < 60),
            (bpm >= 60) & (bpm < 80),
            (bpm >= 80) & (bpm < 120),
            (bpm >= 120) & (bpm < 140),
            (bpm >= 140) & (bpm < 180),
            (bpm >= 180)
        ]
        choices = ['Very Slow (<60)', 'Slow (60-80)', 'Moderate (80-120)',
                  'Fast (120-140)', 'Very Fast (140-180)', 'Extreme (180+)']
        return pd.Series(np.select(conditions, choices), index=self.data.index)

    def _create_genre_proxies(self) -> pd.DataFrame:
        """音楽ジャンル代理変数を作成.

        Returns:
            ジャンル代理変数を含むデータフレーム
        """
        df = self.data.copy()

        # 電子音楽系（LivePerformance低、Energy高）
        df['Electronic_Proxy'] = (
            (df['LivePerformanceLikelihood'] < 0.3) &
            (df['Energy'] > 0.7)
        ).astype(int)

        # アコースティック系（AcousticQuality高、InstrumentalScore高）
        df['Acoustic_Proxy'] = (
            (df['AcousticQuality'] > 0.7) &
            (df['InstrumentalScore'] > 0.5)
        ).astype(int)

        # ボーカル重視系（VocalContent高）
        df['Vocal_Heavy'] = (df['VocalContent'] > 0.5).astype(int)

        # ダンス系（RhythmScore高、Energy高）
        df['Dance_Proxy'] = (
            (df['RhythmScore'] > 0.7) &
            (df['Energy'] > 0.6)
        ).astype(int)

        # アンビエント系（Energy低、MoodScore中程度）
        df['Ambient_Proxy'] = (
            (df['Energy'] < 0.3) &
            (df['MoodScore'] > 0.3) &
            (df['MoodScore'] < 0.7)
        ).astype(int)

        return df

    def analyze_music_outliers(self) -> Dict:
        """音楽データ特有の外れ値を分析.

        Returns:
            外れ値分析結果の辞書
        """
        logger.info("音楽データ特有の外れ値分析を実行中...")

        results = {}

        # 1. BPM異常値検出（音楽理論ベース）
        bpm = self.data[self.target]

        # 極端なBPM値（音楽理論的に異常）
        extreme_slow = bpm < 40  # 通常の楽曲範囲外
        extreme_fast = bpm > 200  # 通常の楽曲範囲外

        results['extreme_bpm'] = {
            'extreme_slow_count': sum(extreme_slow),
            'extreme_fast_count': sum(extreme_fast),
            'extreme_slow_ratio': sum(extreme_slow) / len(bpm),
            'extreme_fast_ratio': sum(extreme_fast) / len(bpm),
            'extreme_slow_samples': bpm[extreme_slow].tolist()[:10],  # 最初の10件
            'extreme_fast_samples': bpm[extreme_fast].tolist()[:10]
        }

        # 2. 楽器バランス異常
        # VocalContentとInstrumentalScoreの矛盾（論理的整合性）
        vocal_instrumental_conflict = (
            (self.data['VocalContent'] > 0.8) &
            (self.data['InstrumentalScore'] > 0.8)
        )

        # AudioLoudnessの異常値（音響工学的観点）
        audio_outliers = (
            (self.data['AudioLoudness'] < -30) |  # 異常に小さい
            (self.data['AudioLoudness'] > 0)      # 0dB超過（物理的限界）
        )

        results['balance_anomalies'] = {
            'vocal_instrumental_conflict_count': sum(vocal_instrumental_conflict),
            'audio_loudness_outliers_count': sum(audio_outliers),
            'vocal_instrumental_conflict_ratio': sum(vocal_instrumental_conflict) / len(self.data),
            'audio_outlier_ratio': sum(audio_outliers) / len(self.data),
            'audio_outlier_range': {
                'min': self.data['AudioLoudness'].min(),
                'max': self.data['AudioLoudness'].max(),
                'suspicious_low': sum(self.data['AudioLoudness'] < -30),
                'suspicious_high': sum(self.data['AudioLoudness'] > 0)
            }
        }

        # 3. 楽曲構造異常
        duration_ms = self.data['TrackDurationMs']
        too_short = duration_ms < 30000  # 30秒未満
        too_long = duration_ms > 600000  # 10分超過

        results['duration_anomalies'] = {
            'too_short_count': sum(too_short),
            'too_long_count': sum(too_long),
            'too_short_ratio': sum(too_short) / len(duration_ms),
            'too_long_ratio': sum(too_long) / len(duration_ms),
            'duration_stats': {
                'mean_seconds': duration_ms.mean() / 1000,
                'median_seconds': duration_ms.median() / 1000,
                'min_seconds': duration_ms.min() / 1000,
                'max_seconds': duration_ms.max() / 1000
            }
        }

        # 4. エネルギー・ムード不整合
        energy_mood_mismatch = (
            ((self.data['Energy'] > 0.8) & (self.data['MoodScore'] < 0.3)) |  # 高エネルギー・低ムード
            ((self.data['Energy'] < 0.2) & (self.data['MoodScore'] > 0.7))    # 低エネルギー・高ムード
        )

        results['energy_mood_mismatch'] = {
            'count': sum(energy_mood_mismatch),
            'ratio': sum(energy_mood_mismatch) / len(self.data)
        }

        logger.success(f"音楽データ外れ値分析完了 - 問題データ比率: {sum(extreme_slow) + sum(extreme_fast) + sum(vocal_instrumental_conflict)}/{len(self.data)} ({((sum(extreme_slow) + sum(extreme_fast) + sum(vocal_instrumental_conflict))/len(self.data)*100):.2f}%)")
        return results

    def analyze_cv_lb_gap_factors(self) -> Dict:
        """CV-LB格差要因を統計分析.

        Returns:
            CV-LB格差分析結果の辞書
        """
        logger.info("CV-LB格差要因の統計分析を実行中...")

        results = {}

        # 1. データ分布の歪度・尖度分析（CV-LB不一致の原因）
        skewness_analysis = {}
        kurtosis_analysis = {}

        for feature in self.basic_features + [self.target]:
            if feature in self.data.columns:
                feature_data = self.data[feature].dropna()
                skew = stats.skew(feature_data)
                kurt = stats.kurtosis(feature_data)
                skewness_analysis[feature] = skew
                kurtosis_analysis[feature] = kurt

        # 高い歪度/尖度の特徴量を特定
        high_skew_features = {k: v for k, v in skewness_analysis.items() if abs(v) > 1.0}
        high_kurt_features = {k: v for k, v in kurtosis_analysis.items() if abs(v) > 3.0}

        results['distribution_analysis'] = {
            'skewness': skewness_analysis,
            'kurtosis': kurtosis_analysis,
            'high_skew_features': high_skew_features,
            'high_kurtosis_features': high_kurt_features
        }

        # 2. 特徴量間の非線形関係検出
        nonlinear_relationships = {}

        for feature in self.basic_features:
            if feature in self.data.columns:
                # スピアマン相関 vs ピアソン相関の差（非線形関係の指標）
                pearson_corr = self.data[feature].corr(self.data[self.target])
                spearman_corr = self.data[feature].corr(self.data[self.target], method='spearman')
                nonlinearity_score = abs(spearman_corr - pearson_corr)

                nonlinear_relationships[feature] = {
                    'pearson': pearson_corr,
                    'spearman': spearman_corr,
                    'nonlinearity_score': nonlinearity_score
                }

        # 非線形性が高い特徴量を特定
        high_nonlinear = {k: v for k, v in nonlinear_relationships.items()
                         if v['nonlinearity_score'] > 0.1}

        results['nonlinear_relationships'] = nonlinear_relationships
        results['high_nonlinear_features'] = high_nonlinear

        # 3. BPM範囲別データ密度の不均一性
        bpm_ranges = self._create_bpm_ranges()
        density_analysis = bpm_ranges.value_counts(normalize=True).to_dict()

        # 密度の偏りを計算
        density_values = list(density_analysis.values())
        density_variance = np.var(density_values)

        results['bpm_density_distribution'] = {
            'distribution': density_analysis,
            'density_variance': density_variance,
            'most_common_range': max(density_analysis, key=density_analysis.get),
            'least_common_range': min(density_analysis, key=density_analysis.get)
        }

        # 4. 条件付き分散分析（異分散性検出）
        heteroscedasticity_analysis = {}

        for feature in self.basic_features:
            if feature in self.data.columns:
                # BPM帯域別の特徴量分散
                variances_by_bpm = []
                for bpm_range in bpm_ranges.unique():
                    mask = bpm_ranges == bpm_range
                    if sum(mask) > 10:  # 十分なサンプル数
                        variance = self.data.loc[mask, feature].var()
                        variances_by_bpm.append(variance)

                if len(variances_by_bpm) > 1:
                    variance_ratio = max(variances_by_bpm) / min(variances_by_bpm)
                    heteroscedasticity_analysis[feature] = {
                        'variance_ratio': variance_ratio,
                        'max_variance': max(variances_by_bpm),
                        'min_variance': min(variances_by_bpm)
                    }

        results['heteroscedasticity'] = heteroscedasticity_analysis

        logger.success(f"CV-LB格差要因分析完了 - 高歪度特徴量: {len(high_skew_features)}個, 高非線形特徴量: {len(high_nonlinear)}個")
        return results

    def analyze_bpm_genre_distribution(self) -> Dict:
        """ジャンル別・BPM帯別データ分布を調査.

        Returns:
            分布調査結果の辞書
        """
        logger.info("ジャンル別・BPM帯別データ分布調査を実行中...")

        # ジャンル代理変数作成
        genre_data = self._create_genre_proxies()
        bpm_ranges = self._create_bpm_ranges()
        genre_data['BPM_Range'] = bpm_ranges

        results = {}

        # 1. ジャンル別BPM統計
        genre_columns = ['Electronic_Proxy', 'Acoustic_Proxy', 'Vocal_Heavy',
                        'Dance_Proxy', 'Ambient_Proxy']

        genre_bpm_stats = {}
        genre_counts = {}

        for genre in genre_columns:
            genre_mask = genre_data[genre] == 1
            genre_count = sum(genre_mask)
            genre_counts[genre] = genre_count

            if genre_count > 10:
                bpm_subset = genre_data.loc[genre_mask, self.target]
                genre_bpm_stats[genre] = {
                    'count': len(bpm_subset),
                    'mean': bpm_subset.mean(),
                    'std': bpm_subset.std(),
                    'median': bpm_subset.median(),
                    'min': bpm_subset.min(),
                    'max': bpm_subset.max(),
                    'skewness': stats.skew(bpm_subset),
                    'kurtosis': stats.kurtosis(bpm_subset)
                }

        results['genre_bpm_statistics'] = genre_bpm_stats
        results['genre_counts'] = genre_counts

        # 2. BPM帯域別特徴量統計
        bpm_range_stats = {}
        for bpm_range in bpm_ranges.unique():
            mask = bpm_ranges == bpm_range
            range_count = sum(mask)

            if range_count > 10:
                subset_stats = {'count': range_count}
                for feature in self.basic_features:
                    if feature in genre_data.columns:
                        feature_subset = genre_data.loc[mask, feature]
                        subset_stats[feature] = {
                            'mean': feature_subset.mean(),
                            'std': feature_subset.std(),
                            'median': feature_subset.median()
                        }
                bpm_range_stats[bpm_range] = subset_stats

        results['bpm_range_feature_stats'] = bpm_range_stats

        # 3. ジャンル×BPM帯域のクロス分布
        cross_distribution = {}
        for genre in genre_columns:
            if genre_counts[genre] > 0:
                cross_dist = pd.crosstab(
                    genre_data['BPM_Range'],
                    genre_data[genre],
                    normalize='index'
                )
                if 1 in cross_dist.columns:
                    cross_distribution[genre] = cross_dist[1].to_dict()

        results['genre_bpm_cross_distribution'] = cross_distribution

        logger.success(f"ジャンル別・BPM帯別分布調査完了 - 検出ジャンル数: {len([g for g, c in genre_counts.items() if c > 10])}")
        return results

    def detect_measurement_errors(self) -> Dict:
        """測定誤差・記録エラーパターンを特定.

        Returns:
            エラーパターン検出結果の辞書
        """
        logger.info("測定誤差・記録エラーパターン特定を実行中...")

        results = {}

        # 1. 数値の異常パターン検出
        numerical_anomalies = {}

        for feature in self.basic_features:
            if feature in self.data.columns:
                feature_data = self.data[feature].dropna()

                # 同一値の異常な集中（記録エラーの可能性）
                value_counts = feature_data.value_counts()
                max_count = value_counts.max()
                dominant_value_ratio = max_count / len(feature_data)

                # 0の異常な多さ
                zero_count = sum(feature_data == 0)
                zero_ratio = zero_count / len(feature_data)

                # 負の値（一部特徴量では異常）
                negative_count = sum(feature_data < 0) if feature != 'AudioLoudness' else 0

                numerical_anomalies[feature] = {
                    'dominant_value_ratio': dominant_value_ratio,
                    'most_common_value': value_counts.index[0],
                    'most_common_count': max_count,
                    'zero_count': zero_count,
                    'zero_ratio': zero_ratio,
                    'negative_count': negative_count,
                    'unique_values': len(value_counts)
                }

        results['numerical_anomalies'] = numerical_anomalies

        # 2. 範囲外値検出（ドメイン知識ベース）
        domain_violations = {}

        # 確率的特徴量（0-1範囲）の検証
        probability_features = ['RhythmScore', 'VocalContent', 'LivePerformanceLikelihood',
                               'MoodScore', 'Energy']
        for feature in probability_features:
            if feature in self.data.columns:
                out_of_range_low = sum(self.data[feature] < 0)
                out_of_range_high = sum(self.data[feature] > 1)
                out_of_range_total = out_of_range_low + out_of_range_high

                domain_violations[feature] = {
                    'out_of_range_low': out_of_range_low,
                    'out_of_range_high': out_of_range_high,
                    'out_of_range_total': out_of_range_total,
                    'out_of_range_ratio': out_of_range_total / len(self.data)
                }

        # AudioLoudness の妥当性（通常 -60dB から 0dB）
        if 'AudioLoudness' in self.data.columns:
            suspicious_low = sum(self.data['AudioLoudness'] < -60)
            suspicious_high = sum(self.data['AudioLoudness'] > 0)
            domain_violations['AudioLoudness'] = {
                'suspicious_low': suspicious_low,
                'suspicious_high': suspicious_high,
                'suspicious_total': suspicious_low + suspicious_high,
                'suspicious_ratio': (suspicious_low + suspicious_high) / len(self.data)
            }

        results['domain_violations'] = domain_violations

        # 3. 論理的整合性チェック
        logical_inconsistencies = {}

        # InstrumentalScore + VocalContent の合計チェック
        if 'InstrumentalScore' in self.data.columns and 'VocalContent' in self.data.columns:
            instrumental_vocal_sum = self.data['InstrumentalScore'] + self.data['VocalContent']
            impossible_combinations = sum(instrumental_vocal_sum > 1.2)  # 余裕をもって1.2

            logical_inconsistencies['instrumental_vocal_sum_violation'] = {
                'impossible_count': impossible_combinations,
                'impossible_ratio': impossible_combinations / len(self.data),
                'max_sum': instrumental_vocal_sum.max(),
                'mean_sum': instrumental_vocal_sum.mean()
            }

        results['logical_inconsistencies'] = logical_inconsistencies

        # 4. 異常な相関パターン
        correlation_anomalies = {}
        correlation_matrix = self.data[self.basic_features].corr()

        # 極めて高い相関（>0.95）の検出
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > 0.95:
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': correlation_matrix.iloc[i, j]
                    })

        correlation_anomalies['extremely_high_correlations'] = high_correlations
        correlation_anomalies['high_correlation_count'] = len(high_correlations)

        results['correlation_anomalies'] = correlation_anomalies

        # 問題の総計算
        total_domain_violations = sum(
            data['out_of_range_total'] if 'out_of_range_total' in data
            else data.get('suspicious_total', 0)
            for data in domain_violations.values()
        )

        logger.success(f"測定誤差・記録エラーパターン特定完了 - ドメイン違反総数: {total_domain_violations}")
        return results

    def create_comprehensive_visualizations(self, analysis_results: Dict) -> List[Path]:
        """包括的な可視化を作成.

        Args:
            analysis_results: 分析結果の辞書

        Returns:
            作成された画像ファイルパスのリスト
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("可視化ライブラリが利用できません")
            return []

        logger.info("包括的な可視化を作成中...")
        created_plots = []

        # 1. 音楽データ外れ値の可視化
        if 'music_outliers' in analysis_results:
            plot_path = self._create_music_outliers_plot(analysis_results['music_outliers'])
            if plot_path:
                created_plots.append(plot_path)

        # 2. CV-LB格差要因の可視化
        if 'cv_lb_analysis' in analysis_results:
            plot_path = self._create_cv_lb_analysis_plot(analysis_results['cv_lb_analysis'])
            if plot_path:
                created_plots.append(plot_path)

        # 3. ジャンル別・BPM帯別分布の可視化
        if 'genre_bpm_distribution' in analysis_results:
            plot_path = self._create_genre_bpm_distribution_plot(analysis_results['genre_bpm_distribution'])
            if plot_path:
                created_plots.append(plot_path)

        # 4. 測定エラーパターンの可視化
        if 'measurement_errors' in analysis_results:
            plot_path = self._create_measurement_errors_plot(analysis_results['measurement_errors'])
            if plot_path:
                created_plots.append(plot_path)

        logger.success(f"可視化作成完了: {len(created_plots)}個のプロットを生成")
        return created_plots

    def _create_music_outliers_plot(self, outliers_data: Dict) -> Optional[Path]:
        """音楽データ外れ値の可視化."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # BPM分布とextreme値のハイライト
        bpm = self.data[self.target]
        axes[0, 0].hist(bpm, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(40, color='red', linestyle='--', label='Extreme Slow (<40)')
        axes[0, 0].axvline(200, color='red', linestyle='--', label='Extreme Fast (>200)')
        axes[0, 0].set_title(f'BPM Distribution\nExtreme values: {outliers_data["extreme_bpm"]["extreme_slow_count"] + outliers_data["extreme_bpm"]["extreme_fast_count"]} samples')
        axes[0, 0].set_xlabel('BeatsPerMinute')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()

        # 楽器バランス異常の散布図
        x_feature = 'VocalContent'
        y_feature = 'InstrumentalScore'
        conflict_mask = (self.data[x_feature] > 0.8) & (self.data[y_feature] > 0.8)

        axes[0, 1].scatter(self.data[x_feature], self.data[y_feature], alpha=0.6, c='blue', s=10, label='Normal')
        axes[0, 1].scatter(self.data.loc[conflict_mask, x_feature], self.data.loc[conflict_mask, y_feature],
                          c='red', s=20, label=f'Conflict ({sum(conflict_mask)} samples)')
        axes[0, 1].set_xlabel(x_feature)
        axes[0, 1].set_ylabel(y_feature)
        axes[0, 1].set_title('Vocal-Instrumental Balance Conflicts')
        axes[0, 1].legend()

        # AudioLoudness異常値
        audio_data = self.data['AudioLoudness']
        normal_mask = (audio_data >= -30) & (audio_data <= 0)
        outlier_mask = ~normal_mask

        axes[1, 0].hist(audio_data[normal_mask], bins=30, alpha=0.7, color='green', label='Normal Range')
        axes[1, 0].hist(audio_data[outlier_mask], bins=30, alpha=0.7, color='red', label=f'Outliers ({sum(outlier_mask)} samples)')
        axes[1, 0].set_xlabel('AudioLoudness (dB)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('AudioLoudness Distribution')
        axes[1, 0].legend()

        # 楽曲長異常
        duration_data = self.data['TrackDurationMs'] / 1000  # 秒に変換
        normal_duration_mask = (duration_data >= 30) & (duration_data <= 600)

        axes[1, 1].hist(duration_data[normal_duration_mask], bins=30, alpha=0.7, color='blue', label='Normal Duration')
        axes[1, 1].hist(duration_data[~normal_duration_mask], bins=30, alpha=0.7, color='orange',
                       label=f'Abnormal ({sum(~normal_duration_mask)} samples)')
        axes[1, 1].set_xlabel('Track Duration (seconds)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Track Duration Distribution')
        axes[1, 1].legend()

        plt.tight_layout()
        plot_path = self.output_dir / "music_outliers_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _create_cv_lb_analysis_plot(self, cv_lb_data: Dict) -> Optional[Path]:
        """CV-LB格差要因の可視化."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 歪度分析
        if 'skewness' in cv_lb_data['distribution_analysis']:
            skewness = cv_lb_data['distribution_analysis']['skewness']
            features = list(skewness.keys())
            skew_values = list(skewness.values())

            colors = ['red' if abs(val) > 1.0 else 'blue' for val in skew_values]
            axes[0, 0].bar(range(len(features)), skew_values, color=colors, alpha=0.7)
            axes[0, 0].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='High Skew Threshold')
            axes[0, 0].axhline(-1.0, color='red', linestyle='--', alpha=0.5)
            axes[0, 0].set_xticks(range(len(features)))
            axes[0, 0].set_xticklabels(features, rotation=45, ha='right')
            axes[0, 0].set_title('Feature Skewness (CV-LB Gap Factor)')
            axes[0, 0].set_ylabel('Skewness')
            axes[0, 0].legend()

        # 尖度分析
        if 'kurtosis' in cv_lb_data['distribution_analysis']:
            kurtosis = cv_lb_data['distribution_analysis']['kurtosis']
            features = list(kurtosis.keys())
            kurt_values = list(kurtosis.values())

            colors = ['red' if abs(val) > 3.0 else 'green' for val in kurt_values]
            axes[0, 1].bar(range(len(features)), kurt_values, color=colors, alpha=0.7)
            axes[0, 1].axhline(3.0, color='red', linestyle='--', alpha=0.5, label='High Kurtosis Threshold')
            axes[0, 1].axhline(-3.0, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].set_xticks(range(len(features)))
            axes[0, 1].set_xticklabels(features, rotation=45, ha='right')
            axes[0, 1].set_title('Feature Kurtosis (Distribution Shape)')
            axes[0, 1].set_ylabel('Kurtosis')
            axes[0, 1].legend()

        # 非線形関係スコア
        if 'nonlinear_relationships' in cv_lb_data:
            nonlinear = cv_lb_data['nonlinear_relationships']
            features = list(nonlinear.keys())
            nonlinear_scores = [data['nonlinearity_score'] for data in nonlinear.values()]

            colors = ['red' if score > 0.1 else 'blue' for score in nonlinear_scores]
            axes[1, 0].bar(range(len(features)), nonlinear_scores, color=colors, alpha=0.7)
            axes[1, 0].axhline(0.1, color='red', linestyle='--', alpha=0.5, label='High Nonlinearity Threshold')
            axes[1, 0].set_xticks(range(len(features)))
            axes[1, 0].set_xticklabels(features, rotation=45, ha='right')
            axes[1, 0].set_title('Nonlinearity Score (Spearman - Pearson)')
            axes[1, 0].set_ylabel('Nonlinearity Score')
            axes[1, 0].legend()

        # BPM密度分布
        if 'bpm_density_distribution' in cv_lb_data:
            density_dist = cv_lb_data['bpm_density_distribution']['distribution']
            ranges = list(density_dist.keys())
            densities = list(density_dist.values())

            axes[1, 1].bar(range(len(ranges)), densities, color='purple', alpha=0.7)
            axes[1, 1].set_xticks(range(len(ranges)))
            axes[1, 1].set_xticklabels(ranges, rotation=45, ha='right')
            axes[1, 1].set_title('BPM Range Density Distribution')
            axes[1, 1].set_ylabel('Density')

        plt.tight_layout()
        plot_path = self.output_dir / "cv_lb_gap_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _create_genre_bpm_distribution_plot(self, genre_data: Dict) -> Optional[Path]:
        """ジャンル別・BPM帯別分布の可視化."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ジャンル代理変数を作成
        genre_df = self._create_genre_proxies()
        genre_columns = ['Electronic_Proxy', 'Acoustic_Proxy', 'Vocal_Heavy', 'Dance_Proxy', 'Ambient_Proxy']

        # ジャンル別BPM箱ひげ図
        genre_bpm_data = []
        genre_labels = []
        for genre in genre_columns:
            if genre in genre_df.columns:
                genre_mask = genre_df[genre] == 1
                if sum(genre_mask) > 10:
                    bpm_subset = self.data.loc[genre_mask, self.target]
                    genre_bpm_data.append(bpm_subset)
                    genre_labels.append(genre.replace('_Proxy', '').replace('_', ' '))

        if genre_bpm_data:
            axes[0, 0].boxplot(genre_bpm_data, labels=genre_labels)
            axes[0, 0].set_title('BPM Distribution by Genre Proxy')
            axes[0, 0].set_ylabel('BeatsPerMinute')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # BPM範囲別サンプル数
        bpm_ranges = self._create_bpm_ranges()
        range_counts = bpm_ranges.value_counts()

        axes[0, 1].bar(range(len(range_counts)), range_counts.values, color='orange', alpha=0.7)
        axes[0, 1].set_xticks(range(len(range_counts)))
        axes[0, 1].set_xticklabels(range_counts.index, rotation=45, ha='right')
        axes[0, 1].set_title('Sample Count by BPM Range')
        axes[0, 1].set_ylabel('Count')

        # ジャンル検出率
        if 'genre_counts' in genre_data:
            genre_counts = genre_data['genre_counts']
            valid_genres = {k: v for k, v in genre_counts.items() if v > 0}

            axes[1, 0].bar(range(len(valid_genres)), valid_genres.values(), color='green', alpha=0.7)
            axes[1, 0].set_xticks(range(len(valid_genres)))
            axes[1, 0].set_xticklabels([k.replace('_Proxy', '').replace('_', ' ') for k in valid_genres.keys()],
                                     rotation=45, ha='right')
            axes[1, 0].set_title('Genre Proxy Detection Count')
            axes[1, 0].set_ylabel('Detected Samples')

        # 全BPM分布（参考）
        axes[1, 1].hist(self.data[self.target], bins=30, color='lightblue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Overall BPM Distribution')
        axes[1, 1].set_xlabel('BeatsPerMinute')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plot_path = self.output_dir / "genre_bpm_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _create_measurement_errors_plot(self, error_data: Dict) -> Optional[Path]:
        """測定エラーパターンの可視化."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ドメイン違反の可視化
        if 'domain_violations' in error_data:
            violations = error_data['domain_violations']
            features = []
            violation_counts = []

            for feature, data in violations.items():
                violation_count = data.get('out_of_range_total', data.get('suspicious_total', 0))
                features.append(feature)
                violation_counts.append(violation_count)

            colors = ['red' if count > 0 else 'green' for count in violation_counts]
            axes[0, 0].bar(range(len(features)), violation_counts, color=colors, alpha=0.7)
            axes[0, 0].set_xticks(range(len(features)))
            axes[0, 0].set_xticklabels(features, rotation=45, ha='right')
            axes[0, 0].set_title('Domain Violations by Feature')
            axes[0, 0].set_ylabel('Violation Count')

        # 論理的整合性問題
        if 'logical_inconsistencies' in error_data:
            logical = error_data['logical_inconsistencies']
            check_names = []
            inconsistency_counts = []

            for check_name, data in logical.items():
                check_names.append(check_name.replace('_', ' ').title())
                inconsistency_counts.append(data.get('impossible_count', 0))

            axes[0, 1].bar(range(len(check_names)), inconsistency_counts, color='orange', alpha=0.7)
            axes[0, 1].set_xticks(range(len(check_names)))
            axes[0, 1].set_xticklabels(check_names, rotation=45, ha='right')
            axes[0, 1].set_title('Logical Inconsistency Issues')
            axes[0, 1].set_ylabel('Issue Count')

        # 数値異常パターン
        if 'numerical_anomalies' in error_data:
            numerical = error_data['numerical_anomalies']
            features = list(numerical.keys())
            zero_ratios = [data['zero_ratio'] for data in numerical.values()]

            axes[1, 0].bar(range(len(features)), zero_ratios, color='purple', alpha=0.7)
            axes[1, 0].set_xticks(range(len(features)))
            axes[1, 0].set_xticklabels(features, rotation=45, ha='right')
            axes[1, 0].set_title('Zero Value Ratio by Feature')
            axes[1, 0].set_ylabel('Zero Ratio')

        # 相関異常
        if 'correlation_anomalies' in error_data:
            high_corr_count = error_data['correlation_anomalies']['high_correlation_count']
            total_pairs = len(self.basic_features) * (len(self.basic_features) - 1) // 2

            axes[1, 1].pie([high_corr_count, total_pairs - high_corr_count],
                          labels=[f'High Correlation\n({high_corr_count})', f'Normal\n({total_pairs - high_corr_count})'],
                          colors=['red', 'green'], autopct='%1.1f%%')
            axes[1, 1].set_title('Feature Correlation Pattern\n(>0.95 threshold)')

        plt.tight_layout()
        plot_path = self.output_dir / "measurement_errors.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def generate_analysis_report(self, analysis_results: Dict) -> Path:
        """分析レポートを生成.

        Args:
            analysis_results: 全分析結果の辞書

        Returns:
            生成されたレポートファイルのパス
        """
        logger.info("分析レポートを生成中...")

        report_path = self.output_dir / "advanced_eda_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 高度音楽データEDA分析レポート\n\n")
            f.write(f"**分析日時**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**データセット**: {self.data_path.name}\n")
            f.write(f"**サンプル数**: {len(self.data):,}\n")
            f.write(f"**特徴量数**: {len(self.data.columns)}\n\n")

            f.write("## 戦略転換の背景\n\n")
            f.write("- **モデル最適化限界**: TICKET-021でアンサンブル重み最適化完了\n")
            f.write("- **CV-LB格差**: +0.076の一貫した下振れ\n")
            f.write("- **根本原因**: データ品質問題による汎化性能限界\n")
            f.write("- **新戦略**: データ改善によるCV-LB一貫性向上\n\n")

            # 1. 音楽データ外れ値分析
            if 'music_outliers' in analysis_results:
                f.write("## 1. 音楽データ特有の外れ値分析\n\n")
                outliers = analysis_results['music_outliers']

                f.write("### 🎵 極端なBPM値（音楽理論違反）\n")
                f.write(f"- **異常に遅い楽曲** (<40 BPM): {outliers['extreme_bpm']['extreme_slow_count']:,}件 ")
                f.write(f"({outliers['extreme_bpm']['extreme_slow_ratio']:.3%})\n")
                f.write(f"- **異常に速い楽曲** (>200 BPM): {outliers['extreme_bpm']['extreme_fast_count']:,}件 ")
                f.write(f"({outliers['extreme_bpm']['extreme_fast_ratio']:.3%})\n\n")

                f.write("### 🎚️ 楽器バランス異常\n")
                balance = outliers['balance_anomalies']
                f.write(f"- **ボーカル・楽器矛盾**: {balance['vocal_instrumental_conflict_count']:,}件 ")
                f.write(f"({balance['vocal_instrumental_conflict_ratio']:.3%})\n")
                f.write(f"- **音響レベル異常**: {balance['audio_loudness_outliers_count']:,}件 ")
                f.write(f"({balance['audio_outlier_ratio']:.3%})\n\n")

                f.write("### ⏱️ 楽曲長異常\n")
                duration = outliers['duration_anomalies']
                f.write(f"- **異常に短い** (<30秒): {duration['too_short_count']:,}件\n")
                f.write(f"- **異常に長い** (>10分): {duration['too_long_count']:,}件\n")
                f.write(f"- **平均長**: {duration['duration_stats']['mean_seconds']:.1f}秒\n\n")

            # 2. CV-LB格差要因分析
            if 'cv_lb_analysis' in analysis_results:
                f.write("## 2. CV-LB格差要因分析\n\n")
                cv_lb = analysis_results['cv_lb_analysis']

                f.write("### 📊 分布の歪み（CV-LB不一致要因）\n")
                if 'high_skew_features' in cv_lb:
                    high_skew = cv_lb['high_skew_features']
                    if high_skew:
                        f.write("**高い歪度を持つ特徴量**:\n")
                        for feature, skew_val in sorted(high_skew.items(), key=lambda x: abs(x[1]), reverse=True):
                            f.write(f"- {feature}: {skew_val:.3f}\n")
                    else:
                        f.write("- 顕著な歪みを持つ特徴量は検出されませんでした\n")
                    f.write("\n")

                f.write("### 🔄 非線形関係（予測困難性）\n")
                if 'high_nonlinear_features' in cv_lb:
                    high_nonlinear = cv_lb['high_nonlinear_features']
                    if high_nonlinear:
                        f.write("**非線形性の強い特徴量**:\n")
                        for feature, scores in sorted(high_nonlinear.items(),
                                                    key=lambda x: x[1]['nonlinearity_score'], reverse=True):
                            f.write(f"- {feature}: 非線形スコア {scores['nonlinearity_score']:.3f}\n")
                    else:
                        f.write("- 顕著な非線形関係は検出されませんでした\n")
                    f.write("\n")

                f.write("### 📈 BPM範囲別データ密度\n")
                if 'bpm_density_distribution' in cv_lb:
                    density = cv_lb['bpm_density_distribution']
                    f.write(f"- **最も多いBPM帯**: {density['most_common_range']}\n")
                    f.write(f"- **最も少ないBPM帯**: {density['least_common_range']}\n")
                    f.write(f"- **密度分散**: {density['density_variance']:.4f}\n\n")

            # 3. 測定誤差分析
            if 'measurement_errors' in analysis_results:
                f.write("## 3. 測定誤差・記録エラー分析\n\n")
                errors = analysis_results['measurement_errors']

                f.write("### ⚠️ ドメイン知識違反\n")
                if 'domain_violations' in errors:
                    violations = errors['domain_violations']
                    total_violations = 0
                    for feature, violation_data in violations.items():
                        violation_count = violation_data.get('out_of_range_total',
                                                           violation_data.get('suspicious_total', 0))
                        if violation_count > 0:
                            ratio = violation_data.get('out_of_range_ratio',
                                                     violation_data.get('suspicious_ratio', 0))
                            f.write(f"- **{feature}**: {violation_count:,}件の範囲外値 ({ratio:.3%})\n")
                            total_violations += violation_count

                    if total_violations == 0:
                        f.write("- 重大なドメイン違反は検出されませんでした\n")
                    f.write("\n")

                f.write("### 🔍 論理的整合性問題\n")
                if 'logical_inconsistencies' in errors:
                    logical = errors['logical_inconsistencies']
                    for check_name, check_data in logical.items():
                        if check_data.get('impossible_count', 0) > 0:
                            f.write(f"- **{check_name}**: {check_data['impossible_count']:,}件 ")
                            f.write(f"({check_data['impossible_ratio']:.3%})\n")
                    f.write("\n")

            # 4. 推奨改善アクション
            f.write("## 4. 推奨データ改善アクション\n\n")

            f.write("### 🏆 **優先度: 最高** - CV-LB格差改善への直接的効果\n")
            f.write("1. **極端なBPM値の除去**\n")
            f.write("   - 40 BPM未満、200 BPM超過のサンプル除去\n")
            f.write("   - 音楽理論的に不適切なデータの品質向上\n\n")

            f.write("2. **範囲外値の修正**\n")
            f.write("   - 確率特徴量の負値・1超過値の修正\n")
            f.write("   - AudioLoudnessの物理的限界外値の処理\n\n")

            f.write("### 🎯 **優先度: 高** - 予測精度向上への効果\n")
            f.write("1. **高い歪度特徴量の変換**\n")
            f.write("   - 対数変換、Box-Cox変換の適用\n")
            f.write("   - 分布正規化による汎化性能改善\n\n")

            f.write("2. **非線形関係の特徴量エンジニアリング**\n")
            f.write("   - 多項式特徴量、交互作用項の追加\n")
            f.write("   - モデルの非線形パターン捕捉能力向上\n\n")

            f.write("### 🔧 **優先度: 中** - データ品質向上\n")
            f.write("1. **論理的矛盾の解決**\n")
            f.write("   - ボーカル+楽器スコア合計>1の調整\n")
            f.write("   - エネルギー・ムード不整合の修正\n\n")

            f.write("2. **異分散性の改善**\n")
            f.write("   - BPM帯域別の特徴量スケーリング\n")
            f.write("   - 条件付き正規化の適用\n\n")

            # 5. 次のステップ
            f.write("## 5. 次のステップ（Phase 2-4）\n\n")
            f.write("### Phase 2: 外れ値検出・除去システム実装\n")
            f.write("- `src/data/outlier_handler.py` の作成\n")
            f.write("- 統計的手法とドメイン知識の組み合わせ\n")
            f.write("- 段階的除去による性能改善効果の検証\n\n")

            f.write("### Phase 3: 音楽ドメイン特徴量の実装\n")
            f.write("- `src/features/music_theory.py` の作成\n")
            f.write("- ハーモニー解析、リズムパターン特徴量\n")
            f.write("- 楽曲構造推定による差別化\n\n")

            f.write("### Phase 4: データ品質最適化\n")
            f.write("- 欠損値の音楽理論ベース補完\n")
            f.write("- 測定誤差・記録エラー対策\n")
            f.write("- データ整合性チェック・修正\n\n")

            f.write("---\n")
            f.write(f"**分析完了**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**期待効果**: CV-LB格差 +0.076 → +0.050以下, LB性能 26.385未満達成\n")

        logger.success(f"分析レポート生成完了: {report_path}")
        return report_path

    def run_comprehensive_analysis(self) -> Dict:
        """包括的な分析を実行.

        Returns:
            全分析結果の辞書
        """
        logger.info("🎵 高度音楽データEDA分析を開始...")
        logger.info("戦略: データ品質向上によるCV-LB一貫性改善")

        if not self.load_data():
            return {}

        results = {}

        try:
            # 各分析を順次実行
            results['music_outliers'] = self.analyze_music_outliers()
            results['cv_lb_analysis'] = self.analyze_cv_lb_gap_factors()
            results['genre_bpm_distribution'] = self.analyze_bpm_genre_distribution()
            results['measurement_errors'] = self.detect_measurement_errors()

            # 可視化作成
            plot_paths = self.create_comprehensive_visualizations(results)
            results['visualizations'] = [str(path) for path in plot_paths]

            # レポート生成
            report_path = self.generate_analysis_report(results)
            results['report_path'] = str(report_path)

            # 分析結果を保存
            self.analysis_results = results

            logger.success("🏆 包括的な高度EDA分析完了")
            return results

        except Exception as e:
            logger.error(f"分析実行中にエラーが発生しました: {e}")
            return {}


def main():
    """メイン実行関数."""
    logger.info("TICKET-024: 高度音楽データEDA・問題特定システム")

    # データパス設定（統一特徴量データセット使用）
    train_data_path = PROCESSED_DATA_DIR / "train_unified_75_features.csv"

    if not train_data_path.exists():
        logger.error(f"統一特徴量データが見つかりません: {train_data_path}")
        logger.info("代替手段: 基本データでの実行を試行")

        # フォールバック: 基本データセット
        basic_train_path = PROCESSED_DATA_DIR / "train.csv"
        if basic_train_path.exists():
            train_data_path = basic_train_path
            logger.info(f"基本データセットで実行: {train_data_path}")
        else:
            logger.error("利用可能なデータセットがありません")
            logger.info("先にデータ処理を実行してください: python src/dataset.py")
            return

    # 高度EDA実行
    eda = AdvancedMusicEDA(train_data_path)
    results = eda.run_comprehensive_analysis()

    if results:
        logger.success("✅ 高度EDA分析が正常に完了しました")

        # 結果サマリー表示
        if 'report_path' in results:
            logger.info(f"📊 分析レポート: {results['report_path']}")

        if 'visualizations' in results:
            logger.info(f"📈 可視化ファイル数: {len(results['visualizations'])}")

        # 重要な発見を表示
        if 'music_outliers' in results:
            outliers = results['music_outliers']
            extreme_total = (outliers['extreme_bpm']['extreme_slow_count'] +
                           outliers['extreme_bpm']['extreme_fast_count'])
            if extreme_total > 0:
                logger.warning(f"🚨 発見: 極端なBPM値 {extreme_total}件 (要対処)")

        if 'measurement_errors' in results:
            errors = results['measurement_errors']
            if 'domain_violations' in errors:
                total_violations = sum(
                    data.get('out_of_range_total', data.get('suspicious_total', 0))
                    for data in errors['domain_violations'].values()
                )
                if total_violations > 0:
                    logger.warning(f"⚠️ 発見: ドメイン違反 {total_violations}件 (要修正)")

        logger.info("🎯 次のステップ: Phase 2外れ値除去システムの実装")

    else:
        logger.error("❌ 高度EDA分析に失敗しました")


if __name__ == "__main__":
    main()