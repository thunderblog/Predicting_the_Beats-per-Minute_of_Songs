"""
TICKET-005: データ可視化機能の実装
KaggleのBPM予測コンペティション用EDAとプロット機能
"""

# %%

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import typer
from loguru import logger

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as mplstyle
    import seaborn as sns
    import scipy.stats as stats
    import japanize_matplotlib
    
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("可視化ライブラリ (matplotlib, seaborn) がインストールされていません")

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR
from scripts.my_config import config

app = typer.Typer()

# プロットスタイル設定
if VISUALIZATION_AVAILABLE:
    plt.style.use('default')
    sns.set_palette("husl")
    mplstyle.use('fast')


def setup_plot_style() -> None:
    """プロット用のスタイルを設定する."""
    if not VISUALIZATION_AVAILABLE:
        return

    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9


def create_target_distribution_plot(
    data: pd.DataFrame,
    target_col: str = "BeatsPerMinute",
    save_path: Optional[Path] = None
) -> Optional[Path]:
    """ターゲット変数の分布をプロットする.

    Args:
        data: データフレーム
        target_col: ターゲット変数のカラム名
        save_path: 保存パス（Noneの場合は自動生成）

    Returns:
        保存されたファイルパス
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("可視化ライブラリが利用できません")
        return None

    logger.info(f"ターゲット変数 {target_col} の分布をプロット中...")
    
    fig, axes = plt.subplots(2,2, figsize=(15, 10))
        
    sns.histplot(data=data, x=target_col, ax=axes[0,0])
    axes[0,0].set_title('Histogram')

    sns.boxplot(data=data, x=target_col, ax=axes[0,1])
    axes[0,1].set_title('Box Plot')

    sns.violinplot(data=data, x=target_col, ax=axes[1,0])
    axes[1,0].set_title('Violin Plot')

    stats.probplot(data[target_col], dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot (Normal Distribution)')

    plt.suptitle(f'{target_col} Distribution Analysis', fontsize=16)
    plt.tight_layout()    

    if save_path is None:
        save_path = FIGURES_DIR / "target_distribution.png"

    # プロット保存
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.success(f"ターゲット分布プロットを保存: {save_path}")
    return save_path


def create_feature_distribution_plots(
    data: pd.DataFrame,
    feature_cols: List[str],
    save_path: Optional[Path] = None
) -> Optional[Path]:
    """特徴量の分布をプロットする.

    Args:
        data: データフレーム
        feature_cols: 特徴量のカラム名リスト
        save_path: 保存パス（Noneの場合は自動生成）

    Returns:
        保存されたファイルパス
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("可視化ライブラリが利用できません")
        return None

    logger.info(f"{len(feature_cols)}個の特徴量分布をプロット中...")

    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, col in enumerate(feature_cols):
        if i < len(axes):
            sns.histplot(data=data, x=col, ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {col}')
            plt.setp(axes[i].get_xticklabes(), rotation=45)

    # 余ったサブプロットを非表示
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Feature Distributions', fontsize=16)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "feature_distributions.png"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.success(f"特徴量分布プロットを保存: {save_path}")
    return save_path


def create_correlation_heatmap(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "BeatsPerMinute",
    save_path: Optional[Path] = None
) -> Optional[Path]:
    """相関ヒートマップを作成する.

    Args:
        data: データフレーム
        feature_cols: 特徴量のカラム名リスト
        target_col: ターゲット変数のカラム名
        save_path: 保存パス（Noneの場合は自動生成）

    Returns:
        保存されたファイルパス
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("可視化ライブラリが利用できません")
        return None

    logger.info("相関ヒートマップを作成中...")

    # 相関を計算
    correlation_cols = feature_cols + [target_col]
    correlation_matrix = data[correlation_cols].corr()

    plt.figure(figsize=(12,10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={'shrink': 0.8}
    )
    
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "correlation_heatmap.png"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.success(f"相関ヒートマップを保存: {save_path}")
    return save_path


def create_target_vs_features_plots(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "BeatsPerMinute",
    save_path: Optional[Path] = None
) -> Optional[Path]:
    """ターゲット vs 特徴量の散布図を作成する.

    Args:
        data: データフレーム
        feature_cols: 特徴量のカラム名リスト
        target_col: ターゲット変数のカラム名
        save_path: 保存パス（Noneの場合は自動生成）

    Returns:
        保存されたファイルパス
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("可視化ライブラリが利用できません")
        return None

    logger.info(f"ターゲット vs 特徴量の散布図を作成中...")

    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    # TODO(human): ターゲットvs特徴量の散布図作成
    # 各特徴量とターゲット変数の関係を可視化してください：
    for i, col in enumerate(feature_cols):
        if i < len(axes):
            # TODO(human): 散布図の描画
            # 1. sns.scatterplot(data=data, x=col, y=target_col, ax=axes[i], alpha=0.6)
            # 2. axes[i].set_title(f'{target_col} vs {col}')
            # 3. 相関係数を計算: corr = data[col].corr(data[target_col])
            # 4. 相関係数をプロット上に表示 (axes[i].text使用)
            # pass  # TODO(human): ここに実装してください

    # 余ったサブプロットを非表示
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'{target_col} vs Features Scatter Plots', fontsize=16)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "target_vs_features.png"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.success(f"ターゲット vs 特徴量プロットを保存: {save_path}")
    return save_path


def create_outlier_detection_plot(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "BeatsPerMinute",
    save_path: Optional[Path] = None
) -> Optional[Path]:
    """外れ値検出プロットを作成する.

    Args:
        data: データフレーム
        feature_cols: 特徴量のカラム名リスト
        target_col: ターゲット変数のカラム名
        save_path: 保存パス（Noneの場合は自動生成）

    Returns:
        保存されたファイルパス
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("可視化ライブラリが利用できません")
        return None

    logger.info("外れ値検出プロットを作成中...")

    # TODO(human): 外れ値検出のための複数手法プロット
    # 以下の4つの手法で外れ値を可視化してください：
    # 1. Z-scoreによる外れ値検出 (scipy.stats.zscore使用)
    # 2. IQR法による外れ値検出
    # 3. Isolation Forestによる異常検知 (sklearn.ensemble.IsolationForest)
    # 4. ターゲット変数の外れ値とのクロス分析
    #
    # 2x2のサブプロットを作成し、各手法の結果を可視化してください
    # fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # TODO(human): ここに外れ値検出ロジックを実装してください
    # ヒント：異なる色で正常データと外れ値を区別して表示
    pass

    if save_path is None:
        save_path = FIGURES_DIR / "outlier_detection.png"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()

    logger.success(f"外れ値検出プロットを保存: {save_path}")
    return save_path


@app.command()
def create_eda_plots(
    train_data_path: Path = PROCESSED_DATA_DIR / "train.csv",
    output_dir: Path = FIGURES_DIR,
) -> None:
    """包括的なEDAプロットを作成する.

    Args:
        train_data_path: 訓練データのパス
        output_dir: 出力ディレクトリ
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("可視化ライブラリがインストールされていません")
        logger.info("以下を実行してください: pip install matplotlib seaborn scipy")
        return

    logger.info("包括的なEDAプロットの作成を開始...")
    setup_plot_style()

    # データ読み込み
    if not train_data_path.exists():
        logger.error(f"訓練データが見つかりません: {train_data_path}")
        logger.info("先に 'python src/dataset.py' を実行してください")
        return

    data = pd.read_csv(train_data_path)
    logger.info(f"データ読み込み完了: {data.shape}")

    # 特徴量とターゲットの定義
    feature_cols = config.features
    target_col = config.target

    output_dir.mkdir(parents=True, exist_ok=True)

    # 各種プロットの作成
    plots_created = []

    # TODO(human): EDAプロットパイプラインの実装
    # 以下の順序で各プロット関数を呼び出し、結果をplots_createdリストに追加してください：
    #
    # 1. create_target_distribution_plot() - ターゲット分布分析
    # 2. create_feature_distribution_plots() - 特徴量分布
    # 3. create_correlation_heatmap() - 相関ヒートマップ
    # 4. create_target_vs_features_plots() - ターゲットvs特徴量散布図
    # 5. create_outlier_detection_plot() - 外れ値検出プロット（新機能）
    #
    # 各関数が戻り値を返した場合のみplots_created.append()で追加してください

    # TODO(human): ここに5つのプロット作成コードを実装してください
    pass

    logger.success(f"EDAプロット作成完了! 作成されたファイル数: {len(plots_created)}")
    for plot_path in plots_created:
        logger.info(f"  - {plot_path}")


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "train.csv",
    output_path: Path = FIGURES_DIR / "eda_plots",
) -> None:
    """メインのプロット生成処理（create_eda_plotsのエイリアス）."""
    create_eda_plots(train_data_path=input_path, output_dir=output_path)


if __name__ == "__main__":
    app()
# %%
