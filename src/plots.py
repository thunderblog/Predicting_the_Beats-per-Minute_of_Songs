"""
TICKET-005: データ可視化機能の実装
KaggleのBPM予測コンペティション用EDAとプロット機能
"""

# %%

from pathlib import Path
import platform
import subprocess
import sys
from typing import List, Optional
import webbrowser

from loguru import logger
import pandas as pd
import typer

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

try:
    import japanize_matplotlib  # 日本語フォント設定用 # noqa: F401
    import matplotlib.pyplot as plt
    import matplotlib.style as mplstyle
    import scipy.stats as stats
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("可視化ライブラリ (matplotlib, seaborn) がインストールされていません")

from scripts.my_config import config

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# プロットスタイル設定
if VISUALIZATION_AVAILABLE:
    plt.style.use("default")
    sns.set_palette("husl")
    mplstyle.use("fast")


def setup_plot_style() -> None:
    """プロット用のスタイルを設定する."""
    if not VISUALIZATION_AVAILABLE:
        return

    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9


def _open_image_file(file_path: Path) -> None:
    """システムの既定アプリで画像ファイルを開く.

    Args:
        file_path: 開く画像ファイルのパス
    """
    try:
        if platform.system() == "Windows":
            subprocess.run(["start", str(file_path)], shell=True, check=True)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", str(file_path)], check=True)
        else:  # Linux
            subprocess.run(["xdg-open", str(file_path)], check=True)
        logger.success(f"画像ファイルを開きました: {file_path.name}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"画像ファイルを開けませんでした: {file_path.name} - {e}")
    except FileNotFoundError:
        logger.warning(f"画像表示コマンドが見つかりません。手動で確認してください: {file_path}")


def _create_html_gallery(image_paths: List[Path], output_path: Path) -> None:
    """画像ギャラリーのHTMLファイルを作成しブラウザで開く.

    Args:
        image_paths: 画像ファイルパスのリスト
        output_path: HTMLファイルの出力パス
    """
    try:
        relative_paths = [path.relative_to(output_path.parent) for path in image_paths]

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>EDA Results - BPM Prediction</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .image-container {{ margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 10px 0; }}
        .image-title {{ font-size: 18px; font-weight: bold; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>EDA Results - Beats Per Minute Prediction</h1>
    <p>生成された画像数: {len(image_paths)}</p>
"""

        for i, (path, relative_path) in enumerate(zip(image_paths, relative_paths)):
            title = path.stem.replace("_", " ").title()
            html_content += f"""
    <div class="image-container">
        <div class="image-title">{i + 1}. {title}</div>
        <img src="{relative_path}" alt="{title}">
    </div>
"""

        html_content += """
</body>
</html>
"""

        output_path.write_text(html_content, encoding="utf-8")
        webbrowser.open(f"file://{output_path.absolute()}")
        logger.success(f"HTMLギャラリーをブラウザで開きました: {output_path}")

    except Exception as e:
        logger.error(f"HTMLギャラリーの作成に失敗しました: {e}")


def _display_plots(plot_paths: List[Path]) -> None:
    """作成されたプロットを表示する.

    Args:
        plot_paths: 表示する画像ファイルパスのリスト
    """
    if not plot_paths:
        logger.warning("表示する画像がありません")
        return

    # HTMLギャラリーを優先的に作成
    html_path = plot_paths[0].parent / "eda_gallery.html"
    _create_html_gallery(plot_paths, html_path)

    # 個別ファイルも開く（最大3つまで）
    for i, plot_path in enumerate(plot_paths[:3]):
        _open_image_file(plot_path)
        if i < len(plot_paths) - 1:
            # ファイルが開かれるまで少し待機
            import time

            time.sleep(0.5)


def create_target_distribution_plot(
    data: pd.DataFrame, target_col: str = "BeatsPerMinute", save_path: Optional[Path] = None
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

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.histplot(data=data, x=target_col, ax=axes[0, 0])
    axes[0, 0].set_title("Histogram")

    sns.boxplot(data=data, x=target_col, ax=axes[0, 1])
    axes[0, 1].set_title("Box Plot")

    sns.violinplot(data=data, x=target_col, ax=axes[1, 0])
    axes[1, 0].set_title("Violin Plot")

    stats.probplot(data[target_col], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot (Normal Distribution)")

    plt.suptitle(f"{target_col} Distribution Analysis", fontsize=16)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "target_distribution.png"

    # プロット保存
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"ターゲット分布プロットを保存: {save_path}")
    return save_path


def create_feature_distribution_plots(
    data: pd.DataFrame, feature_cols: List[str], save_path: Optional[Path] = None
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
            axes[i].set_title(f"Distribution of {col}")
            plt.setp(axes[i].get_xticklabels(), rotation=45)

    # 余ったサブプロットを非表示
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Feature Distributions", fontsize=16)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "feature_distributions.png"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"特徴量分布プロットを保存: {save_path}")
    return save_path


def create_correlation_heatmap(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "BeatsPerMinute",
    save_path: Optional[Path] = None,
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

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "correlation_heatmap.png"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"相関ヒートマップを保存: {save_path}")
    return save_path


def create_target_vs_features_plots(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "BeatsPerMinute",
    save_path: Optional[Path] = None,
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

    logger.info("ターゲット vs 特徴量の散布図を作成中...")

    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, col in enumerate(feature_cols):
        if i < len(axes):
            sns.scatterplot(data=data, x=col, y=target_col, ax=axes[i], alpha=0.6)
            axes[i].set_title(f"{target_col} vs {col}")
            corr = data[col].corr(data[target_col])
            axes[i].text(
                0.05,
                0.95,
                f"r = {corr:.3f}",
                transform=axes[i].transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                ha="left",
                va="top",
            )

    # 余ったサブプロットを非表示
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"{target_col} vs Features Scatter Plots", fontsize=16)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "target_vs_features.png"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"ターゲット vs 特徴量プロットを保存: {save_path}")
    return save_path


def detect_outliers_zscore(
    data: pd.DataFrame, feature_cols: List[str], threshold: float = 3.0
) -> pd.Series:
    """Z-score法による外れ値検出.

    Args:
        data: データフレーム
        feature_cols: 対象特徴量のカラム名リスト
        threshold: Z-scoreの閾値

    Returns:
        外れ値のブール値シリーズ
    """
    import numpy as np

    numeric_features = data[feature_cols].select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numeric_features, nan_policy="omit"))
    outliers = (z_scores > threshold).any(axis=1)
    return outliers


def detect_outliers_iqr(data: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    """IQR法による外れ値検出.

    Args:
        data: データフレーム
        feature_cols: 対象特徴量のカラム名リスト

    Returns:
        外れ値のブール値シリーズ
    """
    outlier_indices = set()
    for col in feature_cols:
        if data[col].dtype in ["int64", "float64"]:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
            outlier_indices.update(outliers)

    outliers_series = pd.Series(False, index=data.index)
    outliers_series.iloc[list(outlier_indices)] = True
    return outliers_series


def detect_outliers_isolation_forest(
    data: pd.DataFrame, feature_cols: List[str], contamination: float = 0.1
) -> pd.Series:
    """Isolation Forest法による外れ値検出.

    Args:
        data: データフレーム
        feature_cols: 対象特徴量のカラム名リスト
        contamination: 異常値の割合

    Returns:
        外れ値のブール値シリーズ
    """
    try:
        from sklearn.ensemble import IsolationForest

        numeric_features = data[feature_cols].select_dtypes(include=["int64", "float64"])
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers_pred = iso_forest.fit_predict(numeric_features)
        # -1が異常値、1が正常値
        outliers = pd.Series(outliers_pred == -1, index=data.index)
        return outliers
    except ImportError:
        logger.warning(
            "scikit-learn がインストールされていません。Isolation Forest をスキップします"
        )
        return pd.Series(False, index=data.index)


def detect_target_outliers(data: pd.DataFrame, target_col: str) -> pd.Series:
    """ターゲット変数の外れ値検出.

    Args:
        data: データフレーム
        target_col: ターゲット変数のカラム名

    Returns:
        外れ値のブール値シリーズ
    """
    Q1 = data[target_col].quantile(0.25)
    Q3 = data[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data[target_col] < lower_bound) | (data[target_col] > upper_bound)
    return outliers


def create_outlier_detection_plot(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "BeatsPerMinute",
    save_path: Optional[Path] = None,
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

    outliers_zscore = detect_outliers_zscore(data, feature_cols)
    outliers_iqr = detect_outliers_iqr(data, feature_cols)
    outliers_isolation_forest = detect_outliers_isolation_forest(data, feature_cols)
    outliers_target = detect_target_outliers(data, target_col)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 最も相関の高い特徴量を選択（X軸用）
    best_feature = data[feature_cols].corrwith(data[target_col]).abs().idxmax()

    # 1. Z-Score法（左上）
    colors_zscore = ["red" if outlier else "blue" for outlier in outliers_zscore]
    axes[0, 0].scatter(data[best_feature], data[target_col], c=colors_zscore, alpha=0.6)
    axes[0, 0].set_title(f"Z-Score Method (Outliers: {sum(outliers_zscore)})")
    axes[0, 0].set_xlabel(best_feature)
    axes[0, 0].set_ylabel(target_col)

    # 2. IQR法（右上）
    colors_iqr = ["red" if outlier else "blue" for outlier in outliers_iqr]
    axes[0, 1].scatter(data[best_feature], data[target_col], c=colors_iqr, alpha=0.6)
    axes[0, 1].set_title(f"IQR Method (Outliers: {sum(outliers_iqr)})")
    axes[0, 1].set_xlabel(best_feature)
    axes[0, 1].set_ylabel(target_col)

    # 3. Isolation Forest法（左下）
    colors_iso = ["red" if outlier else "blue" for outlier in outliers_isolation_forest]
    axes[1, 0].scatter(data[best_feature], data[target_col], c=colors_iso, alpha=0.6)
    axes[1, 0].set_title(f"Isolation Forest (Outliers: {sum(outliers_isolation_forest)})")
    axes[1, 0].set_xlabel(best_feature)
    axes[1, 0].set_ylabel(target_col)

    # 4. ターゲット変数外れ値（右下）
    colors_target = ["red" if outlier else "blue" for outlier in outliers_target]
    axes[1, 1].scatter(data[best_feature], data[target_col], c=colors_target, alpha=0.6)
    axes[1, 1].set_title(f"Target Outliers (Outliers: {sum(outliers_target)})")
    axes[1, 1].set_xlabel(best_feature)
    axes[1, 1].set_ylabel(target_col)

    # 凡例を追加
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="blue", label="Normal"),
        Patch(facecolor="red", label="Outlier"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)

    plt.suptitle("Outlier Detection Methods Comparison", fontsize=16, y=0.95)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "outlier_detection.png"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"外れ値検出プロットを保存: {save_path}")
    return save_path


@app.command()
def create_eda_plots(
    train_data_path: Path = PROCESSED_DATA_DIR / "train.csv",
    output_dir: Path = FIGURES_DIR,
    show: bool = typer.Option(False, "--show", "-s", help="プロット作成後に画像を表示する"),
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

    # 1. ターゲット分布分析
    plot_path = create_target_distribution_plot(
        data, target_col, output_dir / "target_distribution.png"
    )
    if plot_path is not None:
        plots_created.append(plot_path)

    # 2. 特徴量分布
    plot_path = create_feature_distribution_plots(
        data, feature_cols, output_dir / "feature_distributions.png"
    )
    if plot_path is not None:
        plots_created.append(plot_path)

    # 3. 相関ヒートマップ
    plot_path = create_correlation_heatmap(
        data, feature_cols, target_col, output_dir / "correlation_heatmap.png"
    )
    if plot_path is not None:
        plots_created.append(plot_path)

    # 4. ターゲットvs特徴量散布図
    plot_path = create_target_vs_features_plots(
        data, feature_cols, target_col, output_dir / "target_vs_features.png"
    )
    if plot_path is not None:
        plots_created.append(plot_path)

    # 5. 外れ値検出プロット
    plot_path = create_outlier_detection_plot(
        data, feature_cols, target_col, output_dir / "outlier_detection.png"
    )
    if plot_path is not None:
        plots_created.append(plot_path)

    logger.success(f"EDAプロット作成完了! 作成されたファイル数: {len(plots_created)}")
    for plot_path in plots_created:
        logger.info(f"  - {plot_path}")

    # プロット表示機能
    if show and plots_created:
        logger.info("画像ファイルを表示中...")
        _display_plots(plots_created)


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
