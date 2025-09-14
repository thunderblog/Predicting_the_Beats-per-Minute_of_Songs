from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# Configクラスを定義
class CFG:
    # プロジェクト構造パス（bmp/config.pyから統合）
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    raw_data_dir = data_dir / "raw"
    interim_data_dir = data_dir / "interim"
    processed_data_dir = data_dir / "processed"
    external_data_dir = data_dir / "external"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    
    # 実験設定
    exp_name = 'exp01'
    log_dir = "./log/" + exp_name
    test_size = 0.2
    random_state = 42
    learning_rate = 0.1
    num_leaves = 31
    n_estimators = 10000
    feature_fraction = 0.9
    stopping_rounds = 50
    log_evaluation = 100
    objective = 'regression'
    metric = 'rmse'
    features = [
        "RhythmScore", 
        "AudioLoudness", 
        "VocalContent", 
        "AcousticQuality", 
        "InstrumentalScore", 
        "LivePerformanceLikelihood", 
        "MoodScore", 
        "TrackDurationMs", 
        "Energy"
    ]
    target = "BeatsPerMinute"
    
    # データファイル名設定
    raw_data_files = {
        "train": "train.csv",
        "test": "test.csv",
        "sample_submission": "sample_submission.csv"
    }
    
    processed_data_files = {
        "train": "train.csv",
        "validation": "validation.csv", 
        "test": "test.csv",
        "sample_submission": "sample_submission.csv",
        "feature_summary" : "feature_summary.csv"
    }
    
    # パス取得メソッド（遅延評価でインポート循環を回避）
    def get_raw_path(self, file_key: str) -> Path:
        """生データファイルの完全パスを取得"""
        return self.raw_data_dir / self.raw_data_files[file_key]
        
    def get_processed_path(self, file_key: str) -> Path:
        """処理済みデータファイルの完全パスを取得"""
        return self.processed_data_dir / self.processed_data_files[file_key]

# CFGクラスのインスタンスを作成
config = CFG()