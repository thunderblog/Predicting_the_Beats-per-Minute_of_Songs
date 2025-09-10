# Configクラスを定義
class CFG:
    exp_name = 'exp004'
    log_dir = "./tensorboard_logs/" + exp_name
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

# CFGクラスのインスタンスを作成
config = CFG()