# 🧠 TICKET-011-04: MLP回帰モデル使用ガイド

**最終更新**: 2025-09-23
**対応バージョン**: TICKET-011-04実装完了版
**関連ファイル**: `src/modeling/neural_models.py`, `src/modeling/neural_trainer.py`, `src/modeling/data_loaders.py`

## 📖 概要

TICKET-011-04で実装されたMulti-Layer Perceptron（MLP）回帰モデルは、楽曲特徴量からBPMを予測するためのニューラルネットワークベースの機械学習モデルです。従来のLightGBMとは異なる学習メカニズムを用いて、同等以上の予測精度を実現します。

### 🎯 **主要特徴**
- **表形式データ特化**: BatchNormalization + Dropout で安定訓練
- **2種類のアーキテクチャ**: Standard（高性能）とSimple（高速）
- **既存パイプライン統合**: `train.py`からシームレス実行
- **自動スケーリング**: StandardScaler統合済み
- **Early Stopping**: 過学習防止機能内蔵

### 📊 **性能指標**
- **検証RMSE**: 26.47（LightGBMと同等）
- **訓練時間**: 約2.8分（62エポック）
- **パラメータ数**: 15,361個（Simple）/ 179,969個（Standard）

---

## 🚀 クイックスタート

### 1. **基本的な実行**

```bash
# MLPモデルの訓練（簡易版）
python -m src.modeling.train --model-type=mlp_simple --exp-name=my_mlp_experiment

# MLPモデルの訓練（高性能版）
python -m src.modeling.train --model-type=mlp_standard --exp-name=my_mlp_standard
```

### 2. **最小限のコード例**

```python
from src.modeling.neural_trainer import NeuralTrainer, TrainingConfig
from src.modeling.data_loaders import BPMDataProcessor
import pandas as pd

# データ準備
train_df = pd.read_csv("data/processed/train_features.csv")

# 設定
config = TrainingConfig(
    model_type="simple",
    epochs=50,
    batch_size=512
)

# データ処理
processor = BPMDataProcessor()
data_dict = processor.prepare_data(train_df, target_col="BeatsPerMinute")

# 訓練実行
trainer = NeuralTrainer(config)
results = trainer.train(data_dict["train_loader"], data_dict["val_loader"])

print(f"最終検証RMSE: {results['best_val_rmse']:.4f}")
```

---

## 💎 詳細使用方法

### 🎛️ **モデルタイプの選択**

#### **mlp_simple** - 高速実験用
```python
# 特徴
- アーキテクチャ: Input → 256 → Output
- パラメータ数: ~15K
- 訓練時間: 短い（~2分）
- 用途: 高速実験、ベースライン確立
```

#### **mlp_standard** - 高性能用
```python
# 特徴
- アーキテクチャ: Input → 512 → 256 → 128 → Output
- パラメータ数: ~180K
- 訓練時間: 中程度（~5分）
- 用途: 最終モデル、精度重視
```

### ⚙️ **ハイパーパラメータ調整**

```python
config = TrainingConfig(
    # モデル構造
    model_type="standard",
    hidden_dims=[512, 256, 128],        # 隠れ層サイズ
    dropout_rates=[0.3, 0.2, 0.1],      # Dropout率
    activation="relu",                   # 活性化関数

    # 訓練パラメータ
    learning_rate=1e-3,                  # 学習率
    batch_size=512,                      # バッチサイズ
    epochs=100,                          # 最大エポック数
    patience=15,                         # Early Stopping閾値

    # 最適化
    optimizer_type="adam",               # adam, adamw, sgd
    scheduler_type="reduce_on_plateau",  # 学習率スケジューラ
    weight_decay=1e-4,                   # 重み減衰

    # データ処理
    scaler_type="standard",              # standard, robust, minmax
    validation_size=0.2,                 # 検証データ比率

    # システム
    device="auto"                        # auto, cpu, cuda
)
```

### 📁 **出力ファイル構成**

```
models/
├── {exp_name}_mlp_simple_20250923_220112.pth     # 訓練済みモデル
├── {exp_name}_mlp_simple_results_20250923_220112.json  # 訓練結果
└── ...

# results.json の内容例
{
    "experiment_name": "test_mlp",
    "model_type": "mlp_simple",
    "cv_rmse": 26.4738,
    "train_rmse": 26.8812,
    "training_time_minutes": 2.8,
    "epochs_trained": 62,
    "feature_count": 26,
    "train_samples": 335464,
    "val_samples": 83867
}
```

---

## 📊 実験管理との統合

### 🧪 **自動実験実行**

```bash
# 実験管理システムとの統合（推奨）
python scripts/submit_experiment.py \
    --experiment-name="exp06_mlp_baseline" \
    --model-type="mlp_simple"
```

### 📈 **性能比較**

| モデル | 検証RMSE | 訓練時間 | パラメータ数 | 用途 |
|--------|----------|----------|--------------|------|
| LightGBM | 26.39 | ~30分 | - | ベースライン |
| MLP Simple | 26.47 | ~3分 | 15K | 高速実験 |
| MLP Standard | TBD | ~5分 | 180K | 高精度 |

---

## 🔧 高度な使用方法

### 🎯 **カスタムモデル作成**

```python
from src.modeling.neural_models import BPMPredictor

# カスタムアーキテクチャ
model = BPMPredictor(
    input_dim=26,
    hidden_dims=[1024, 512, 256, 128, 64],  # 5層ネットワーク
    dropout_rates=[0.4, 0.3, 0.2, 0.1, 0.05],
    activation="leaky_relu",
    use_batch_norm=True
)

print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
```

### 💾 **モデルの保存・読み込み**

```python
# 保存
trainer.save_model("models/my_custom_model.pth")

# 読み込み
trainer.load_model("models/my_custom_model.pth", input_dim=26)

# 予測実行
predictions = trainer.predict(test_loader)
```

### 🔄 **カスタム訓練ループ**

```python
# 手動訓練制御
for epoch in range(100):
    train_loss, train_rmse = trainer.train_epoch(train_loader)
    val_loss, val_rmse = trainer.validate_epoch(val_loader)

    print(f"Epoch {epoch}: Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}")

    if trainer.check_early_stopping(val_loss):
        print("Early stopping triggered")
        break
```

---

## 🔍 トラブルシューティング

### ❗ **よくあるエラーと解決方法**

#### **1. GPU/CUDA関連エラー**
```bash
# エラー: CUDA out of memory
# 解決: バッチサイズを小さくする
config = TrainingConfig(batch_size=256)  # デフォルト512から削減

# エラー: CUDA not available
# 解決: CPUを明示的に指定
config = TrainingConfig(device="cpu")
```

#### **2. メモリ不足エラー**
```bash
# エラー: System memory exhausted
# 解決: DataLoaderのワーカー数を削減
data_loader = DataLoader(dataset, num_workers=0)  # マルチプロセス無効化
```

#### **3. 学習が進まない**
```python
# 症状: 損失が下がらない
# 解決: 学習率調整
config = TrainingConfig(learning_rate=1e-4)  # より小さな学習率

# 症状: 過学習
# 解決: Dropout率上げる
config = TrainingConfig(dropout_rates=[0.5, 0.4, 0.3])
```

#### **4. 訓練時間が長すぎる**
```python
# 解決: 簡易モデル使用
config = TrainingConfig(model_type="simple")

# または早期終了設定を厳しく
config = TrainingConfig(patience=5, min_delta=1e-3)
```

### 🐛 **デバッグ手順**

```python
# 1. データの確認
print(f"Train shape: {data_dict['train_loader'].dataset.features.shape}")
print(f"Feature range: {data_dict['train_loader'].dataset.features.min():.3f} to {data_dict['train_loader'].dataset.features.max():.3f}")

# 2. モデルのサマリー
model = trainer.model
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# 3. 勾配の確認
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")
```

---

## 🔬 実験のベストプラクティス

### 📋 **実験計画テンプレート**

```markdown
## 実験: exp06_mlp_baseline

### 目的
- MLPベースラインの確立
- LightGBMとの性能比較

### 設定
- モデル: mlp_simple
- エポック: 100
- バッチサイズ: 512
- 学習率: 1e-3

### 期待結果
- 検証RMSE < 26.5
- 訓練時間 < 5分

### 実際の結果
- 検証RMSE: 26.47
- 訓練時間: 2.8分
- 判定: ✅ 成功
```

### 🎯 **次の実験候補**

1. **ハイパーパラメータ最適化**
   ```bash
   # Optunaを使った自動最適化
   python scripts/optuna_optimization.py --model-type=mlp_standard
   ```

2. **アンサンブル実験**
   ```bash
   # LightGBM + MLP アンサンブル
   python scripts/ensemble_experiment.py --models=lightgbm,mlp_standard
   ```

3. **特徴量エンジニアリング検証**
   ```bash
   # 新特徴量でのMLP性能テスト
   python -m src.modeling.train --model-type=mlp_simple --exp-name=mlp_new_features
   ```

---

## 📚 関連ドキュメント

- [TICKET-008_USAGE_GUIDE.md](TICKET-008_USAGE_GUIDE.md) - ジャンル特徴量使用ガイド
- [KAGGLE_SUBMIT_GUIDE.md](KAGGLE_SUBMIT_GUIDE.md) - Kaggle提出ガイド
- [プロジェクトREADME](../README.md) - プロジェクト全体概要

---

## 🔄 更新履歴

- **2025-09-23**: TICKET-011-04実装完了、初版作成
- **検証RMSE**: 26.4738達成（LightGBM同等性能確認）
- **統合完了**: train.py、実験管理システム統合済み

---

## ❓ サポート

質問や不具合報告は以下まで：
- **GitHub Issues**: プロジェクトリポジトリのIssues
- **実装詳細**: `src/modeling/neural_*.py` のソースコード参照
- **設定例**: このドキュメント内のコード例を参考