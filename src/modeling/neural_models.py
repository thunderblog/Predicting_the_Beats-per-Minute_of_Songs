"""
BPM予測用ニューラルネットワークモデル。

表形式データの回帰タスクに特化したPyTorchベースのニューラルネットワークモデルを実装する。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class BPMPredictor(nn.Module):
    """
    楽曲の表形式特徴量からBPMを予測するマルチレイヤーパーセプトロン。

    Beats-per-Minute予測タスクに特化して設計されており、以下の特徴を持つ：
    - BatchNormalization による安定した訓練
    - Dropout による正則化
    - 実験用の設定可能なアーキテクチャ
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rates: Optional[List[float]] = None,
        use_batch_norm: bool = True,
        activation: str = "relu"
    ):
        """
        BPM予測MLPを初期化する。

        Args:
            input_dim: 入力特徴量の数
            hidden_dims: 隠れ層の次元数のリスト。デフォルト: [512, 256, 128]
            dropout_rates: 各層のドロップアウト率のリスト。デフォルト: [0.3, 0.2, 0.1]
            use_batch_norm: バッチ正規化を使用するかどうか
            activation: 活性化関数 ('relu', 'leaky_relu', 'elu')
        """
        super(BPMPredictor, self).__init__()

        # Default architecture optimized for tabular data
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        if dropout_rates is None:
            dropout_rates = [0.3, 0.2, 0.1]

        # Ensure dropout_rates match hidden_dims length
        if len(dropout_rates) != len(hidden_dims):
            dropout_rates = [dropout_rates[0]] * len(hidden_dims)

        self.use_batch_norm = use_batch_norm
        self.activation_name = activation

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.01, inplace=True))
            elif activation == "elu":
                layers.append(nn.ELU(inplace=True))
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/He初期化を使用してネットワークの重みを初期化する。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation_name == "relu":
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ネットワークの順伝播を実行する。

        Args:
            x: 形状(batch_size, input_dim)の入力テンソル

        Returns:
            形状(batch_size, 1)の予測BPM値
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        推論用の予測メソッド（squeeze された出力を返す）。

        Args:
            x: 形状(batch_size, input_dim)の入力テンソル

        Returns:
            形状(batch_size,)の予測BPM値
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x).squeeze()


class BPMPredictorSimple(nn.Module):
    """
    高速実験とベースライン比較用の簡略化MLP。

    高速な反復実験とデバッグのための最小限の2層MLP。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout_rate: float = 0.2):
        """
        簡単なBPM予測器を初期化する。

        Args:
            input_dim: 入力特徴量の数
            hidden_dim: 隠れ層の次元数
            dropout_rate: ドロップアウト率
        """
        super(BPMPredictorSimple, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播を実行する。"""
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推論用の予測メソッド。"""
        self.eval()
        with torch.no_grad():
            return self.forward(x).squeeze()


def create_bpm_model(
    input_dim: int,
    model_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    BMP予測モデルを作成するファクトリー関数。

    Args:
        input_dim: 入力特徴量の数
        model_type: モデルのタイプ ('standard', 'simple')
        **kwargs: モデルコンストラクタに渡される追加引数

    Returns:
        初期化されたPyTorchモデル
    """
    if model_type == "standard":
        return BPMPredictor(input_dim, **kwargs)
    elif model_type == "simple":
        # Filter kwargs for simple model (only accepts input_dim, hidden_dim, dropout_rate)
        simple_kwargs = {}
        if 'dropout_rates' in kwargs and kwargs['dropout_rates']:
            simple_kwargs['dropout_rate'] = kwargs['dropout_rates'][0]
        if 'hidden_dims' in kwargs and kwargs['hidden_dims']:
            simple_kwargs['hidden_dim'] = kwargs['hidden_dims'][0]
        return BPMPredictorSimple(input_dim, **simple_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation and forward pass
    input_dim = 26  # Example: current feature count after dimensionality reduction
    batch_size = 32

    # Test standard model
    model = create_bpm_model(input_dim, "standard")
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print(f"Standard model output shape: {output.shape}")

    # Test simple model
    simple_model = create_bpm_model(input_dim, "simple")
    simple_output = simple_model(x)
    print(f"Simple model output shape: {simple_output.shape}")

    # Print model architectures
    print("\n=== Standard Model ===")
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n=== Simple Model ===")
    print(simple_model)
    print(f"Parameters: {sum(p.numel() for p in simple_model.parameters()):,}")