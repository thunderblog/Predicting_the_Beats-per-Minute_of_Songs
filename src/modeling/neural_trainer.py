"""
Training and evaluation system for neural network models.

This module provides comprehensive training loop, evaluation, and model management
for PyTorch models in the BPM prediction task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from loguru import logger
from tqdm.auto import tqdm


@dataclass
class TrainingConfig:
    """Configuration class for neural network training."""

    # Model parameters
    model_type: str = "standard"
    hidden_dims: List[int] = None
    dropout_rates: List[float] = None
    activation: str = "relu"

    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 512
    epochs: int = 100
    patience: int = 15
    min_delta: float = 1e-4

    # Optimizer and scheduler
    optimizer_type: str = "adam"
    scheduler_type: str = "reduce_on_plateau"
    weight_decay: float = 1e-4

    # Data parameters
    scaler_type: str = "standard"
    validation_size: float = 0.2

    # System parameters
    device: str = "auto"
    random_seed: int = 42

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
        if self.dropout_rates is None:
            self.dropout_rates = [0.3, 0.2, 0.1]

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralTrainer:
    """
    Comprehensive trainer for neural network models.

    Handles training loop, validation, early stopping, model saving,
    and performance tracking.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_rmse": [],
            "val_rmse": [],
            "learning_rate": []
        }

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

        logger.info(f"Initialized NeuralTrainer with device: {self.device}")

    def _setup_model(self, input_dim: int) -> None:
        """Setup model architecture."""
        try:
            from .neural_models import create_bpm_model
        except ImportError:
            from neural_models import create_bpm_model

        self.model = create_bpm_model(
            input_dim=input_dim,
            model_type=self.config.model_type,
            hidden_dims=self.config.hidden_dims,
            dropout_rates=self.config.dropout_rates,
            activation=self.config.activation
        ).to(self.device)

        logger.info(f"Created {self.config.model_type} model with {input_dim} input features")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        if self.config.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")

        # Setup scheduler
        if self.config.scheduler_type.lower() == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True,
                min_lr=1e-7
            )
        elif self.config.scheduler_type.lower() == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-7
            )

        logger.info(f"Setup {self.config.optimizer_type} optimizer with {self.config.scheduler_type} scheduler")

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x).squeeze()
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / total_samples
        rmse = np.sqrt(avg_loss)

        return avg_loss, rmse

    def validate_epoch(self, val_loader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)

                batch_size = batch_x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        rmse = np.sqrt(avg_loss)

        return avg_loss, rmse

    def check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping condition."""
        if val_loss < self.best_val_loss - self.config.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_model_state = self.model.state_dict().copy()
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience

    def train(self, train_loader, val_loader) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training results dictionary
        """
        logger.info("Starting neural network training")
        start_time = time.time()

        # Setup model and optimizer
        input_dim = next(iter(train_loader))[0].shape[1]
        self._setup_model(input_dim)
        self._setup_optimizer()

        # Training loop
        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train and validate
            train_loss, train_rmse = self.train_epoch(train_loader)
            val_loss, val_rmse = self.validate_epoch(val_loader)

            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Track metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_rmse"].append(train_rmse)
            self.history["val_rmse"].append(val_rmse)
            self.history["learning_rate"].append(current_lr)

            # Logging
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )

            # Early stopping check
            if self.check_early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with validation RMSE: {np.sqrt(self.best_val_loss):.4f}")

        training_time = time.time() - start_time

        results = {
            "best_val_rmse": np.sqrt(self.best_val_loss),
            "final_train_rmse": self.history["train_rmse"][-1],
            "training_time_minutes": training_time / 60,
            "epochs_trained": len(self.history["train_loss"]),
            "history": self.history,
            "config": self.config.__dict__
        }

        logger.success(
            f"Training completed. Best validation RMSE: {results['best_val_rmse']:.4f} "
            f"in {training_time/60:.1f} minutes"
        )

        return results

    def predict(self, data_loader) -> np.ndarray:
        """
        Generate predictions using trained model.

        Args:
            data_loader: Data loader for prediction

        Returns:
            Predictions array
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_x, _ in tqdm(data_loader, desc="Predicting"):
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x).squeeze()
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def save_model(self, filepath: Path, include_optimizer: bool = False) -> None:
        """Save trained model to file."""
        if self.model is None:
            raise RuntimeError("No model to save")

        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "history": self.history,
            "best_val_loss": self.best_val_loss
        }

        if include_optimizer and self.optimizer is not None:
            save_dict["optimizer_state_dict"] = self.optimizer.state_dict()

        torch.save(save_dict, filepath)
        logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path, input_dim: int) -> None:
        """Load trained model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore config and setup model
        self.config = TrainingConfig(**checkpoint["config"])
        self._setup_model(input_dim)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", {})
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        logger.info(f"Model loaded from: {filepath}")


if __name__ == "__main__":
    # Test training system with dummy data
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from data_loaders import BPMDataProcessor
    import pandas as pd

    # Create dummy data for testing
    np.random.seed(42)
    dummy_df = pd.DataFrame({
        **{f"feature_{i}": np.random.randn(1000) for i in range(10)},
        "BeatsPerMinute": np.random.uniform(60, 180, 1000)
    })

    # Setup data
    processor = BPMDataProcessor(batch_size=64, device="cpu")
    data_dict = processor.prepare_data(dummy_df)

    # Setup training
    config = TrainingConfig(
        epochs=5,
        batch_size=64,
        patience=3,
        device="cpu",
        hidden_dims=[128, 64],
        dropout_rates=[0.2, 0.1]
    )

    trainer = NeuralTrainer(config)

    # Train
    results = trainer.train(data_dict["train_loader"], data_dict["val_loader"])

    print("=== Training Results ===")
    print(f"Best validation RMSE: {results['best_val_rmse']:.4f}")
    print(f"Training time: {results['training_time_minutes']:.1f} minutes")
    print(f"Epochs trained: {results['epochs_trained']}")

    # Test prediction
    predictions = trainer.predict(data_dict["val_loader"])
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.1f}, {predictions.max():.1f}]")