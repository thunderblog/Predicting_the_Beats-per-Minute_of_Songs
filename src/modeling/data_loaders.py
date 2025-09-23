"""
Data loading and preprocessing utilities for neural network training.

This module provides PyTorch-compatible data loaders and preprocessing
pipelines for the BPM prediction task.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from loguru import logger


class BPMDataset(Dataset):
    """
    PyTorch Dataset for BPM prediction task.

    Handles feature-target pair creation and tensor conversion.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        device: str = "cpu"
    ):
        """
        Initialize BPM dataset.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            targets: Target values of shape (n_samples,). None for test set.
            device: Device to place tensors on ('cpu' or 'cuda')
        """
        self.features = torch.FloatTensor(features).to(device)
        self.targets = None
        if targets is not None:
            self.targets = torch.FloatTensor(targets).to(device)
        self.device = device

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        else:
            # For test set (no targets)
            return self.features[idx], torch.tensor(0.0, device=self.device)


class BPMDataProcessor:
    """
    Data preprocessing pipeline for neural network training.

    Handles scaling, splitting, and DataLoader creation.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        batch_size: int = 512,
        validation_size: float = 0.2,
        random_state: int = 42,
        device: str = "cpu"
    ):
        """
        Initialize data processor.

        Args:
            scaler_type: Type of scaler ('standard', 'robust', 'minmax', 'none')
            batch_size: Batch size for DataLoaders
            validation_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            device: Device for tensors ('cpu' or 'cuda')
        """
        self.scaler_type = scaler_type
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.device = device

        # Initialize scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        self.is_fitted = False
        logger.info(f"Initialized BPMDataProcessor with {scaler_type} scaler")

    def prepare_data(
        self,
        train_df: pd.DataFrame,
        target_col: str = "BeatsPerMinute",
        test_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Prepare training, validation, and test data.

        Args:
            train_df: Training DataFrame with features and target
            target_col: Name of target column
            test_df: Optional test DataFrame (features only)

        Returns:
            Dictionary containing DataLoaders and metadata
        """
        logger.info("Preparing data for neural network training")

        # Separate features and target
        feature_cols = [col for col in train_df.columns if col != target_col]
        X = train_df[feature_cols].values
        y = train_df[target_col].values

        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

        # Split training data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.validation_size,
            random_state=self.random_state,
            shuffle=True
        )

        # Fit and transform scaler on training data
        if self.scaler is not None:
            logger.info(f"Fitting {self.scaler_type} scaler on training data")
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            self.is_fitted = True
        else:
            logger.info("No scaling applied")

        # Create datasets
        train_dataset = BPMDataset(X_train, y_train, self.device)
        val_dataset = BPMDataset(X_val, y_val, self.device)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,  # For stable batch norm
            pin_memory=(self.device == "cuda")
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=(self.device == "cuda")
        )

        result = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "input_dim": X_train.shape[1],
            "train_size": len(X_train),
            "val_size": len(X_val),
            "feature_names": feature_cols
        }

        # Handle test data if provided
        if test_df is not None:
            X_test = test_df[feature_cols].values
            if self.scaler is not None:
                X_test = self.scaler.transform(X_test)

            test_dataset = BPMDataset(X_test, device=self.device)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=(self.device == "cuda")
            )

            result["test_loader"] = test_loader
            result["test_size"] = len(X_test)

        logger.success(
            f"Data preparation complete. "
            f"Train: {result['train_size']}, Val: {result['val_size']}, "
            f"Features: {result['input_dim']}"
        )

        return result

    def transform_new_data(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """
        Transform new data using fitted scaler.

        Args:
            df: DataFrame to transform
            feature_cols: List of feature column names

        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted and self.scaler is not None:
            raise RuntimeError("Scaler not fitted. Call prepare_data first.")

        X = df[feature_cols].values
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    def get_scaler_info(self) -> Dict[str, Any]:
        """Get information about the fitted scaler."""
        if not self.is_fitted or self.scaler is None:
            return {"scaler_type": self.scaler_type, "fitted": False}

        info = {
            "scaler_type": self.scaler_type,
            "fitted": True
        }

        if hasattr(self.scaler, "mean_"):
            info["mean"] = self.scaler.mean_
        if hasattr(self.scaler, "scale_"):
            info["scale"] = self.scaler.scale_
        if hasattr(self.scaler, "center_"):
            info["center"] = self.scaler.center_

        return info


def load_processed_data(
    data_dir: Path,
    train_file: str = "train_features.csv",
    test_file: str = "test_features.csv"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load processed data from CSV files.

    Args:
        data_dir: Directory containing processed data files
        train_file: Name of training data file
        test_file: Name of test data file

    Returns:
        Tuple of (train_df, test_df). test_df is None if file doesn't exist.
    """
    train_path = data_dir / train_file
    test_path = data_dir / test_file

    logger.info(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)

    test_df = None
    if test_path.exists():
        logger.info(f"Loading test data from: {test_path}")
        test_df = pd.read_csv(test_path)
    else:
        logger.warning(f"Test file not found: {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    # Test data processing pipeline
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import DATA_DIR

    try:
        # Load sample data for testing
        processed_dir = DATA_DIR / "processed"
        train_df, test_df = load_processed_data(processed_dir)

        # Initialize processor
        processor = BPMDataProcessor(
            scaler_type="standard",
            batch_size=128,
            validation_size=0.2,
            device="cpu"
        )

        # Prepare data
        data_dict = processor.prepare_data(train_df, test_df=test_df)

        print("=== Data Preparation Results ===")
        for key, value in data_dict.items():
            if not isinstance(value, (DataLoader, list)):
                print(f"{key}: {value}")

        # Test a batch
        train_loader = data_dict["train_loader"]
        for batch_x, batch_y in train_loader:
            print(f"\nBatch shapes - Features: {batch_x.shape}, Targets: {batch_y.shape}")
            print(f"Feature range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
            print(f"Target range: [{batch_y.min():.3f}, {batch_y.max():.3f}]")
            break

        # Scaler info
        scaler_info = processor.get_scaler_info()
        print(f"\nScaler info: {scaler_info}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        # Create dummy data for testing if real data not available
        np.random.seed(42)
        dummy_df = pd.DataFrame({
            **{f"feature_{i}": np.random.randn(1000) for i in range(10)},
            "BeatsPerMinute": np.random.uniform(60, 180, 1000)
        })

        processor = BPMDataProcessor(batch_size=64)
        data_dict = processor.prepare_data(dummy_df)
        print("=== Dummy Data Test ===")
        print(f"Input dim: {data_dict['input_dim']}")
        print(f"Train size: {data_dict['train_size']}")
        print(f"Val size: {data_dict['val_size']}")