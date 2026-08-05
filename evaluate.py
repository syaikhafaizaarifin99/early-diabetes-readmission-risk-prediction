from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from model import readmission_model

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

processed_data_dir = data_dir / "processed"
models_dir = base_dir / "models"

test_data_path = processed_data_dir / "test.csv"
model_path = models_dir / "readmission_model.pth"
preprocessor_path = models_dir / "preprocessor.pkl"

# define output file path
predictions_path = (
    processed_data_dir / "test_predictions.csv"
)


def check_required_files():
    """
    check whether all required files exist.
    """

    required_files = [
        test_data_path,
        model_path,
        preprocessor_path,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"file not found: {file_path}"
            )


def load_test_data():
    """
    load the testing dataset.
    """

    test_data = pd.read_csv(test_data_path)

    return test_data


def separate_features_and_target(data):
    """
    separate model features from the target.
    """

    x_test = data.drop(
        columns=["readmitted_binary"]
    ).copy()

    y_test = data["readmitted_binary"].to_numpy(
        dtype=np.float32
    )

    return x_test, y_test


def main():
    # check all required files
    check_required_files()

    # choose gpu when available, otherwise cpu
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"device: {device}")

    # load testing data
    test_data = load_test_data()

    print("\ntesting data shape")
    print(test_data.shape)

    # separate features and target
    x_test, y_test = (
        separate_features_and_target(
            test_data
        )
    )


if __name__ == "__main__":
    main()