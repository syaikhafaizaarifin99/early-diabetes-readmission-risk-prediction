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

from model import ReadmissionModel as readmission_model

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
        columns=["target"]
    ).copy()

    y_test = data["target"].to_numpy(
        dtype=np.float32
    )

    return x_test, y_test


def preprocess_test_data(x_test):
    """
    load the saved preprocessor
    and transform testing features.
    """

    preprocessor = joblib.load(
        preprocessor_path
    )

    x_test_processed = (
        preprocessor
        .transform(x_test)
        .astype(np.float32)
    )

    return x_test_processed


def create_test_loader(
    x_test_processed,
    y_test,
):
    """
    convert testing data into a pytorch data loader.
    """

    x_test_tensor = torch.tensor(
        x_test_processed,
        dtype=torch.float32,
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.float32,
    )

    test_dataset = TensorDataset(
        x_test_tensor,
        y_test_tensor,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
    )

    return test_loader


def load_trained_model(device):
    """
    load the saved model checkpoint.
    """

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    input_size = checkpoint["input_size"]

    model = readmission_model(
        input_size=input_size
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return model, checkpoint


def collect_predictions(
    model,
    test_loader,
    device,
):
    """
    collect actual targets and predicted probabilities.
    """

    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)

            logits = model(features)

            probabilities = torch.sigmoid(
                logits
            )

            all_targets.extend(
                targets.numpy()
            )

            all_probabilities.extend(
                probabilities
                .cpu()
                .numpy()
            )

    return (
        np.array(all_targets),
        np.array(all_probabilities),
    )


def evaluate_predictions(
    targets,
    probabilities,
    threshold=0.5,
):
    """
    calculate and display evaluation metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    pr_auc = average_precision_score(
        targets,
        probabilities,
    )

    roc_auc = roc_auc_score(
        targets,
        probabilities,
    )

    result_matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )

    true_negative = result_matrix[0, 0]
    false_positive = result_matrix[0, 1]
    false_negative = result_matrix[1, 0]
    true_positive = result_matrix[1, 1]

    print("\nevaluation threshold")
    print(threshold)

    print("\naccuracy")
    print(round(accuracy, 4))

    print("\npr-auc")
    print(round(pr_auc, 4))

    print("\nroc-auc")
    print(round(roc_auc, 4))

    print("\nclassification report")
    print(
        classification_report(
            targets,
            predictions,
            labels=[0, 1],
            target_names=[
                "not readmitted within 30 days",
                "readmitted within 30 days",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("\nconfusion matrix")
    print(result_matrix)

    print("\ntrue negative")
    print(true_negative)

    print("\nfalse positive")
    print(false_positive)

    print("\nfalse negative")
    print(false_negative)

    print("\ntrue positive")
    print(true_positive)

    return predictions


def save_predictions(
    test_data,
    probabilities,
    predictions,
):
    """
    save predicted probabilities and classes.
    """

    prediction_data = test_data.copy()

    prediction_data[
        "predicted_probability"
    ] = probabilities

    prediction_data[
        "predicted_target"
    ] = predictions

    prediction_data.to_csv(
        predictions_path,
        index=False,
    )

    print("\npredictions saved")
    print(predictions_path)


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

    # preprocess testing features
    x_test_processed = (
        preprocess_test_data(x_test)
    )

    print("\nprocessed testing shape")
    print(x_test_processed.shape)

    # create testing data loader
    test_loader = create_test_loader(
        x_test_processed,
        y_test,
    )

    # load the trained model
    model, checkpoint = (
        load_trained_model(device)
    )

    print("\nsaved validation pr-auc")
    print(
        round(
            checkpoint[
                "validation_pr_auc"
            ],
            4,
        )
    )

    # collect model predictions
    (
        test_targets,
        test_probabilities,
    ) = collect_predictions(
        model,
        test_loader,
        device,
    )

    # calculate evaluation results
    test_predictions = (
        evaluate_predictions(
            targets=test_targets,
            probabilities=test_probabilities,
            threshold=0.5,
        )
    )

    # save prediction results
    save_predictions(
        test_data,
        test_probabilities,
        test_predictions,
    )


if __name__ == "__main__":
    main()