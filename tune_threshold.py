from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from model import ReadmissionModel as readmission_model


base_dir = Path(__file__).resolve().parent

processed_dir = base_dir / "data" / "processed"
models_dir = base_dir / "models"

validation_data_path = (
    processed_dir / "validation.csv"
)

model_path = (
    models_dir / "readmission_model.pth"
)

preprocessor_path = (
    models_dir / "preprocessor.pkl"
)


def load_validation_data():
    validation_data = pd.read_csv(
        validation_data_path
    )

    x_validation = validation_data.drop(
        columns=["readmitted_binary"]
    )

    y_validation = validation_data[
        "readmitted_binary"
    ].to_numpy(dtype=np.float32)

    return x_validation, y_validation


def preprocess_data(x_validation):
    preprocessor = joblib.load(
        preprocessor_path
    )

    x_validation_processed = (
        preprocessor
        .transform(x_validation)
        .astype(np.float32)
    )

    return x_validation_processed


def load_model(
    input_size,
    device,
):
    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model = readmission_model(
        input_size=input_size
    ).to(device)

    if (
        isinstance(checkpoint, dict)
        and "model_state" in checkpoint
    ):
        model_state = checkpoint["model_state"]

    elif (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model_state = checkpoint[
            "model_state_dict"
        ]

    else:
        model_state = checkpoint

    model.load_state_dict(model_state)

    model.eval()

    return model


def get_probabilities(
    model,
    x_validation,
    device,
):
    x_tensor = torch.tensor(
        x_validation,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():
        logits = model(x_tensor)

        probabilities = torch.sigmoid(
            logits
        )

    return probabilities.cpu().numpy()


def test_thresholds(
    targets,
    probabilities,
):
    thresholds = np.arange(
        0.30,
        0.81,
        0.05,
    )

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            targets,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            targets,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            targets,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    results = pd.DataFrame(results)

    return results


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    x_validation, y_validation = (
        load_validation_data()
    )

    x_validation_processed = (
        preprocess_data(x_validation)
    )

    model = load_model(
        input_size=(
            x_validation_processed.shape[1]
        ),
        device=device,
    )

    probabilities = get_probabilities(
        model,
        x_validation_processed,
        device,
    )

    results = test_thresholds(
        y_validation,
        probabilities,
    )

    print("\nthreshold results")
    print(
        results.to_string(
            index=False
        )
    )

    best_row = results.loc[
        results["f1"].idxmax()
    ]

    print("\nbest threshold based on f1")
    print(best_row)


if __name__ == "__main__":
    main()