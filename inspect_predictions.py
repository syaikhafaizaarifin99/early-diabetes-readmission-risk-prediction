from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

processed_data_dir = data_dir / "processed"
predictions_path = processed_data_dir / "test_predictions.csv"

def main():
    predictions = pd.read_csv(predictions_path)

    print("\nprediction data shape")
    print(predictions.shape)

    print("\nfirst five predictions")
    print(
        predictions[
            [
                "readmitted_binary",
                "predicted_probability",
                "predicted_target",
            ]
        ].head()
    )

    print("\nprediction distribution")
    print(
        predictions[
            "predicted_target"
        ].value_counts()
    )

    print("\naverage predicted probability")
    print(
        predictions[
            "predicted_probability"
        ].mean()
    )

    false_negatives = predictions[
        (predictions["readmitted_binary"] == 1)
        & (predictions["predicted_target"] == 0)
    ]

    false_positives = predictions[
        (predictions["readmitted_binary"] == 0)
        & (predictions["predicted_target"] == 1)
    ]

    correct_predictions = predictions[
        predictions["readmitted_binary"]
        == predictions["predicted_target"]
    ]

    print("\ncorrect predictions")
    print(len(correct_predictions))

    print("\nfalse negatives")
    print(len(false_negatives))

    print("\nfalse positives")
    print(len(false_positives))

    print("\nhighest risk patients")
    print(
        predictions
        .sort_values(
            by="predicted_probability",
            ascending=False,
        )
        [
            [
                "readmitted_binary",
                "predicted_probability",
                "predicted_target",
            ]
        ]
        .head(10)
    )


if __name__ == "__main__":
    main()