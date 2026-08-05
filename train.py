from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import ReadmissionModel as readmission_model

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
raw_data_dir = data_dir / "raw"
processed_data_dir = data_dir / "processed"
models_dir = base_dir / "models"

diabetic_data_path = raw_data_dir / "diabetic_data.csv"
ids_mapping_path = raw_data_dir / "IDS_mapping.csv"

train_data_path = processed_data_dir / "train.csv"
validation_data_path = processed_data_dir / "validation.csv"

model_path = models_dir / "readmission_model.pth"
preprocessor_path = models_dir / "preprocessor.pkl"


numerical_columns = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


categorical_columns = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
]


def load_training_data():
    """
    load training and validation data.
    """

    if not train_data_path.exists():
        raise FileNotFoundError(
            f"file not found: {train_data_path}"
        )

    if not validation_data_path.exists():
        raise FileNotFoundError(
            f"file not found: {validation_data_path}"
        )

    train_data = pd.read_csv(train_data_path)
    validation_data = pd.read_csv(
        validation_data_path
    )

    return train_data, validation_data


def separate_features_and_target(data):
    """
    separate model features from the target.
    """

    x = data.drop(columns=["readmitted_binary"]).copy()

    y = data["readmitted_binary"].to_numpy(
        dtype=np.float32
    )

    return x, y


def create_preprocessor():
    """
    create preprocessing steps for numerical
    and categorical columns.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "missing_values",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "missing_values",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    return preprocessor


def preprocess_data(
    preprocessor,
    x_train,
    x_validation,
):
    """
    fit preprocessing only on training data
    and transform both datasets.
    """

    x_train_processed = (
        preprocessor
        .fit_transform(x_train)
        .astype(np.float32)
    )

    x_validation_processed = (
        preprocessor
        .transform(x_validation)
        .astype(np.float32)
    )

    return (
        x_train_processed,
        x_validation_processed,
    )


def create_data_loaders(
    x_train,
    y_train,
    x_validation,
    y_validation,
):
    """
    convert numpy arrays into pytorch data loaders.
    """

    x_train_tensor = torch.tensor(
        x_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    x_validation_tensor = torch.tensor(
        x_validation,
        dtype=torch.float32,
    )

    y_validation_tensor = torch.tensor(
        y_validation,
        dtype=torch.float32,
    )

    train_dataset = TensorDataset(
        x_train_tensor,
        y_train_tensor,
    )

    validation_dataset = TensorDataset(
        x_validation_tensor,
        y_validation_tensor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1024,
        shuffle=False,
    )

    return (
        train_loader,
        validation_loader,
    )


def calculate_positive_weight(y_train):
    """
    calculate a larger loss weight
    for the minority positive class.
    """

    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)

    if positive_count == 0:
        raise ValueError(
            "training data has no positive target"
        )

    positive_weight = (
        negative_count / positive_count
    )

    return float(positive_weight)


def collect_predictions(
    model,
    data_loader,
    device,
):
    """
    collect targets and predicted probabilities.
    """

    model.eval()

    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for features, targets in data_loader:
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


def train_model(
    model,
    train_loader,
    validation_loader,
    loss_function,
    optimizer,
    device,
    epochs,
    input_size,
):
    """
    train the model and save the best version.
    """

    best_pr_auc = -1.0

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            # remove gradients from the previous batch
            optimizer.zero_grad()

            # make predictions
            logits = model(features)

            # calculate prediction error
            loss = loss_function(
                logits,
                targets,
            )

            # calculate gradients
            loss.backward()

            # update model weights
            optimizer.step()

            total_loss += (
                loss.item() * features.size(0)
            )

        average_loss = (
            total_loss / len(train_loader.dataset)
        )

        (
            validation_targets,
            validation_probabilities,
        ) = collect_predictions(
            model,
            validation_loader,
            device,
        )

        validation_pr_auc = (
            average_precision_score(
                validation_targets,
                validation_probabilities,
            )
        )

        validation_roc_auc = roc_auc_score(
            validation_targets,
            validation_probabilities,
        )

        print(
            f"epoch {epoch + 1}/{epochs} | "
            f"loss: {average_loss:.4f} | "
            f"pr-auc: {validation_pr_auc:.4f} | "
            f"roc-auc: {validation_roc_auc:.4f}"
        )

        if validation_pr_auc > best_pr_auc:
            best_pr_auc = validation_pr_auc

            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "input_size": input_size,
                    "validation_pr_auc": (
                        validation_pr_auc
                    ),
                },
                model_path,
            )

            print("best model saved")

    return best_pr_auc


def main():
    # use the same random result on repeated runs
    np.random.seed(42)
    torch.manual_seed(42)

    # create the models folder
    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # choose gpu when available, otherwise cpu
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"device: {device}")

    # load prepared csv files
    train_data, validation_data = (
        load_training_data()
    )

    print("\ntraining data shape")
    print(train_data.shape)

    print("\nvalidation data shape")
    print(validation_data.shape)

    # separate features and targets
    x_train, y_train = (
        separate_features_and_target(
            train_data
        )
    )

    x_validation, y_validation = (
        separate_features_and_target(
            validation_data
        )
    )

    # create and run preprocessing
    preprocessor = create_preprocessor()

    (
        x_train_processed,
        x_validation_processed,
    ) = preprocess_data(
        preprocessor,
        x_train,
        x_validation,
    )

    print("\nprocessed training shape")
    print(x_train_processed.shape)

    print("\nprocessed validation shape")
    print(x_validation_processed.shape)

    # save preprocessing rules
    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    print("\npreprocessor saved")

    # create pytorch data loaders
    (
        train_loader,
        validation_loader,
    ) = create_data_loaders(
        x_train_processed,
        y_train,
        x_validation_processed,
        y_validation,
    )

    # get the number of processed input columns
    input_size = x_train_processed.shape[1]

    # create the neural network
    model = readmission_model(
        input_size=input_size
    ).to(device)

    print("\nmodel")
    print(model)

    # calculate weight for the minority class
    positive_weight = (
        calculate_positive_weight(y_train)
    )

    print("\npositive weight")
    print(round(positive_weight, 4))

    positive_weight_tensor = torch.tensor(
        positive_weight,
        dtype=torch.float32,
        device=device,
    )

    # create the loss function
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight_tensor
    )

    # create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # train the model
    best_pr_auc = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        epochs=15,
        input_size=input_size,
    )

    print("\ntraining completed")
    print(
        f"best validation pr-auc: "
        f"{best_pr_auc:.4f}"
    )

    print("\nmodel file")
    print(model_path)

    print("\npreprocessor file")
    print(preprocessor_path)


if __name__ == "__main__":
    main()