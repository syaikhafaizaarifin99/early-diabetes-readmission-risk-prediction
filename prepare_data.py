from pathlib import Path
import pandas as pd
import numpy as np

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
raw_data_dir = data_dir / "raw"

diabetic_data_path = raw_data_dir / "diabetic_data.csv"
ids_mapping_path = raw_data_dir / "IDS_mapping.csv"

def load_data():
    """
    Load diabetic data and mapping data from the CSV file.

    """
    if not diabetic_data_path.exists():
        raise FileNotFoundError(f"Diabetic data file not found at {diabetic_data_path}")

    if not ids_mapping_path.exists():
        raise FileNotFoundError(f"IDS mapping file not found at {ids_mapping_path}")
    
    diabetic_data = pd.read_csv(diabetic_data_path)
    mapping_data = pd.read_csv(ids_mapping_path)

    return diabetic_data, mapping_data


def clean_data(data):
    """
    Clean the diabetic data by handling missing values and create binary target.
    
    """
    data = data.replace("?", np.nan)

    # create binary target variable: 1 (True) if readmitted within 30 days, else 0 (False)
    data['readmitted_binary'] = (data['readmitted'] == '<30').astype(int)

    return data


def remove_invalid_rows(data):
    """
    remove patients who cannot be included in readmission prediction

    """
    # discharge codes associated with patient death
    death_codes = [11, 19, 20, 21]

    filtered_data = data[
        ~data["discharge_disposition_id"].isin(death_codes)
    ].copy()

    return filtered_data


def select_features(data):
    """
    select numerical and categorical features for the model.
    """

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

    feature_columns = numerical_columns + categorical_columns

    x = data[feature_columns].copy()
    y = data["readmitted_binary"].copy()
    groups = data["patient_nbr"].copy()

    return (
        x,
        y,
        groups,
        numerical_columns,
        categorical_columns,
    )


def show_data_summary(data):
    """
    display summary statistics of the dataset.
    
    """
    print("\ndataset shape")
    print(data.shape)

    print("\nreadmission distribution")
    print(data["readmitted"].value_counts())

    print("\ntarget distribution")
    print(data["readmitted_binary"].value_counts())

    print("\ntarget percentage")
    print(
        data["readmitted_binary"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\ntop 10 columns with missing values")

    missing_percentage = (
        data.isnull()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print(missing_percentage.head(10))


def main():
     # load both datasets
    diabetes_data, mapping_data = load_data()

    print("datasets loaded successfully")

    # clean the dataset
    diabetes_data = clean_data(diabetes_data)

    print("\nrows before filtering")
    print(len(diabetes_data))

    # remove invalid rows
    diabetes_data = remove_invalid_rows(diabetes_data)

    print("\nrows after filtering")
    print(len(diabetes_data))

    # display dataset summary
    show_data_summary(diabetes_data)

    # select model features
    (
        x,
        y,
        groups,
        numerical_columns,
        categorical_columns,
    ) = select_features(diabetes_data)

    print("\nfeature data shape")
    print(x.shape)

    print("\ntarget data shape")
    print(y.shape)

    print("\npatient group shape")
    print(groups.shape)

    print("\nnumerical columns")
    print(numerical_columns)

    print("\ncategorical columns")
    print(categorical_columns)

    print("\nfirst five feature rows")
    print(x.head())


if __name__ == "__main__":
    main()
