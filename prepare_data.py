from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
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


def split_data(x, y, groups):
    """
    split the dataset by patient (groups) into training, validation, and testing data.
    
    """
    # split the  data into training and temporary data
    first_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
    random_state=42,
    )

    train_index, temporary_index = next(
        first_splitter.split(
            x,
            y,
            groups=groups,
        )
    )

    x_train = x.iloc[train_index].copy()
    y_train = y.iloc[train_index].copy()
    groups_train = groups.iloc[train_index].copy()

    x_temporary = x.iloc[temporary_index].copy()
    y_temporary = y.iloc[temporary_index].copy()
    groups_temporary = groups.iloc[temporary_index].copy()

    # split the temporary data into validation and testing data
    second_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42,
    )

    validation_index, test_index = next(
        second_splitter.split(
            x_temporary,
            y_temporary,
            groups=groups_temporary,
        )
    )

    x_validation = x_temporary.iloc[validation_index].copy()
    y_validation = y_temporary.iloc[validation_index].copy()
    groups_validation = groups_temporary.iloc[
        validation_index
    ].copy()

    x_test = x_temporary.iloc[test_index].copy()
    y_test = y_temporary.iloc[test_index].copy()
    groups_test = groups_temporary.iloc[test_index].copy()

    return (
        x_train,
        y_train,
        groups_train,
        x_validation,
        y_validation,
        groups_validation,
        x_test,
        y_test,
        groups_test,
    )


def show_split_summary(
    x_train,
    y_train,
    groups_train,
    x_validation,
    y_validation,
    groups_validation,
    x_test,
    y_test,
    groups_test,
):
    """
    display the data split summary and check patient overlap.
    """

    print("\ntraining data shape")
    print(x_train.shape)

    print("\nvalidation data shape")
    print(x_validation.shape)

    print("\ntesting data shape")
    print(x_test.shape)

    print("\ntraining target percentage")
    print(
        y_train
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nvalidation target percentage")
    print(
        y_validation
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\ntesting target percentage")
    print(
        y_test
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    train_patients = set(groups_train)
    validation_patients = set(groups_validation)
    test_patients = set(groups_test)

    train_validation_overlap = (
        train_patients.intersection(validation_patients)
    )

    train_test_overlap = (
        train_patients.intersection(test_patients)
    )

    validation_test_overlap = (
        validation_patients.intersection(test_patients)
    )

    print("\npatient overlap between training and validation")
    print(len(train_validation_overlap))

    print("\npatient overlap between training and testing")
    print(len(train_test_overlap))

    print("\npatient overlap between validation and testing")
    print(len(validation_test_overlap))


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
        categorical_columns ) = select_features(diabetes_data)

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

    (
        x_train,
        y_train,
        groups_train,
        x_validation,
        y_validation,
        groups_validation,
        x_test,
        y_test,
        groups_test,
    ) = split_data(
        x,
        y,
        groups,
    )

    show_split_summary(
        x_train,
        y_train,
        groups_train,
        x_validation,
        y_validation,
        groups_validation,
        x_test,
        y_test,
        groups_test,
    )

if __name__ == "__main__":
    main()
