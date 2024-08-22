# churn_script_logging_and_tests.py

"""
Module containing unit tests for the churn_library.py functions.
It logs ERROR and INFO related messages.

Author: Fabian Löw
Date: 29 June 2024
"""

import os
import logging
import joblib
import pandas as pd
import pytest
import constants as const
from churn_library import ChurnLibrary

# Set up a logging configuration
logging.basicConfig(
    filename='logs/churn_library_tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')

churn_lib = ChurnLibrary(const)


@pytest.fixture(scope="module")
def data_set():
    """Fixture for loading data."""
    df = churn_lib.import_data(const.DATA_PATH, const.KEEP_COLUMNS)
    df = df.sample(frac=0.1, random_state=42)
    return df


@pytest.fixture(scope="module")
def target_data_set(data_set):
    """Fixture for creating target data."""
    return churn_lib.create_target(data_set)


@pytest.fixture(scope="module")
def encoded_data_set(target_data_set):
    """Fixture for encoding categorical variables."""
    return churn_lib.encoder_helper(target_data_set, const.CAT_COLUMNS)


@pytest.fixture(scope="module")
def feature_engineered_data_set(encoded_data_set):
    """Fixture for performing feature engineering."""
    return churn_lib.perform_feature_engineering(encoded_data_set, const.RESPONSE)


@pytest.fixture(scope="module")
def models(feature_engineered_data_set):
    """Fixture for training models and loading them."""
    X_train, X_test, y_train, y_test = feature_engineered_data_set
    churn_lib.train_models(X_train, X_test, y_train, y_test)
    rfc_model = joblib.load(const.MODELS_PATH + 'rfc_model.pkl')
    lr_model = joblib.load(const.MODELS_PATH + 'logistic_model.pkl')
    return rfc_model, lr_model, X_train, X_test, y_train, y_test


def test_import(data_set):
    """
    Test import_data() function.
    """
    try:
        logging.info("Testing import_data: SUCCESS - Data imported.")
        assert data_set.shape[0] > 0 and data_set.shape[1] > 0
        assert all(column in data_set.columns for column in const.KEEP_COLUMNS)
        assert data_set.dtypes['Customer_Age'] == 'int64'
        assert not data_set.isnull().values.any(), "Data contains missing values."
    except FileNotFoundError as err:
        logging.error("Testing import_data: ERROR - The file wasn't imported.")
        raise err
    except AssertionError as err:
        logging.error(
            "Testing import_data: ERROR - Data validation failed.")
        raise err


def test_create_target(data_set, target_data_set):
    """
    Test create_target() function.
    """
    try:
        logging.info(
            "Testing create_target: SUCCESS - Created target 'churn'.")
        assert const.RESPONSE in target_data_set.columns.tolist()
        assert len(data_set) == len(target_data_set)
        assert target_data_set[const.RESPONSE].nunique() == 2, "Target variable should have two unique values."
    except KeyError as err:
        logging.error(
            "Testing create_target: ERROR - Response column not found.")
        raise err
    except AssertionError as err:
        logging.error(
            "Testing create_target: ERROR - Target variable not created or data length mismatch.")
        raise err


def test_perform_eda(target_data_set):
    """
    Test perform_eda() function.
    """
    try:
        churn_lib.perform_eda(target_data_set, const.CAT_COLUMNS)
        image_lst = [
            "Churn_Distribution.png",
            "Customer_Age_Distribution.png",
            "Correlation_Heatmap.png",
            "Marital_Status_Distribution.png",
            "Total_Transaction_Distribution.png"
        ]
        for image in image_lst:
            path = os.path.join(const.EDA_IMAGES_PATH, image)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            logging.info(f"SUCCESS: {image} created and saved.")
    except AssertionError as err:
        logging.error(f"ERROR: {image} not found or empty.")
        raise err
    except Exception as err:
        logging.error(f"ERROR: perform_eda failed with error: {str(err)}")
        raise err


def test_encoder_helper(encoded_data_set):
    """
    Test encoder_helper() function.
    """
    try:
        # Ensure all values in the encoded DataFrame are numerical
        for column in encoded_data_set.columns:
            assert pd.api.types.is_numeric_dtype(encoded_data_set[column]), \
                f"Column '{column}' contains non-numerical values."

        # Ensure all original categorical columns have their corresponding encoded columns
        for col in const.CAT_COLUMNS:
            assert any(col in col_name for col_name in encoded_data_set.columns), \
                f"Encoded column for '{col}' not found."

        logging.info("Testing encoder_helper: SUCCESS - Only numerical values present.")
    except AssertionError as err:
        logging.error("Testing encoder_helper: ERROR - Non-numerical values found in encoded data.")
        raise err
    except Exception as err:
        logging.error(f"Testing encoder_helper: ERROR - {str(err)}")
        raise err


def test_perform_feature_engineering(feature_engineered_data_set):
    """
    Test perform_feature_engineering() function.
    """
    try:
        X_train, X_test, y_train, y_test = feature_engineered_data_set
        assert X_train.shape[0] > 0 and X_test.shape[0] > 0
        assert y_train.shape[0] > 0 and y_test.shape[0] > 0
        assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch between train and test sets."
        logging.info(
            "Testing perform_feature_engineering: SUCCESS - Data split.")
    except KeyError as err:
        logging.error(
            "Testing perform_feature_engineering: ERROR - Column not found.")
        raise err
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: ERROR - Data not split correctly.")
        raise err


def test_classification_report_image():
    """
    Test classification_report_image() function.
    """
    try:
        assert os.path.exists(
            os.path.join(const.RESULTS_PATH, 'logistic_model_classification_report.png'))
        assert os.path.exists(
            os.path.join(const.RESULTS_PATH, 'rfc_model_classification_report.png'))
        logging.info(
            "Testing classification_report_image: SUCCESS - Class. report images created.")
    except AssertionError as err:
        logging.error(
            "Testing classification_report_image: ERROR - Class. report images not created.")
        raise err



def test_feature_importance_plot(models):
    """
    Test feature_importance_plot() function.
    """
    try:
        rfc_model, _, X_train, _, _, _ = models
        churn_lib.feature_importance_plot(rfc_model, X_train)
        assert os.path.exists(
            os.path.join(const.RESULTS_PATH, 'random_forest_feature_importance.png'))
        logging.info(
            "Testing feature_importance_plot: SUCCESS - Feature importance plot created.")
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: ERROR - Feature importance plot not created.")
        raise err



def test_train_models(models):
    """
    Test train_models() function.
    """
    try:
        assert os.path.exists(
            os.path.join(const.MODELS_PATH, 'rfc_model.pkl'))
        assert os.path.exists(
            os.path.join(const.MODELS_PATH, 'logistic_model.pkl'))
        logging.info(
            "Testing train_models: SUCCESS - Models trained and saved.")
    except AssertionError as err:
        logging.error("Testing train_models: ERROR - Models not saved.")
        raise err


if __name__ == '__main__':
    pytest.main()
