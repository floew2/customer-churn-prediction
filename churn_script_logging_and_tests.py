'''
Module containing unit tests for the churn_library.py functions.
It logs SUCCESS, ERROR, and INFO related messages.

Author: Fabian Löw
Date: 27 June 2023
'''

# Load libraries
import os
import logging
import churn_library as clb
from churn_library import DATA_PATH, CAT_COLUMNS, EDA_IMAGES_PATH, \
MODELS_PATH, RESULTS_PATH, RESPONSE, KEEP_COLUMNS

# Set up logging configuration
logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(data_path):
    '''
    Test test_import() function.
    '''
    try:
        data = clb.import_data(data_path)
        logging.info("Testing import_data: SUCCESS - Data imported.")
        return data
    except FileNotFoundError as err:
        logging.error("Testing import_eda: ERROR - The file wasn't imported.")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: ERROR - The file doesn't appear to have rows and columns.")
        raise err

def test_create_target(dataframe, response):
    '''
    Test create_target() function.
    '''
    try:
        data_with_target = clb.create_target(dataframe, response)
        logging.info("Testing create_target: SUCCESS - Created target 'churn'.")
        return data_with_target
    except FileNotFoundError as err:
        logging.error("Testing create_target: ERROR - The target wasn't created.")
        raise err

    try:
        assert response in data_with_target.columns.tolist()
        logging.info(f"Testing create_target: SUCCESS - code \
        generated {response} column.")
    except FileNotFoundError as err:
        logging.error("Testing create_target: ERROR - code did not \
        generate {response} column.")
        raise err


def test_eda(dataframe):
    '''
    Test test_eda() function.
    '''
    try:
        clb.perform_eda(dataframe)

        # Try open previously crerated images
        image_lst = [
            "Barplot_Marital_Status.png",
            "Heatmap_Correlation.png",
            "Histogram_Customer_Age.png",
            "Histogram_Existing_Customer.png",
            "Histplot_Kde_Total_Trans_Ct.png",
        ]
        for image in image_lst:
            assert os.path.exists(EDA_IMAGES_PATH + image)
            logging.info(f"Testing perform_eda: SUCCESS - Created {image}.")
    except AssertionError as err:
        logging.error(
            f"Testing perform_eda: ERROR - Could not found the {image} image")
        raise err


def test_encoder_helper(dataframe, category_lst, response):
    '''
    Test test_encoder_helper() function.
    '''

    new_columns = ['Gender_Churn',
                   'Education_Level_Churn',
                   'Marital_Status_Churn',
                   'Income_Category_Churn',
                   'Card_Category_Churn']

    try:
        dataframe = clb.encoder_helper(dataframe, category_lst, response)
        for column in new_columns:
            assert column in dataframe.columns.tolist()
            logging.info(f"SUCCESS: encoder generated {column} column.")
        return dataframe
    except AssertionError:
        logging.error("ERROR: One or more columns are missing from encoding.")


def test_perform_feature_engineering(dataframe, keep_columns, response):
    '''
    Test test_perform_feature_engineering() function.
    '''
    try:
        train_feature, testing_feature, train_label, test_label = clb.perform_feature_engineering(
            dataframe, keep_columns, response)
        assert len(train_feature) == len(train_label)
        assert len(testing_feature) == len(test_label)
        logging.info("Testing perform_feature_engineering: SUCCESS")
        return train_feature, testing_feature, train_label, test_label
    except AssertionError as err:
        logging.info("Testing perform_feature_engineering: \
           ERROR - Feature and Target should have the same lenght")
        raise err


def test_train_models(train_feature, testing_feature, train_label, test_label):
    '''
    Test test_train_models() function.
    '''
    try:
        clb.train_models(
            train_feature,
            testing_feature,
            train_label,
            test_label)
        image_lst = [
            "classification_report_LR.png",
            "classification_report_RF.png",
            "RF_feature_importance_plot.png",
            "roc_curve_result.png"
        ]
        for image in image_lst:
            assert os.path.exists(RESULTS_PATH + image)
        logging.info("Testing train_models: SUCCESS - Classification \
        reports and feature importance statistics calculated.")
    except AssertionError as err:
        logging.info(
            f"Testing train_models: The {image} is not found")
        raise err

    try:
        model_lst = ["logistic_model_v2.pkl", "rfc_model_v2.pkl"]
        for model in model_lst:
            assert os.path.exists(MODELS_PATH + model)
    except AssertionError as err:
        logging.info("Testing train_models: ERROR - could not found \
        all the models")
        raise err


if __name__ == "__main__":
    data_frame = test_import(DATA_PATH)
    data_frame = test_create_target(data_frame, RESPONSE)
    test_eda(data_frame)
    data_frame = test_encoder_helper(data_frame,
                                     CAT_COLUMNS,
                                     RESPONSE)
    x_training, x_testing, y_training, y_testing = test_perform_feature_engineering(data_frame,
                                                                                    KEEP_COLUMNS,
                                                                                    RESPONSE)
    test_train_models(x_training,
                      x_testing,
                      y_training,
                      y_testing)
