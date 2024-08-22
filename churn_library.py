# churn_library.py

"""
Module containing functions to process and analyze customer churn data.
Functions are organized within the ChurnLibrary class.

Author: Fabian Löw
Date: 28.06.2024
"""

# Load standard Python modules
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
import joblib
import pandas as pd
import numpy as np

# Load machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV

# Load modules for visualization
import shap
import matplotlib
from typing import List, Tuple

matplotlib.use('Agg')

# Load constants from project module
import constants as const

# Define class containing the complete pipeline
class ChurnLibrary:
    """
    ChurnLibrary class containing methods for data import, processing,
    EDA, encoding, feature engineering, model training, and creating visualizations.
    """

    def __init__(self,
                 constants: object) -> None:
        """
        Initialize the ChurnLibrary with constants.

        Parameters:
        - constants (object): module containing constants used throughout the class
        """
        self.constants = constants
        self.model_and_params = {}
        self.best_models = {}
        self.accuracies = {}
        self.accuracy = None
        self.feature_names = None

        # Initialize the logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # Create a file handler and set the log level
        file_handler = logging.FileHandler('logs/churn_library.log')
        file_handler.setLevel(logging.INFO)

        # Define the log format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(file_handler)

    def import_data(self,
                    data_path: str,
                    keep_columns: List[str]) -> pd.DataFrame:
        """
        Import data from csv file.

        Parameters:
        - data_path (str): Path to the data file.
        - keep_columns (list[str]): List of columns to keep.

        Returns:
        - pd.DataFrame: DataFrame containing the imported data.

        Raises:
        - FileNotFoundError: If the file at data_path is not found.
        - AssertionError: If data_path is not a valid file path.
        """
        assert os.path.exists(
            data_path), f"{data_path} is not a valid file path."

        try:
            data = pd.read_csv(data_path)
            data = data[keep_columns]

            assert not data.empty, "DataFrame is empty"
            self.logger.info(
                "SUCCESS (import_data): Data imported from %s",
                data_path)
        except FileNotFoundError as err:
            self.logger.error(
                "ERROR (import_data): File not found %s",
                data_path)
            raise err
        else:
            return data

    def create_target(self,
                      dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing the data.

        Returns:
        - pd.DataFrame: DataFrame with the target variable created.

        Raises:
        - KeyError: If the required columns for target creation are not found.
        - AssertionError: If dataframe is not a Pandas DataFrame.
        """
        assert isinstance(
            dataframe, pd.DataFrame), "Variable 'dataframe' is not a Pandas DataFrame."

        try:
            # Create a new 'Churn' column
            dataframe[self.constants.RESPONSE] = dataframe[self.constants.RESPONSE_OLD].apply(
                lambda val: 0 if val == "Existing Customer" else 1)
            dataframe.drop(self.constants.RESPONSE_OLD, axis=1, inplace=True)
            self.logger.info(
                "SUCCESS (create_target): Target variable %s created",
                self.constants.RESPONSE)
        except KeyError as err:
            self.logger.error(
                "ERROR (create_target): Column %s does not exist",
                self.constants.RESPONSE)
            raise err
        else:
            return dataframe

    def perform_eda(self,
                    dataframe: pd.DataFrame,
                    categorical: List[str]) -> None:
        """
        Perform EDA and save figures.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing the data.
        - categorical (list[str]): List of categorical columns.

        Returns:
        - None

        Raises:
        - AssertionError: If dataframe is not a DataFrame or categorical is not a valid list.
        - Exception: If an error occurs during EDA.
        """
        assert isinstance(
            dataframe, pd.DataFrame), "Variable 'dataframe' is not a Pandas DataFrame."
        assert isinstance(categorical, list), "Variable is not a list."
        assert all(isinstance(item, str)
                   for item in categorical), "List contains non-string elements."
        assert len(categorical) >= 1, "List does not have at least one element."

        try:
            # Churn distribution
            plt.figure(figsize=(10, 5))
            dataframe[self.constants.RESPONSE].value_counts().plot(kind='bar')
            plt.title('Churn Distribution', fontsize=18)
            plt.xlabel('Churn')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(
                self.constants.EDA_IMAGES_PATH +
                'Churn_Distribution.png')
            plt.close()
            self.logger.info("...SUCCESS: Churn distribution plot saved")

            # Customer age distribution
            plt.figure(figsize=(10, 5))
            sns.histplot(dataframe['Customer_Age'], kde=True)
            plt.title('Customer Age Distribution', fontsize=18)
            plt.xlabel('Customer Age')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(
                self.constants.EDA_IMAGES_PATH +
                'Customer_Age_Distribution.png')
            plt.close()
            self.logger.info(
                "...SUCCESS: Customer age distribution plot saved")

            # Heatmap of all numeric variables
            plt.figure(figsize=(15, 10))
            df_excluded = dataframe.drop(categorical + ['Churn'], axis=1)
            sns.heatmap(
                df_excluded.corr(),
                annot=True,
                cmap='coolwarm',
                linewidths=2)
            plt.title('Correlation of Variables', fontsize=18)
            plt.tight_layout()
            plt.savefig(
                self.constants.EDA_IMAGES_PATH +
                'Correlation_Heatmap.png')
            plt.close()
            self.logger.info("...SUCCESS: Correlation heatmap plot saved")

            # Marital status distribution
            plt.figure(figsize=(10, 5))
            dataframe['Marital_Status'].value_counts().plot(kind='bar')
            plt.title('Marital Status Distribution', fontsize=18)
            plt.xlabel('Marital Status')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(
                self.constants.EDA_IMAGES_PATH +
                'Marital_Status_Distribution.png')
            plt.close()
            self.logger.info(
                "...SUCCESS: Marital status distribution plot saved")

            # Total transaction distribution
            plt.figure(figsize=(10, 5))
            sns.histplot(dataframe['Total_Trans_Ct'], kde=True)
            plt.title('Total Transaction Distribution', fontsize=18)
            plt.xlabel('Total Transactions')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(
                self.constants.EDA_IMAGES_PATH +
                'Total_Transaction_Distribution.png')
            plt.close()
            self.logger.info(
                "...SUCCESS: Total transaction distribution plot saved")
        except Exception as err:
            self.logger.error(
                "ERROR (perform_eda): Could not save EDA plot(s): %s",
                str(err))
            raise err

    def encoder_helper(self,
                       dataframe: pd.DataFrame,
                       category_lst: List[str]) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot-encoding.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing the data.
        - category_lst (list[str]): List of categorical columns to be encoded.

        Returns:
        - pd.DataFrame: DataFrame with encoded categorical variables.

        Raises:
        - KeyError: If encoding fails for any category.
        """
        try:
            for category in category_lst:
                encoded_cols = pd.get_dummies(
                    dataframe[category], prefix=category + '_enc').astype(int)
                dataframe = pd.concat([dataframe, encoded_cols], axis=1)

                # Remove original column
                dataframe.drop(columns=[category], inplace=True)

                # Store feature variable names for later use, e.g. in random
                # forest feature importance plots
                self.feature_names = [
                    col for col in dataframe if col != self.constants.RESPONSE]

                self.logger.info(
                    "SUCCESS (encoder_helper): One-hot encoded and removed %s", category)
        except KeyError as err:
            self.logger.error(
                "ERROR (encoder_helper): Encoding failed for %s", category)
            raise err
        else:
            return dataframe

    def perform_feature_engineering(self,
                                    dataframe: pd.DataFrame,
                                    response: str,
                                    test_size: float = 0.3) -> Tuple[np.ndarray,
                                                                     np.ndarray,
                                                                     np.ndarray,
                                                                     np.ndarray]:
        """
        Split data into separate train and test sets and apply feature scaling.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing the features.
        - response (str): Name of the target variable.
        - test_size (float): Proportion of the test data

        Returns:
        - tuple: Containing train-test split arrays (X_train, X_test, y_train, y_test).

        Raises:
        - KeyError: If feature scaling or train-test split fails.
        - AssertionError: If test_size is not float or not larger than zero and smaller than 1.0
        """
        assert isinstance(test_size, float), "test_size must be a float."
        assert 0.0 < test_size < 1.0, "test_size must be greater than 0.0 and less than 1.0."

        try:
            X = dataframe.drop(columns=[response])
            y = dataframe[response]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            self.logger.info(
                "SUCCESS (perform_feature_engineering): Feature scaling & train-test split completed")
        except KeyError as err:
            self.logger.error(
                "ERROR (perform_feature_engineering): Feature scaling failed")
            raise err
        else:
            return X_train_scaled, X_test_scaled, y_train, y_test

    def _calculate_accuracy(self,
                            y_test: np.ndarray,
                            y_test_preds: np.ndarray) -> float:
        """
        Calculate overall classification accuracy based on test data.

        Parameters:
        - y_test (np.ndarray): Test labels.
        - y_test_preds (np.ndarray): Predictions on test set.

        Returns:
        - float: Overall classification accuracy based on test data

        Raises:
        - Exception: If the classification report generation fails.
        """
        try:
            self.accuracy = accuracy_score(y_test, y_test_preds)
            self.logger.info(
                "SUCCESS (_calculate_accuracy): Calculated overall accuracy successfull")
            return self.accuracy
        except Exception as err:
            self.logger.error(
                "ERROR (_calculate_accuracy): Calculating overall accuracy failed")
            raise err

    def roc_plots(self,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  model_key: str,
                  model) -> None:
        """
        Create and save roc plots.

        Parameters:
        - y_test (np.ndarray): Test labels.
        - y_test_preds (np.ndarray): Predictions on test set.
        - model_key (str): Name of the algorithm, for example rfc or logistic
        - model (Classifier): Trained model.

        Returns:
        - None

        Raises:
        - Exception: If the roc  generation fails.
        """

        # Compute ROC curve and ROC area
        probabilities = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        area_under_curve = auc(fpr, tpr)

        # Plotting the ROC curves
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'{model_key} AUC = {area_under_curve:.2f}')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)

        # Save the plot
        plt.savefig(self.constants.RESULTS_PATH + f'{model_key}_roc_curves.png')
        plt.close()

    def classification_report_image(self,
                                    y_train: np.ndarray,
                                    y_test: np.ndarray,
                                    y_train_preds: np.ndarray,
                                    y_test_preds: np.ndarray,
                                    algorithm: str) -> None:
        """
        Create and save classification report.

        Parameters:
        - y_train (np.ndarray): Training labels.
        - y_test (np.ndarray): Test labels.
        - y_train_preds (np.ndarray): Predictions on training set.
        - y_test_preds (np.ndarray): Predictions on test set.
        - algorithm (str): Name of the algorithm, for example rfc or logistic

        Returns:
        - None

        Raises:
        - Exception: If the classification report generation fails.
        """
        try:
            plt.figure(figsize=(10, 5))
            plt.text(
                0.01, 1.25, str(f'Confusion matrix, training data, {algorithm} model'), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.05, str(
                    classification_report(
                        y_test, y_test_preds)), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.6, str(f'Confusion matrix, test data, {algorithm} model'), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.7, str(
                    classification_report(
                        y_train, y_train_preds)), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(self.constants.RESULTS_PATH +
                        f'{algorithm}_model_classification_report.png')
            plt.close()
            self.logger.info(
                "...SUCCESS: %s classification report saved", algorithm)
        except Exception as err:
            self.logger.error(
                "ERROR (classification_report_image): Classification report generation for %s failed", algorithm)
            raise err

    def feature_importance_plot(self,
                                model: RandomForestClassifier,
                                X_test: np.ndarray) -> None:
        """
        Create and save feature importance plot.

        Parameters:
        - model (RandomForestClassifier): Trained Random Forest model.
        - X_test (np.ndarray): Test features, used in shap.

        Returns:
        - None

        Raises:
        - Exception: If the feature importance plot generation fails.
        """
        try:

            # Feature importance plot created with matplotlib
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(20, 10))
            plt.title("Random Forest Feature Importance", fontsize=20)
            plt.ylabel('Importance score', fontsize=15)
            plt.bar(range(X_test.shape[1]), importances[indices])
            plt.xticks(range(X_test.shape[1]), self.feature_names, rotation=90)
            plt.tight_layout()
            plt.savefig(
                self.constants.RESULTS_PATH +
                'random_forest_feature_importance.png')
            plt.close()

            # Alternative plot created with shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, plot_type="bar",
                              feature_names=self.feature_names)
            plt.tight_layout()
            plt.savefig(
                self.constants.RESULTS_PATH +
                'random_forest_shap_feature_importance.png')

            self.logger.info(
                "SUCCESS (feature_importance_plot): Feature importance plot saved")
        except Exception as err:
            self.logger.error(
                "ERROR (feature_importance_plot): Feature importance plot generation failed")
            raise err

    def _setup_param_grids(self) -> None:
        """
        Initialize model instances and their respective parameter grids.

        Sets up RandomForestClassifier and LogisticRegression models with predefined
        parameter grids and appends them to self.model_and_params list.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - Exception: If setting up the parameter grid fails.
        """
        try:
            rfc = RandomForestClassifier(random_state=42)
            param_grid_rf = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True, False]
            }

            self.model_and_params["rfc"] = (rfc, param_grid_rf)
            self.logger.info(
                "SUCCESS (_setup_param_grids): Initialized Random Forest model and grid search parameter grid")

            lrc = LogisticRegression(solver='liblinear', random_state=42)
            param_grid_lr = {
                "penalty": ["l1", "l2"],
                "C": np.logspace(0, 4, 10)
            }

            self.model_and_params["logistic"] = (lrc, param_grid_lr)
            self.logger.info(
                "SUCCESS (_setup_param_grids): Initialized Logistic Regression model and grid search parameter grid")
        except Exception as err:
            self.logger.error(
                "ERROR (_setup_param_grids): Model initialization and creation of grid search parameter grids failed")
            raise err

    def _grid_search(self,
                     X_train: np.ndarray,
                     y_train: np.ndarray) -> None:
        """
        Define parameters for GridSearchCV.

        Parameters:
        - X_train (np.ndarray): Training features.
        - y_train (np.ndarray): Training labels.

        Returns:
        - None

        Raises:
        - Exception: If the classification report generation fails.
        """

        for model_key, values in self.model_and_params.items():
            model, params = values
            grid = GridSearchCV(estimator=model,
                                param_grid=params,
                                cv=5,
                                verbose=0,
                                n_jobs=-1)
            grid.fit(X_train, y_train)
            self.best_models[model_key] = grid.best_estimator_

    def _model_inference(self,
                         model,
                         X: np.ndarray,
                         model_key: str) -> np.array:
        """
        Apply trained model with the best parameters to predict.

        Parameters:
        - model (RandomForestClassifier or LogisticRegression): Trained model.
        - X (np.ndarray): Test or train features (X).
        - model_key (str): Indicates the model being used (e.g. rfc or logistic)


        Returns:
        - np.array: Array containing the predictions (y).

        Raises:
        - Exception: If model inference fails.
        """
        try:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            self.logger.info(
                "SUCCESS (model_inference): %s model inference to features was successfull", model_key)
        except Exception as err:
            self.logger.error(
                "ERROR (model_inference): %s model inference failed", model_key)
            raise err
        else:
            return predictions, probabilities

    def train_models(self,
                     X_train: np.ndarray,
                     X_test: np.ndarray,
                     y_train: np.ndarray,
                     y_test: np.ndarray) -> None:
        """
        Train models and save models as pkl.

        Parameters:
        - X_train (np.ndarray): Training features.
        - X_test (np.ndarray): Test features.
        - y_train (np.ndarray): Training labels.
        - y_test (np.ndarray): Test labels.

        Returns:
        - None

        Raises:
        - Exception: If model training or predictions fail.
        """
        self._setup_param_grids()
        self._grid_search(X_train, y_train)

        try:
            for model_key, best_model in self.best_models.items():

                # Use best models from grid search
                best_model.fit(X_train, y_train)
                y_train_preds, _ = self._model_inference(
                    best_model, X_train, model_key)
                y_test_preds, _ = self._model_inference(
                    best_model, X_test, model_key)

                # Save best model as pkl-file
                joblib.dump(
                    best_model,
                    self.constants.MODELS_PATH +
                    f"{model_key}_model.pkl")

                self.logger.info(
                    "SUCCESS (train_models): %s model trained and saved as pkl-file", model_key)

                # Create classification report, roc, and feature importance score
                # plots for the best model
                self.classification_report_image(y_train,
                                                 y_test,
                                                 y_train_preds,
                                                 y_test_preds,
                                                 model_key)

                self.roc_plots(X_test,
                               y_test,
                               model_key,
                               best_model)

                self.accuracies[model_key] = self._calculate_accuracy(
                    y_test, y_test_preds)

                # Create feature importance score only if model is Random
                # Forest
                if model_key == "rfc":
                    self.feature_importance_plot(best_model, X_test)
        except Exception as err:
            self.logger.error("ERROR (train_models): Model training failed")
            raise err

    def __str__(self):
        accuracy_strings = [
            f"The overall classification accuracy of the {model} model is: {accuracy:.2f}"
            for model, accuracy in self.accuracies.items()
        ]
        return "\n".join(accuracy_strings)


if __name__ == '__main__':
    print(">> Pipeline start\n")

    # Create an instance of class ChurnLibrary
    churn_pipeline = ChurnLibrary(const)

    # Load data, create target variable
    imported_data = churn_pipeline.import_data(
        const.DATA_PATH, const.KEEP_COLUMNS)
    data_with_target = churn_pipeline.create_target(imported_data)

    # Perform explanatory data analysis (EDA)
    churn_pipeline.perform_eda(data_with_target, const.CAT_COLUMNS)

    # Encoding of categorical variables
    encoded_data = churn_pipeline.encoder_helper(
        data_with_target, const.CAT_COLUMNS)

    # Split data into training and test data and apply feature scaling
    x_training_scaled, x_testing_scaled, y_training, y_testing = churn_pipeline.perform_feature_engineering(
        encoded_data, const.RESPONSE)

    # Model training and model evaluation (based on test data predictions)
    churn_pipeline.train_models(
        x_training_scaled,
        x_testing_scaled,
        y_training,
        y_testing)

    print(churn_pipeline)
    print("\n>> Pipeline end")
