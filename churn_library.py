'''
Module with functions to create a model that identifies customers who are likely to churn.

Author: Fabian Löw
Date: 26 June 2023
'''

import os
import pandas as pd
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_PATH = "./data/bank_data.csv"  # Path to *.csv file
EDA_IMAGES_PATH = "./images/eda/"
MODELS_PATH = "./models/"
RESULTS_PATH = "./images/results/"
RESPONSE = 'Churn'

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Define columns with categorical features
CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

# Define columns to keep
KEEP_COLUMNS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']


def import_data(pth):
    '''
    Returns a dataframe for the csv found at a given path 'pth'.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    # Read the *.csv file and return as dataframe
    dataframe = pd.read_csv(pth)

    print('Loaded data set')
    return dataframe


def create_target(dataframe, response):
    '''
    Returns dataframe with a target column.

    input:
            dataframe: pandas dataframe
    output:
            df_target: pandas dataframe
    '''

    df_target = dataframe.copy()

    # Create the attribute 'Churn'
    df_target[response] = df_target['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return df_target


def perform_eda(dataframe):
    '''
    Perform an EDA on a dataframe and save figures to images folder.

    input:
            dataframe: pandas dataframe

    output:
            None (save figures to image folder)
    '''

    # Create the 'images' folder if it doesn't exist.
    os.makedirs(EDA_IMAGES_PATH, exist_ok=True)

    # Create histogram plot showing number of existing customers vs. new
    # customers
    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.savefig(EDA_IMAGES_PATH + 'Histogram_Existing_Customer.png')

    # Create histogram plot showing customer age
    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.savefig(EDA_IMAGES_PATH + 'Histogram_Customer_Age.png')

    # Create bar plot showing marital status
    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(EDA_IMAGES_PATH + 'Barplot_Marital_Status.png')

    # Create histplot showing the values of Total_Trans_Ct, add kernel density
    # function
    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(EDA_IMAGES_PATH + 'Histplot_Kde_Total_Trans_Ct.png')

    # Create heatmap showing the features correlation among each other
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(EDA_IMAGES_PATH + 'Heatmap_Correlation.png')

    plt.close()


def encoder_helper(dataframe, category_lst, response):
    '''
    Turn each categorical column into a new column containing the
    propotion of churn for each category

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            dataframe: pandas dataframe with new columns
    '''

    for category in category_lst:
        groups = dataframe.groupby(category).mean()['Churn']
        lst = [groups.loc[val] for val in dataframe[category]]
        dataframe[category + '_' + response] = lst

    return dataframe


def perform_feature_engineering(dataframe, keep_columns, response):
    '''
    Split dataframe into seperate training and testing data sets, keep only required columns

    input:
              dataframe: pandas dataframe
              keep_columns: list with features to keep for analysis & classification
              response: string of response name

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # Create seperate dataframes for response (y) and predictor (X) variables
    # / features
    y_data = dataframe[response]
    x_data = dataframe[keep_columns]

    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)

    # Return the training and testing data sets
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    Creates classification reports for each tested ML algorithm, based on training/testing
    and stores report as a fugure in the 'images' folder

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    model_names = ['Random Forest', 'Logistic Regresson']
    images_names = ['classification_report_RF', 'classification_report_LR']
    y_train_preds = [y_train_preds_rf, y_train_preds_lr]
    y_test_preds = [y_test_preds_rf, y_test_preds_lr]


    # Create classification_report for each ML algorithm
    for model_index, model_name in enumerate(model_names):
        plt.figure(figsize=(10, 10))
        plt.text(0.01, 1.25, str(f'{model_name} Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds[model_index])), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str(f'{model_name} Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds[model_index])), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(f'images/results/{images_names[model_index]}.png')
        plt.close()


def feature_importance_plot(model, x_train, out_path):
    '''
    Creates and stores the feature importance plots in the 'images' folder

    input:
            model: model object containing 'feature_importances_'
            X_test: pandas dataframe of predictor variables from test data

    output:
             None
    '''

    importances = pd.Series(
        model.best_estimator_.feature_importances_,
        index=x_train.columns)

    importances_plot = sns.barplot(x=importances.index, y=importances.values)
    importances_plot.set_title("Feature Importance")
    importances_plot.set_ylabel("Importance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_path + 'RF_feature_importance_plot.png')
    plt.close()


def create_roc_curves(rfc_model, lr_model, x_test, y_test, out_path):
    '''
    Calculates receiver operator curves (ROC) and save the figures in the 'images' folder.

    input:
              rfc_model, lr_model: model objects
              X_test: X testing data
              y_test: y testing data
    output:
              None
    '''

    plt.figure(figsize=(15, 8))
    plot_axis = plt.gca()
    plot_roc_curve(rfc_model, x_test, y_test, ax=plot_axis, alpha=0.8)
    plot_roc_curve(lr_model, x_test, y_test, ax=plot_axis, alpha=0.8)
    plot_axis.figure.savefig(out_path + 'roc_curve_result.png')
    # plt.show()
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    Train and save ML model results: images, scores, and models.

    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              y_train_preds_rf: random forest predictions on the training data
              y_test_preds_rf: random forest predictions on the test data
              y_train_preds_lr: logistic regression predictions on the training data
              y_test_preds_lr: logistic regression predictions on the test data
    '''

    # Define two classifier algorithms, random forest and linear regression
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Define a parameter grid for the grid search
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Perform grid search with the RFC classifier algorithm and the training
    # data
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Fit the RFC model with the training data
    cv_rfc.fit(x_train, y_train)

    # Fit the LRC model using the training data
    lrc.fit(x_train, y_train)

    # Create the folder if doesn't exists.
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Save the models to a folder (models)
    joblib.dump(cv_rfc, MODELS_PATH + 'rfc_model_v2.pkl')
    joblib.dump(lrc, MODELS_PATH + 'logistic_model_v2.pkl')

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Create classification reports and save as figure
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Create the receiver operator curves for both models and save as figure
    create_roc_curves(cv_rfc.best_estimator_, lrc, x_test, y_test, RESULTS_PATH)

    # Create feature variable importance score plot for the RF model and save
    # as figure
    feature_importance_plot(cv_rfc, x_test, RESULTS_PATH)


if __name__ == '__main__':

    loaded_data = import_data(DATA_PATH)

    dataframe_target = create_target(loaded_data, RESPONSE)

    perform_eda(dataframe_target)

    dataframe_encoded = encoder_helper(dataframe_target, CAT_COLUMNS, RESPONSE)

    X_training, X_testing, y_training, y_testing = perform_feature_engineering(
        dataframe_encoded, KEEP_COLUMNS, RESPONSE)

    X = dataframe_encoded[KEEP_COLUMNS]

    train_models(X_training, X_testing, y_training, y_testing)
