# Predict Customer Churn

Project **Predict Customer Churn**

## Project Description

The workflow reads a csv-file with different attributes of more than 10,000
customers. The workflow calibrates and evaluates two machine learning (ML) algorithms,
random forest (RF) and linear regression (LR). It creates various figures to inform 
about the performance of these algorithms. Finally, the workflow creates a prediction
about customers churning or not. The workflow includes unit testing and logging.

The workflow consists of five steps:

- EDA
- Feature Engineering (including encoding of categorical variables)
- Model Training
- Prediction
- Model Evaluation


## Files and data description

Overview of the most important files and data present in the root directory:

- Folders:
  - `data`: Contains the input data in `csv` format
  -  `images`: Main folder
    - `eda`: Used to store the EDA visualizations
    - `results`: Used to store the classification reports and evaluation of the models
  - `logs`: Stores the logs for the unit tests on the churn_library.py file
  - `models`: Stores model objects


- Files:
  - `churn_library.py`: A library of functions to find customers who are likely to churn, based on machine learning.
  - `churn_script_logging_and_tests.py`: Contains the unit tests for the functions in churn_library.py


The project repository has the following structure:  

```
.
├── churn_library.py        --------> This file contains the functions to the churn model
├── churn_script_logging_and_tests.py  ------> This file contains the tests functions to test churn functions
├── data
│   └── bank_data.csv   ------> data in *.csv format
├── images
│   ├── eda 
│   │   ├── Histogram_Existing_Customer.png  ------> churn distribution
│   │   ├── Histogram_Customer_Age.png  ------> customer age distribution
│   │   ├── Heatmap_Correlation.png  ------> heatmap
│   │   ├── Barplot_Marital_Status.png  ------> marital status distribution
│   │   └── Histplot_Kde_Total_Trans_Ct.png  ------> total transaction distribution
│   └── results
│       ├── feature_importances.png  ------> feature importances
│       ├── classification_report_LR.png  ------> logistic model classification report
│       ├── classification_report_RF.png  ------> random forest classification report
│       └── roc_curve_result.png  ------> roc curve
├── logs
│   └── churn_library_v2.log  ------> logging file
├── models
│   ├── logistic_model_v2.pkl ------> logistic model file
│   └── rfc_model_v2.pkl ------> random forest file
│
├── README.md
├── requirements_py3.6.txt
└── requirements_py3.8.txt

```

## Running Files

**Install the required libraries**
<br>`pip install -r requirements_py3.8.txt`<br>

With this command you will install all the required libraries.

**Use ipython or python to run the python files**
<br>`ipython churn_library.py`<br>
or 
<br>`python churn_library.py`<br>

This will trigger the workflow, i.e. the random forest and logistric regression models 
will be calibrated, validated, and saved in the models folder. A couple of EDA related
figures such as heatmaps and distribution plots will be created and stored in the images/eda folder. 
Classifciation reports and receiver operator curve plots of the two models will be created in images/results folder.  

**Run unit tests**
<br>`ipython churn_script_logging_and_tests.py`<br>
or 
<br>`python churn_script_logging_and_tests.py`<br>

All log messages (success or error) from the implemented unit tests will be saved in churn_library.log file inside logs folder.

## Dependencies

- Python 3.7
- Pandas 1.2.4
- Scikit-learn 0.24.2
- Matplotlib 3.4.2
- Seaborn 0.11.1
- Shap 0.39.0
- Joblib 1.0.1