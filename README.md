# Analyzing credit card fraud with machine learning
CSU Channel Islands Fall 2021 Senior Capstone Project
<br/>
Contributions by Jeffrey Romero

# Project Description
Credit card fraud is a problem that occurs more often than necessary. Cases of fraud may be hard to detect if transactions are disguised as legitimate ones. By using Machine Learning, it will be easier to predict fraudulent transactions.

The goal of this project is to analyze and process a data set containing credit card fraud data. Different functions are used to analyze and process the data. For example, check statistics such as the size of the file containing the data set, the size of the contents of the file, the distribution of values, as well as data interpolation methods to fill in non-applicable values.

After the data set is processed, it is passed to a Machine Learning algorithm which will train a model. This trained model can be used to predict cases of fraudulent credit card transactions, as long as the input is readable and sanitized.

## How to view this project
The HTML file **JeffreyRomeroCapstoneML-CCF.html** has all of the content/research which relates to the project at hand. If you want to run the code yourself, you can download the program JupyterLab and its dependencies listed further below, then download and run the file **JeffreyRomeroCapstoneML-CCF.ipynb** The run-time of this .ipynb roject file will be 20-30 minutes due to the training time of the machine learning models.

---

## Input/Data Set
The data set that will be used to train the machine learning model has been downloaded from:
<br/>
https://www.kaggle.com/mishra5001/credit-card

## Functions
### Data processing
Before the model can make predictions, input data must be processed/sanitized to make it program-readable. For example, the program does not understand the meaning behind input terms such as "employment status" or "fraudulent transaction," so these terms must be encoded as 0's, 1's, 2's, 3's, etc. Some values may also be missing or not-applicable. Such values can be filled in through mathematical interpolation techniques.

### Training a model
Multiple machine learning models are created with different machine learning algorithms. Each algorithm has its own method of outcome predictions, but there will either be a few or only one algorithm which can efficiently solve the problem at hand.

### Model evaluation
Machine learning models can be evaluated through an accuracy metric which calculates the rate of correct fraud predictions. For example, a machine learning model with an accuracy score of 90% out of 100 total predictions will have 90/100 correct predictions of fraud. The efficiency of a model is also calculated by the time taken to train the model. If two models have the same prediction accuracy rating of 75.0%, but one has a training time of 4 seconds and the other 18 minutes, the former model will be more efficient.

### How are machine learning models trained?
The machine learning algorithm that I have chosen to train my model is called Logistic Regresion. This model has a prediction accuracy of 92.05% and a training time of 7 seconds. Two more models have tied with my chosen model at 92.05%, however the other two models had a training time of 200 seconds and 18 minutes, making Logistic Regression the most efficient model to solve this problem of credit card fraud prediction.
<br/>
With Logistic Regresion, it is possible to predict whether a credit card transaction is fraudulent or not based off features from the data set such as income, credit, or annuity.

# Conclusions
- Before predictions can be made for fraud, a machine learning model must make sense of and learn from sanitized data.
- Imbalanced data can be fixed through an algorithm known as oversampling but does not guarantee better fraud prediction.
- There are only 5 pairs of two columns/features which strongly influences a fraudulent transaction.

### The five most positive correlations leading to fraudulent transactions
AMT_CREDIT + AMT_GOODS_PRICE

REGION_RATING_CLIENT_W_CITY + REGION_RATING_CLIENT

CNT_CHILDREN + CNT_FAM_MEMBERS

AMT_CREDIT + AMT_ANNUITY

AMT_GOODS_PRICE + AMT_ANNUITY
<br/>
<br/>
### Description of the columns above
AMT_CREDIT - Credit amount of the loan.

AMT_GOODS_PRICE - For consumer loans it is the price of the goods for which the loan is given.

AMT_ANNUITY - Loan annuity.

REGION_RATING_CLIENT_W_CITY - Our rating of the region where client lives with taking city into account (1,2,3).

REGION_RATING_CLIENT - Our rating of the region where client lives (1,2,3).

CNT_CHILDREN - Number of children the client has.

CNT_FAM_MEMBERS - How many family members does client have.

# Dependencies
The entire project is coded in the Python programming language. The minimum Python version required to run the project itself is at least 3.0. The recommended Python version is 3.7. Aside from Python, there are other dependencies such as:
- Jupyter Lab
    - https://jupyter.org/install
- pip
    - https://pip.pypa.io/en/stable/installation/
- numpy
    - \$ pip install numpy
- pandas
    - \$ pip install pandas
- seaborn
    - \$ pip install seaborn
- scipy
    - \$ pip install scipy
- scikit-learn
    - \$ pip install -U scikit-learn
- matplotlib
    - \$ pip install -U matplotlib
- imbalanced-learn
    - \$ pip install -U imbalanced-learn

# References
Credit card fraud detection data set
<br/>
https://www.kaggle.com/mishra5001/credit-card
<br/>
<br/>
Imbalanced data classification
<br/>
https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/
<br/>
<br/>
Logistic Regression
<br/>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3936971/
<br/>
<br/>
Logistic Regression on imbalanced data
<br/>
https://machinelearningmastery.com/cost-sensitive-logistic-regression/
<br/>
<br/>
Machine Learning model accuracy
<br/>
https://developers.google.com/machine-learning/crash-course/classification/accuracy
<br/>
<br/>
Seaborn Correlation Heatmap Creation
<br/>
https://heartbeat.comet.ml/seaborn-heatmaps-13-ways-to-customize-correlation-matrix-visualizations-f1c49c816f07
