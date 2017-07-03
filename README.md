# A_fraud_experiment
Coursera Data Science Community project

Overview

Identify the fraud propensity for a retail company based on a 4K rows worth of data with 1 target variable (Fraud Instance) and 11 predictor variables

Basic Information

    Length – 1-2 week
    Group Size – 1-2 individuals
    Difficulty - Beginner
    Prerequisite Knowledge – Understanding of basic statistics, data types and usage of data mining tool is required.
    Required/Recommended Technology – Use R based GUI tools to perform this task (Recommended GUIs: Rattle, Rcmdr, Deducer). Being a very small dataset even excel can be really helpful

Background

    Background/Context – Every retail chain faces a potential fraud instances where people order a product and then return it after some days claiming either the product doesn’t work or doesn’t provide desired utility. However, each such transaction has some precursors that may point towards a potential fraud instances.
    Target Audience – Retail Industry’s Risk Management Team will highly benefit from this analysis
    Portfolio Development - Upon completion of this project the students will be able to learn:
    Specific analytical techniques deployment based on a specific data types
    How to use one of the most in demand data science tool R and its GUI based data mining packages
    How to model data set with binary, categorical and continuous data

Details

    Description – The project revolves around creating a working predictive model that predict the propensity of a fraud instance given certain conditions (predictor variables). The aim of this project is to create a fully deployable predictive algorithm with the capability to predict fraud occurrence.

    It is expected that the individual or the team should be able to clearly explain the entire analytical and algorithm building process steps by step in a form of a presentation. The individual or the teams can leverage any statistical tool of their choice to deliver the work output.

    Success Criteria – Step by Step model building presentation and a working algorithm
    
    Data Access and Usage limitations – The users can create features out of the data if required with a valid reason for taking that route. Considering the fact that it is fairly and easy a light dataset creating too many features may reduce the chances of success.
    Data – The dataset has 4k rows worth of data with 12 variables out of which 1 is Response variable (Fraud Instance) remaining variable are predictor variables.

## Initial approach : see Fraud_start.ipynb

* [Dataset](https://docs.google.com/spreadsheets/d/1TufF3QBHK8RsC06V0arvF3PwN3gfz5kg5eV6BjRxEjc/edit#gid=581816440) Analysis show binary, continuous and categorical data.

![](https://github.com/experimentboy/A_fraud_experiment/blob/master/dataset.png)

* Binary data are used asis.
* Continuous data are normalized.
* Categorical data are simplified (transformed in binary).
* Dataset is splited in Train and Test set (ratio 75/25).
* Train a Decision Tree model (limited to 3 level) to identify the most important features.

### Decision Tree Classifier : 

* We use here a simple Decision Tree classifier with only 3 layer

![](https://github.com/experimentboy/A_fraud_experiment/blob/master/Fraud_start_dtplot.png)

### Initial Results : 

* Accuracy of Decision Tree classifier on training set: 0.91
* Accuracy of Decision Tree classifier on test set: 0.90
* AUC on validation set: 0.9028665063372497

![](https://github.com/experimentboy/A_fraud_experiment/blob/master/Fraud_dt_ROC.png)

### Top feature per importance : 
1. 'Damaged Item'
1. 'Item Not Avaiable' 
1. 'Product Care Plan'
1. 'Item Not In Stock'

![](https://github.com/experimentboy/A_fraud_experiment/blob/master/Fraud_start_FIplot.png)

## Next steps:
* Increase of amount of layer in Decision Tree seems to overfit
* Use of another classifier also seems to overfit
