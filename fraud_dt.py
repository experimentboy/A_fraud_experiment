# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 07:15:12 2017

@author: Eric Vos
"""


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
from re import sub
from decimal import Decimal
#avoid noise
warnings.filterwarnings("ignore")
#feature normalization
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = pd.read_csv('Fraud_raw.csv')
train_data["Marital Status"][train_data["Marital Status"] == "In-Relationship"] = 1
train_data["Marital Status"][train_data["Marital Status"] == "Married"] = 1
train_data["Marital Status"][train_data["Marital Status"] == "Unmarried,"] = 0
train_data["Accomodation Type"][train_data["Accomodation Type"] == "Owns a house"] = 1
train_data["Accomodation Type"][train_data["Accomodation Type"] == "Staying with Family"] = 1
train_data["Accomodation Type"][train_data["Accomodation Type"] == "Rented"] = 0
train_data['Claim Amount'] = (train_data['Claim Amount'].replace( '[\$,)]','', regex=True).replace( '[(]','-',   regex=True ).astype(float))
train_data['Claim Amount'] = scaler.fit_transform(train_data['Claim Amount'])
train_data['Age Group'] = scaler.fit_transform(train_data['Age Group'])
train_data['Height (cms)'] = scaler.fit_transform(train_data['Height (cms)'])
#splt train and test data
X_train = train_data.iloc[:,2:]
y_train = train_data.iloc[:,1]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size = 0.75, random_state=0)
#train Deision tree model
clf = DecisionTreeClassifier(random_state = 0 ,max_depth = 3).fit(X_train2, y_train2)
#model score
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train2, y_train2)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test2, y_test2)))
probs = clf.predict(X_test2)
print('Accuracy of Decision Tree classifier on validation set: {:.2f}'.format(clf.score(X_test2, probs)))
# compute fpr, tpr, and threshold for the validation test set
fpr_lr, tpr_lr, _ = roc_curve(y_test2, probs)
roc_auc_lr = auc(fpr_lr, tpr_lr)
print('auc on validation set {}'.format(roc_auc_lr))
#Plot the ROC curve
plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, label='DecisionTree ROC curve (area = {:0.2f})'.format(roc_auc_lr))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve (Fraud classifier)', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()
#Display feature importance
top_feature = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False).iloc[:5].index.tolist()
print("Top feature per importance :", top_feature)
