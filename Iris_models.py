# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:12:33 2019

@author: Jeel Patel
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

#Load preloaded Iris dataset from    
dataset = load_iris()

#features and target
X = dataset.data[:,:]
y = dataset.target

#=================Selecting Algorithms================
#SVC
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X, y)
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model.fit(X, y)
#XGBoost
from xgboost import XGBRegressor
xgb_model = XGBRegressor()
xgb_model.fit(X, y)

#predicting result
y_pred_svc = svc_model.predict(X)
y_pred_knn = knn_model.predict(X)
y_pred_xgb = xgb_model.predict(X).round() + 0 #Added positive zero to change -0 to +0

#Metric Scores(Evaluation)
from sklearn import metrics 
print("SVC model Accuracy: {:.4f}".format(metrics.accuracy_score(y, y_pred_svc))) #OUTPUT : 0.9867
print("KNN model Accuracy: {:.4f}".format(metrics.accuracy_score(y, y_pred_knn))) #OUTPUT : 0.9867
print("XGB model Accuracy: {:.4f}".format(metrics.accuracy_score(y, y_pred_xgb))) #OUTPUT : 1.0
