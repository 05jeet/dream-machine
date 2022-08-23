#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 23:08:39 2022

@author: addynobi
"""

#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
data= pd.read_csv('/Users/addynobi/Documents/Essex Jan 22/CE 888 7 SP/Re/in-vehicle-coupon-recommendation.csv')
data
# Exploratory Data Snalysis
data.shape
data.info()
data.describe()
data.dtypes
data['temperature']=data['temperature'].astype('category')
data.isnull().sum()
data.isnull().sum().sort_values(ascending=False) * 100 /len(data)
data.drop_duplicates()
sns.pairplot(data, vars= ['occupation', 'weather', 'temperature', 'time', 'car', 'Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50', 'direction_same' ])
data.isin(['?']).sum(axis=0)
plt.figure(figsize=(8,4))
plt.pie(data.Y.value_counts(),labels=['Coupon Accepted','Coupon Rejected'], autopct='%.1f%%')
plt.title("Percentage of Coupon Acceptance")
plt.show()
# dropping the "Car' variable since it  has too many null values
data.drop('car',axis=1,inplace=True)
data.head(3)
#replacing null values with mode
data=data.fillna(data.mode().iloc[0])
data.isnull().sum()
#cconverting object data types to categorical 

data_object = data.select_dtypes(include=['object']).copy()

for col in data_object.columns:
    data[col]=data[col].astype('category')
data.dtypes
data.select_dtypes('int64').nunique()
data.drop(columns=['toCoupon_GEQ5min'], inplace=True)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
enc = OneHotEncoder(dtype='int64')

data_category = data.select_dtypes(include=['category']).copy()
data_integer = data.select_dtypes(include=['int64']).copy()

data_enc = pd.DataFrame()
for col in data_category.columns:
    enc_output = enc.fit_transform(data_category[[col]])
    df0 = pd.DataFrame(enc_output.toarray(), columns=enc.categories_)
    data_enc = pd.concat([data_enc,df0], axis=1)
    
data_encoded = pd.concat([data_enc, data_integer], axis=1)
data_encoded
### Modelling
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import accuracy_score 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
import xgboost
from sklearn.model_selection import StratifiedKFold
# Train Test Split
data_target = data_encoded[['Y']]
data_encoded.drop('Y',inplace=True,axis=1)
X_train, X_test, y_train, y_test = train_test_split(data_encoded, data_target, test_size=0.3, random_state=42,stratify=data_target)

print(f'X_train :{X_train.shape}')
print(f'y_train :{y_train.shape}')
print(f'X_test :{X_test.shape}')
print(f'y_test :{y_test.shape}')

#### Baseline Model
from sklearn.model_selection import cross_val_score
cross_val_score(RandomForestClassifier(n_estimators=60), X_train, y_train)

#### Random Forest with hyper-parametric tunning
clf1 = GridSearchCV(RandomForestClassifier(), {
    'n_estimators': [10,20,40,60],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [2,7]
    
}, cv=10, return_train_score=False)
clf1.fit(X_train, y_train)
clf1.cv_results_

dfrf = pd.DataFrame(clf1.cv_results_)
dfrf
dfrf[['param_n_estimators','param_max_features', 'param_max_depth','mean_test_score']]

# Best Model with Best parameters with hyper-parametric tuning
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear'],
            'random_state' : [None]
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
             'n_estimators': [10,20,40,60],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [2,7]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10],
            'random_state' : [None],
            'fit_intercept': [True]
        }
    }
}

# Best Model with Best Params
scores = [ ]

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df
# Bsst model is SVM with 74.36% accoracy.
############################