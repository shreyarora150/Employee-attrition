# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:44:15 2021

@author: shrey
"""

##Load libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

##load data

test = pd.read_csv("test_hXY9mYw.csv")
train = pd.read_csv("train_MpHjUjU.csv")

##Exploring Train data
train.shape
test.shape

train.columns

train.head()
train.info()
train.describe()
train.var()

## check missing values
train.isnull().sum()/train.shape[0] *100

#categorical features
categorical = train.select_dtypes(include =[np.object])
print("Categorical Features in Train Set:",categorical.shape[1])

##cast as date and MMM-YY

train['Dateofjoining'] = pd.to_datetime(train['Dateofjoining'], errors='raise')
train['MMM-YY'] = pd.to_datetime(train['MMM-YY'], errors='raise')
train['LastWorkingDate'] = pd.to_datetime(train['LastWorkingDate'], errors='raise')



#numerical features
numerical= train.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features in Train Set:",numerical.shape[1])

train.head()
train.info()
train.describe()
train.var()

#Feature engineering


def last_reporting_month(row):
    return train[train['Emp_ID']==row['Emp_ID']]['MMM-YY'].max()   


def total_business_value(row):
    return train[train['Emp_ID']==row['Emp_ID']]['Total Business Value'].mean()

def avergae_quaterly_rating(row):
     return train[train['Emp_ID']==row['Emp_ID']]['Quarterly Rating'].mean()
 

def attrition(row):
    if pd.isnull(row['LastWorkingDate']):
        return 0
    else :
        return 1

train['last_reporting_month'] = train.apply(lambda x: last_reporting_month(x),axis =1)
train['Net_Business_Value'] = train.apply(lambda x: total_business_value(x),axis =1)
train['avergae_quaterly_rating'] = train.apply(lambda x: avergae_quaterly_rating(x),axis =1)
train1 = train[train['MMM-YY']==train['last_reporting_month']]

train1['Gender'].value_counts()
train1['City'].value_counts()
train1['Education_Level'].value_counts()

train1.dtypes

le = LabelEncoder()
var_mod = train1.select_dtypes(include='object').columns
for i in var_mod:
    train1[i] = le.fit_transform(train1[i])
    
train1['Y'] = train1.apply(lambda x: attrition(x),axis=1)
train1['Months_from_joining'] = (( train1['MMM-YY'] -  train1['Dateofjoining'])/np.timedelta64(1, 'M'))
train1['Months_from_joining'] = train1['Months_from_joining'].astype(int)
train1.dtypes

# Seperate Features and Target
X= train1.drop(columns = ['Emp_ID','LastWorkingDate','Y','MMM-YY','last_reporting_month','Dateofjoining'], axis=1)
y= train1['Y']
#X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=22)



#regressor = DecisionTreeRegressor(random_state = 0)
#regressor = RandomForestClassifier(n_estimators = 100) 
#regressor = GradientBoostingClassifier()

## Model selection 
scores = pd.DataFrame()
models = [DecisionTreeRegressor(random_state = 0),RandomForestClassifier(n_estimators = 100) ,
          GradientBoostingClassifier(),XGBClassifier()]
for m in models:
    regressor = m
    s = cross_val_score(regressor, X, y, cv=5, scoring='f1_macro')
    #regressor.fit(X_train, y_train)
    #  y_pred = regressor.predict(X_valid).astype(int)
    
    print(m)
    #y = f1_score(y_valid, y_pred, average='macro')*100
    print(s)
    scores = scores.append({"model":m,"avg score":s.mean()},ignore_index= True)
#sample_submission_znWiLZ4

# Gradient boost had the highest score so I'm considering it as my final model and training the model on the whole training set   
    
regressor = GradientBoostingClassifier()
regressor.fit(X, y)



test1= pd.merge(train1,test,how = 'inner',on='Emp_ID')
test1 = test1.drop(columns = ['LastWorkingDate','Y','MMM-YY','last_reporting_month','Dateofjoining'], axis=1)
test2 = test1.drop(columns = ['Emp_ID'],axis=1)

def set_targets(row):
    return int(test1[test1['Emp_ID']==row['Emp_ID']]['Y'].values[0])

submission = pd.read_csv('sample_submission_znWiLZ4.csv')
final_predictions = regressor.predict(test2)
final_predictions.astype(int)
test1['Y'] = final_predictions

#set_targets(submission.iloc[0])

submission['Target'] = submission.apply(lambda x: set_targets(x),axis=1)
submission.to_csv('Final_Submission_GradBoost.csv', index=False)