# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 20:54:29 2016

@author: Ganapriya
"""

import pandas as pd
#import numpy as np
##import datetime as DT
#import math
from sklearn import preprocessing, cross_validation

#import zipfile
#import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#
#def llfun(act, pred):
#    """ Logloss function for 1/0 probability
#    """
#    return (-(~(act == pred)).astype(int) * math.log(1e-15)).sum() / len(act)
#

#train = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'])[['X', 'Y', 'Category']]
train1 = pd.read_csv(r'C:\Users\Ganapriya\Documents\AppliedProject\Kaggle\SF_Crime\train.csv',parse_dates = ["Dates"], index_col= False)
test = pd.read_csv(r'C:\Users\Ganapriya\Documents\AppliedProject\Kaggle\SF_Crime\test.csv',parse_dates = ["Dates"], index_col= False)

def datesplit(data):
    data["Year"] = data["dates"].dt.year
    data["Month"] = data["dates"].dt.month
    data["Day"] = data["dates"].dt.day
    data["Hour"] = data["dates"].dt.hour
    data["Minute"] = data["dates"].dt.minute
    return data
    
#sfT = pd.read_csv(z.open('train.csv'), 
#                  sep=r'[\s,]',              # 1
#                  header=None, skiprows=1,
#                  converters={               # 2
#                      0: lambda x: DT.Dates.decode.strptime(x, '%Y%m%d'),  
#                      1: lambda x: DT.time.decode(*map(int, x.split(':')))},
#                  names=['Date', 'Time', 'Category', 'Descript', 'DayOfWeek','PdDistrict','Resolution','Address','X','Y'])
#
train1.columns = ['dates', 'category_predict', 'description_ignore', 'day_of_week', 'pd_district', 'resolution', 'address', 'x', 'y']
test.columns = ['id', 'dates', 'day_of_week', 'pd_district', 'address', 'x', 'y']


train1 = datesplit(train1)
test = datesplit(test)


day_of_week_encoded = preprocessing.LabelEncoder()
day_of_week_encoded.fit(train1['day_of_week'])
train1['day_of_week_encoded'] = day_of_week_encoded.transform(train1['day_of_week'])

pd_district_encoded=preprocessing.LabelEncoder()
pd_district_encoded.fit(train1['pd_district'])
train1['pd_district_encoded'] = pd_district_encoded.transform(train1['pd_district'])


day_of_week_encoded = preprocessing.LabelEncoder()
day_of_week_encoded.fit(test['day_of_week'])
test['day_of_week_encoded'] = day_of_week_encoded.transform(test['day_of_week'])

pd_district_encoded=preprocessing.LabelEncoder()
pd_district_encoded.fit(test['pd_district'])
test['pd_district_encoded'] = pd_district_encoded.transform(test['pd_district'])

scaler = preprocessing.StandardScaler().fit(train1[['day_of_week_encoded','Year','Month','Day', 'pd_district_encoded', 'x', 'y', 'Hour']])

knn=KNeighborsClassifier(n_neighbors=121)

knn.fit(scaler.transform(train1[['day_of_week_encoded', 'pd_district_encoded','Year','Month','Day', 'x', 'y', 'Hour']]),
       train1['category_predict'])
#test['hour'] = test['date'].str[11:13]
# Separate test and train set out of orignal train set.

train1['pred'] = knn.predict(scaler.transform(train1[['day_of_week_encoded', 'pd_district_encoded', 'x', 'y','Year','Month','Day','Hour']]))
test_pred = knn.predict_proba(scaler.transform(test[['day_of_week_encoded', 'pd_district_encoded', 'x', 'y','Year','Month','Day','Hour']]))

# CHECK TRAINING SET ACCURACY.

# Compute training set accuracy.
print('Training Set Accuracy :', sum(train1['category_predict'] == train1['pred']) / len(train1['dates']))

# CROSS VALIDATION.

# Get cross validation scores.
cv_scores = cross_validation.cross_val_score(knn,
                                             scaler.transform(train1[['day_of_week_encoded', 'pd_district_encoded', 'x', 'y','Year','Month','Day','Hour']]),
                                             train1['category_predict'],
                                             cv = 2)

# Take the mean accuracy across all cross validation segments.
print('Cross Validation Accuracy: ', cv_scores.mean())
                                            
# EXPORT TEST SET PREDICTIONS.
# This section exports test predictions to a csv in the format specified by Kaggle.com.

# Turn 'test_pred' into data frame.
test_pred = pd.DataFrame(test_pred)

# Add column names to 'test_pred'.
test_pred.columns = knn.classes_

# Name index column.
test_pred.index.name = 'Id'

# Write csv.
test_pred.to_csv('test_pred_knn.csv')
