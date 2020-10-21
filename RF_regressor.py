import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from ipywidgets import interact
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import os
from sklearn.ensemble import RandomForestRegressor
import json
from util import evaluation

train_df=pd.read_csv('./train_data_df.csv',index_col='Unnamed: 0')
train_df['pickup_hour']=pd.to_datetime(train_df['pickup_hour'])
train_df['zip_code']=(train_df['zip_code']).astype(str)
train_df.fillna(method='backfill',inplace=True)

test_df=pd.read_csv('./test_data_df.csv',index_col='Unnamed: 0')
test_df['pickup_hour']=pd.to_datetime(test_df['pickup_hour'])
test_df['zip_code']=(test_df['zip_code']).astype(str)
test_df.fillna(method='backfill',inplace=True)

print('data imported')

"""label encoding zip-code"""
le = LabelEncoder()
train_df['zip_code_le'] = le.fit_transform(train_df['zip_code'])
test_df['zip_code_le'] = le.fit_transform(test_df['zip_code'])

del train_df['zip_code']
del train_df['pickup_hour']
del test_df['zip_code']
del test_df['pickup_hour']

y_train = train_df.pop('cnt')
y_test = test_df.pop('cnt')
x_train = train_df.copy()
x_test = test_df.copy()

"""전처리 끝!"""



rf_reg = RandomForestRegressor(n_estimators=20, n_jobs=-1)
rf_reg.fit(x_train, y_train)
rf_pred = rf_reg.predict(x_test)

print(evaluation(y_test, rf_pred))
