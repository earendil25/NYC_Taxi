#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
import warnings
import matplotlib.pyplot as plt
from ipywidgets import interact
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

PROJECT_ID='mobility-293009' # 여기에 여러분들의 프로젝트 ID를 넣어주세요


# In[44]:


get_ipython().run_cell_magic('time', '', 'base_query = """\nWITH base_data AS \n(\n  SELECT nyc_taxi.*, gis.* EXCEPT (zip_code_geom)\n  FROM (\n    SELECT *\n    FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2015`\n    WHERE \n        EXTRACT(MONTH from pickup_datetime) = 1\n        and pickup_latitude  <= 90 and pickup_latitude >= -90\n    ) AS nyc_taxi\n  JOIN (\n    SELECT zip_code, state_code, state_name, city, county, zip_code_geom\n    FROM `bigquery-public-data.geo_us_boundaries.zip_codes`\n    WHERE state_code=\'NY\'\n    ) AS gis \n  ON ST_CONTAINS(zip_code_geom, st_geogpoint(pickup_longitude, pickup_latitude))\n)\n\nSELECT \n    zip_code,\n    DATETIME_TRUNC(pickup_datetime, hour) as pickup_hour,\n    EXTRACT(MONTH FROM pickup_datetime) AS month,\n    EXTRACT(DAY FROM pickup_datetime) AS day,\n    CAST(format_datetime(\'%u\', pickup_datetime) AS INT64) -1 AS weekday,\n    EXTRACT(HOUR FROM pickup_datetime) AS hour,\n    CASE WHEN CAST(FORMAT_DATETIME(\'%u\', pickup_datetime) AS INT64) IN (6, 7) THEN 1 ELSE 0 END AS is_weekend,\n    COUNT(*) AS cnt\nFROM base_data \nGROUP BY zip_code, pickup_hour, month, day, weekday, hour, is_weekend\nORDER BY pickup_hour\n\n\n"""\n\ndata_df = pd.read_gbq(query=base_query, dialect=\'standard\', project_id=PROJECT_ID)\ndata_df.to_csv(\'./data_df.csv\')\n\ndata_df.info()')


# In[70]:


base_df=pd.read_csv('./data_df.csv',index_col='Unnamed: 0')
base_df['pickup_hour']=pd.to_datetime(base_df['pickup_hour'])
base_df['zip_code']=(base_df['zip_code']).astype(str)
base_df.info()


# In[71]:


def split_train_and_test(df, date):
    """
    Dataframe에서 train_df, test_df로 나눠주는 함수
    
    df : 시계열 데이터 프레임
    date : 기준점 날짜
    """
    train_df = df[df['pickup_hour'] < date]
    test_df = df[df['pickup_hour'] >= date]
    return train_df, test_df

def evaluation(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true-y_pred)/y_true))*100
    mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    score = pd.DataFrame([mape,mae,mse],
                         index=['mape','mae','mse'],
                        columns = ['score']).T
    return score


# In[75]:


'''one-hot encording'''

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(base_df[['zip_code']])
ohe_output = enc.transform(base_df[['zip_code']]).toarray()
ohe_df = pd.concat([base_df, pd.DataFrame(ohe_output, columns='zip_code_'+ enc.categories_[0])], axis=1)


'''test/train set'''

ohe_df['log_cnt']=np.log10(ohe_df['cnt'])
train_df, test_df = split_train_and_test(ohe_df,'2015-01-24')
del train_df['zip_code']
del train_df['pickup_hour']
del test_df['zip_code']
del test_df['pickup_hour']

y_train_raw = train_df.pop('cnt')
y_train_log = train_df.pop('log_cnt')
y_test_raw = test_df.pop('cnt')
y_test_log = test_df.pop('log_cnt')

x_train = train_df.copy()
x_test = test_df.copy()

'''linea regression'''

lr_reg = LinearRegression()
lr_reg.fit(x_train, y_train_log)
prev = lr_reg.predict(x_train)
pred = lr_reg.predict(x_test)

'''result input'''
train_df['prev_log']=prev
test_df['pred_log']=pred
train_df['prev_raw']=10**prev
test_df['pred_raw']=10**pred

train_df['real_log']=y_train_log
test_df['real_log']=y_test_log
train_df['real_raw']=y_train_raw
test_df['real_raw']=y_test_raw

train_df = train_df[np.isfinite(train_df).all(1)]
test_df = test_df[np.isfinite(test_df).all(1)]


# In[73]:


evaluation(test_df['real_log'],test_df['pred_log'])


# In[76]:


evaluation(test_df['real_raw'],test_df['pred_raw'])


# In[77]:


plt.scatter(train_df['real_raw'],train_df['prev_raw'])


# In[78]:


plt.scatter(test_df['real_raw'],test_df['pred_raw'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




