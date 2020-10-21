import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from util import evaluation

train_df=pd.read_csv('./train_data_df.csv',index_col='Unnamed: 0')
train_df['pickup_hour']=pd.to_datetime(train_df['pickup_hour'])
train_df['zip_code']=(train_df['zip_code']).astype(str)
train_df.fillna(method='backfill',inplace=True)

test_df=pd.read_csv('./test_data_df.csv',index_col='Unnamed: 0')
test_df['pickup_hour']=pd.to_datetime(test_df['pickup_hour'])
test_df['zip_code']=(test_df['zip_code']).astype(str)
test_df.fillna(method='backfill',inplace=True)

print("data set imported")

"""One-Hot encoding zip-code"""
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(train_df[['zip_code']])
ohe_output = enc.transform(train_df[['zip_code']]).toarray()
train_df = pd.concat([train_df, pd.DataFrame(ohe_output, columns='zip_code_'+ enc.categories_[0])], axis=1)

ohe_output = enc.transform(test_df[['zip_code']]).toarray()
test_df = pd.concat([test_df, pd.DataFrame(ohe_output, columns='zip_code_'+ enc.categories_[0])], axis=1)

"""전처리"""
del train_df['pickup_hour']
del test_df['pickup_hour']
del train_df['zip_code']
del test_df['zip_code']

y_train = train_df.pop('cnt')
y_test = test_df.pop('cnt')
x_train = train_df.copy()
x_test = test_df.copy()

print("전처리 끝!")

"""linear regression"""
lr_reg = LinearRegression()
lr_reg.fit(x_train, y_train)
pred = lr_reg.predict(x_test)

"""evaluation"""
print(evaluation(y_test, pred))