import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('E:/projects/machine-learing/data/data.csv')
data = data.drop(columns=['date', 'waterfront', 'view', 'street', 'country'])

data['city'] = data['city'].astype('category').cat.codes
data['statezip'] = data['statezip'].astype('category').cat.codes

features = data.columns.difference(['price'])
data[features] = (data[features] - data[features].mean()) / data[features].std()

# 单位：万元
data['price'] = data['price'] / 10000 

train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)
train_data.to_excel('E:/projects/machine-learing/data/train.xlsx', index=False)
test_data.to_excel('E:/projects/machine-learing/data/test.xlsx', index=False)

