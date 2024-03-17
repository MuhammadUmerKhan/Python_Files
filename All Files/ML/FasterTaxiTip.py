import time
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import gc, sys
# Dataset Analysis
df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\yellow_tripdata_2019-06.csv\yellow_tripdata_2019-06.csv")
df.head()

df = df[df['tip_amount'] > 0]
# we also remove some outliers, namely those where the tip was larger than the fare cost
df = df[(df['tip_amount'] <= df['fare_amount'])]
# we remove trips with very large fare cost
df = df[((df['fare_amount'] >=2) & (df['fare_amount'] < 200))]
# we drop variables that include the target variable in it, namely the total_amount
clean_data = df.drop(['total_amount'], axis=1)

del df
gc.collect()

# Dataset Preprocessing
clean_data.head()
# Convert 'tpep_dropoff_datetime' and 'tpep_pickup_datetime' columns to datetime objects
clean_data['tpep_dropoff_datetime'] = pd.to_datetime(clean_data['tpep_dropoff_datetime'])
clean_data['tpep_pickup_datetime'] = pd.to_datetime(clean_data['tpep_pickup_datetime'])

# Extract pickup and dropoff hour
clean_data['pickup_hour'] = clean_data['tpep_pickup_datetime'].dt.hour
clean_data['dropoff_hour'] = clean_data['tpep_dropoff_datetime'].dt.hour

# Extract pickup and dropoff day of the week (0 = Monday, 6 = Sunday)
clean_data['pickup_day'] = clean_data['tpep_pickup_datetime'].dt.weekday
clean_data['dropoff_day'] = clean_data['tpep_dropoff_datetime'].dt.weekday

# Calculate trip time in seconds
clean_data['trip_time'] = (clean_data['tpep_dropoff_datetime'] - clean_data['tpep_pickup_datetime']).dt.total_seconds()

clean_data.shape
clean_data.head()

first_n_rows = 200000
clean_data = clean_data.head(first_n_rows)

# drop the pickup and dropoff datetimes
clean_data = clean_data.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)
# some features are categorical, we need to encode them
# to encode them we use one-hot encoding from the Pandas package
get_dummy_col = ["VendorID","RatecodeID","store_and_fwd_flag","PULocationID", "DOLocationID","payment_type", "pickup_hour", "dropoff_hour", "pickup_day", "dropoff_day"]
proc_data = pd.get_dummies(clean_data, columns = get_dummy_col)
# release memory occupied by clean_data as we do not need it anymore
# we are dealing with a large dataset, thus we need to make sure we do not run out of memory
del clean_data
gc.collect()

# extract the labels from the dataframe
y = proc_data[['tip_amount']].values.astype('float32')
# drop the target variable from the feature matrix
proc_data = proc_data.drop(['tip_amount'], axis=1)
# get the feature matrix used for training
X = proc_data.values
# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)

# print the shape of the features matrix and the labels vector
print('X.shape=', X.shape, 'y.shape=', y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train.shape
x_test.shape

# Build a Decision Tree Regressor model with Scikit-Learn
model = DecisionTreeRegressor(max_depth=8, random_state=35)
t0 = time.time()
model.fit(x_train, y_train)
sklearn_time = time.time()-t0
print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))

y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
score # 0.7325729535212653
mse = mean_squared_error(y_test, y_pred)
mse

tree = DecisionTreeRegressor(max_depth=12, random_state=45)
tree.fit(x_train, y_train)
score = tree.score(x_test, y_test) # 0.6994274701282022
pred = tree.predict(x_test)

print("MSE: ", mean_squared_error(y_test, pred)) # 1.766354864347884
# We learned that increasing the max_depth parameter to 12 increases the MSE