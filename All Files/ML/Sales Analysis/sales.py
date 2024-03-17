from turtle import title
import pandas as pd
# import plotly.express  as px
# import plotly.graph_objects as go
import numpy as np
from pyparsing import C
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

feature = pd.read_csv("./features.csv")
store = pd.read_csv("./stores.csv")
data = pd.read_csv("./train.csv")

feature.head()
feature.shape
store.shape
data.shape
store.head()
data.head()
feature.head()
data.isnull().sum()
feature.isnull().sum()
store.isnull().sum()

feature['CPI'].fillna(feature['CPI'].median(), inplace=True)
feature['Unemployment'].fillna((feature['Unemployment'].median()), inplace=True)
# feature['MarkDown'].unique()
for i in range(1, 6):
    feature['MarkDown' + str(i)] = feature['MarkDown' + str(i)].apply(lambda x:0 if x<0 else x)
    feature['MarkDown' + str(i)].fillna(value=0, inplace=True)    

feature.isnull().sum()
feature.head()
data.head()
store.head()

data = pd.merge(data, store, on='Store', how='left')
data = pd.merge(data, feature, on=['Store', 'Date'], how='left')
data.head()

data.info()
data['IsHoliday_y'] = data['IsHoliday_y'].astype('int')
data['IsHoliday_x'] = data['IsHoliday_x'].astype('int')
data.head()

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data.sort_values(by=['Date'], inplace=True)
data.set_index(data['Date'], inplace=True)
data.head()

data['IsHoliday_x'].isin(data['IsHoliday_y']).all()
data.drop(columns=['IsHoliday_x'], inplace=True)
data.rename(columns={'IsHoliday_y' : 'IsHoliday'}, inplace=True)

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month_name()
data.head()

agg_data = data.groupby(['Store', 'Dept']).Weekly_Sales.agg(['min', 'max', 'mean', 'median', 'std']).reset_index()
agg_data.head()

store_data = pd.merge(left=data, right=agg_data, on=['Store', 'Dept'], how='left')
store_data.head()
store_data.dropna(inplace=True)
store_data.shape

data = store_data.copy()
data['Date'] = pd.to_datetime(data['Date'],errors='coerce')
data.sort_values(by=['Date'],inplace=True)
data.set_index(data.Date, inplace=True)
data.head()

data['Total_MarkDown'] = data['MarkDown1']+data['MarkDown2']+data['MarkDown3']+data['MarkDown4']+data['MarkDown5']
data.drop(columns=['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], inplace=True)

data.shape
numeric = []
for i in data.columns:
    if data[i].dtype != 'object':
        numeric.append(i)
numeric_col = numeric

data_numeric = data[['Weekly_Sales','Size','Temperature','Fuel_Price','CPI','Unemployment','Total_MarkDown']].copy()
data_numeric.head()
data_numeric.shape

data = data[(np.abs(stats.zscore(data_numeric)) < 3).all(axis = 1)]
data.shape

data = data[data['Weekly_Sales'] >= 0]
data.shape
data.head()

# Visualization
#Average Monthly Sales
# fig = px.bar(data, x='Month', y='Weekly_Sales')
# fig.update_layout(
#     title='Average Monthly Sales',
#     xaxis_title='Month',
#     yaxis_title='Weekly Sales'
# )
# fig.show()
plt.figure(figsize=(12, 10))
sns.barplot(data=data, x=data['Month'], y=data['Weekly_Sales'])
plt.title('Average Monthly Sales')
plt.show()
data_monthly = pd.crosstab(data['Year'], data['Month'], values=data['Weekly_Sales'], aggfunc='sum')
fig, axes = plt.subplots(3, 4, figsize=(16, 8))
plt.suptitle('Monthly Sales for each Year', fontsize=18)
k = 1
for i in range(3):
    for j in range(4):
        sns.lineplot(ax=axes[i, j], data=data_monthly, x=data_monthly.index, y=data_monthly.columns[k-1])
        axes[i, j].set_ylabel('Sales', fontsize=12)
        axes[i, j].set_xlabel('Years', fontsize=12)
        axes[i, j].set_title(data_monthly.columns[k-1], fontsize=12)
        k += 1
plt.subplots_adjust(wspace=0.4, hspace=0.32)
plt.show()

# #Average Weekly Sales Store wise
data.head()

plt.figure(figsize=(20, 10))
sns.barplot(x='Store', y='Weekly_Sales', data=data)
plt.title("Average Weekly Sales Store wise")
plt.grid(True)
plt.show()

data.head()
plt.figure(figsize=(20, 10))
sns.lineplot(x='Year', y='Unemployment', data=data)
plt.title("Unemployment Rate by the change of time")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate")
plt.show()

plt.figure(figsize=(20, 10))
sns.lineplot(x=data['Year'], y=data['Fuel_Price'], data=data)
plt.title("Fuel Price Rate by the change of time")
plt.show()

data.head()

plt.figure(figsize=(16, 8))
sns.barplot(x='Dept', y='Weekly_Sales', data=data)
plt.title("Average Weekly Sales Dept wise")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,8))
sns.histplot(data['Temperature'])
plt.title('Effect of Temperature',fontsize=15)
plt.xlabel('Temperature',fontsize=14)
plt.ylabel('Density',fontsize=14)
# plt.savefig('effect_of_temp.png')
plt.show()

holiday_counts = data['IsHoliday'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(x=holiday_counts, labels=['Holiday', 'No Holiday'], autopct='%0.2f%%')
plt.title("Holiday Distribution", fontsize=18)
plt.grid()
plt.legend()
plt.show()
# Building the model
data.dtypes
cat_col = ['Store', 'Dept', 'Type']
data_cat = data[cat_col].copy()
data_cat.head()

data_cat = pd.get_dummies(data_cat, columns=cat_col)
data_cat.head()
data_cat = data_cat.astype('int')
data_cat.shape

data = pd.concat([data, data_cat], axis=1)
data.shape

data.drop(columns=cat_col, inplace=True)
data.drop(columns=['Date'], inplace=True)
data.shape

num_col = ['Weekly_Sales','Size','Temperature','Fuel_Price','CPI','Unemployment','Total_MarkDown','max','min','mean','median','std']
minmax_scale = MinMaxScaler(feature_range=(0, 1))
def normalization(df, col):
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = minmax_scale.fit_transform(arr.reshape(len(arr), 1))
    return df
data.head()
data = normalization(data.copy(), num_col)
data.head()

# Correlation

corr = data[num_col].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix", fontsize=18)
plt.show()
feature_col = data.columns.difference(['Weekly_Sales'])
feature_col

data.columns
data['Date'] = data.index
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.weekday
data.drop(columns=['Date'], inplace=True)
data.head()
data.to_csv("Data.csv", index=False)
X = data[['median', 'mean', 'Week', 'Temperature', 'max', 'CPI', 'Fuel_Price', 'min', 'std', 'Unemployment', 'Month', 'Total_MarkDown', 'Dept_16', 'Dept_18', 'Dept_3', 'IsHoliday', 'Size', 'Year', 'Dept_11', 'Dept_1', 'Dept_9', 'Dept_5', 'Dept_55', 'Dept_56', 'Dept_7', 'Dept_72']]
Y = data['Weekly_Sales']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)  
x_train.shape, x_test.shape
model = LinearRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test) # 0.9271714435597925
score = cross_val_score(LinearRegression(), X, Y, cv=5)
score.mean() # 0.9217650259718685
y_pre = model.predict(x_test)

print("MAE" , metrics.mean_absolute_error(y_test, y_pre)) # 0.027264566987851065
print("MSE" , metrics.mean_squared_error(y_test, y_pre)) # 0.0029529897410963575
print("RMSE" , np.sqrt(metrics.mean_squared_error(y_test, y_pre))) # 0.054341418283813295
print("R2" , metrics.explained_variance_score(y_test, y_pre)) # 0.9271714464427763

prediction_table = pd.DataFrame({'Actual':y_test, 'Predicted Value': y_pre})
prediction_table.head()

plt.figure(figsize=(15,8))
plt.title('Comparison between actual and predicted values',fontsize=16)
plt.plot(model.predict(x_test[:100]), label="prediction", linewidth=3.0,color='blue')
plt.plot(y_test[:100].values, label="real_values", linewidth=3.0,color='red')
plt.legend(loc="best")
#plt.savefig('lr_real_pred.png')
plt.show()

# Random Forest Regressor
rft = RandomForestRegressor()
rft.fit(x_train, y_train)
print(rft.score(x_test, y_test)*100) # 97.44673501002458
score_rft = cross_val_score(RandomForestRegressor(), X, Y, cv=5)
score_rft.mean()
y_pre_rft = rft.predict(x_test)

print("MAE" , metrics.mean_absolute_error(y_test, y_pre_rft)) # 0.015566224450155056
print("MSE" , metrics.mean_squared_error(y_test, y_pre_rft)) # 0.0010352759535867042
print("RMSE" , np.sqrt(metrics.mean_squared_error(y_test, y_pre_rft))) # 0.03217570439923117
print("R2" , metrics.explained_variance_score(y_test, y_pre_rft)) # 0.9744688894898691

rf_df = pd.DataFrame({'Actual':y_test, 'Predicted Value': y_pre_rft})


plt.figure(figsize=(15,8))
plt.title('Comparison between actual and predicted values',fontsize=16)
plt.plot(rft.predict(x_test[:100]), label="prediction", linewidth=3.0,color='blue')
plt.plot(y_test[:100].values, label="real_values", linewidth=3.0,color='red')
plt.legend(loc="best")
#plt.savefig('lr_real_pred.png')
plt.show()

# KNN
KNN = KNeighborsRegressor()
KNN.fit(x_train, y_train)
print(KNN.score(x_test, y_test)*100) # 95.67979483509247
score_KNN = cross_val_score(RandomForestRegressor(), X, Y, cv=5)
score_KNN.mean()
y_pre_KNN = KNN.predict(x_test)

print("MAE" , metrics.mean_absolute_error(y_test, y_pre_KNN)) 
print("MSE" , metrics.mean_squared_error(y_test, y_pre_KNN)) 
print("RMSE" , np.sqrt(metrics.mean_squared_error(y_test, y_pre_KNN))) 
print("R2" , metrics.explained_variance_score(y_test, y_pre_KNN)) 

knn_df = pd.DataFrame({'Actual':y_test, 'Predicted Value': y_pre_KNN})


plt.figure(figsize=(15,8))
plt.title('Comparison between actual and predicted values',fontsize=16)
plt.plot(KNN.predict(x_test[:100]), label="prediction", linewidth=3.0,color='blue')
plt.plot(y_test[:100].values, label="real_values", linewidth=3.0,color='red')
plt.legend(loc="best")
#plt.savefig('lr_real_pred.png')
plt.show()

# XGBoost Regressor
xgb = XGBRegressor()
xgb.fit(x_train, y_train)
print(xgb.score(x_test, y_test)*100)
score_xgb = cross_val_score(RandomForestRegressor(), X, Y, cv=5)
score_xgb.mean()
y_pre_xgb = xgb.predict(x_test)

print("MAE" , metrics.mean_absolute_error(y_test, y_pre_xgb)) 
print("MSE" , metrics.mean_squared_error(y_test, y_pre_xgb)) 
print("RMSE" , np.sqrt(metrics.mean_squared_error(y_test, y_pre_xgb))) 
print("R2" , metrics.explained_variance_score(y_test, y_pre_xgb)) 

xgb_df = pd.DataFrame({'Actual':y_test, 'Predicted Value': y_pre_xgb}) 


plt.figure(figsize=(15,8))
plt.title('Comparison between actual and predicted values',fontsize=16)
plt.plot(xgb.predict(x_test[:100]), label="prediction", linewidth=3.0,color='blue')
plt.plot(y_test[:100].values, label="real_values", linewidth=3.0,color='red')
plt.legend(loc="best")
#plt.savefig('lr_real_pred.png')
plt.show()