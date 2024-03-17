import seaborn as sb
from email import errors
import math
from os import error
from turtle import color, mode
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

df_insurance = pd.read_csv("insurance_data.csv")
df_insurance.head()


plt.scatter(df_insurance['age'], df_insurance['bought_insurance'], marker='.', color="blue")
plt.xlabel('Age')
plt.ylabel("bought_insurance")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(df_insurance[['age']],df_insurance.bought_insurance,train_size=0.8)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

df_insurance.head()
df_insurance.isnull().sum()

log = LogisticRegression()
log.fit(X_train, y_train)
log.predict(X_test)
y_test

log.score(X_test, y_test)
log.predict_proba(X_test) # predict probability of yes or no

# Lets defined sigmoid function now and do the math with hand

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def prediction_function(age):
    z = 0.042 * age - 1.53
    y = sigmoid(z)
    return y
age = 35
prediction_function(age)
# 0.485 is less than 0.5 which means person with 35 age will not buy insurance
prediction_function(43)
# 0.485 is more than 0.5 which means person with 43 will buy the insurance

# -------------------------------------------
df_HR = pd.read_csv("HR_comma_sep.csv")
df_HR.head()

df_HR.columns
df_HR.isnull().sum()

df_HR_numeric = df_HR.copy()
# df_HR_numeric.corr()

for i in df_HR_numeric.columns:
    if(df_HR_numeric[i].dtype == 'object'):
        df_HR_numeric[i] = df_HR_numeric[i].astype('category')
        df_HR_numeric[i] = df_HR_numeric[i].cat.codes
df_HR_numeric.corr()

plt.figure(figsize=(20, 7))
sb.heatmap(df_HR_numeric.corr(), annot=True)
plt.title("Correlation Matric For Numeric Features")
plt.show()

left = df_HR[df_HR.left==1]
left.shape

retained = df_HR[df_HR.left==0]
retained.shape

df_HR.columns
df_HR.groupby('left')[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident', 'left',
       'promotion_last_5years']].mean()


pd.crosstab(df_HR.salary, df_HR.left).plot(kind='bar')
pd.crosstab(df_HR.Department, df_HR.left).plot(kind='bar')

subdf = df_HR[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
subdf.head()

salary_dummies = pd.get_dummies(subdf.salary, prefix='salary')
salary_dummies = salary_dummies.astype('int')
salary_dummies

df_HR_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')
df_HR_with_dummies

df_HR_with_dummies.drop(columns='salary', inplace=True)
df_HR_with_dummies.head()

X = df_HR_with_dummies
Y = df_HR.left

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train, y_train)
model.predict(x_test)
model.score(x_test, y_test)

