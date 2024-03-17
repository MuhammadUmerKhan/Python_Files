import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model 

homePrice_df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\homeprices.csv")
homePrice_df.head()

plt.scatter(homePrice_df.area, homePrice_df.price, color='red', marker='+')
plt.xlabel("Area")
plt.ylabel('Price')
plt.show()

price = homePrice_df.price
areea = homePrice_df.drop(columns='price')

lr = linear_model.LinearRegression()
#  fit(X,Y) X=independent var, Y=dependent var
lr.fit(areea, price)
# predicting random value price
lr.predict([[5000]])

plt.scatter(homePrice_df.area, homePrice_df.price, color='red', marker='+')
plt.plot(homePrice_df.area, lr.predict(areea), color='blue')
plt.xlabel("Area")
plt.ylabel('Price')
plt.show()
# Task is to predict per capita income in 2020 year 
canada_df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\canada_per_capita_income.csv")
canada_df.head()

plt.scatter(canada_df.year, canada_df['per capita income (US$)'], marker="+", color="orange")
plt.xlabel("Year")
plt.ylabel("Per capita income (US$)")
plt.show()

X = canada_df['year'].values.reshape(-1, 1)
Y = canada_df['per capita income (US$)']

lr_can = linear_model.LinearRegression()
lr_can.fit(X, Y)
# Predicting "per capita income (US$)" in year 2020
lr_can.predict([[2020]])
# 
plt.scatter(canada_df.year, canada_df['per capita income (US$)'], marker="+", color="orange")
plt.plot(X, lr_can.predict(X), color='blue')
plt.xlabel("Year")
plt.ylabel("Per capita income (US$)")
plt.show()