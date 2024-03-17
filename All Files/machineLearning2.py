from bs4 import BeautifulSoup
from sklearn import linear_model
from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
df = pd.read_csv(path)
df.head()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel('CO2 Emmision')
plt.show()

# Creating training and testing data set
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='green')
plt.xlabel("Engine Size")
plt.ylabel('CO2 Emmision')
plt.show()

reg = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
reg.fit(x, y)
print('The cofficients are: ', reg.coef_)

y_hat = reg.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
x1 = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y1 = np.asanyarray(test[['CO2EMISSIONS']])
print("Mean Squared error(MSE): %.2f" % np.mean((y_hat - y1) ** 2))
print('Variance score: %.2f' % reg.score(x1, y1))
y_hat[0:4]
# Practice
# Try to use a multiple linear regression with the same dataset, 
# but this time use FUELCONSUMPTION_CITY and FUELCONSUMPTION_HWY instead of FUELCONSUMPTION_COMB. 
# Does it result in better accuracy?
reg1 = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
reg1.fit(train_x, train_y)
print("Cofficient:", reg1.coef_)

y_hat1 = reg1.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]) 
test_x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
test_y = np.asanyarray(test[['CO2EMISSIONS']]) 
y_hat1[0:4]
print("Mean Squared error(MSE): %.2f" % np.mean((y_hat1 - test_y) ** 2))
print('Variance score: %.2f' % reg1.score(test_x, test_y))