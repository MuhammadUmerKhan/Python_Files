from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import date
import seaborn as sns
import pandas as pd
import pip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(filepath)
df.head()

# 1. Linear Regression and Multiple Li  near Regression
lm = LinearRegression()
lm

X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)

Yhat = lm.predict(X)
Yhat[0:5]

lm.intercept_
lm.coef_

lm1 = LinearRegression()
X1 = df[['engine-size']]
Y1 = df['price']
lm1.fit(X,Y)
Yhat1 = lm.predict(X)
Yhat1[0:5]

# intercept
lm1.intercept_
#  slope
lm1.coef_

Yhat1 =-7963.34 + 166.86*X
Price=-7963.34 + 166.86*df['engine-size']

# Multiple Linear Regression
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z,df['price'])
lm.intercept_
lm.coef_

Yhat2 = -15806.624626329223 + 53.49574423*df['horsepower'] + 4.70770099*df['curb-weight'] + df['engine-size']*81.53026382 + df['highway-mpg']*36.05748882
F = df[['normalized-losses','highway-mpg']]
lm2 = LinearRegression()
lm2.fit(F,df['price'])
lm2.coef_
lm2.intercept_

# Regression Plot
width = 12
height = 10
plt.figure(figsize=(width,height))
sns.regplot(x="highway-mpg",y="normalized-losses",data=df)
plt.ylim(0,)

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

df[["peak-rpm","highway-mpg","price"]].corr()


width = 12
height = 10
plt.figure(figsize=(width,height))
sns.residplot(x="highway-mpg",y="normalized-losses",data=df)
plt.ylim(0,)


Y_hat = lm.predict(Z)
plt.figure(figsize=(width,height))
ax1 = sns.distplot(df['price'],hist=False,color='r',label="Actual Value")
sns.distplot(Y_hat,hist=False,color='b',label="Fitted Value",ax=ax1)
plt.title("Actual vs Fitted value for price")
plt.xlabel("Price (in dollar)")
plt.ylabel("Proportion of Cars")
plt.show()
plt.close()


def PlotPoolly(model,independent_variables,dependent_variable,Name):
    x_new = np.linspace(15,55,100)
    y_new = model(x_new)
    
    
    plt.plot(independent_variables,dependent_variable,'.',x_new,y_new,'-')
    plt.title("Polynomial fit with Matplotlib fro Price ~ Length")
    ax = plt.gca()
    ax.set_facecolor((0.898,0.898,0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    
    plt.show()
    plt.close()
    
x = df['highway-mpg']
y = df['price']
f=np.polyfit(x,y,3)
p=np.poly1d(f)
print(p)
PlotPoolly(p,x,y,'highway-mpg')
np.polyfit(x,y,3)


f1=np.polyfit(x,y,11)
p1=np.poly1d(f1)
print(p1)
PlotPoolly(p1,x,y,'highway-mpg')
np.polyfit(x,y,3)

pr = PolynomialFeatures(degree=2)
Z_pr = pr.fit_transform(Z)
Z.shape
Z_pr.shape


Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe = Pipeline(Input)

Z = Z.astype('float')
pipe.fit(Z,y)
ypipe = pipe.predict(Z)
ypipe[0:4]

Input1 = [('scale',StandardScaler()),('model',LinearRegression())]
pipe1 = Pipeline(Input1)
pipe1.fit(Z,y)
ypipe1 = pipe1.predict(Z)
ypipe1[0:10]

lm.fit(X,Y)
print("The R-square is: ",lm.score(X,Y))

Yhat = lm.predict(X)
print("The Output of the first four predicted value is: ",Yhat[0:4])

mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

# Model 2: Multiple Linear Regression
# let's calculate R^2
lm.fit(Z,df['price'])
print("The Value of R-squared is: ",lm.score(Z,df['price']))

Y_predict_multifit = lm.predict(Z)
print("The mean square error of price and predicted value using multifit is: ",mean_squared_error(df['price'],Y_predict_multifit))
r_squared = r2_score(y,p(x))
print("The R-square value is:",r_squared)
mean_squared_error(df['price'],p(x))

# Prediction and Decision Making
new_input = np.arange(1,100,1).reshape(-1,1)
lm.fit(X,Y)
lm

yhat = lm.predict(new_input)
yhat[0:5]
plt.plot(new_input,yhat)
plt.show()