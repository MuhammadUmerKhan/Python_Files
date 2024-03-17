from cProfile import label
from pickle import FALSE
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

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv'
df = pd.read_csv(filepath)

lm = LinearRegression()
X = df[['CPU_frequency']]
Y = df['Price']
lm.fit(X,Y)
Yhat = lm.predict(X)
Yhat[0:5]

ax1 = sns.distplot(df['Price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()

mse = mean_squared_error(df['Price'],Yhat)
print('The mean square error of price and predicted value is: ', mse)

r_squared = lm.score(X,Y)
print("The mean square error of price and predicted value is: ",r_squared)
lm1 = LinearRegression()
Z = df[['CPU_frequency','RAM_GB','Storage_GB_SSD','CPU_core','OS','GPU','Category']]
lm.fit(Z,Y)
Y_hat = lm.predict(Z)
Y_hat[0:5]

ax2 = sns.distplot(df['Price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax2)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value','Fitted Value'])
plt.show()

mse1 = mean_squared_error(df['Price'],Y_hat)
print('The mean square error of price and predicted value is: ', mse1)
r_squared2 = lm.score(Z,Y)
print("The mean square error of price and predicted value is: ",r_squared2)

X = X.flatten()
f1 = np.polyfit(X,Y,1)
p1=np.poly1d(f1)

f2 = np.polyfit(X,Y,3)
p2 = np.poly1d(f2)

f3 = np.polyfit(X,Y,5)
p3 = np.poly1d(f3)


def PlotPolly(model,independent_variable,dependent_variable,Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)
    
    plt.plot(independent_variable,dependent_variable,'.',x_new,y_new,'-')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.89,0.89,0.89))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Laptop')
    plt.show()    
    plt.close()
    
PlotPolly(p1,X,Y,'CPU_frequency')
PlotPolly(p2,X,Y,'CPU_frequency')
PlotPolly(p3,X,Y,'CPU_frequency')


print("The Mean Squared error of 1st degree polynomial is:",mean_squared_error(Y,p1(X)))
print("The Mean Squared error of 2nd degree polynomial is:",mean_squared_error(Y,p2(X)))
print("The Mean Squared error of 3rd degree polynomial is:",mean_squared_error(Y,p3(X)))

print("The R Squared value of first degree polynomial is: ",r2_score(Y,p1(X)))
print("The R Squared value of Second degree polynomial is: ",r2_score(Y,p2(X)))
print("The R Squared value of third degree polynomial is: ",r2_score(Y,p3(X)))

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)
ypipe[0:4]

print("The Mean Squared error of 3rd degree polynomial is:",mean_squared_error(Y,ypipe))
print("The R Squared value of first degree polynomial is: ",r2_score(Y,ypipe))