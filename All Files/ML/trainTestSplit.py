from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model

df = pd.read_csv("carprices.csv")
df.head()

df.drop(columns="Car Model", inplace=True)
df.head()


plt.scatter(df['Sell Price($)'], df['Mileage'], marker='*', color="green")
plt.xlabel("Sell Price($)")
plt.ylabel("Mileage")
plt.show()

plt.scatter(df['Sell Price($)'], df['Age(yrs)'], marker='*', color="red")
plt.xlabel("Sell Price($)")
plt.ylabel("Age(yrs)")
plt.show()


# train_test_split
# Looking at above two scatter plots, using linear regression model makes 
# sense as we can clearly see a linear relationship between our dependant 
# (i.e. Sell Price) and independant variables (i.e. car age and car mileage)

# The approach we are going to use here is to split available data in two 
# sets

# Training: We will train our model on this dataset
# Testing: We will use this subset to make actual predictions using trained model
# The reason we don't use same training set for testing is because our model
# has seen those samples before, using same samples for making predictions might give us wrong impression about accuracy of our model. 
# It is like you ask same questions in exam paper as you tought the students in the class.

X = df[['Mileage','Age(yrs)']]
y = df['Sell Price($)']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
lr.predict(X_test)
y_test
lr.score(X_test, y_test)