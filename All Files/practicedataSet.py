import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, ridge_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict,cross_val_score,train_test_split

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
df = pd.read_csv(filepath)
# Task 1 : Import the dataset
# Print the first 10 rows of the dataframe to confirm successful loading.
df.head(10)
# Add the headers to the dataframe, as mentioned in the project scenario.
headers = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df.columns = headers
df.replace('?',np.NaN, inplace=True)
# Use dataframe.info() to identify the columns that have some 'Null' (or NaN) information.
df.dtypes
df.describe(include="all")
df.info()

missingData = df.isnull()

for column in missingData.columns.values.tolist():
    print(column)
    print(missingData[column].value_counts())
    print("")


df['age'].notnull()
avg_age = df['age'].astype('float').mean(axis=0)
df["age"].replace(np.NaN,avg_age, inplace = True)

smoker = df['smoker'].value_counts().idxmax()
df["smoker"].replace(np.nan, smoker, inplace=True)

df[['age','smoker']] = df[['age','smoker']].astype('float')
df.info()
# Also note, that the charges column has values which are more than 2 decimal places long
#. Update the charges column such that all values are rounded to nearest 2 
# decimal places. Verify conversion by printing the first 5 values of the updated 
# dataframe.
df[['charges']] = np.round(df[['charges']],2)
df['charges'].head(10)

# Implement the regression plot for charges with respect to bmi.
sns.regplot(x='bmi', y='charges', data=df, line_kws={'color':'red'})
plt.ylim(0,)
# Implement the box plot for charges with respect to smoker.
sns.boxplot(x='bmi', y='charges', data=df)
plt.ylim(0,)
# Print the correlation matrix for the dataset.
df.corr()
# Task 4 : Model Development
# Fit a linear regression model that may be used to predict the charges value, 
# just by using the smoker attribute of the dataset. Print the R^2 score of this model.
lr = LinearRegression()
X = df[['smoker']]
Y = df[['charges']]
lr.fit(X, Y)
# R^2
lr.score(X, Y)
# Fit a linear regression model that may be used to predict the charges value, 
# just by using all other attributes of the dataset. Print the  
# R^2 score of this model. You should see an improvement in the performance.
Z = df[['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region']]
lr.fit(Z,Y)
lr.score(Z, Y)
# Create a training pipeline that uses StandardScaler(), PolynomialFeatures() and 
# LinearRegression() to create a model that can predict the charges value using all
# the other attributes of the dataset. There should be even further improvement in the 
# performance.
Input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model', LinearRegression())]
pipe = Pipeline(Input)
Z = Z.astype('float')
pipe.fit(Z,Y)
ypipe = pipe.predict(Z)
ypipe[0:4]

# Task 5 : Model Refinement
# Split the data into training and testing subsets,
# assuming that 20% of the data will be reserved for testing.
x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size=0.20,random_state=1)
# Initialize a Ridge regressor that used hyperparameter  
# α=0.1. Fit the model using training data data subset. 
# Print the R^2 score for the testing data.
R = Ridge(alpha=0.1)
R.fit(x_train, y_train)
yhat = R.predict(x_test)
yhat[0:4]
r2_score(y_test,yhat)
# Apply polynomial transformation to the training parameters with degree=2.
# Use this transformed feature set to fit the same regression model, as above, 
# using the training subset. Print the  
# R2 score for the testing subset.
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
R.fit(x_train_pr,y_train)
y_hat = R.predict(x_test_pr)
r2_score(y_test, y_hat)
y_hat[0:4]