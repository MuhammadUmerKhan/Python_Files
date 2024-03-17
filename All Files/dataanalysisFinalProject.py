from sklearn.linear_model import Ridge
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(file_name)
df.head()

# Question 1
df.dtypes

# Question 2
df.drop(['Unnamed: 0','id'],axis=1, inplace=True)
df.describe()

print("number of NaN _calues for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

mean_bedroom = df['bedrooms'].mean()
df['bedrooms'].replace(np.NaN, mean_bedroom, inplace= True)
mean_bathrooms = df['bathrooms'].mean()
df['bathrooms'].replace(np.NaN, mean_bathrooms, inplace= True)

# Question 3
# Use the method value_counts to count the number of houses with unique floor values,
# use the method .to_frame() to convert it to a dataframe.
floor_counts = df['floors'].value_counts()
df_floor_counts = floor_counts.to_frame().reset_index()

# Question 4
# Use the function boxplot in the seaborn library to determine whether houses 
# with a waterfront view or without a waterfront view have more price outliers.
sns.boxplot(x='waterfront', y='price', data=df)
plt.xlabel('Waterfront View')
plt.ylabel('Price')
plt.title('Boxplot of Prices for Houses with and without Waterfront View')
plt.show()

# Question 5
# Use the function regplot in the seaborn library to determine 
# if the feature sqft_above is negatively or positively correlated with price.
sns.regplot(x='sqft_above', y='price', data=df)
plt.xlabel('Price')
plt.ylabel('Square footage above')
plt.title('Regression Plot of Price vs. Square Footage Above')
plt.show()

X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)

# Question 6
# Fit a linear regression model to predict the 'price' using the feature 'sqft_living' 
# then calculate the R^2. Take a screenshot of your code and the value of the R^2.
Z = df[['sqft_living']]
lm.fit(Z,  df['price'])
Yhat = lm.predict(Z)
Yhat[0:4]
lm.score(Z,Y)

# Question 7
# Fit a linear regression model to predict the 'price' using the list of features:
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms",
           "sqft_living15","sqft_above","grade","sqft_living"]     
lm.fit(df[features], Y)
yhat = lm.predict(df[features])
yhat[0:4]
# Then calculate the R^2. Take a screenshot of your code.
lm.score(df[features], Y)
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

# Question 8
# Use the list to create a pipeline object to predict the 'price',
# fit the object using the features in the list features, and calculate the R^2.
pipe = Pipeline(Input)
F = df[features]
F = F.astype('float')
pipe.fit(F, Y)
ypipe = pipe.predict(F)
ypipe[0:4]
pipe.fit(F, Y)
print("The value of R-Squared is: ",lm.score(F,Y))

# Module 5: Model Evaluation and Refinement
x_train, x_test, y_train, y_test = train_test_split(df[features], Y, test_size=0.15, random_state=1)
print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

# Question 9
# Create and fit a Ridge regression object using the training data,
# set the regularization parameter to 0.1, and calculate the R^2 using the test data.
RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_test, y_test)
RidgeModel.score(x_test, y_test)

# Question 10
# Perform a second order polynomial transform on both the training data and testing data.
# Create and fit a Ridge regression object using the training data, set the regularisation
# parameter to 0.1, and calculate the R^2 utilising the test data provided. 
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
RidgeModel.score(x_train_pr, y_train)