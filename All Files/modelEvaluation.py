from sklearn.model_selection import GridSearchCV
from cProfile import label
from turtle import width
from matplotlib.pylab import f
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
df = pd.read_csv(filepath)
df.head()
df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
df.to_csv('module_5_auto.csv')
df=df._get_numeric_data()
df.head()

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName,Title):
    width = 12
    height = 10
    plt.figure(figsize=(width,height))
    
    ax1 = sns.kdeplot(RedFunction, color = 'r', label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color = 'b', label=BlueName,ax=ax1)
    
    plt.title(Title)
    plt.xlabel('Price (in dollar)')
    plt.ylabel('Proportion of Cars')
    plt.legend()
    plt.show()
    plt.close()
    
    
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    
y_data = df['price']
x_data=df.drop('price',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# Question #1):
# Use the function "train_test_split" to split up the dataset such that 40% of the data samples will be utilized for testing. Set the parameter "random_state" equal to zero. The output of the function should be the following: "x_train1" , "x_test1", "y_train1" and "y_test1".
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.4,random_state=0)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


lre = LinearRegression()
# We fit the model using the feature "horsepower":
lre.fit(x_train[['horsepower']], y_train)
# Let's calculate the R^2 on the test data:
lre.score(x_test[['horsepower']],y_test)
# We can see the R^2 is much smaller using the test data compared to the training data.
lre.score(x_train[['horsepower']],y_train)

# Question #2): 
# Find the R^2 on the test data using 40% of the dataset for testing.

x_train1, y_train1, x_test1, y_test1 = train_test_split(x_data,y_data,test_size=0.4,random_state=0)
lre.fit(x_test[['horsepower']],y_test)
lre.score(x_test[['horsepower']],y_test)


Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)

print("The mean of the fold is ", Rcross.mean()," and standard deviation is: ",Rcross.std())
-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')


# Question #3): 
# Calculate the average R^2 using two folds, then find the average R^2 for the second fold utilizing the "horsepower" feature:
Rcross1 = cross_val_score(lre, x_data[['horsepower']], y_data, cv=2)
Rcross1.mean()

yhat = cross_val_predict(lre, x_data[['horsepower']],y_data,cv=4)
yhat[0:4]

# Part 2: Overfitting, Underfitting and Model Selection
lr = LinearRegression() 
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_train)
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train[0:4]

yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test[0:4]




# Figure 1: Plot of predicted values using the training data compared to the actual values of the training data.
Title1 = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)",Title1)
# Figure 2: Plot of predicted value using the test data compared to the actual values of the test data.
Title2='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test),Title",Title2)


# Overfitting
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.transform(x_test[['horsepower']]) 

poly = LinearRegression()
poly.fit(x_train_pr, y_train)

yhat_1 = poly.predict(x_test_pr)
yhat_1[0:4]
print("Predicted Value: ",yhat_1[0:4])
print("True Values: ",y_test[0:4].values)

# Figure 3: A polynomial regression model where red dots represent training data, green dots represent test data, and the blue line represents the model prediction.
PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)

poly.score(x_train_pr,y_train)
poly.score(x_test_pr,y_test)

Rsq_test = []
order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures()
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    
    lr.fit(x_train_pr, y_train)
    
    Rsq_test.append(lr.score(x_test_pr, y_test))
    
plt.plot(order, Rsq_test)
plt.xlabel('Order')
plt.ylabel('R^2')
plt.title('R^2 using square data')
plt.text(3, 0.75, 'Maximum R^2')

pr1 = PolynomialFeatures(degree=2)
# Transform the training and testing samples for the features 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg'. Hint: use the method 
x_train_pr1 = pr.fit_transform(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
x_test_pr1 = pr.fit_transform(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])
# How many dimensions does the new feature have? Hint: use the attribute "shape".
x_train_pr1.shape
# Create a linear regression model "poly1". Train the object using the method "fit" using the polynomial features.
poly1 = LinearRegression()
# Create a linear regression model "poly1". Train the object using the method "fit" using the polynomial features.
poly.fit(x_train_pr1,y_train)
# Use the method "predict" to predict an output on the polynomial features, then use the function "DistributionPlot" to display the distribution of the predicted test output vs. the actual test data.
yhat_pr1 = poly.predict(x_test_pr1)
yhat_pr1[0:4]
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_pr1,"Actual Values (Test)", "Predicted Values (Test)", Title)

# Ridge Regression
pr2 = PolynomialFeatures(degree=2)
x_train_pr2 = pr2.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr2 = pr2.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
RidgeModel = Ridge(alpha = 1)
RidgeModel.fit(x_train_pr2,y_train)
yhat_train_pr2 = RidgeModel.predict(x_train_pr2)
yhat_train_pr2[0:4]
print("Predicted", yhat_train_pr2[0:4])
print("Test Set ",y_test[0:4].values)

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)
for alpha in pbar:
    RidgeModel = Ridge(alpha = alpha)
    RidgeModel.fit(x_train_pr2,y_train)
    test_score, train_score = RidgeModel.score(x_test_pr2, y_test), RidgeModel.score(x_train_pr2, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha, Rsqu_test, label='Validation Data')
plt.plot(Alpha, Rsqu_train, 'r', label='Training data')
plt.xlabel('Alpha')
plt.ylabel("R^2")
plt.legend()
plt.show()
plt.close()

# Perform Ridge regression. Calculate the R^2 using the polynomial features,
# use the training data to train the model and use the test data to test the model. 
# The parameter alpha should be set to 10.
RidgeModel1 = Ridge(alpha = 10)
RidgeModel1.fit(x_train_pr2,y_train)
RidgeModel.score(x_test_pr2,y_test)

parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
RR = Ridge()
GridSeacrch = GridSearchCV(RR, parameters1, cv = 4)

GridSeacrch.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
BestRR = GridSeacrch.best_estimator_
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)

# Perform a grid search for the alpha parameter 
# and the normalization parameter, then find the best values of the parameters:
parameters2 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000]}]
Grid2 = GridSearchCV(Ridge(), parameters2, cv=4)
Grid2.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
best_alpha = Grid2.best_params_['alpha']
best_ridge_model = Ridge(alpha=best_alpha)
best_ridge_model.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
