from cProfile import label
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

from symbol import parameters

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv'
df = pd.read_csv(filepath)
df.columns
df.head()
df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
y_data = df['Price']
x_data = df.drop('Price',axis =1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

lr = LinearRegression()
lr.fit(x_train[['CPU_frequency']], y_train)
lr.score(x_test[['CPU_frequency']], y_test)
lr.score(x_train[['CPU_frequency']],y_train)

Rcross = cross_val_score(lr, x_data[['CPU_frequency']], y_data, cv=4)
print("The mean of the fold is ", Rcross.mean()," and standard deviation is: ",Rcross.std())

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=1)


Rsq_test = []
order = [1, 2, 3, 4, 5]
for i in order:
    pr = PolynomialFeatures()
    x_train_pr = pr.fit_transform(x_train[['CPU_frequency']])  
    x_test_pr = pr.fit_transform(x_test[['CPU_frequency']])
    lr.fit(x_train_pr, y_train)
    Rsq_test.append(lr.score(x_test_pr, y_test))
    
plt.plot(order, Rsq_test)
plt.xlabel('Order')
plt.ylabel('R^2')
plt.title('R^2 using test data')
# plt.legend()
plt.show()
plt.close()

pr1 = PolynomialFeatures(degree=2)
x_train_pr1 = pr1.fit_transform(x_train[['CPU_frequency','RAM_GB', 'Storage_GB_SSD','CPU_core','GPU','Category']])
x_test_pr1 = pr1.fit_transform(x_test[['CPU_frequency','RAM_GB', 'Storage_GB_SSD','CPU_core','GPU','Category']])

Rsqu_test = []
Rsqu_train = []
dummy = []
Alpha = np.arange(0.001, 1, 0.001)
pbar = tqdm(Alpha)
for alpha in pbar:
    RidgeModel = Ridge(alpha=alpha)
    RidgeModel.fit(x_train_pr1, y_train)
    test_score, train_score = RidgeModel.score(x_test_pr1, y_test), RidgeModel.score(x_train_pr1, y_train)
    pbar.set_postfix({"Test score": test_score, "Train Score": train_score})
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

width = 10
height = 6
plt.figure(figsize=(width, height))
plt.plot(Alpha, Rsqu_test, label='Validation Data')
plt.plot(Alpha, Rsqu_train,'r', label='Train Data')
plt.xlabel('Alpha')
plt.ylabel('R^2')
# plt.ylim(0, 1)
plt.legend()
plt.show()
plt.close()

parameters = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}]
RR = Ridge()
GridSearch = GridSearchCV(RR, parameters, cv=4)
GridSearch.fit(x_train[['CPU_frequency','RAM_GB', 'Storage_GB_SSD','CPU_core','GPU','Category']],y_train)
BestRR = GridSearch.best_estimator_
GridSearch.fit(x_test[['CPU_frequency','RAM_GB', 'Storage_GB_SSD','CPU_core','GPU','Category']],y_test)
print(BestRR.score(x_test[['CPU_frequency','RAM_GB', 'Storage_GB_SSD','CPU_core','GPU','Category']],y_test))