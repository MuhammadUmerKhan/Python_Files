from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
df = pd.read_csv(path)
df.head()
df.shape
df[['custcat']].value_counts()

df.hist(column='income', bins = 20)

df.columns
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values
X[0:5]

y = df['custcat'].values
y[0:5]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set: ', x_train.shape, y_train.shape)
print('Test set: ', x_test.shape, y_test.shape)

# Normalize Data
# Data Standardization gives the data zero mean and unit variance, 
# it is good practice, especially for algorithms such as KNN which 
# is based on the distance of data points:
x_train_norm = preprocessing.StandardScaler().fit(x_train).transform(x_train.astype(float))
# x_train_norm[0:5]

# Classification
# K nearest neighbor (KNN)
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(x_train_norm, y_train)
x_test_norm = preprocessing.StandardScaler().fit(x_test).transform(x_test.astype(float))
# x_test_norm[0:5]

# Predicting
yhat = neigh.predict(x_test_norm)
yhat[0:5]
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train_norm)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
# Practice
# Can you build the model again, but this time with k=6?
k1 = 6
x_train_norm1 = preprocessing.StandardScaler().fit(x_train).transform(x_train.astype(float))
x_test_norm1 = preprocessing.StandardScaler().fit(x_test).transform(x_test.astype(float))
neigh = KNeighborsClassifier(n_neighbors=k1).fit(x_train_norm1, y_train)
yhat1 = neigh.predict(x_test_norm1)
yhat1[0:5]


print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train_norm1)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train_norm,y_train)
    yhat=neigh.predict(x_test_norm)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

# mean_acc
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 