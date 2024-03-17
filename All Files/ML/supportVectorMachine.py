import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_digits

iris = load_iris()
dir(iris)
iris.feature_names
iris.target_names

df  = pd.DataFrame(iris.data, columns = iris.feature_names)
df.head()

df['target'] = iris.target
df.head()

len(df[df['target']==1])
len(df[df['target']==2])
len(df[df['target']==0])

df['flower_name'] = df['target'].apply(lambda x: iris.target_names[x])
df.head()

X = df.drop(columns=['target', 'flower_name'])
Y = df.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
len(x_train)
len(x_test)
model = SVC()
model.fit(x_train, y_train)
x_test
model.predict([[6.4, 2.8, 5.6, 2.1]])
model.score(x_test, y_test)
y_test

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')


plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')

# Tune Parameters 
# 1. C (Regularization)

model_C = SVC(C=1)
model_C.fit(x_train, y_train)
model_C.score(x_test, y_test)

model_C = SVC(C=10)
model_C.fit(x_train, y_train)
model_C.score(x_test, y_test)

model_g = SVC(gamma=10)
model_g.fit(x_train, y_train)
model_g.score(x_test, y_test)

model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(x_train, y_train)
model_linear_kernal.score(x_test, y_test)

# ------------
digit = load_digits()
dir(digit)
digit.data
digit.target_names
df_digit = pd.DataFrame(digit.data,digit.target)
df_digit.head()

df_digit['target'] = digit.target
df_digit.head()

X1 = df_digit.drop(columns='target')
Y1 = df_digit['target']
x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.8, random_state=4)
len(x_train1)
len(x_test1)
df_digit.shape
df_digit.isnull().sum()

model_digit = SVC(kernel='linear')
model_digit.fit(x_train1, y_train1)
model_digit.score(x_test1, y_test1)
y_test1
model_digit.predict(x_test1)