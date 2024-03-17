from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import seaborn as sns
from sklearn.datasets import load_digits
from sqlalchemy import true

digits = load_digits()

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])
    
dir(digits)
digits.data[0]

lr = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
lr.fit(x_train, y_train)
lr.predict(x_test)
y_test

lr.score(x_test, y_test)
lr.predict(digits.data[:5])

y_pred = lr.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm

sns.heatmap(cm, annot=True)

iris = load_iris()
dir(iris)

iris.data[0]
iris.target_names

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2)
lr1 = LogisticRegression()
lr1.fit(X_train, Y_train)
lr1.predict(X_test)

lr1.predict(iris.data[:5])
Y_test

y_pred1 = lr1.predict(X_test)
cm = confusion_matrix(Y_test, y_pred1)
cm

sns.heatmap(cm, annot=True)


target_index = 98
target_index_predicted = lr1.predict([iris.data[target_index]])
iris.target_names[target_index_predicted]

iris.target_names[iris.target[target_index]]
cm

plt.figure(figsize=(5,3))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()