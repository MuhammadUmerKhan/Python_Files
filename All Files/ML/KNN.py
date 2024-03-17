import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_digits, load_iris, load_breast_cancer

iris = load_iris()

df_iris = pd.DataFrame(iris.data,columns = iris.feature_names)
df_iris.head()

df_iris['target'] = iris.target
df_iris.head()

df_iris['flower_name'] =df_iris.target.apply(lambda x: iris.target_names[x])

df0 = df_iris[:50]
df1 = df_iris[50:100]
df2 = df_iris[100:]

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')
X = df_iris.drop(columns=['target', 'flower_name'])
Y = df_iris['target']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
knn.score(x_test, y_test)
x_test
knn.predict([[6.4,	2.8, 5.6, 2.1]])
y_pred = knn.predict(x_test)
y_pred
y_test
df_iris['target'].unique()
df_iris['flower_name'].unique()

cm = confusion_matrix(y_test, y_pred)

cm

sns.heatmap(cm, annot=True)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

# -----------------------
digit = load_digits()
df_digit = pd.DataFrame(digit.data, digit.target)

df_digit.head()
knn = KNeighborsClassifier(n_neighbors=5)
df_digit['target'] = digit.target
df_digit.head()

X = df_digit.drop(columns='target')
Y = df_digit['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
len(x_train)
knn.fit(x_train, y_train)
knn.score(x_test, y_test)

cm = confusion_matrix(y_test, knn.predict(x_test))
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

print(classification_report(y_test, knn.predict(x_test)))