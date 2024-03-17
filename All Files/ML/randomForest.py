import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits, load_iris
from sqlalchemy import column


digits = load_digits()
dir(digits)

plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])
    
df = pd.DataFrame(digits.data)
df.head()
df['target'] = digits.target
df.head()

X = df.drop(columns='target')
Y = df.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
len(x_train)
len(x_test)

model = RandomForestClassifier(n_estimators=50)
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

model.score(x_test, y_test)
cm = confusion_matrix(y_test, y_predicted)


plt.figure(figsize=(20, 10))
sns.heatmap(cm, annot=True)
plt.ylabel('Predicted')
plt.xlabel('Truth')
plt.show()

# ---------------
irirs = load_iris()
dir(irirs)
irirs.feature_names
df_iris = pd.DataFrame(irirs.data, columns = irirs.feature_names)
df_iris.head()

df_iris['target'] = irirs.target
df_iris.head()
irirs.target_names
df_iris['flower_name'] = df_iris['target'].apply(lambda x: irirs.target_names[x])

df_iris.head()

X = df_iris.drop(columns=['target', 'flower_name'])
Y = df_iris['target']

x_train1, x_test1, y_train1, y_test1 = train_test_split(X, Y, test_size=0.2, random_state=4)
len(x_train1)
len(x_test1)

model_flower = RandomForestClassifier(n_estimators=100)
model_flower.fit(x_train1, y_train1)
model_flower.score(x_test1, y_test1)
model_flower.predict(x_test1)

