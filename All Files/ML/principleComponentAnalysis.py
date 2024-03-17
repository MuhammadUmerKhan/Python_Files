from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

digit = load_digits()
df_digits = pd.DataFrame(digit.data, digit.target)
df_digits.head()
df_digits['target'] = digit.target

df_digits.head()

X = df_digits.drop(columns='target')
Y = df_digits.target

scaler = StandardScaler().fit_transform(X)
X_scaled = scaler

X_scaled

x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=4)
len(x_train)
len(x_test)

model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)
model.score(x_test, y_test) # 0.972222

pca = PCA(0.95)
X_pca = pca.fit_transform(X)
X_scaled.shape
X_pca.shape

pca.explained_variance_ratio_
pca.n_components_


pca = PCA(n_components=30)
X_pca = pca.fit_transform(X)
x_train_pca, x_test_pca, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=4)
model = LogisticRegression(solver='liblinear')
model.fit(x_train_pca, y_train)
model.score(x_test_pca, y_test) # 0.9666
