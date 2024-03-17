from matplotlib.pyplot import xscale
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

df = pd.read_csv("pima-indians-diabetes.csv")
df.head()

df.isnull().sum()
df.describe()

X = df.drop(columns='Outcome', axis='columns')
Y = df.Outcome

scaler = StandardScaler().fit_transform(X)
X_scaled = scaler

X_scaled

x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, random_state=10, stratify=Y)
x_train.shape
x_test.shape


log = cross_val_score(LogisticRegression(solver='lbfgs'), X_scaled, Y, cv=5)
log.mean()

# Train using bagging
bagging = BaggingClassifier(
    LogisticRegression(solver='lbfgs'),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)
bagging.fit(x_train, y_train)
bagging.oob_score_ # out of bag
bagging.score(x_test, y_test)

rrandom = cross_val_score(RandomForestClassifier(n_estimators=50), X_scaled, Y, cv=5)
rrandom.mean()

bagging1 = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)
bagging1.fit(x_train, y_train)
bagging1.oob_score_ # out of bag
bagging1.score(x_test, y_test)
