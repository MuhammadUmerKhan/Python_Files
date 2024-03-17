from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from cgi import test
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sqlalchemy import column

digits = load_digits()

df = pd.DataFrame(digits.data, digits.target)

df.head()
dir(digits)
df['target'] = digits.target
df.head()
X = df.drop(columns='target')
Y = df['target']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

len(x_train)
len(x_test)
# Logistic Regresssion
lr = LogisticRegression(solver='lbfgs')
lr.fit(x_train, y_train)
lr.score(x_test, y_test)

# SVM
model_svm = SVC(gamma='auto')
model_svm.fit(x_train, y_train)
model_svm.score(x_test, y_test)

# Random Forest
model_R = RandomForestClassifier(n_estimators=40)
model_R.fit(x_train, y_train)
model_R.score(x_test, y_test)

# KFold Cross validation
kf = KFold(n_splits=3)
kf
for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print(train_index, test_index)
    
# Use KFold for our digits example
def getScore(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

folds = StratifiedKFold(n_splits=3)
scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    scores_logistic.append(getScore(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))  
    scores_svm.append(getScore(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    scores_rf.append(getScore(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))
    
    
scores_logistic
scores_svm
scores_rf

# OR
cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), X, Y, cv=3) 
cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), X, Y, cv=5) 

cross_val_score(RandomForestClassifier(n_estimators=40), X, Y, cv=3)
cross_val_score(SVC(gamma='auto'), X, Y, cv=3)

score1 = cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), X, Y, cv=10) 
np.average(score1)

score2 = cross_val_score(RandomForestClassifier(n_estimators=40), X, Y, cv=10)
np.average(score2)

score3 = cross_val_score(SVC(gamma='auto'), X, Y, cv=3)
np.average(score3)

# ----------------------
iris = load_iris()
dir(iris)
df_iris = pd.DataFrame(iris.data,columns = iris.feature_names)
df_iris['target'] = iris.target
df_iris['flower_names'] = df_iris['target'].apply(lambda x: iris.target_names[x])
df_iris.head()
X_iris = df_iris.drop(columns=['target', 'flower_names'])
Y_iris = df_iris['target']

score_log = cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), X_iris, Y_iris, cv=10)
score_svm = cross_val_score(SVC(gamma='auto'), X_iris, Y_iris, cv=10)
score_randomForest = cross_val_score(RandomForestClassifier(n_estimators=40), X_iris, Y_iris, cv=10)
score_dTree = cross_val_score(DecisionTreeClassifier(), X_iris, Y_iris, cv=10)

np.average(score_log)
np.average(score_dTree)
np.average(score_svm)
np.average(score_randomForest)