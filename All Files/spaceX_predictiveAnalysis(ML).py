from sklearn.model_selection import GridSearchCV
from re import X
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv("dataset_part_2.csv")
data.head()

X = pd.read_csv("dataset_part_3.csv")
X.head()

def plot_confusion_metrics(y, y_predict):
    "This function confusion metrics"
    
    cm = confusion_matrix(y, y_predict)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show();
    
# TASK 1
# Create a NumPy array from the column Class in data, 
# by applying the method to_numpy() then assign it to the variable Y,
# make sure the output is a Pandas series (only one bracket df['name of column']).
Y = data['Class'].to_numpy()

# TASK 2
# Standardize the data in X then reassign it to the variable X 
# using the transform provided below.
transform = preprocessing.StandardScaler().fit(X)
X = transform.transform(X)
# X
# TASK 3
# Use the function train_test_split to split the data X and Y 
# into training and test data. Set the parameter test_size to 0.2 and random_state to 2. 
# The training data and test data should be assigned to the following labels.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

X_train.shape, Y_train.shape
X_test.shape, Y_test.shape

# TASK 4
# Create a logistic regression object then create a GridSearchCV object 
# logreg_cv with cv = 10. Fit the object to find the best parameters from \
# the dictionary parameters.
parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}
lr = LogisticRegression()
logreg_cv = GridSearchCV(lr, parameters, cv=10)
logreg_cv.fit(X_train, Y_train)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# TASK 5
# Calculate the accuracy on the test data using the method score:
logreg_cv.score(X_test, Y_test)
yhat = logreg_cv.predict(X_test)
yhat

plot_confusion_metrics(Y_test, yhat)

# TASK 6
# Create a support vector machine object then create a GridSearchCV object 
# svm_cv with cv - 10. Fit the object to find the best parameters from the 
# dictionary parameters.
parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()
svm_vc = GridSearchCV(svm, parameters, cv=10)
svm_vc.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",svm_vc.best_params_)
print("accuracy :",svm_vc.best_score_)

# TASK 7
# Calculate the accuracy on the test data using the method score:
svm_vc.score(X_test, Y_test)

yhat_svm = svm_vc.predict(X_test)
yhat_svm

plot_confusion_metrics(Y_test, yhat_svm)

# TASK 8
# Create a decision tree classifier object then create a GridSearchCV object 
# tree_cv with cv = 10. Fit the object to find the best parameters from the 
# dictionary parameters.
parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, parameters, cv=10)
tree_cv.fit(X_train, Y_train)

yhat_tree = tree_cv.predict(X_test)

print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)

tree_cv.score(X_test, Y_test)
plot_confusion_metrics(Y_test, yhat_tree)


# TASK 10
# Create a k nearest neighbors object then create a GridSearchCV object 
# knn_cv with cv = 10. Fit the object to find the best parameters from the 
# dictionary parameters.
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()
knn_cv = GridSearchCV(KNN, parameters, cv=10)
knn_cv.fit(X_train, Y_train)
print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)

yhat_knn = knn_cv.predict(X_test)
knn_cv.score(X_test, Y_test)

plot_confusion_metrics(Y_test, yhat_knn)

# TASK 12
# Find the method performs best:
print(f'LogReg: {logreg_cv.score(X_test,Y_test)}, SVM: {svm_vc.score(X_test,Y_test)}, Tree: {tree_cv.score(X_test,Y_test)}, KNN: {knn_cv.score(X_test,Y_test)}')