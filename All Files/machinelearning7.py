# Logistic Regression
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pylab as py
url ="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
df = pd.read_csv(url)

# df.to_csv('ChurnData.csv', index=False)
churn_df = pd.read_csv("ChurnData.csv")

churn_df.head()
churn_df.dtypes

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.dtypes
churn_df.head()
churn_df.shape

x = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
x[0:5]

y = np.asarray(churn_df['churn'])
y[0:5]

x = preprocessing.StandardScaler().fit(x).transform(x)
x[0:5]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
x_train.shape
x_test.shape
y_train.shape
y_test.shape

# Modeling (Logistic Regression with Scikit-learn)
LR = LogisticRegression(C=0.001, solver='liblinear').fit(x_train, y_train)
yhat = LR.predict(x_test)
yhat[0:5]

yhat_prob = LR.predict_proba(x_test)
yhat_prob[0:5]

# Evaluation
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

print(classification_report(y_test, yhat))

# log loss
log_loss(y_test, yhat_prob)


# Practice
# Try to build Logistic Regression model 
# again for the same dataset, but this time, use different 
# __solver__ and __regularization__ values? What is new __logLoss__ value?
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=4)
x_train1.shape
x_test1.shape
y_train1.shape
y_test1.shape

LR1 = LogisticRegression(C=0.001, solver='newton-cg').fit(x_train1, y_train1)
yhat1 = LR1.predict(x_test1)
yhat1[0:5]
yhat_prob1 = LR1.predict_proba(x_test1)
yhat_prob1[0:5]
log_loss(y_test1, yhat_prob1)

LR2 = LogisticRegression(C=0.001, solver='sag').fit(x_train1, y_train1)
yhat2 = LR2.predict(x_test1)
yhat1[0:5]
yhat_prob2 = LR2.predict_proba(x_test1)
yhat_prob2[0:5]
print('Log-loss: %.2f' % log_loss(y_test1, yhat_prob2))