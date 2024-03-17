from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn import svm
from cProfile import label
from turtle import color, pos
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv'
df = pd.read_csv(url)
df.head()
df.to_csv('cell_samples.csv', index=False)
df = pd.read_csv('cell_samples.csv')
# df.head()
df.drop(columns='Unnamed: 0', inplace=True)
df.head()
cell_df = df
cell_df.head()

cell_df.shape
cell_df.columns
cell_df.describe()
cell_df[['Class']]

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

# Data pre-processing and selectionc
cell_df.dtypes
cell_df[['BareNuc']]
# one row contain non numeric data now we are dropping this row
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
feature_df.head()
x = np.asarray(feature_df)
x[0:5]

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
y[0:5]

# Train/Test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
x_train.shape
x_test.shape
y_train.shape
y_test.shape

# Modeling (SVM with Scikit-learn)
clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)
yhat = clf.predict(x_test)
yhat[0:5]

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
    
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

f1_score(y_test, yhat, average='weighted')

jaccard_score(y_test, yhat, pos_label=2)

# Practice
# Can you rebuild the model, but this time with a __linear__ kernel? 
# You can use __kernel='linear'__ option, when you define the svm. 
# How the accuracy changes with the new kernel function?
clf1 = svm.SVC(kernel='linear')
clf1.fit(x_train, y_train)
yhat1 = clf1.predict(x_test)
yhat1[0:5]
yhat[0:5]
print("Average f1-score is : %.4f"% f1_score(y_test, yhat1, average='weighted'))
print("Jaccord SCore is: %.4f"% jaccard_score(y_test, yhat1, pos_label=2))