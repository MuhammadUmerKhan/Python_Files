from snapml import SupportVectorMachine
from sklearn.svm import LinearSVC
from snapml import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from cProfile import label
import time
import opendatasets as od
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

url = "C:\DATA SCIENCE\Python-git-files\dataset\creditCardFraud\creditcard.csv"
df = pd.read_csv(url)

print("There are " + str(len(df)) + " observations in the credit card fraud dataset.")
print("There are " + str(len(df.columns)) + " variables in the dataset.")

df.shape
df.head()

n_replicas = 10
big_raw_data = pd.DataFrame(np.repeat(df.values, n_replicas, axis = 0), columns=df.columns)
big_df = big_raw_data
big_raw_data.shape

labels = big_df.Class.unique()
sizes = big_df.Class.value_counts().values

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct="%1.3f%%")
ax.set_title("Target Variable Value Counts")
plt.show()



plt.hist(big_df.Amount.values, 6, histtype='bar', facecolor='g')
plt.show()
print("Minimum amount value is {}".format(np.min(big_df.Amount.values)))
print("Maximum amount value is {}".format(np.max(big_df.Amount.values)))
print("90% of the transactions have an amount less or equal than {}".format(np.percentile(df.Amount.values, 90)))


# Dataset Preprocessing
big_df.iloc[:, 1:30] = StandardScaler().fit_transform(big_df.iloc[:, 1:30])
df_matrix = big_df.values

# X: feature matrix (for this analysis, 
# we exclude the Time variable from the dataset)
x = df_matrix[:, 1:30]
# y: label matrix
y = df_matrix[:, 30]
# data normalization
x = normalize(x, norm="l1")
print("x.shape is {}, y.shape is {}".format(x.shape, y.shape))

# Dataset Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# Checking Shapes training and testing data
x_train.shape
y_train.shape
x_test.shape
y_test.shape

# Build a Decision Tree Classifier model with Scikit-Learn
# compute the sample weights to be used as input to the train routine so that 
# it takes into account the class imbalance present in this dataset
w_train = compute_sample_weight('balanced', y_train)
# for reproducible output across multiple function calls, set random_state to a given integer value
sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)

# train a Decision Tree Classifier using scikit-learn
t0 = time.time()
sklearn_dt.fit(x_train, y_train, sample_weight=w_train)
sklearn_time = time.time()-t0
 
# Build a Decision Tree Classifier model with Snap ML
# if not already computed, 
# compute the sample weights to be used as input to the train routine so that 
# it takes into account the class imbalance present in this dataset
w_train = compute_sample_weight('balanced', y_train)

# Snap ML offers multi-threaded CPU/GPU training of decision trees, unlike scikit-learn
# to use the GPU, set the use_gpu parameter to True
snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, use_gpu=True)

# to set the number of CPU threads used at training time, set the n_jobs parameter
# for reproducible output across multiple function calls, set random_state to a given integer value
snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)

# train a Decision Tree Classifier model using Snap ML
t0 = time.time()
snapml_dt.fit(x_train, y_train, sample_weight=w_train)
snapml_time = time.time()-t0
print("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))


# Evaluate the Scikit-Learn and Snap ML Decision Tree Classifier Models
# Snap ML vs Scikit-Learn training speedup
training_speedup = sklearn_time/snapml_time
print('[Decision Tree Classifier] Snap ML vs. Scikit-Learn speedup : {0:.2f}x '.format(training_speedup))

# run inference and compute the probabilities of the test samples 
# to belong to the class of fraudulent transactions
sklearn_pred = sklearn_dt.predict_proba(x_test)[:,1]

# evaluate the Compute Area Under the Receiver Operating Characteristic 
# Curve (ROC-AUC) score from the predictions
sklearn_roc_auc = roc_auc_score(y_test, sklearn_pred)
print('[Scikit-Learn] ROC-AUC score : {0:.3f}'.format(sklearn_roc_auc))

# run inference and compute the probabilities of the test samples
# to belong to the class of fraudulent transactions
snapml_pred = snapml_dt.predict_proba(x_test)[:,1]

# evaluate the Compute Area Under the Receiver Operating Characteristic
# Curve (ROC-AUC) score from the prediction scores
snapml_roc_auc = roc_auc_score(y_test, snapml_pred)   
print('[Snap ML] ROC-AUC score : {0:.3f}'.format(snapml_roc_auc))


# Build a Support Vector Machine model with Scikit-Learn

# instatiate a scikit-learn SVM model
# to indicate the class imbalance at fit time, set class_weight='balanced'
# for reproducible output across multiple function calls, set random_state to a given integer value
sklearn_svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

# train a linear Support Vector Machine model using Scikit-Learn
t0 = time.time()
sklearn_svm.fit(x_train, y_train)
sklearn_time = time.time() - t0
print("[Scikit-Learn] Training time (s):  {0:.2f}".format(sklearn_time))

# Build a Support Vector Machine model with Snap ML

# in contrast to scikit-learn's LinearSVC, Snap ML offers multi-threaded CPU/GPU training of SVMs
# to use the GPU, set the use_gpu parameter to True
snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, use_gpu=True, fit_intercept=False)

# to set the number of threads used at training time, one needs to set the n_jobs parameter
snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)
print(snapml_svm.get_params())

# train an SVM model using Snap ML
t0 = time.time()
model = snapml_svm.fit(x_train, y_train)
snapml_time = time.time() - t0
print("[Snap ML] Training time (s):  {0:.2f}".format(snapml_time))

# Evaluate the Scikit-Learn and Snap ML Support Vector Machine Models
# compute the Snap ML vs Scikit-Learn training speedup
training_speedup = sklearn_time/snapml_time
print('[Support Vector Machine] Snap ML vs. Scikit-Learn training speedup : {0:.2f}x '.format(training_speedup))

# run inference using the Scikit-Learn model
# get the confidence scores for the test samples
sklearn_pred = sklearn_svm.decision_function(X_test)

# evaluate accuracy on test set
acc_sklearn  = roc_auc_score(y_test, sklearn_pred)
print("[Scikit-Learn] ROC-AUC score:   {0:.3f}".format(acc_sklearn))

# run inference using the Snap ML model
# get the confidence scores for the test samples
snapml_pred = snapml_svm.decision_function(X_test)

# evaluate accuracy on test set
acc_snapml  = roc_auc_score(y_test, snapml_pred)
print("[Snap ML] ROC-AUC score:   {0:.3f}".format(acc_snapml))

# In this section we will evaluate the quality of the SVM models trained 
# using the hinge loss metric (
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html).
# Run inference on the test set using both Scikit-Learn and Snap ML models. Compute the hinge loss metric for both sets of predictions.
# Print the hinge losses of Scikit-Learn and Snap ML.
# get the confidence scores for the test samples
sklearn_pred = sklearn_svm.decision_function(x_test)
snapml_pred  = snapml_svm.decision_function(x_test)

# evaluate the hinge loss from the predictions
loss_snapml = hinge_loss(y_test, snapml_pred)
print("[Snap ML] Hinge loss:   {0:.3f}".format(loss_snapml))

# evaluate the hinge loss metric from the predictions
loss_sklearn = hinge_loss(y_test, sklearn_pred)
print("[Scikit-Learn] Hinge loss:   {0:.3f}".format(loss_snapml))

# the two models should give the same Hinge loss