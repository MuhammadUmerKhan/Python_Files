import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\creditcard.csv\creditcard.csv")
df.head()

df.describe()


df.isnull().sum()

df.dtypes
df.shape # (284807, 31)
df.columns
# Treating Outliers
# z_score = (df['Time'] - df['Time'].mean())/df['Time'].std()
# df['Z_score'] = z_score
# df1 = df[(df['Z_score']>-3) & (df['Z_score']<3)]
# df1.drop(columns='Z_score', inplace=True)

for i in df.columns:
    if i != 'Class':
        df['Z_score'] = (df[i] - df[i].mean())/df[i].std()
        df = df[(df['Z_score']>-3) & (df['Z_score']<3)]
        df.drop(columns='Z_score', inplace=True)
df.shape # (207614, 31)
df.columns
#  EDA
dfClass = df['Class'].value_counts()

sns.barplot(x=dfClass.index, y=dfClass.values)
plt.title("Title Distribution")
plt.show()

df.head()
legit = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

fraud.describe()
legit.shape
fraud.shape

# Modeling
X = df.drop(columns='Class')
X.columns
Y = df['Class']
df.shape
# 
x_scaled = StandardScaler().fit_transform(X)
x_scaled

x_train, x_test, y_train, y_test = train_test_split(x_scaled, Y, test_size=0.2,random_state=4, stratify=Y)
x_train.shape
x_test.shape

lr = LogisticRegression(solver='lbfgs')
lr.fit(x_train, y_train)
lr.score(x_test, y_test) # 0.9998555017701033

cm = confusion_matrix(y_test, lr.predict(x_test))
LABELS = ['Normal', 'Fraud']
sns.heatmap(cm, annot=True, xticklabels=LABELS, yticklabels=LABELS, fmt='d')
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show()

# cross val score
log_score = cross_val_score(LogisticRegression(solver='lbfgs'), x_scaled, Y, cv=5)
log_score_mean = np.average(log_score) # 0.9998651344054231

# Support Vector Machine
svm = SVC(gamma='auto')
svm.fit(x_train, y_train)
svm.score(x_test, y_test) # 0.9998555017701033
svm_score = cross_val_score(SVC(gamma='scale'), x_scaled, Y, cv=5)
svm_score_mean = np.average(svm_score) # 0.9998651344054231

# Random Forest
random = RandomForestClassifier(n_estimators=30)
random.fit(x_train, y_train)
random.score(x_test, y_test) # 0.9998555017701033
random_score = cross_val_score(RandomForestClassifier(n_estimators=40), x_scaled, Y, cv=5)
random_score_mean = np.average(random_score) # 0.9998266015441173


dict = {'Model' :['LogisticRegression', 'SVC', 'RandomForestClassifier'],
        'Score': [log_score_mean, svm_score_mean, random_score_mean]}
model = pd.DataFrame(dict)
model