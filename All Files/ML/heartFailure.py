from matplotlib.pyplot import xscale
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("heart_No_Outlier.csv")
df.head()
df.shape
df.describe()
df.columns

le = LabelEncoder()
df['RestingECG'] = le.fit_transform(df['RestingECG'])

df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])

df['ST_Slope'] = le.fit_transform(df['ST_Slope'])

df2 = pd.get_dummies(df)
df2.head()
df2.columns

df2[['Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA']] = df2[['Sex_F',
       'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP',
       'ChestPainType_TA']].astype('int')
df2.head()
df2.isnull().sum()

X = df2.drop(columns='HeartDisease')
Y = df2.HeartDisease

x_scaled = StandardScaler().fit_transform(X)
# Train a model using svm tree and then using bagging
x_train, x_test, y_train, y_test = train_test_split(x_scaled, Y, stratify=Y, random_state=10)
x_train.shape
x_test.shape
# 225/674
svc = SVC(gamma='auto')
svc.fit(x_train, y_train)
svc.score(x_test, y_test) # 0.906666666

score = cross_val_score(SVC(gamma='auto'), x_scaled, Y, cv=5)
score.mean()
score

bagging = BaggingClassifier(
    SVC(gamma='auto'), n_estimators=50, max_samples=0.8,
    oob_score=True, random_state=0
)
bagging.fit(x_train, y_train)
bagging.oob_score_
bagging.score(x_test, y_test) # 0.906666666

# Train a model using decision tree and then using bagging
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
tree.score(x_test, y_test) # 0.81777777

score_ = cross_val_score(DecisionTreeClassifier(), x_scaled, Y, cv=5)
score_.mean()

bagging_ = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=50, max_samples=0.8, oob_score=True, random_state=0
)
bagging_.fit(x_train, y_train)
bagging_.score(x_test, y_test) # 0.8755555555555555