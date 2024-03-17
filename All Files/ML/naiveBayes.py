from matplotlib.pyplot import xscale
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
df = pd.read_csv("titanic.csv")
df.head()

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.isnull().sum()
df.shape
df['Age'].replace(np.NaN, df.Age.median(), inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

df.head()

X = df.drop(columns='Survived')
Y = df.Survived
x_scaled = StandardScaler().fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, Y, test_size=0.2, stratify=Y, random_state=4)

model = GaussianNB()
model.fit(x_train, y_train)
model.score(x_test, y_test) # 0.7597765

cross_score = cross_val_score(GaussianNB(), x_scaled, Y, cv=10)
cross_score.mean() # 0.7766666666666666