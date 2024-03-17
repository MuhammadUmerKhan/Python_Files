from sklearn.datasets import load_wine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from matplotlib.pyplot import xscale
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("spam.csv")
df.head()

df.groupby('Category').describe()
le = LabelEncoder()
df['spam'] = le.fit_transform(df['Category'])
df.head()

x_train, x_test, y_train, y_test = train_test_split(df.Message, df.spam)

v = CountVectorizer()
x_train_values = v.fit_transform(x_train.values)
x_train_values.toarray()[:2]

model = MultinomialNB()
model.fit(x_train_values, y_train)

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]

email_count = v.transform(emails)
model.predict(email_count) 
X_test_count = v.transform(x_test)
model.score(X_test_count, y_test) # 0.9813352476669059

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(x_train, y_train)
clf.score(x_test, y_test) # 0.9813352476669059
clf.predict(emails)
clf.predict_proba(x_test)

#  -----------
wine = load_wine()
dir(wine)
wine.target

df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
df_wine.head()
df_wine['target'] = wine.target

df_wine.head()

df_wine['target'].unique()
X = df_wine.drop(columns='target')
Y = df_wine.target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

x_train.shape
x_test.shape

model_wine = GaussianNB()
model_wine.fit(x_train, y_train)
model_wine.score(x_test, y_test) # 1.0

clf_wine = Pipeline([
    ('nb', MultinomialNB())
])
clf_wine.fit(x_train, y_train)
clf_wine.score(x_test, y_test) # 0.8888888888888888