import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

from symbol import trailer

df_salary = pd.read_csv("salaries.csv")
df_salary.head()

inputs = df_salary.drop('salary_more_then_100k', axis='columns')
target = df_salary['salary_more_then_100k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['compnay_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

inputs.drop(columns=['company', 'job', 'degree'], inplace=True, axis='columns')

inputs.head()

model = tree.DecisionTreeClassifier()
model.fit(inputs, target)
model.score(inputs, target)
#              [[google, business manager, masters]] 
model.predict([[2, 0, 1]])
# -------------------------------
df_titanic = pd.read_csv('titanic.csv')
df_titanic.head()
# input variable
inputs_titanic = df_titanic[['Pclass', 'Sex', 'Age', 'Fare']]
# target variable
target_titanic = df_titanic['Survived']

inputs_titanic.isnull().sum()
inputs_titanic['Age'].median()
inputs_titanic['Age'].replace(np.NAN, inputs_titanic['Age'].median(), inplace=True)
target_titanic.isnull().sum()

inputs_titanic.info()
le_sex = LabelEncoder()
inputs_titanic['Sex_n'] = le_sex.fit_transform(inputs_titanic['Sex'])
inputs_titanic.drop(columns='Sex', axis='columns', inplace=True)
inputs_titanic['Age'] = inputs_titanic['Age'].astype('int')
inputs_titanic

x_train, x_test, y_train, y_test = train_test_split(inputs_titanic, target_titanic, test_size=0.2, random_state=4)
len(x_train)
len(x_test)
len(y_train)
len(y_test)

model_titanic = tree.DecisionTreeClassifier()
model_titanic.fit(x_train, y_train)
model_titanic.score(x_test, y_test)
model_titanic.predict([[3, 19, 7.8958, 1]])