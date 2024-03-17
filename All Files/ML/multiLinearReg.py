import math
import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n

homePrice_df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\homeprices multi reg.csv")
homePrice_df.head()

homePrice_df.isnull().sum()

nullMedian = homePrice_df.bedrooms.median()

homePrice_df.bedrooms = homePrice_df.bedrooms.fillna(nullMedian)

homePrice_df.head()

X = homePrice_df[['area', 'bedrooms', 'age']]
Y = homePrice_df['price']

lr = linear_model.LinearRegression()
lr.fit(X, Y)
# Predict randomvalues
lr.predict([[5000, 1, 3]])

# 
hiring_df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\hiring.csv")
hiring_df.head()
hiring_df.info()
def word_to_number(row):
    if isinstance(row, str) and row.strip().isalpha():
        return w2n.word_to_num(row)
    else:
        return np.nan

hiring_df['experience'] = hiring_df['experience'].apply(word_to_number)
hiring_df
hiring_df['experience'] = hiring_df['experience'].fillna(0)
median_test_score = math.floor(hiring_df['test_score(out of 10)'].median())
hiring_df['test_score(out of 10)'] = hiring_df['test_score(out of 10)'].fillna(median_test_score)
hiring_df.columns


X1 = hiring_df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']]
Y1 = hiring_df['salary($)']

lr_hiring = linear_model.LinearRegression()
lr_hiring.fit(X1, Y1)
lr_hiring.predict([[2, 9, 6]])
lr_hiring.predict([[12, 10, 10]])