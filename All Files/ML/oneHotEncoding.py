import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df_homePrice = pd.read_csv("homeprices.csv")
df_homePrice.shape
df_homePrice.head(13)

dummies = pd.get_dummies(df_homePrice['town'])
dummies

dummies = dummies.astype('int')

merged = pd.concat([df_homePrice, dummies], axis='columns')
merged

final = merged.drop(['town'], axis='columns')
final

final.drop(['west windsor'], axis='columns', inplace=True)
final

X = final.drop(['price'], axis='columns')

Y = final.price

lr = linear_model.LinearRegression()
lr.fit(X, Y)

lr.predict(X)

lr.score(X, Y)

lr.predict([[3400, 0, 0]])

# Using sklearn OneHotEncoder
le = LabelEncoder()
dfle = df_homePrice
dfle.town = le.fit_transform(dfle.town)
dfle

X1 = dfle[['town', 'area']].values
X1

Y1 = dfle.price.values
Y1

ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder='passthrough')
ct

X1 = ct.fit_transform(X1)
X1

X1 = X1[:, 1:]
X1

lr.fit(X1, Y1)

lr.predict([[0, 1, 3400]])
lr.predict([[1, 0, 2800]])
# ---------------------------
df_carPrice = pd.read_csv("carprices.csv")
df_carPrice.shape
df_carPrice.head(13)

plt.scatter(df_carPrice.Mileage, df_carPrice['Sell Price($)'], marker='+', color='red')
plt.xlabel("Mileage")
plt.ylabel("Sell Price($)")
plt.show()

df_carPrice
dummies_carPrice = pd.get_dummies(df_carPrice['Car Model'])
dummies_carPrice
dummies_carPrice = dummies_carPrice.astype('int')
dummies_carPrice

final = pd.concat([df_carPrice, dummies_carPrice], axis='columns')
final

final.drop(columns=['Car Model'], inplace=True)
final

final.drop(columns='Mercedez Benz C class', inplace=True)
final

X_carPrice = final.drop(columns=['Sell Price($)'])
Y_carPrice = final['Sell Price($)']

lr_carPrice = linear_model.LinearRegression()
lr_carPrice.fit(X_carPrice, Y_carPrice)

lr_carPrice.predict([[45000, 4, 0, 0]])
lr_carPrice.predict([[86000, 7, 0, 1]])

# using sklearn one encoder
le_carPrice = LabelEncoder()
df_carPriceLe = df_carPrice.copy()

df_carPriceLe['Car Model'] = le_carPrice.fit_transform(df_carPriceLe['Car Model'])
df_carPriceLe

X1_carPrice = df_carPriceLe[['Car Model', 'Mileage', 'Age(yrs)']].values
Y1_carPrice = df_carPriceLe['Sell Price($)'].values

ct_carPrice = ColumnTransformer(
    [('Car Model', OneHotEncoder(drop='first'), [0])],
    remainder='passthrough'
)
X1_carPrice = ct_carPrice.fit_transform(X1_carPrice)
X1_carPrice



lr_carPriceLr = linear_model.LinearRegression()
lr_carPriceLr.fit(X1_carPrice, Y1_carPrice)
lr_carPriceLr.predict([[0, 1, 45000, 4]])