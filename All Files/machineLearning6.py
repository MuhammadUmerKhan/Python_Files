from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv"
df = pd.read_csv(URL)

df.head()
df.isna().sum()
df.dropna(inplace=True)

x = df.drop(columns="MEDV")
y = df["MEDV"]
x.head()
y.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Regression Tree
regression_tree = DecisionTreeRegressor(criterion = "squared_error")
regression_tree.fit(x_train, y_train)
regression_tree.score(x_test, y_test)
prediction = regression_tree.predict(x_test)
print("$",(prediction - y_test).abs().mean()*1000)

