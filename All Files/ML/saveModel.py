import pickle
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model 

homePrice_df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\homeprices.csv")
homePrice_df.head()

plt.scatter(homePrice_df.area, homePrice_df.price, color='red', marker='+')
plt.xlabel("Area")
plt.ylabel('Price')
plt.show()

price = homePrice_df.price
areea = homePrice_df.drop(columns='price')

lr = linear_model.LinearRegression()
#  fit(X,Y) X=independent var, Y=dependent var
lr.fit(areea, price)
# predicting random value price
lr.predict([[5000]])

# Save Model To a File Using Python Pickle
with open('model_picke', 'wb') as file:
    pickle.dump(lr, file);
# Load Saved Model
with open('model_picke', 'rb') as f:
    mp = pickle.load(f)

mp.coef_
mp.intercept_
mp.predict([[5000]])

# Save Trained Model Using joblib
joblib.dump(lr, 'model_joblib')
# load
mj = joblib.load('model_joblib')
mj.coef_
mj.intercept_
mj.predict([[5000]])