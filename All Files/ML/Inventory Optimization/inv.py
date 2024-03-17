from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\demand_inventory.csv")
df.drop(columns="Unnamed: 0", inplace= True)
df.head()

df.shape
df.describe()
df.info()
df.columns
df['Product_ID'].unique()

df.head()
df.isnull().sum()

df['Month'] = pd.to_datetime(df['Date']).dt.month_name()
df.head()
df['Month'].unique()
df['Year'] = pd.to_datetime(df['Date']).dt.year
df.head()
df['Year'].unique()

demand_via_month = df.groupby(['Month'])['Demand'].sum()
demand_via_month.index


# Visualization
fig = px.bar(demand_via_month, x=demand_via_month.index, y=demand_via_month.values, title="Demand by rate of time")
fig.update_layout(xaxis_title = "Month", yaxis_title = "Demand")
fig.show()
df.head()

fig = px.line(df, x=df.Date, y=df.Demand, title="Demand Over time")
fig.show()


fig = px.line(df, x=df.Date, y=df.Inventory, title="Inventory Over time")
fig.show()


df.head()
from statsmodels.tsa.statespace.sarimax import SARIMAX
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
time_series = df.set_index('Date')['Demand']

differenced_series = time_series.diff().dropna()

# Plot ACF and PACF of differenced time series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(differenced_series, ax=axes[0])
plot_pacf(differenced_series, ax=axes[1])
plt.show()
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 2)
model = SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)
future_steps = 10
predictions = model_fit.predict(len(time_series), len(time_series) + future_steps - 1)
predictions = predictions.astype(int)
print(predictions)

