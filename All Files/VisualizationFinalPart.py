from pkg_resources import safe_extra
from pyodide.http import pyfetch
from plotly import express as px
from plotly import graph_objects as go
from cProfile import label
from cgitb import reset
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyparsing import line
import seaborn as sns
import folium as fp
import numpy as np
import pandas as pd
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"
df = pd.read_csv(URL)
df.head()
df.describe()
df.columns
# TASK 1.1: 
# Develop a *Line chart* using the functionality of
# pandas to show how automobile sales fluctuate from year to year
line_data = df.groupby(df['Year'])['Automobile_Sales'].mean()
plt.figure(figsize=(10, 6))
line_data.plot(kind='line')
plt.title('Automobile_Sales over time')
plt.xlabel('Year')
plt.xticks(list(range(1980, 2024)), rotation=75)
plt.ylabel('Automobile_Sales')
plt.text(1982, 650, '1981-82 Recession')
plt.show()
# TASK 1.2: 
# Plot different lines for categories of vehicle type and analyse the trend to 
# answer the question Is there a noticeable difference in sales trends 
# between different vehicle types during recession periods?
df_Mline = df.groupby(['Year','Vehicle_Type'], as_index=False)['Automobile_Sales'].sum()
df_Mline.set_index('Year', inplace=True)
df_Mline = df_Mline.groupby(['Vehicle_Type'])['Automobile_Sales']
plt.figure(figsize=(10, 6))
df_Mline.plot(kind='line')
plt.title('Sales Trend Vehicle-wise during Recession')
plt.xlabel('Year')
plt.xticks(list(range(1980, 2024)), rotation=75)
plt.ylabel('Automobile_Sales')
plt.legend()
plt.show()
#TASK 1.3: 
# Use the functionality of **Seaborn Library** to create a visualization to
# compare the sales trend per vehicle type for a recession period with a 
# non-recession period.
df_sns = df.groupby('Recession')['Automobile_Sales'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Recession', y='Automobile_Sales', hue='Recession', data=df_sns)
plt.title('Average Automobile Sales during Recession and Non-Recession')
plt.xlabel('Recession')
plt.ylabel('Automobile_Sales')
plt.xticks(ticks=[0, 1], labels=['Recession', 'Non-Recession'])
plt.show()

recession_data = df[df['Recession'] == 1]
df_sns_ = df.groupby(['Recession', 'Vehicle_Type'])['Automobile_Sales'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Recession', y='Automobile_Sales',hue='Vehicle_Type' , data= df)
plt.title('Vehicle-Wise Sales during Recession and Non-Recession Period')
plt.xlabel('Period')
plt.ylabel('Average Sales')
plt.xticks(ticks=[0, 1], labels=['Recession', 'Non-Recession'])
plt.show()
# TASK 1.4: 
# Use sub plotting to compare the variations in GDP during recession and 
# non-recession period by developing line plots for each period.
rec_data = df[df['Recession'] == 1]
non_rec_data = df[df['Recession'] == 0]

fig = plt.figure(figsize=(12, 6))
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2)

sns.lineplot(x='Year', y='GDP', data=rec_data, label='Recession',ax=ax0)
ax0.set_xlabel('Year')
ax0.set_ylabel('GDP')
ax0.set_title('GDP Variation during Recession Period')

sns.lineplot(x='Year', y='GDP', data=non_rec_data, ax=ax1)
ax1.set_xlabel('Year')
ax1.set_ylabel('GDP')
ax1.set_title('GDP Variation during Non-Recession Period')

plt.tight_layout()
plt.show()
# TASK 1.5: 
# Develop a Bubble plot for displaying the impact of seasonality on 
# Automobile Sales.
non_rec_data = df[df['Recession'] == 0]
size = non_rec_data['Seasonality_Weight']
sns.scatterplot(x='Month', y='Automobile_Sales', data=df, size=size
                )
plt.xlabel('Year')
plt.ylabel('Automobile_Sales')
plt.title('Seasonality impact on Automobile Sales')
plt.show()
# TASK 1.6: 
# Use the functionality of Matplotlib to develop a scatter plot to identify the 
# correlation between average vehicle price relate to the sales volume during recessions.
rec_data = df[df['Recession'] == 1]
plt.scatter(x='Consumer_Confidence', y='Automobile_Sales', data=rec_data)
plt.title('Correlation between average vehicle price relate to the sales volume during recessions.')
plt.xlabel('Consumer_Confidence')
plt.ylabel('Automobile_Sales')
plt.show()  
# How does the average vehicle price relate to the sales volume during recessions?
plt.scatter(x='Price', y='Automobile_Sales', data=rec_data)
plt.title('Relationship between Average Vehicle Price and Sales during Recessions')
plt.xlabel('Price')
plt.ylabel('Automobile_Sales')
plt.show()
# TASK 1.7: 
# Create a pie chart to display the portion of advertising expenditure of 
# XYZAutomotives during recession and non-recession periods.
non_rec_data = df[df['Recession'] == 0]
rec_data = df[df['Recession'] == 1]
df_AE = rec_data['Advertising_Expenditure'].sum()
df_non_AE = non_rec_data['Advertising_Expenditure'].sum()
plt.figure(figsize=(10, 6))
labels = ['Recession', 'Non-Recession']
sizes = [df_AE, df_non_AE]
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title(' Advertising Expenditure during Recession and Non-Recession Periods')
plt.show()
# TASK 1.8: 
# Develop a pie chart to display the total Advertisement expenditure for each 
# vehicle type during recession period.
non_rec_data = df[df['Recession'] == 0]
rec_data = df[df['Recession'] == 1]
df_AV = rec_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum()
labels_AV = df_AV.index
sizes_AV = df_AV.values
plt.figure(figsize=(10, 6))
plt.pie(df_AV.values, labels = labels_AV, autopct = "%1.1f%%", startangle = 90)
plt.title('Share of Each Vehicle Type in Total Sales during Recessions')
plt.show()
# TASK 1.9: Develop a countplot to analyse the effect of the unemployment rate on 
# vehicle type and sales during the Recession Period.
rec_data = df[df['Recession'] == 1]
plt.figure(figsize=(10, 6))
sns.countplot(x='unemployment_rate', hue='Vehicle_Type', data=rec_data)
plt.xlabel('Unemployment_Rate')
plt.ylabel('Count')
plt.title('Effect of Unemployment Rate on Vehicle Type and Sales')
plt.legend()
plt.show()
# OPTIONAL : 
# TASK 1.10 Create a map on the hightest sales region/offices of the 
# company during recession period
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/us-states.json'
rec_data = df[df['Recession'] == 1]
sales_by_city  = rec_data.groupby('City')['Automobile_Sales'].sum().reset_index()
sales_map = fp.Map(location=[37.0902, -95.7129], zoom_start=4)
cloropath = fp.Choropleth(
    geo_data=path,
    data=sales_by_city,
    columns=['City', 'Automobile_Sales'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.6,
    line_opacity=0.2,
    legend_name='Automobile Sales during Recession'
).add_to(sales_map)
cloropath.geojson.add_child(
    fp.features.GeoJsonTooltip(['name'], labels = True)
)
sales_map