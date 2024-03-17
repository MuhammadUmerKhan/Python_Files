import io
from tkinter import font
from turtle import color, title, width
from matplotlib import figure
from matplotlib.pylab import annotations
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import os

from sqlalchemy import true

df = pd.read_csv("dataset_part_2.csv")
df.head(10)

plt.figure(figsize=(20, 10))
sns.catplot(x='FlightNumber', y="PayloadMass", data=df, hue="Class", aspect=5)
plt.title("Relation between Flight number and Payload Mass")
plt.xlabel("Flight Number", fontsize = 10)
plt.ylabel("Payload Mass", fontsize = 10)
plt.show()

### TASK 1: Visualize the relationship between Flight Number and Launch Site                    
plt.figure(figsize=(20, 10))
sns.catplot(x='LaunchSite', y="PayloadMass", data=df, hue="Class", aspect=5)
plt.title("Relation between Launch Site and Payload Mass")
plt.xlabel("Launch Site", fontsize = 10)
plt.ylabel("Payload Mass", fontsize = 10)
plt.show()

plt.figure(figsize=(20, 10))
sns.scatterplot(x='LaunchSite', y="PayloadMass", data=df, hue="Class")
plt.title("Relation between Launch Site and Payload Mass")
plt.xlabel("Launch Site", fontsize = 10)
plt.ylabel("Payload Mass", fontsize = 10)
plt.show()


plt.figure(figsize=(20, 10))
sns.scatterplot(x='FlightNumber', y="LaunchSite", data=df, hue="Class")
plt.title("Relation between Launch Site and Payload Mass")
plt.xlabel("Flight Number", fontsize = 10)
plt.ylabel("Launch Site", fontsize = 10)
plt.show()
# OR                    
# Using plotly
# import plotly.graph_objs as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['FlightNumber'], y=df['LaunchSite'], mode='markers', marker=dict(color='blue')))
fig.update_layout(title="Relation between Launch Site and Payload Mass", xaxis_title='Flight Number', yaxis_title='Launch Site')
fig.show()

## TASK 2: Visualize the relationship between Payload and Launch Site                                                                                                    
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['PayloadMass'],
    y=df['LaunchSite'],
    mode='markers',
    marker=dict(color="red", size=10, line=dict(color='black', width=2)),
    text=df['LaunchSite'],  # Hover text
    showlegend=False  # Hide legend for this trace
))

# Layout updates
fig.update_layout(
    title="Relation between Launch Site and Payload Mass",
    xaxis_title='Payload Mass',
    yaxis_title='Launch Site',
    height=500,  # Set the height of the plot
    width=800,   # Set the width of the plot
    font=dict(family="Arial, sans-serif", size=12, color="RebeccaPurple"),  # Set font properties
    plot_bgcolor='lightgray',  # Set background color of the plot
    xaxis=dict(type='log', title='Logarithmic Payload Mass'),  # Use logarithmic scale for x-axis
    yaxis=dict(title='Lau/nch Site', tickangle=45),  # Rotate y-axis labels for better visibility
    hovermode='closest',  # Display closest data point's information on hover
)

# Annotations for additional information
fig.update_layout(annotations=[
    dict(
        x=df['PayloadMass'].max(),
        y=df['LaunchSite'].iloc[-1],
        xref="x",
        yref="y",
        text="Max Payload Mass",  # Annotation text
        showarrow=True,
        arrowhead=7,
        ax=0,
        ay=-40
    )
])
fig.show()
# OR
plt.figure(figsize=(20, 10))
sns.scatterplot(x='PayloadMass', y="LaunchSite", data=df, hue="Class")
plt.title("Relation between Launch Site and Payload Mass")
plt.xlabel("Payload Mass", fontsize = 10)
plt.ylabel("Launch Site", fontsize = 10)
plt.show()

# Plot a scatter point chart with x axis to be Pay Load Mass (kg) and y axis to be the launch site, and hue to be the class value
plt.figure(figsize=(20, 10))
sns.scatterplot(x='PayloadMass', y="LaunchSite", data=df, hue="Class")
plt.title("Relation between Launch Site and Payload Mass")
plt.xlabel("Payload Mass (kg)", fontsize = 10)
plt.ylabel("Launch Site", fontsize = 10)
plt.show()


### TASK  3: Visualize the relationship between success rate of each orbit type                    
df.head()
df_succesRate = df.groupby('Orbit')['Class'].mean().reset_index()
fig = px.bar(x = df_succesRate['Orbit'], y = df_succesRate['Class'], title="Relationship between success rate of each orbit type")
fig.show()

sns.barplot(x = 'Orbit', y = 'Class', data=df_succesRate)
plt.title("Relationship between success rate of each orbit type")
plt.show()
# ### TASK  4: Visualize the relationship between FlightNumber and Orbit type
df.head()
plt.figure(figsize=(20, 10))
sns.catplot(x='FlightNumber', y="Orbit", data=df, hue="Class", aspect=5)
plt.title("Relation between Flight Number and Orbit type")
plt.xlabel("Flight Number", fontsize = 10)
plt.ylabel("Orbit type", fontsize = 10)
plt.show()

### TASK  5: Visualize the relationship between Payload and Orbit type
plt.figure(figsize=(20, 10))
sns.catplot(x='PayloadMass', y="Orbit", data=df, hue="Class", aspect=5)
plt.title("Relation between Payload Mass and Orbit type")
plt.xlabel("Payload Mass", fontsize = 10)
plt.ylabel("Orbit type", fontsize = 10)
plt.show()


### TASK  6: Visualize the launch success yearly trend                    
  
year = []
def extract_year():
    for i in df["Date"]:
        year.append(i.split('-')[0])
    return year
extract_year()
df['Year'] = year
df.head() 
df.columns
new_formated = ['FlightNumber', 'Date', 'Year', 'BoosterVersion', 'PayloadMass', 'Orbit',
       'LaunchSite', 'Outcome', 'Flights', 'GridFins', 'Reused', 'Legs',
       'LandingPad', 'Block', 'ReusedCount', 'Serial', 'Longitude', 'Latitude',
       'Class']
df = df[new_formated]
df.head()

yearlySucess = df.groupby('Year')['Class'].mean().reset_index()
xData = yearlySucess['Year']
yData = yearlySucess['Class']
fig = go.Figure()
fig.add_trace(go.Scatter(x=xData, y=yData, mode='lines', marker = dict(color='green')))
fig.update_layout(title='Success Yearly Trend', xaxis_title = 'Year', yaxis_title = 'Class')
fig.show()



features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()

### TASK  7: Create dummy variables to categorical columns

features_one_hot = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'])
features_one_hot.shape
features_one_hot.head()

features_one_hot = features_one_hot.astype('float64')

features_one_hot.to_csv('dataset_part_3.csv', index=False)
