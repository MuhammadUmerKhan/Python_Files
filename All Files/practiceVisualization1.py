import folium as fp
from folium import plugins
import datetime as dt
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from dash import dcc
import dash
import dash_core_components
import dash_html_components
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Historical_Wildfires.csv"
df = pd.read_csv(URL)
df.head()
df.shape
df.dtypes
df.columns

df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month
df[['Year']]
df[['Month']]
df.head()
df.dtypes
# TASK 1.1: 
# Let's try to understand the change in average estimated fire area over time
# (use pandas to plot)
plt.figure(figsize=(12, 6))
df_AverageEstimated_Time = df.groupby('Year')['Estimated_fire_area'].mean() 
df_AverageEstimated_Time.plot(x=df_AverageEstimated_Time.index, y=df_AverageEstimated_Time.values)
plt.xlabel('Year')
plt.ylabel('Estimated_fire_area')
plt.title('Estimated Fire Area over Time')
plt.show()

# TASK 1.2: 
# You can notice the peak in the plot between 2010 to 2013. 
# Let's narrow down our finding, by plotting the estimated fire area for year grouped together with month
df_AverageEstimated_month = df.groupby(['Year', 'Month'])['Estimated_fire_area'].mean()
df_AverageEstimated_month.plot(x=df_AverageEstimated_month.index, y=df_AverageEstimated_month.values)
plt.xlabel('Year, Month')
plt.ylabel('Estimated_fire_area')
plt.title('Estimated Fire Area over Time (km²)')
plt.show()
# TASK 1.3: 
# Let's have an insight on the distribution of mean estimated fire brightness across 
# the regions
# use the functionality of seaborn to develop a barplot
df['Region'].unique()
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Region', y='Mean_estimated_fire_brightness')
plt.xlabel('Region')
plt.ylabel('Mean_estimated_fire_brightness')
plt.title('Distribution of Mean Estimated Fire Brightness across Regions')
plt.show()

# TASK 1.4: 
# Let's find the portion of count of pixels for presumed 
# vegetation fires vary across regions
# we will develop a pie chart for this
explode_list= [0 ,0, 0, 0, 0.2, 0, 0]
plt.figure(figsize=(12, 6))
region_counts = df.groupby('Region')['Count'].sum()
region_counts.plot(kind='pie', labels=region_counts.index,
         figsize = (12, 6))
plt.title('Percentage of Pixels for Presumed Vegetation Fires by Region')
plt.legend([(i,round(k/region_counts.sum()*100,2)) for i,k in zip(region_counts.index, region_counts)])
plt.axis('equal')
plt.show()
# TASK 1.6: 
# Let's try to develop a histogram of the mean estimated fire brightness
# Using Matplotlib to create the histogram
df_hist = df['Mean_estimated_fire_brightness']
df_hist.plot(kind='hist', figsize=(12, 6), bins=20)
plt.xlabel('Histogram of Mean Estimated Fire Brightness(kelvin)')
plt.ylabel('Count')
plt.title('Histogram of Mean Estimated Fire Brightness')
plt.show()

# TASK 1.7: 
# What if we need to understand the distribution of estimated fire brightness 
# across regions? Let's use the functionality of seaborn and pass region as hue
sns.histplot(data=df, x='Mean_estimated_fire_brightness', hue='Region', multiple='stack')
plt.show()
# TASK 1.8: 
# Let's try to find if there is any correlation between mean estimated fire radiative 
# power and mean confidence level?
plt.figure(figsize=(8, 6))
sns.scatterplot(y='Mean_estimated_fire_radiative_power', x='Mean_confidence', data=df)
plt.xlabel('Mean Estimated Fire Radiative Power (MW)')
plt.ylabel('Mean Confidence')
plt.title('Mean Estimated Fire Radiative Power (MW) vs Mean_confidence')
plt.show()
# TASK 1.9:
# Let's mark these seven regions on the Map of Australia using Folium
region_data = {'region':['NSW','QL','SA','TA','VI','WA','NT'], 'Lat':[-31.8759835,-22.1646782,-30.5343665,-42.035067,-36.5986096,-25.2303005,-19.491411], 
               'Lon':[147.2869493,144.5844903,135.6301212,146.6366887,144.6780052,121.0187246,132.550964]}
reg=pd.DataFrame(region_data)

# TASK 1.9: 
# Let's mark these seven regions on the Map of Australia using Folium
world_map = fp.Map()
world_map = fp.Map(location=[-31.875984, 147.286949], zoom_start=4)

aus_reg = fp.map.FeatureGroup()
aus_map = fp.Map(location=[-25, 135], zoom_start=4)

for lat, long, lab in zip(reg.Lat, reg.Lon, reg.region):
    aus_reg.add_child(
        fp.vector_layers.CircleMarker(
            [lat, long],
            popup=lab,
            radius=5,
            fill=True,
            fill_color='blue',
            color='red',
            fill_opacity='0.6'
        )
    )
aus_map.add_child(aus_reg)
