import folium as fp
import branca
import branca.colormap as cm
import folium
import seaborn as sns
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objects as go
from plotly import express as px
from matplotlib import legend
from folium import plugins

# Un comment and run this bolck of code

url = 'https://raw.githubusercontent.com/VicmanGT/KaggleCompetition/b39212ebd9c8106a00d7e9823183c6e7cb9305c6/Data/cities_air_quality_water_pollution.18-10-2021%20(1)%20with_coordiantes.csv'
# df = pd.read_csv(url)
# df.head()
# df.drop(columns="Unnamed: 0", inplace=True)

# df = df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
# df = df.rename(columns=lambda x: x.replace('"', ''))
# df.to_csv("cities_air_quality_water_pollution.csv", index=False)

df = pd.read_csv("cities_air_quality_water_pollution.csv")
df.head()
df.shape
df.isnull().sum()

df.describe()
df.info()

df.duplicated().any()

# We Found no any outlier and no any missing values and its time to visualize it
df.head()
# Visualization
# Top 20 Water poluted Country
df.columns
df_top_20=df.groupby(' Country')[' WaterPollution'].mean().sort_values(ascending=True).head(20)
# df_top_20

fig = px.bar(df_top_20, x=df_top_20.index, 
             y=df_top_20,
             color=df_top_20.index)
fig.update_layout(xaxis_title='Countriies', 
                  yaxis_title = "Water Pollution",
                  title="Top 20 Water Popluted Countries"
                  )
fig.show()
# 
df.head()
# Top 20 countries with dirty air
df.columns
df_top_20_air = df.groupby(' Country')[' AirQuality'].mean().sort_values(ascending=False).head(20)
fig = px.bar(df_top_20_air, 
             x=df_top_20_air.values, 
             y=df_top_20_air.index,
             color=df_top_20_air.index,
             orientation='h')  

fig.update_layout(
    xaxis_title="Air Quality",
    yaxis_title="Country",
    title="Top 20 countries with dirty air",
    yaxis_categoryorder='total ascending' 
)

fig.show()

# 
df.head()

# Top 20 countries with clean air
df_top_20_Dirt_air = df.groupby(' Country')[' AirQuality'].mean().sort_values(ascending=True).head(20)
fig = px.bar(df_top_20_Dirt_air, 
             x=df_top_20_Dirt_air.index, 
             y=df_top_20_Dirt_air,
             color=df_top_20_Dirt_air.index,
            #  orientation='h'
            )  

fig.update_layout(
    xaxis_title="Air Quality",
    yaxis_title="Country",
    title="Top 20 countries with Clean air",
    yaxis_categoryorder='total ascending' 
)

fig.show()
# 
# Top 20 countries with Clean water
df.columns
df_top_20_water = df.groupby(' Country')[' WaterPollution'].mean().sort_values(ascending=False).head(20)

fig = px.bar(df_top_20_water, 
             x=df_top_20_water.index, 
             y=df_top_20_water,
             color=df_top_20_water.index, 
            #  orientation='h'
            )  

fig.update_layout(
    xaxis_title="Air Quality",
    yaxis_title="Country",
    title="Top 20 countries with Clean Water",
    yaxis_categoryorder='total ascending' ,
    height=400,
    width=800
)

fig.show()

# Top 20 countries with dirty water
df_top_20_Dwater = df.groupby(' Country')[' WaterPollution'].mean().sort_values(ascending=True).head(20)
fig = px.bar(df_top_20_Dwater, 
             x=df_top_20_Dwater.index, 
             y=df_top_20_Dwater,
             color=df_top_20_Dwater.index, 
            #  orientation='h'
            )  

fig.update_layout(
    xaxis_title="Air Quality",
    yaxis_title="Country",
    title="Top 20 countries with Dirty Water",
    yaxis_categoryorder='total ascending' ,
    height=400,
    width=800
)

fig.show()

#lets plot a line graph to compare between air polution and water polution in different countries
print(df_top_20_Dirt_air, df_top_20_Dwater)
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=df_top_20_Dwater, y=df_top_20_Dirt_air,
               mode='markers', marker= dict(color='orange')
               , marker_color = df_top_20_Dirt_air)
)
fig.update_layout(
    xaxis_title = "Water", yaxis_title = 'Air', 
    title = "Showing Realation bwtween Top 20 Country with Poluted Air and Water"
)
fig.show()
# ######
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=df_top_20_water, y=df_top_20_air,
               mode='markers', marker= dict(color='orange'),
                marker_color = df_top_20_water
                )
)
fig.update_layout(
    xaxis_title = "Water", yaxis_title = 'Air', 
    title = "Showing Realation bwtween Top 20 Country with Air and Water Quality"
)
fig.update_layout(height=500)
fig.show()
# 
#lets plot a heatmap of water and air
df.columns
water_air = [' AirQuality', ' WaterPollution']
df1 = df[water_air].corr()
# Plotting Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df1, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of the AirQuality and WaterPollution')
plt.show()

# Geovisualization
df.head()
Map = fp.Map()
Map = fp.Map(locations=[40.712728, -74.006015], zoom_start=8)
# print(Map)

# Air Quality
# Plotting for city
water_cm = cm.LinearColormap(["#80FFFF", "#000000"], vmin=0, vmax=100, caption='Water Pollution').to_step(5)
air_cm = cm.LinearColormap(["#000000", "#D6D6D6"], vmin=0, vmax=100, caption='Air Quality').to_step(5)
Lat = list(df['Lat'])
Lon = list(df['Lon'])
air = list(df[' AirQuality'])
water = list(df[' WaterPollution'])

air_world_map = fp.Map(location=[0, 0], zoom_start=4) 

for loc, a in zip(zip(Lat, Lon), air):
    fp.Circle(
        location=loc,
        radius = 5000,
        fill = True,
        color = air_cm(a),
        popup=f'Air Quality {a}',
        weight = 5        
    ).add_to(air_world_map)
air_world_map.add_child(air_cm)

# Water Quality
water_world_map = fp.Map(location=[0, 0], zoom_start=4)
for loc, w in zip(zip(Lat, Lon), water):
    fp.Circle(
        location=loc,
        radius = 5000,
        fill = True,
        color = water_cm(w),
        popup=f'Water Polution {w}',
        weight = 5        
    ).add_to(water_world_map)
water_world_map.add_child(water_cm)

# Plotting for country or Choropleths
df_country = df.groupby(' Country')[[' AirQuality', ' WaterPollution']].mean()
df_country
#Removes the doble quotation marks from the country's name and puts the result in a different coulumn
def quotation(row):
    return row.replace(' "', '').replace('"', '')
df_country['Country'] = sorted(df[' Country'].apply(quotation).unique())
df_country.reset_index(inplace=True)
df_country.drop(' Country', inplace=True, axis=1)
df_country

df_country.loc[119, 'Country'] = 'China'

#json file with the contours of each country
url = ( "https://raw.githubusercontent.com/python-visualization/folium/main/examples/data")
country_geo = f"{url}/world-countries.json"

air_dict = df_country.set_index('Country')[' AirQuality']
water_dict = df_country.set_index('Country')[' WaterPollution']

# Air Quality:
air_choropleth = fp.Map(location=[0, 0], zoom_start=2)
fp.GeoJson(
    data=country_geo, 
    style_function= lambda feature: { 
        'fillColor': air_cm(air_dict[feature['properties']['name']])
        if feature['properties']['name'] in air_dict.index
        else 'white',
        'color' : 'black', 
        'fillOpacity': 0.6
    }, 
    
).add_to(air_choropleth)
air_cm.add_to(air_choropleth)
air_choropleth

# Water Polution
water_choropleth = fp.Map(location=[0, 0], zoom_start=2)
fp.GeoJson(
    data=country_geo, 
    style_function= lambda feature: { 
        'fillColor': water_cm(water_dict[feature['properties']['name']])
        if feature['properties']['name'] in water_dict.index
        else 'white',
        'color' : 'black', 
        'fillOpacity': 0.6
    }, 
    
).add_to(water_choropleth)
water_cm.add_to(water_choropleth)
water_choropleth