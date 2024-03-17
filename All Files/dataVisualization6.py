from folium import plugins
import folium
from matplotlib import legend
import pandas as pd
import numpy as np
import folium as fp

world_wap = fp.Map()
world_wap = fp.Map(location=[56.130, -106.35], zoom_start=8)
# location = [latitude, logitude]
mexio_map = fp.Map(location=[23.6345, -102.5528], zoom_start= 4)
mexio_map = fp.Map(location=[23.6345, -102.5528], zoom_start= 4, tiles='Cartodb positron')

world_wap = fp.Map(location=[56.130, -106.35], zoom_start=4, tiles='Cartodb dark matter')
world_wap = fp.Map(location=[56.130, -106.35], zoom_start=4, tiles='Cartodb positron')

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Police_Department_Incidents_-_Previous_Year__2016_.csv'
df = pd.read_csv(url)
df.head()

df.shape
limit = 100
df = df.iloc[0:limit, :]
df_incidents = df
df_incidents.shape

# San Fransisco
sanFransiscoMap = fp.Map(location=[37.77, -122.42], zoom_start=4)

incidents = fp.map.FeatureGroup()
for lat, lng, in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        fp.vector_layers.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )
sanFransiscoMap.add_child(incidents)
latitudes = list(df_incidents.Y)
longitudes = list(df_incidents.X)
labels = list(df_incidents.Category)

for lat, lng, label in zip(latitudes, longitudes, labels):
    fp.Marker([lat, lng], popup=label).add_to(sanFransiscoMap)
    
sanFransiscoMap.add_child(incidents)

sanFransiscoMap = fp.Map(location=[37.77, -122.42], zoom_start=12)
incidents = plugins.MarkerCluster().add_to(sanFransiscoMap)

for lat, lng, label in zip(df_incidents.Y, df_incidents.Y, df_incidents.Category):
    fp.Marker(
        location=[lat, lng],
        icon=None,
        popup=label        
    ).add_to(incidents)
sanFransiscoMap

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')
df_can.head()

world_geo = r'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/world_countries.json' # geojson file
world_map = folium.Map(location=[0, 0], zoom_start=4)

folium.Choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada'
).add_to(world_map)

world_geo = r'world_countries.json'
threshold_scale = np.linspace(df_can['Total'].min(),
                              df_can['Total'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist()
threshold_scale[-1] = threshold_scale[-1] + 1
world_map = folium.Map(location=[0, 0], zoom_start=2)

folium.Choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    threshold_scale=threshold_scale,
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada',
    reset=True
).add_to(world_map)