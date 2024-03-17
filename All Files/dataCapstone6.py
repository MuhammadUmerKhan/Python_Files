from math import cos, sin, sqrt, atan2, radians
import site
from textwrap import fill
from turtle import color
import folium as fp
import pandas as pd
from folium.plugins import MarkerCluster, MousePosition
from folium.features import DivIcon
import io
import os

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
space_X_df = pd.read_csv(URL)
space_X_df.head()
space_X_df.to_csv("Space_X.csv")

space_X_df.head()

# Task 1: Mark all launch sites on a map
space_X_df = space_X_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_site_df = space_X_df.groupby(['Launch Site'], as_index=False).first()
launch_site_df.shape
launch_site_df = launch_site_df[['Launch Site', 'Lat', 'Long']]
launch_site_df.shape
launch_site_df.head(50)

nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = fp.Map(location=nasa_coordinate, zoom_start=10)

circle = fp.Circle(nasa_coordinate, radius=1000, color="#d34500", fill=True).add_child(fp.Popup('Nasa Jognson Space Center'))
marker = fp.map.Marker(
    nasa_coordinate,
    
    icon = DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC'
    )
)
site_map.add_child(circle)
site_map.add_child(marker)

# Initial the map                                        
  
  
site_map = fp.Map(location=nasa_coordinate, zoom_start=5)
# For each launch site, add a Circle object based on its coordinate (Lat, Long) values. In addition, add Launch site name as a popup label
for index, row in launch_site_df.iterrows():
    coordinate = [row['Lat'], row['Long']]
    
    
    fp.Circle(
        location=coordinate,
        radius=1000,
        color='#000000',
        fill=True
    ).add_child(fp.Popup(row['Launch Site'])).add_to(site_map)
    
    
    fp.map.Marker(
        location=coordinate,
        icon=fp.DivIcon(
            icon_size=(20, 20),
            icon_anchor=(0, 0),
            html='<div style="font-size:12; color:#d35400;"><b>%s</b></div>' % row['Launch Site']
        )
    ).add_to(site_map)
site_map

# Task 2: Mark the success/failed launches for each site on the map
space_X_df.tail(10)
markerCluster = MarkerCluster()

def class_checker(launch_color):
    if launch_color == 1:
        return 'green'
    else:
        return 'red'
space_X_df['marker_color'] = space_X_df['class'].apply(class_checker)    
space_X_df.head()


# Add marker_cluster to current site_map
site_map.add_child(markerCluster)

for index, row in space_X_df.iterrows():
    coordinate = [row['Lat'], row['Long']]
    fp.map.Marker(coordinate, 
                  icon=fp.Icon(color='white', 
                               icon_color=row['marker_color']
                               )
                  ).add_to(markerCluster)
site_map

# TASK 3: Calculate the distances between a launch site to its proximities
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter
)
site_map.add_child(mouse_position)


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance 

launch_site_lat = 28.56367
launch_site_lon = -80.57163
coastline_lat = 28.56367
coastline_lon = -80.57163
distance_coastline = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)

distance_marker = fp.map.Marker(
    [coastline_lat, coastline_lon],
    icon = DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_coastline),
    )
)
site_map.add_child(distance_marker)

coordinates = [[launch_site_lat, launch_site_lon], [coastline_lat, coastline_lon]]
lines = fp.PolyLine(locations=coordinates, weight=1)
site_map.add_child(lines)


# Create a marker with distance to a closest city, railway, highway, etc.                    
# Draw a line between the marker to the launch site
closest_highway = 28.56335, -80.57085
closest_railroad = 28.57206, -80.58525
closest_city = 28.10473, -80.64531

distance_highway = calculate_distance(launch_site_lat, launch_site_lon, closest_highway[0], closest_highway[1])
distance_railroad = calculate_distance(launch_site_lat, launch_site_lon, closest_railroad[0], closest_railroad[1])
distance_city = calculate_distance(launch_site_lat, launch_site_lon, closest_city[0], closest_highway[1])
# Closest Highway marker
distance_marker = fp.map.Marker(
    closest_highway,
    icon = DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_highway)
    )
)
site_map.add_child(distance_marker)
# Closest highway line
coordinates_highway = [[launch_site_lat, launch_site_lon], closest_highway]
lines_highway = fp.PolyLine(locations=coordinates_highway, weight=1)
site_map.add_child(lines_highway)

# Closest railroad
distance_marker = fp.map.Marker(
    closest_railroad,
    icon = DivIcon(
        icon_size=(20, 20), icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_railroad)

    )
)
# Closest railroad line
site_map.add_child(distance_marker)
coordinates_railroad = [[launch_site_lat, launch_site_lon], closest_railroad]
lines_railroad = fp.PolyLine(locations=coordinates_railroad, weight =1 )
site_map.add_child(lines_railroad)
# Closest city
distance_marker = fp.map.Marker(
    closest_city,
    icon = DivIcon(
        icon_size=(20, 20), icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_city)

    )
)# Closest city line
site_map.add_child(distance_marker)
coordinates_city = [[launch_site_lat, launch_site_lon], closest_city]
lines_city = fp.PolyLine(locations=coordinates_city, weight =1 )
site_map.add_child(lines_city)