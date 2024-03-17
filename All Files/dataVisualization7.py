import io
from turtle import color, mode, title
from matplotlib import markers
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pyparsing import col

# Scatter plot
age_array = np.random.randint(22, 25, 60)
income_array=np.random.randint(300000,700000,3000000)
fig=go.Figure()
fig.add_trace(go.Scatter(x=age_array, y=income_array, mode='markers', marker=dict(color='blue')))
fig.update_layout(title='Economic Servey', xaxis_title='Age', yaxis_title='Income')
fig.show()

# Line Plot
numberOfBicycleSold_array = [50, 100, 40, 150, 160, 70, 60, 45]
monthArray = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'Oct', 'Nov', 'Dec']
fig = go.Figure()
fig.add_trace(go.Scatter(x=monthArray, y=numberOfBicycleSold_array, mode='lines', marker=dict(color='green')))
fig.update_layout(title='Bicycles Sales', xaxis_title='Months', yaxis_title='Number of Bicycles Sold')
fig.show()

# Bar Plot
score_array = [80, 90, 56, 88, 95]
grade_array = ['Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']
fig = px.bar(x=grade_array, y=score_array, title='Pass Percentage of Class')
fig.show()

# Histogram
heights_array = np.random.normal(160, 11, 200)
fig = px.histogram(x=heights_array, title='Distribution of Heights')
fig.show()

# Bubble Plot
crime_details = {
    'City' : ['Chicago', 'Chicago', 'Austin', 'Austin','Seattle','Seattle'],
    'Numberofcrimes' : [1000, 1200, 400, 700,350,1500],
    'Year' : ['2007', '2008', '2007', '2008','2007','2008'],
}
df = pd.DataFrame(crime_details)

bub_data = df.groupby('City')['Numberofcrimes'].sum().reset_index()
fig = px.scatter(bub_data, x='City', y='Numberofcrimes', 
                 size='Numberofcrimes', hover_name='City', 
                 title='Crime Statistics', size_max=50)
fig.show()

# Pie Chart
exp_percent= [20, 50, 10,8,12]
house_holdcategories = ['Grocery', 'Rent', 'School Fees','Transport','Savings']
fig = px.pie(values=exp_percent, names=house_holdcategories, title='House Expenditure')
fig.show()

# Sub Burst Chart
data = dict(
    character=["Eve", "Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],
    parent=["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve" ],
    value=[10, 14, 12, 10, 2, 6, 6, 4, 4])
fig = px.sunburst(data, names='character', parents='parent', values='value', title='Family Chart')
fig.show()

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv'
df_airline = pd.read_csv(url)
df_airline.head()
df_airline.shape
data = df_airline.sample(n=500, random_state=42)
data.shape
data[['Distance']]
data[['DepTime']]
df_airline = data
df_airline.shape

# Scatter Plot
scatter_xdata = df_airline['Distance']
scatter_ydata = df_airline['DepTime']
fig = go.Figure()
fig.add_trace(go.Scatter(x=scatter_xdata, y=scatter_ydata, mode='markers', marker= dict(color='orange')))
fig.update_layout(title='Distance vs Departure Time', xaxis_title='Distance', yaxis_title='DepTime')
fig.show()

# Line Plot
line_data = df_airline.groupby('Month')['ArrDelay'].mean().reset_index()
line_xdata = line_data['Month']
line_ydata = line_data['ArrDelay']
fig = go.Figure()
fig.add_trace(go.Scatter(x=line_xdata, y=line_ydata, mode='lines', marker=dict(color='blue')))
fig.update_layout(title='Month vs Average flight Dalay', xaxis_title = 'Month', yaxis_title = 'ArrDelay')
fig.show()

# bar plot
bar_data = df_airline.groupby('DestState')['Flights'].sum().reset_index()
bar_xdata = bar_data['DestState']
bar_ydata = bar_data['Flights']
fig = px.bar(x=bar_xdata, y=bar_ydata, title=' Total number of flights to the destination state split by reporting air')
fig.show()
    
# Histogram
df_airline['ArrDelay'] = df_airline['ArrDelay'].fillna(0)
hist_data = df_airline['ArrDelay']
fig = px.histogram(x=hist_data, title='Total number of flights to the destination state split by reporting air.')
fig.show()

# Bubble Plot
bub_data = df_airline.groupby('Reporting_Airline')['Flights'].sum().reset_index()
fig = px.scatter(bub_data, x='Reporting_Airline', y='Flights', 
                 size='Flights', hover_name='Reporting_Airline', 
                 title='Reporting Airline vs Number of Flights.', size_max=60
                )
fig.show()

# Pie Chart
fig = px.pie(df_airline,values='Flights', names='DistanceGroup', title='Flight propotion by Distance Group.')
fig.show()

# SunBurst Chart
fig = px.sunburst(df_airline, path=['Month', 'DestStateName'], values='Flights', title='Flight Distribution Hierarchy')
fig.show()