from cgitb import text
from doctest import debug
from gc import callbacks
import dash_core_components as core
from dash.dependencies import Input, Output
import dash_html_components as html
from matplotlib import colors
from numpy import number
import pandas as pd
import plotly.express as px
import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
# Read the airline data into pandas dataframe
airline_data =  pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv', 
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})

# Randomly sample 500 data points. Setting the random state to be 42 so that we get same result.
data = airline_data.sample(n=500, random_state=42)
data.head()
data['Reporting_Airline']
data[['DestState']]
data.columns
# Pie Chart Creation
fig = px.pie(data, values='Flights', names='DistanceGroup', title='Distance group proportion by flights')
fig.show()

app = dash.Dash(__name__)
# Get the layout of the application and adjust it.
# Create an outer division using html.Div and add title to the dashboard using html.H1 
# component
# Add description about the graph using HTML P (paragraph) component
# Finally, add graph component.
app.layout = html.Div(children=[html.H1('Airline Dashboard', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
                                html.P('Proportion of distance group (250 mile distance interval group) by flights.', style={'textAlign':'center', 'color': '#F57241'}),
                                dcc.Graph(figure=fig),
                                               
                    ])

if __name__ == '__main__':
       app.run_server()

# Import required libraries


# Read the airline data into the pandas dataframe
airline_data =  pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv', 
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})
# Create a dash application
app = dash.Dash(__name__)
                               
app.layout = html.Div(children=[ html.H1('Airline Performance Dashboard',style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
                                html.Div(["Input Year: ", dcc.Input(id='input-year', value='2010', 
                                type='number', style={'height':'30px', 'font-size': 15}),], 
                                style={'font-size': 40}),
                                html.Br(),
                                html.Br(),
                                html.Div(dcc.Graph(id='bar-plot')),
                                ])

# add callback decorator
@app.callback( Output(component_id='bar-plot', component_property='figure'),
               Input(component_id='input-year', component_property='value'))

# Add computation to callback function and return graph
def get_graph(entered_year):
       df = airline_data[airline_data['Year'] == int(entered_year)]
       g1 = df.groupby(['Reporting_Airline'])['Flights'].sum().nlargest(10).reset_index()
       fig1 = px.bar(g1, x='Reporting_Airline', y='Flights', 
                     title='Top 10 airline carrirer in year' + str(entered_year) + 'in terms of number of flights')
       fig1.update_layout()
       return fig1

# Run the app
if __name__ == '__main__':
    app.run_server()
    
# Import required libraries


# Read the airline data into the pandas dataframe
airline_data =  pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv', 
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})
# Create a dash application
app = dash.Dash(__name__)
                               
app.layout = html.Div(children=[ html.H1('Airline Performance Dashboard',style={'textAlign': 'center', 
                                   'color': '#503D36', 'font-size': 40}),
                                html.Div(["Input Year: ", dcc.Input(id='input-year', value='2010', 
                                type='number', style={'height':'50px', 'font-size': 35}),], 
                                style={'font-size': 40}),
                                html.Br(),
                                html.Br(),
                                html.Div(dcc.Graph(id='line-plot')),
                                ])

# add callback decorator
@app.callback( Output(component_id='line-plot', component_property='figure'),
               Input(component_id='input-year', component_property='value'))

# Add computation to callback function and return graph
def get_graph(entered_year):
    # Select 2019 data
    df =  airline_data[airline_data['Year']==int(entered_year)]
    
    # Group the data by Month and compute average over arrival delay time.
    line_data = df.groupby('Month')['ArrDelay'].mean().reset_index()

    fig = go.Figure(data=go.Scatter(x=line_data['Month'], y=line_data['ArrDelay'], mode='lines', marker=dict(color='green')))
    fig.update_layout(title='Month vs Average Flight Delay Time', xaxis_title='Month', yaxis_title='ArrDelay')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server()
    

# Give the title to the dashboard to 
# “Total number of flights to the destination state split by reporting air” 
# using HTML H1 component and font-size as 50.
app = dash.Dash(__name__)
app.layout = html.Div(children=[html.H1('Total number of flights to the destination state split by reporting air',
                                        style={'textAlign':'center', 'color':'#E414C2', 'font-size':'50'}),
                                html.Div(["Input Year: ", dcc.Input(id='input-yr', value='2013',
                                          type='number', style={'height':'50px', 'font-size':'35'}),],style={'font-size':40, 'color':'#EF6109 '}),
                                html.Br(), html.Br(), html.Div(dcc.Graph(id='bar-plot')),])

@app.callback(Output(component_id='bar-plot', component_property='figure'),
              Input(component_id='input-yr', component_property='value'))
def get_graph(entered_year):
    df =  airline_data[airline_data['Year']==int(entered_year)]
    bar_data = df.groupby('DestState')['Flights'].sum().reset_index()
    fig = px.bar(bar_data, x= "DestState", y= "Flights", title='Total number of flights to the destination state split by reporting airline') 
    fig.update_layout(title='Flights to Destination State in ' + str(entered_year), xaxis_title='DestState', yaxis_title='Flights')
    return fig        
if __name__ == '__main__':
       app.run_server()
       
