import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import numpy as np
import pandas as pd

url = 'FlightDelayTime.csv'
df = pd.read_csv(url)
df.head()
df.columns
df.drop(columns = 'Unnamed: 0', inplace=True)
df.head()
year_list = [i for i in range(1980, 2024)]
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1('Automobile Sales Statistics Dashboard', 
            style={'text-align':'center', 'color':'#03F909','font-size':24}),
    html.Div([
        html.H2('Select Statistics type: ', style={'font-size':20, 'color':'#06F1F0', 'text-align':'left'}),
        html.Div([
            dcc.Dropdown(id='dropdown-statistics', 
                         options=[
                             {'label':'Yearly Statistics', 'value':'Yearly Statistics'},
                             {'label':'Recession Period Statistics', 'value':'Recession Period Statistics'}
                         ])
        ])
    ]),
    html.Div([
        html.H2('Select year: ', style={'font-size':20, 'color':'#06F1F0', 'text-align':'left'}),
        html.Div([
            dcc.Dropdown(id='select_year',
                         options=[{'label': year, 'value': year} for year in year_list])
        ])
    ])
])

if __name__ == '__main__':
    app.run(debug=True)