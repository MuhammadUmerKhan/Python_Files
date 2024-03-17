from doctest import debug
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv')
df.head()

year_list = [i for i in range(1980, 2024, 1)]
app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1('Automobile Sales Statistics Dashboard', style={
        'textAlign':'center', 'color':'#503D36', 'font-size':'24' }),
    html.Div([
      dcc.Dropdown(id='dropdown-statistics',
                   options=[{'label':'Yearly Statistics', 'value':'Yearly Statistics'},
                            {'label':'Recession Period Statistics', 'value':'Recession Period Statistics'}
                            ],placeholder='Select a report type', style={'width':'80%', 'font-size':'20px', 'textAlign':'center', 'padding':'3px'})  
    ]),
    html.Div(
        dcc.Dropdown(id='select-year',
                     options=[{'label':i, 'value':i} for i in year_list],
                     )
        ),
    html.Div([
        html.Div(id='output-container', className='chart-grid', style={'display':'flex'}),
    ])
])
# @app.callback(
#     Output(component_id='select-year', component_property='value'),
#     Input(component_id='dropdown-statistics',component_property='value'))

# def update_input_container(yearly_statistics):
#     if yearly_statistics =='Yearly Statistics': 
#         return False
#     else: 
#         return True

@app.callback(
    Output(component_id='output-container', component_property='children'),
    [Input(component_id='select-year', component_property='value'), 
     Input(component_id='dropdown-statistics', component_property='value')])
def update_output_container(year_value, yearly_statistics):
    if yearly_statistics == 'Recession Period Statistics':
        rec_data = df[df['Recession'] == 1]
        yearly_rec = rec_data.groupby('Year')['Automobile_Sales'].mean().reset_index()
        rchart1 = dcc.Graph(
            figure = px.line(yearly_rec, x='Year', y='Automobile_Sales', title='Automobile sales fluctuate over Recession Period (year wise) using line chart')
        )
        veh_data = rec_data.groupby('Vehicle_Type')['Competition'].mean().reset_index()
        rchart2 = dcc.Graph(
            figure = px.bar(veh_data, x='Vehicle_Type', y='Competition', 
                          title='The average number of vehicles sold by vehicle type and represent as a Bar chart')
        )
        exp_rec = rec_data.groupby('Vehicle_Type')['Advertising_Expenditure'].mean().reset_index()
        rchart3 = dcc.Graph(
            figure=px.pie(exp_rec, values='Advertising_Expenditure', names='Vehicle_Type',
                         title='Pie chart for total expenditure share by vehicle type during recessions')
        )
        unemplyment_rate = rec_data.groupby('Vehicle_Type')['unemployment_rate'].mean().reset_index()
        rchart4 = dcc.Graph(
            figure =  px.bar(unemplyment_rate, x='Vehicle_Type', y='unemployment_rate',
                   title='Bar chart for the effect of unemployment rate on vehicle type and sales')
        )
        return [
            html.Div(
                className='chart-item',
                children=[
                    html.Div(children=rchart1),html.Div(children=rchart2),
                ]
            ),
            html.Div(
                className='chart-item',
                children=[
                    html.Div(children=rchart3),html.Div(children=rchart4),
                ]
            )
        ]
    elif(yearly_statistics == 'Yearly Statistics'):
        yearly_data = df[df['Year'] == year_value] 
        yas = df.groupby('Year')['Automobile_Sales'].mean().reset_index()
        Ychart1 = dcc.Graph(
            figure=px.line(yas, x='Year', y='Automobile_Sales', title=':Yearly Automobile sales using line chart for the whole period')
         )  
        mon = df.groupby('Month')['Automobile_Sales'].mean().reset_index()
        Ychart2 = dcc.Graph(
            figure=px.line(mon,x='Month', y='Automobile_Sales', title='Total Monthly Automobile sales using line chart')
         )
        given_year = yearly_data.groupby('Year')['Competition'].mean().reset_index()
        Ychart3 = dcc.Graph(
             figure=px.bar(given_year, x='Year', y='Competition',
                           title='Average Vehicles Sold by Vehicle Type in the year {}'.format(year_value))
         )
        tot_ad = yearly_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
        Ychart4 = dcc.Graph(
             figure=px.pie(tot_ad, names='Vehicle_Type', values='Advertising_Expenditure',
                           title='Total Advertisement Expenditure for each vehicle using pie chart')
         )
        return [
            html.Div(
                className='chart-item',
                children=[
                    html.Div(children=Ychart1),html.Div(children=Ychart2),
                ]
            ),
            html.Div(
                className='chart-item',
                children=[
                    html.Div(children=Ychart3),html.Div(children=Ychart4),
                ]
            )
        ]

if __name__ == '__main__':
    app.run_server(debug=True)