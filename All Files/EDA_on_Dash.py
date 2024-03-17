from doctest import debug
from turtle import title
from fsspec import Callback
import pandas as pd
import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
from plotly import express as px
from plotly import graph_objects as go
from sqlalchemy import false

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv")
df.drop(columns='Unnamed: 0', inplace=True)
df.to_csv("spacex_launch_dash.csv", index=False)
df = pd.read_csv("spacex_launch_dash.csv")
df.head()
df['Launch Site'].unique()
sites = ['CCAFS LC-40', 'VAFB SLC-4E', 'KSC LC-39A', 'CCAFS SLC-40']
min_value = df['Payload Mass (kg)'].min()
max_value = df['Payload Mass (kg)'].max()

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.Div([
        html.H1("SpaceX Launch Dash", style={'textAlign': 'center', 'color': '#EEFAF8', 'font-size': 30}),
        html.H2("Which one you like to see first:", style={'textAlign': 'left', 'color': '#DDFBD0', 'font-size': 20})
    ]),
    # TASK 1: Add a Launch Site Drop-down Input Component
    html.Div([
        dcc.Dropdown(
            id='site-dropdown',
            options=[
                {'label': 'All Sites', 'value': 'ALL'},
                * [{'label': site, 'value': site} for site in sites]
            ],
            value='ALL', placeholder="Select a Launch Site here", searchable=True,
            style={'width': '80%', 'padding': '3px', 'font-size': '20px', 'text-align-last': 'center'}
        ),
    ]),
    html.Br(),
    html.Div(dcc.Graph(id="success-pie-chart")),
    html.P("Range slider to select Payload(kg):",
           style={'textAlign': 'left', 'color': '#3C73FB'}),
    # TASK 3: Add a Range Slider to Select Payload
    html.Div(
        dcc.RangeSlider(
            id="payload-slider",
            min=min_value,
            max=max_value,
            step=1000,
            marks={min_value: str(min_value), max_value: str(max_value)},
            value=[min_value, max_value]
        )
    ),
    html.Div(
        dcc.Graph(id="success-payload-scatter-chart")
    )
])

# TASK 2: Add a callback function to render success-pie-chart based on selected site dropdown
@app.callback(
    Output(component_id="success-pie-chart", component_property="figure"),
    Input(component_id="site-dropdown", component_property="value")
)
def get_pie_chart(entered_site):
    if entered_site == "ALL":
        fig = px.pie(
            df,
            values="class",
            names="Launch Site",
            title="Success Count for all launch sites",
            color="Launch Site",
            color_discrete_map={"CCAFS LC-40": "green", "VAFB SLC-4E": "blue", "KSC LC-39A": "orange", 'CCAFS SLC-40': "pink"},
            labels={"class": "Success"},
            hole=0.2)
        fig.update_layout(title_x=0.5, title_font=dict(size=20))
        return fig
    else:
        filtered_df = df[df['Launch Site'] == entered_site]
        filtered_df = filtered_df.groupby(['Launch Site', 'class']).size().reset_index(name="class count")
        fig = px.pie(
            filtered_df,
            values="class count",
            names="class",
            title=f"Total Success Launches for site {entered_site}",
            color="class",
            color_discrete_map={0: "red", 1: "green"},
            labels={"class": "Launch Outcome"},
            hole=0.2,
            category_orders={"class": [0, 1]}
        )
        fig.update_layout(
            title_x=0.5,
            title_font=dict(size=20),
            legend_title_text="Launch Outcome",
        )
        return fig
# TASK 4: Add a callback function to render the success-payload-scatter-chart scatter plot
@app.callback(
    Output(component_id="success-payload-scatter-chart", component_property="figure"),
    [Input(component_id='site-dropdown', component_property='value'),
     Input(component_id="payload-slider", component_property="value")]
)
def scatter_plot(entered_site, payload):
    filtered_df = df[(df['Payload Mass (kg)'] >= payload[0]) & (df['Payload Mass (kg)'] <= payload[1])]

    if entered_site == 'ALL':
        fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class',
                         color='Booster Version Category',
                         title='Success count on Payload mass for all sites'
                         )
        return fig
    else:
        filtered_df = df[df['Launch Site'] == entered_site]
        fig = px.scatter(filtered_df, x='Payload Mass (kg)',
                         y='class', color='Booster Version Category',
                         title=f"Success count on Payload mass for site {entered_site}")
        return fig


if __name__ == '__main__':
    app.run_server(debug=True)
