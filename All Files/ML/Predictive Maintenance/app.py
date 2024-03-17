# import dash
# from doctest import debug
from dash import Dash
from dash import dcc, html, Input, Output, State
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df_encoded = pd.read_csv("model_data.csv")
df_encoded.tail(3)
label = LabelEncoder()
df_encoded['Type'] = label.fit_transform(df_encoded['Type']) # M=2, L=1, H=0
X = df_encoded.drop(columns=['Target', 'Failure Type', 'Failure Type_encoded'])
Y = df_encoded['Failure Type_encoded']
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(x_train_scaled, Y, test_size=0.2, random_state=42)


ran_model = RandomForestClassifier()
ran_model.fit(x_train, y_train)
ran_model.score(x_test, y_test)*100 # 98.25
prediction_xgb = ran_model.predict(x_test)
ran_model.predict([[0,	299,	306.6,	1645,	33.4,	22]])

X.columns


app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Machine Predictive Maintenance Classification Dash Board", style={'text-align': 'center'}),
        html.Div([
            dcc.Input(id='temperature', placeholder="Enter Air Temperature in Kelvin", type='number',
                      style={'margin': '10px', 'padding': '10px', 'width': '70%', 'height': '35px'}),
            dcc.Input(id='process', placeholder="Enter Process Temperature in Kelvin", type='number',
                      style={'margin': '10px', 'padding': '10px', 'width': '70%', 'height': '35px'}),
            dcc.Input(id='rotational', placeholder="Rotational Speed [rpm]", type='number',
                      style={'margin': '10px', 'padding': '10px', 'width': '70%', 'height': '35px'}),
            dcc.Input(id='torque', placeholder="Torque", type='number',
                      style={'margin': '10px', 'padding': '10px', 'width': '70%', 'height': '35px'}),
            dcc.Input(id='tool_wear', placeholder="Enter Tool Wear in min", type='number',
                      style={'margin': '10px', 'padding': '10px', 'width': '70%', 'height': '35px'})
        ], style={'text-align':'center'}),
        html.Div([
            dcc.Dropdown(
                id='type',
                options=[
                    {'label': 'M', 'value': 2},
                    {'label': 'L', 'value': 1},
                    {'label': 'H', 'value': 0}
                ],
                placeholder="Select Type",
                style={'margin': '10px', 'padding': '10px', 'width': '90%', 'height': '35px'}
            )]),html.Br(),
        html.Div([
            html.Button('Predict Price', id='predict_button', n_clicks=0,
                        style={'margin': '10px', 'padding': '10px', 'background-color': '#007BFF', 'color': 'white'})
        ], style={'text-align': 'center'}),
        html.Div(id='prediction_output', style={'text-align': 'center', 'font-size': '20px', 'margin-top': '20px'}),
    ], style={'width': '50%', 'margin': '0 auto', 'border': '2px solid #007BFF', 'padding': '20px', 'border-radius': '10px'})
], style={'background-color': 'white'})

@app.callback(
    Output('prediction_output', 'children'),
    [Input('predict_button', 'n_clicks')],
    [State('temperature', 'value'),
     State('process', 'value'),
     State('rotational', 'value'),
     State('torque', 'value'),
     State('tool_wear', 'value'),
     State('type', 'value')])

def output(n_clicks, temp, process, rot, torq, tool,type):
    if n_clicks > 0 and all(v is not None for v in [temp, process, rot, torq, type]):
        pred = ran_model.predict([[type, temp, process, rot, torq, tool]])
        if pred == 1:
            return 'No Failure'
        elif pred == 0:
            return 'Heat Dissipation Failure'
        elif pred == 3:
            return 'Power Failure'
        elif pred == 2:
            return 'Overstrain Failure'
        elif pred == 5:
            return 'Tool Wear Failure'
        else:
            return 'Random Failures'
        
if __name__ == '__main__':
    app.run_server(debug=True)