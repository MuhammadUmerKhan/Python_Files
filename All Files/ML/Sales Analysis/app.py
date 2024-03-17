import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read the data
df = pd.read_csv("Data.csv")

# Define features and target variable
X = df[['median', 'mean', 'Week', 'Temperature', 'max', 'CPI', 'Fuel_Price', 'min', 'std', 'Unemployment',
        'Month', 'Total_MarkDown', 'Dept_16', 'Dept_18', 'Dept_3', 'IsHoliday', 'Size', 'Year', 'Dept_11',
        'Dept_1', 'Dept_9', 'Dept_5', 'Dept_55', 'Dept_56', 'Dept_7', 'Dept_72']]
Y = df['Weekly_Sales']

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test)

months = [
    {"label": "January", "value": 1},
    {"label": "February", "value": 2},
    {"label": "March", "value": 3},
    {"label": "April", "value": 4},
    {"label": "May", "value": 5},
    {"label": "June", "value": 6},
    {"label": "July", "value": 7},
    {"label": "August", "value": 8},
    {"label": "September", "value": 9},
    {"label": "October", "value": 10},
    {"label": "November", "value": 11},
    {"label": "December", "value": 12},
]

# Create Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.Div([
        html.H1("Weekly Sales Price Prediction", style={'text-align': 'center'}),
        html.Div([
            dcc.Dropdown(
                        id="month",
                        options=months,
                        placeholder="Select a month",
                        style={'margin': '10px', 'padding': '10px'}
                         ),
            dcc.Dropdown(id='holiday',
                         options=[
                             {'label': 'No Holiday', 'value': 0},
                             {'label': 'Holiday', 'value': 1}
                         ], placeholder='Select Holiday', style={'margin': '10px', 'padding': '10px'}),
            dcc.Dropdown(id='year',
                         options=[
                             {'label': '2010', 'value': 2010},
                             {'label': '2011', 'value': 2011},
                             {'label': '2012', 'value': 2012}
                         ], placeholder='Select Year', style={'margin': '10px', 'padding': '10px'}),
        ], style={'text-align': 'center'}),
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
    [State('month', 'value'),
     State('holiday', 'value'),
     State('year', 'value')]
)
def update_output(n_clicks, month, holiday, year):
    if n_clicks > 0 and all(v is not None for v in [month, holiday, year]):
        # Create features for prediction
        features = pd.DataFrame({
            'median': df['median'],
            'mean': df['mean'],
            'Week': df['Week'],
            'Temperature': df['Temperature'],
            'max': df['max'],
            'CPI': df['CPI'],
            'Fuel_Price': df['Fuel_Price'],
            'min': df['min'],
            'std': df['std'],
            'Unemployment': df['Unemployment'],
            'Month': month,
            'Total_MarkDown': df['Total_MarkDown'],
            'Dept_16': df['Dept_16'],
            'Dept_18': df['Dept_18'],
            'Dept_3': df['Dept_3'],
            'IsHoliday': holiday,
            'Size': df['Size'],
            'Year': year,
            'Dept_11': df['Dept_11'],
            'Dept_1': df['Dept_1'],
            'Dept_9': df['Dept_9'],
            'Dept_5': df['Dept_5'],
            'Dept_55': df['Dept_55'],
            'Dept_56': df['Dept_56'],
            'Dept_7': df['Dept_7'],
            'Dept_72': df['Dept_72']
        })

        # Predict
        predictions = model.predict(features)[0]
        return f"Predicted sales price for each record: ${predictions:.4f}"
    else:
        return "Please fill in all the fields to get the prediction."


if __name__ == '__main__':
    app.run_server(debug=True)
