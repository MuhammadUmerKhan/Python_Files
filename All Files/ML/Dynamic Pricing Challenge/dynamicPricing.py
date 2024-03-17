from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from turtle import title
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sqlalchemy import true

df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\dynamic_pricing.csv")
df.shape
df.head()
df.describe()

df.isnull().sum()
# Outlier Treatment
df.head()
df['Location_Category'].unique()
df['Customer_Loyalty_Status'].unique()
df['Time_of_Booking'].unique()
df['Vehicle_Type'].unique()
df.info()

for i in df.columns:
    if df[i].dtype != 'object':
        df['Z_score'] = (df[i] - df[i].mean())/df[i].std()
        df = df[(df['Z_score']>-3) & (df['Z_score']<3)]
        df.drop(columns='Z_score', inplace=True)
df.shape

# Visualization
# Expected Ride vs Historical Cost of Ride
fig = px.scatter(df, x='Expected_Ride_Duration', y='Historical_Cost_of_Ride', title="Expected Ride vs Historical Cost of Ride", trendline='ols')
fig.show()
# Now let’s have a look at the distribution of the historical cost of rides based on the vehicle type
fig = px.box(df, x='Vehicle_Type', y='Historical_Cost_of_Ride', title="Historical Cost of Ride Distribution by Vehicle Type")
fig.show()
df.dtypes

object_columns = df.select_dtypes(include=['int64', 'float64']).columns
df_filtered = df[object_columns]
corr = df_filtered.corr()
fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                ))
fig.update_layout(title="Correlation Matrix")
fig.update_traces(colorscale='Viridis')
fig.show()

veh_eco = df.groupby(['Vehicle_Type'])['Historical_Cost_of_Ride'].mean()
fig = px.bar(x=veh_eco.index, y=veh_eco.values, title="Historical Cost of Ride on Vehicle type")
fig.update_layout(xaxis_title="Vehicle_Type", yaxis_title='Historical_Cost_of_Ride')
fig.show()

df.head()
df.describe(include='all')
high_demand_percentile = 75
low_demand_percentile = 25
df['demand_multiplier'] = np.where(df['Number_of_Riders'] > np.percentile(df['Number_of_Riders'], high_demand_percentile),
                                     df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], high_demand_percentile),
                                     df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], low_demand_percentile))

high_supply_percentile = 75
low_supply_percentile = 25

df['supply_multiplier'] = np.where(df['Number_of_Drivers'] > np.percentile(df['Number_of_Drivers'], low_supply_percentile),
                                     np.percentile(df['Number_of_Drivers'], high_supply_percentile) / df['Number_of_Drivers'],
                                     np.percentile(df['Number_of_Drivers'], low_supply_percentile) / df['Number_of_Drivers'])
demand_threshold_high = 1.2  # Higher demand threshold
demand_threshold_low = 0.8  # Lower demand threshold
supply_threshold_high = 0.8  # Higher supply threshold
supply_threshold_low = 1.2  # Lower supply threshold
df['adjusted_ride_cost'] = df['Historical_Cost_of_Ride'] * (
    np.maximum(df['demand_multiplier'], demand_threshold_low) *
    np.maximum(df['supply_multiplier'], supply_threshold_high)
)
df.head()

df['Profit_percentage'] = (df['Historical_Cost_of_Ride'] - df['adjusted_ride_cost']) * 100
df.head()

profitable_rides = df[df['Profit_percentage'] > 0]
lost_rides = df[df['Profit_percentage'] < 0]

profitable_count = len(profitable_rides)
lost_rides = len(lost_rides)
values = [profitable_count, lost_rides]
labels = ['Profitable Rides', "Loss Rides"]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
fig.update_layout(title='Profitability of Rides (Dynamic Pricing vs. Historical Pricing)')
fig.show()

df.columns
fig = px.scatter(df, y='adjusted_ride_cost', x='Expected_Ride_Duration', 
                 title='Expected Ride Duration vs. Cost of Ride', 
                 trendline='ols')
fig.show()

# Training a Predictive Model
def data_preprocessing_pipeline(data):
    numeric_feature = data.select_dtypes(include=['int', 'float']).columns
    categorical_feature = data.select_dtypes(include=['object']).columns
    
    data[numeric_feature] = data[numeric_feature].fillna(data[numeric_feature].mean())
    #Detect and handle outliers in numeric features using IQR
    for feature in numeric_feature:
        Q1 = data[feature].quantile(0.25) 
        Q2 = data[feature].quantile(0.75) 
        IQR = Q2 - Q1;
        lower_bond = Q1 - (1.5 * IQR)
        upper_bond = Q2 + (1.5 * IQR)
        data[feature] = np.where((data[feature] < lower_bond) | (data[feature].mean(), data[feature]))
        
        data[categorical_feature] = data[categorical_feature].fillna(data[categorical_feature].mode().iloc[0])
        
        return data
    
df.head()
df["Vehicle_Type"] = df["Vehicle_Type"].map({"Premium": 1, 
                                           "Economy": 0})

df.head()

X = np.array(df[['Number_of_Riders', 'Number_of_Drivers', 'Vehicle_Type', 'Expected_Ride_Duration']])
Y = np.array(df['adjusted_ride_cost'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
y_train = y_train.ravel()
y_test = y_test.ravel()

model = RandomForestRegressor()
model.fit(x_train, y_train)
model.score(x_test, y_test) # 0.8818918540942002
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
score = cross_val_score(RandomForestRegressor(), X, Y, cv=cv)
score.mean() # 0.8762383604531083

def get_vehicle_type_numeric(vehicle_type):
    vehicle_type_mapping = {
        "Premium": 1,
        "Economy": 0
    }
    vehicle_type_numeric = vehicle_type_mapping.get(vehicle_type)
    return vehicle_type_numeric
  
# Predicting using user input values
def predict_price(number_of_riders, number_of_drivers, vehicle_type, Expected_Ride_Duration):
    vehicle_type_numeric = get_vehicle_type_numeric(vehicle_type)
    if vehicle_type_numeric is None:
        raise ValueError("Invalid vehicle type")
    
    input_data = np.array([[number_of_riders, number_of_drivers, vehicle_type_numeric, Expected_Ride_Duration]])
    predicted_price = model.predict(input_data)
    return predicted_price

predicted_price = predict_price(50, 25, "Economy", 30)
print("Predicted Price:", predicted_price)
# Predict on the test set
y_pred = model.predict(x_test)

# Create a scatter plot with actual vs predicted values
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=y_test.flatten(),
    y=y_pred,
    mode='markers',
    name='Actual vs Predicted'
))

# Add a line representing the ideal case
fig.add_trace(go.Scatter(
    x=[min(y_test.flatten()), max(y_test.flatten())],
    y=[min(y_test.flatten()), max(y_test.flatten())],
    mode='lines',
    name='Ideal',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title='Actual vs Predicted Values',
    xaxis_title='Actual Values',
    yaxis_title='Predicted Values',
    showlegend=True,
)

fig.show()