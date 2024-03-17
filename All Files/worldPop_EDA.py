from turtle import title
from matplotlib.axis import YAxis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import express as px
from plotly import graph_objects as go

df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\world_population.csv")
df.head()
df.isnull().sum()
df.shape
df.info()
df.describe()

pd.set_option('display.float_format', lambda x: '%.2f' % x)
df.head()
df['2022 Population'].mean()
df['2022 Population'].median()

df.sort_values(by='2022 Population', ascending=True).head()
df.sort_values(by='2022 Population', ascending=False).head()

# Correlation
df_numerized = df
for i in df_numerized.columns:
    if (df_numerized[i].dtype == 'object'):
        df_numerized[i] = df_numerized[i].astype('category')
        df_numerized[i] = df_numerized[i].cat.codes
df_numerized.corr()
# Heatmap
plt.figure(figsize=(20, 7))
sns.heatmap(df_numerized.corr(), annot=True)
plt.xlabel('World Population')
plt.ylabel('World Population Features')
plt.title('Correlation Matric For Numeric Features')
plt.show()
# df_numerized

df['Continent'].unique()
df.head()
# Dealing with missing values
df.isnull().sum()
df['2022 Population'].mean()
df['2022 Population'].replace(np.NaN, df['2022 Population'].mean(), inplace=True)
df['2022 Population']

df['Growth Rate'].mean()
df['Growth Rate'].replace(np.NaN, df['Growth Rate'].mean(), inplace=True)

df['Density (per km²)'].mean()
df['Density (per km²)'].replace(np.NaN, df['Density (per km²)'].mean(), inplace=True)

df['Area (km²)'].mean()
df['Area (km²)'].replace(np.NaN, df['Area (km²)'].mean(), inplace=True)

df.isnull().sum()
df['2020 Population'].mean()
df['2020 Population'].replace(np.NaN, df['2020 Population'].mean(), inplace=True)
df.shape
(5/234)*100
df['2015 Population'].mean()
df['2015 Population'].replace(np.NaN, df['2015 Population'].mean(), inplace=True)

df['2010 Population'].mean()
df['2010 Population'].replace(np.NaN, df['2010 Population'].mean(), inplace=True)

df['2000 Population'].mean()
df['2000 Population'].replace(np.NaN, df['2000 Population'].mean(), inplace=True)

df.isnull().sum()
df['1990 Population'].mean()
df['1990 Population'].replace(np.NaN, df['1990 Population'].mean(), inplace=True)

df['1980 Population'].mean()
df['1980 Population'].replace(np.NaN, df['1980 Population'].mean(), inplace=True)

df['1970 Population'].mean()
df['1970 Population'].replace(np.NaN, df['1970 Population'].mean(), inplace=True)

df.isnull().sum()

df.head()
# _____________________
df.Continent.dtype
df.dtypes
df.head() 
df.isnull().sum()

df[df['Continent'].str.contains('Oceania')]
df['Continent'].value_counts()
df['Continent'].unique()
df.columns
df2 = df_numerized.groupby('Continent').mean().sort_values(by='2022 Population', ascending=False)
df2.plot(figsize=(20, 10))

df3 = df[['Continent', '2022 Population',
       '2020 Population', '2015 Population', '2010 Population',
       '2000 Population', '1990 Population', '1980 Population',
       '1970 Population']]
df3 = df3.groupby('Continent')[[ '1970 Population',
       '1980 Population', '1990 Population', '2000 Population',
       '2010 Population', '2015 Population', '2020 Population',
       '2022 Population']].mean().sort_values(by='2022 Population', ascending=False)
df3 = df3.transpose()


df3.plot(figsize=(20, 10))

df.to_csv("world_pop.csv")

df.boxplot(figsize=(20, 10))

# Visualization
# Population Growth Rate:
df.head()
bar_data = df.groupby('Continent')['Growth Rate'].mean().reset_index()
fig = px.bar(x=bar_data['Continent'], 
                 y=bar_data['Growth Rate'], 
                 title="Continent Population Growth Rate")
fig.update_layout(xaxis_title="Continent", yaxis_title='Growth Rate')
fig.show()

# continent population yearly
df.head()
df3.plot(kind="line", figsize=(20, 10))
plt.title("Continent Population")
plt.xlabel("Year")
plt.ylabel('')
plt.show()

# Population Growth Over Time:
world_pop = df.groupby('Continent')['World Population Percentage'].sum().reset_index()
fig = px.bar(x=world_pop['Continent'], 
             y=world_pop['World Population Percentage'])
fig.update_layout(xaxis_title = "Continent", yaxis_title = "World Population -Percentage", 
                  title="World Population Percentage according to continents")
fig.show()

# Area of Continents
df.head()
area = df.groupby('Continent')['Area (km²)'].mean().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=area['Continent'], y=area['Area (km²)']
                         , mode='lines', marker=dict(color='blue')))
fig.update_layout(title='Continent Area covering', xaxis_title = 'Continent', yaxis_title = 'Area (km²)')
fig.show()

# Regional Population Comparison:
fig = px.pie(world_pop, values='World Population Percentage',
             names='Continent', title='Regional Population Comparison',
             hover_name='Continent',
             labels={'World Population Percentage': 'Percentage'},
             color='Continent',
             color_discrete_map={'Asia': 'blue', 'Africa': 'green', 'Europe': 'red', 'North America': 'purple', 'South America': 'orange', 'Oceania': 'cyan'},
             hole=0.2)
fig.show()
df.head()