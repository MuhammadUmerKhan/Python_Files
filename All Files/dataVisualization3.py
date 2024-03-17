from attr import s
import numpy as np
from cProfile import label
from matplotlib import figure
import pandas as pd
import matplotlib.pyplot as plt
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv'
df_can = pd.read_csv(url)
df_can.head()
df_can.shape
df_can.set_index('Country', inplace=True)
years = list(map(str, range(1980, 2014)))
# Pie Chart
df_Continents = df_can.groupby('Continent', axis=0).sum()
type(df_Continents)
df_Continents.head()

df_Continents['Total'].plot(kind='pie', figsize=(20, 10), startangle=90, autopct='%1.1f%%', shadow=True)
plt.title('Immigration to Canada by Continent [1980 - 2013]')
plt.axis('equal')
plt.legend(labels=df_Continents.index, loc='upper left')
plt.show()      

colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset each wedge.
df_Continents['Total'].plot(kind='pie', figsize=(10, 6), startangle=90, 
                autopct='%1.1f%%', shadow=True, labels=None, pctdistance=1.12)
plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.1, fontsize=15)
plt.axis('equal')
plt.legend(labels=df_Continents.index, loc='upper left', fontsize=7)
plt.show()

# Question: Using a pie chart, explore the proportion
# (percentage) of new immigrants grouped by continents in the year 2013.

df_Continents['Total'].plot(kind='pie', figsize=(10, 6), labels=None, 
                explode = explode_list, startangle=90, shadow=True, autopct='%1.1f%%', pctdistance=1.12)
plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12, fontsize=15)
plt.axis('equal')
plt.legend(labels=df_Continents.index, loc='upper left', fontsize=7)
plt.show()

# Box Plot
# Minimum: The smallest number in the dataset excluding the outliers.
# First quartile: Middle number between the minimum and the median.
# Second quartile (Median): Middle number of the (sorted) dataset.
# Third quartile: Middle number between median and maximum.
# Maximum: The largest number in the dataset excluding the outliers.
df_japan = df_can.loc[['Japan'], years].transpose()
df_japan.plot(kind='box', figsize=(8, 6))
plt.title('Box plot of Japanese Immigrants from 1980 - 2013')
plt.ylabel('Number of Immigrants')
plt.show()
df_japan.describe()

df_CI = df_can.loc[['China', 'India'], years].transpose()
df_CI.plot(kind='box', figsize=(10, 6))
plt.title('Box plot of Chinese and Indian Immigrants from 1980 - 2013')
plt.ylabel('Number of Immigrants')
plt.show()
df_CI.describe()

# Subplots
fig = plt.figure() # create figure

ax0 = fig.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

# Subplot 1: Box plot
df_CI.plot(kind='box', color='blue', vert=False, figsize=(20, 6), ax=ax0) # add to subplot 1
ax0.set_title('Box Plots of Immigrants from China and India (1980 - 2013)')
ax0.set_xlabel('Number of Immigrants')
ax0.set_ylabel('Countries')

# Subplot 2: Line plot
df_CI.plot(kind='line', figsize=(20, 6), ax=ax1) # add to subplot 2
ax1.set_title ('Line Plots of Immigrants from China and India (1980 - 2013)')
ax1.set_ylabel('Number of Immigrants')
ax1.set_xlabel('Years')

plt.show()

# Question: Create a box plot to visualize the distribution of the top 15 countries 
# (based on total immigration) 
# grouped by the decades 1980s, 1990s, and 2000s.
df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)
df_top15 = df_can.head(15)
years_80s = list(map(str, range(1980, 1990)))
years_90s = list(map(str, range(1990, 2000)))
years_00s = list(map(str, range(2000, 2010)))

df_80s = df_top15.loc[:, years_80s].sum(axis=1)
df_90s = df_top15.loc[:, years_90s].sum(axis=1)
df_00s = df_top15.loc[:, years_00s].sum(axis=1)

new_df = pd.DataFrame({'1980s':df_80s, '1990s':df_90s, '2000s':df_00s})
new_df.head()
new_df.describe()

new_df.plot(kind='box', figsize=(10, 6))
plt.title('Immigration from top 15 countries for decades 80s, 90s and 2000s')
plt.show()

new_df = new_df.reset_index()
new_df[new_df['2000s']> 209611.5]

# Scatter Plot
# Using a scatter plot, let's visualize the trend of total immigrantion to Canada 
# (all countries combined) for the years 1980 - 2013.
df_tot = pd.DataFrame(df_can[years].sum(axis=0))
df_tot.head()
df_tot.index = map(int, df_tot.index)
df_tot.reset_index(inplace=True)
df_tot.columns = ['Years', 'Total']

df_tot.plot(kind='scatter', x='Years', y='Total', figsize=(10, 6), color='darkblue')
plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.show()

x = df_tot['Years']
y = df_tot['Total']
fit = np.polyfit(x, y, deg=1)
fit

df_tot.plot(kind='scatter', x='Years', y='Total', figsize=(10, 6), color='darkblue')
plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
# plot line of best fit
plt.plot(x, fit[0] * x + fit[1], color='red') # recall that x is the Years
plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))
plt.show()

df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
df_total = pd.DataFrame(df_countries.sum(axis=1))
df_total.reset_index(inplace=True)
df_total.columns = ['year', 'total']
df_total['year'] = df_total['year'].astype(int)
df_total.head()

df_total.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')
plt.title('Total Immigration of Denmark, Norway and Sweden to Canada from 1980 - 2013')
plt.ylabel('Number of immigirants')
plt.xlabel('Year')
plt.show()

df_can_t = df_can[years].transpose()
df_can_t.index = map(int, df_can_t.index)
df_can_t.index.name = 'Year'
df_can_t.reset_index(inplace=True)


# Bubble Plot
norm_brazil = (df_can_t['Brazil']) - (df_can_t['Brazil'].min()) / (df_can_t['Brazil'].max() - df_can_t['Brazil'].min())
norm_argentina = (df_can_t['Argentina']) - (df_can_t['Argentina'].min()) / (df_can_t['Argentina'].max() - df_can_t['Argentina'].min())

# Brazil
ax0 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='Brazil',
                    figsize=(14, 8),
                    alpha=0.5,  # transparency
                    color='green',
                    s=norm_brazil * 2000 + 10,  # pass in weights 
                    xlim=(1975, 2015)
                    )

# Argentina
ax1 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='Argentina',
                    alpha=0.5,
                    color="blue",
                    s=norm_argentina * 2000 + 10,
                    ax=ax0
                    )

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from Brazil and Argentina from 1980 to 2013')
ax0.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')

norm_india = (df_can_t['India']) - (df_can_t['India'].min()) / (df_can_t['India'].max() - df_can_t['India'].min())
norm_china = (df_can_t['China']) - (df_can_t['China'].min()) / (df_can_t['China'].max() - df_can_t['China'].min())
    
# China
ax0 = df_can_t.plot(kind='scatter',
                        x='Year',
                        y='China',
                        figsize=(14, 8),
                        alpha=0.5,                  # transparency
                        color='green',
                        s=norm_china * 2000 + 10,  # pass in weights 
                        xlim=(1975, 2015)
                       )

# India
ax1 = df_can_t.plot(kind='scatter',
                        x='Year',
                        y='India',
                        alpha=0.5,
                        color="blue",
                        s=norm_india * 2000 + 10,
                        ax = ax0
                       )

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from China and India from 1980 - 2013')
ax0.legend(['China', 'India'], loc='upper left', fontsize='x-large')