from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv'
df_can = pd.read_csv(url)
df_can.head()
df_can.shape
df_can.set_index('Country', inplace=True)
# Area Plot
years = list(map(str, range(1980, 2014)))
df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)
df_top5 = df_can.head()
df_top5 = df_top5[years].transpose()

df_top5.plot(kind='area', figsize=(20, 10), stacked=False)
plt.title('Immigration Trend of Top 5 Countries')
plt.xlabel('Years')
plt.ylabel('Countries')
plt.show()
df_top5.plot(kind='area', alpha = 0.35, figsize=(20, 10), stacked=False) # alpha value default 0 to -1
plt.title('Immigration Trend of Top 5 Countries')
plt.xlabel('Years')
plt.ylabel('Countries')
plt.show()

# Option 2: Artist layer (Object oriented method) -
# using an Axes instance from Matplotlib (preferred)
ax = df_top5.plot(kind='area', alpha=0.35, figsize=(20, 10))
ax.set_title('Immigration Trend of Top 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')

# Option 2: Artist layer (Object oriented method) -
# using an Axes instance from Matplotlib (preferred)
df_can.sort_values(by='Total', ascending=True, axis=0, inplace=True)
df_least = df_can.head()
df_least = df_least[years].transpose()
df_least.index = df_least.index.map(int)
ax1 = df_least.plot(kind='area', alpha=0.45, figsize=(20, 10))
ax1.set_title('Immigration Trend of least 5 Countries')
ax1.set_ylabel('Number of Immigrants')
ax1.set_xlabel('Years')

df_can.sort_values(by='Total', ascending=True, axis=0, inplace=True)
df_least = df_can.head()
df_least = df_least[years].transpose()
df_least.index = df_least.index.map(int)
ax1 = df_least.plot(kind='area', alpha=0.55, figsize=(20, 10), stacked=False)
ax1.set_title('Immigration Trend of least 5 Countries')
ax1.set_ylabel('Number of Immigrants')
ax1.set_xlabel('Years')

# Histogram
df_can['2013'].plot(kind='hist', figsize=(20, 10))
plt.title('Histogram of Immigration from 195 Countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')
plt.show()

counts, bin_edges = np.histogram(df_can['2013'])
df_can['2013'].plot(kind='hist', figsize=(20, 10), xticks=bin_edges)
plt.title('Histogram of Immigration from 195 Countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')
plt.show()

df_can.loc[['Denmark', 'Norway', 'Sweden'], years]
df_can.loc[['Denmark', 'Norway', 'Sweden'], years].plot.hist()
df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()

df_t.plot(kind='hist', figsize=(20, 10))
plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Years')
plt.show()

count, bin_edge = np.histogram(df_t, 15)
df_t.plot(kind='hist', stacked=False, figsize=(20, 10), bins=15, alpha=0.6, color=['coral', 'darkslateblue', 'mediumseagreen'])
plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Years')
plt.show()

count, bin_edges = np.histogram(df_t, 15)
xmin = bin_edges[0] - 10   #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes 
xmax = bin_edges[-1] + 10  #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes

# stacked Histogram
df_t.plot(kind='hist',
          figsize=(10, 6), 
          bins=15,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen'],
          stacked=True,
          xlim=(xmin, xmax)
         )
plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants') 
plt.show()

df_can.loc[['Greece', 'Albania', 'Bulgaria'], years]
df_G = df_can.loc[['Greece', 'Albania', 'Bulgaria'], years].transpose()
df_can.loc[['Greece', 'Albania', 'Bulgaria'], years].plot.hist()

count, bin_edges = np.histogram(df_G, 15)
df_G.plot(kind='hist', figsize=(20, 10), stacked= False, bins =15, alpha=0.35, xticks=bin_edges,color=['coral', 'darkslateblue', 'mediumseagreen'])
plt.title('Histogram of Immigration from Greece Albania and Bulgaria from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Years')
plt.show()

#  Bar Plot
df_iceland = df_can.loc['Iceland',years]
df_iceland.transpose()
df_iceland.transpose().head()
df_iceland.plot(kind='bar', figsize=(20, 10))
plt.xlabel('Year')
plt.ylabel('Number of immigrants')
plt.title('Icelandic immigrants to Canada from 1980 to 2013')
plt.show()

df_iceland.plot(kind='bar', figsize=(10, 6), rot=90)  # rotate the xticks(labelled points on x-axis) by 90 degrees
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.title('Icelandic Immigrants to Canada from 1980 to 2013')

# Annotate arrow
plt.annotate('',  # s: str. Will leave it blank for no text
             xy=(32, 70),  # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),  # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',  # will use the coordinate system of the object being annotated
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
             )

plt.show()

df_iceland.plot(kind='bar', figsize=(10, 6), rot=90)
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.title('Icelandic Immigrants to Canada from 1980 to 2013')
# Annotate arrow
plt.annotate('',  # s: str. will leave it blank for no text
             xy=(32, 70),  # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),  # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',  # will use the coordinate system of the object being annotated
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
             )
# Annotate Text
plt.annotate('2008 - 2011 Financial Crisis',  # text to display
             xy=(28, 30),  # start the text at at point (year 2008 , pop 30)
             rotation=72.5,  # based on trial and error to match the arrow
             va='bottom',  # want the text to be vertically 'bottom' aligned
             ha='left',  # want the text to be horizontally 'left' algned.
             )
plt.show()
df_can.sort_values(by='Total', ascending=True, axis=0, inplace=True)
df_top15 = df_can['Total'].tail(15)

df_top15.plot(kind='barh', figsize=(12, 12), color='steelblue')
plt.xlabel('Number of Immigrants')
plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')

    # annotate value labels to each country
for index, value in enumerate(df_top15): 
    label = format(int(value), ',') # format int with commas
    
    # place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
    plt.annotate(label, xy=(value - 47000, index - 0.10), color='white')

plt.show()