from enum import auto
import os
from tkinter import Y
from turtle import color
from matplotlib import markers
from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')
df_can.shape
df_can.set_index('Country', inplace=True)
years = list(map(str, range(1980, 2014)))
df_can.head()

df_line = df_can[years]
total_immigirants = df_line.sum()

# Line Plot
fig, ax = plt.subplots()
ax.plot(total_immigirants)
ax.set_title('Immigrants between 1980 to 2013')
ax.set_ylabel('Total immigirants')
ax.set_xlabel('Years')
plt.show()
# The plot function populated the x-axis with the index values (years), 
# and the y-axis with the column values (population).
# However, notice how the years were not displayed because they are of type string.
# Therefore, let's change the type of the index values to integer for plotting.¶
total_immigirants.index = total_immigirants.index.map(int)
fig, ax = plt.subplots()
ax.plot(total_immigirants,
        marker='s',
        markersize=5,
        color='green',
        linestyle='dotted'
        )
ax.set_title('Immigrants between 1980 to 2013')
ax.set_ylabel('Total immigirants')
ax.set_xlabel('Years')
ax.legend(['Immigirants'])
plt.show()

# Let's include the background grid, a legend and try to change the limits on the axis
total_immigirants.index = total_immigirants.index.map(int)
fig, ax = plt.subplots()
ax.plot(total_immigirants,
        marker='s',
        markersize=5,
        color='green',
        linestyle='dotted'
        )
ax.set_title('Immigrants between 1980 to 2013')
ax.set_ylabel('Total immigirants')
ax.set_xlabel('Years')
plt.grid(True)
plt.xlim(1975, 2015)
ax.legend(['Immigirants'])
plt.show()


df_can.reset_index(inplace=True)
df_haiti = df_can[df_can['Country'] == 'Haiti']
df_haiti = df_haiti[years].T
df_haiti.columns = ['Total Immigirants']
df_haiti.index = df_haiti.index.map(int)
fig, ax = plt.subplots()
ax.plot(df_haiti,
        marker='s',
        color='blue',
        linestyle='dotted',
        )
ax.set_title('Immigrants between 1980 to 2013')
ax.set_ylabel('Total immigirants')
ax.set_xlabel('Years')
plt.grid(True)
plt.xlim(1975, 2015)
ax.legend(['Immigirants'])
ax.annotate('2010 Earthquake',xy=(2000, 6000))
plt.show()

# Scatter Plot
fig, ax = plt.subplots(figsize=(10, 8))
total_immigirants.index = total_immigirants.index.map(int)
ax.scatter(total_immigirants.index, total_immigirants, marker = 'o', s=20, color='darkblue')
plt.title('Immigrants between 1980 to 2013')
plt.ylabel('Total Immigirants')
plt.legend(['Immigirants'], loc='upper center')
plt.xlabel('Year')
plt.xlim(1975, 2015)
plt.grid(True)
plt.show()

# Bar Plot
df_can.sort_values(['Total'], ascending=False, inplace=True, axis=0)
df_top_5 = df_can.head(5)
df_bar_5 = df_top_5.reset_index()
label = list(df_bar_5.Country)
label[2] = 'UK'

fig, ax = plt.subplots(figsize=(10, 8))
ax.bar(label, df_bar_5['Total'], label=label)
plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of immigirants')
plt.xlabel('Year')
plt.show()

# Least
df_can.sort_values(['Total'], ascending=True, inplace=True, axis=0)
df_least_5 = df_can.head(5)
df_least_bar_5 = df_least_5.reset_index()
least_label = list(df_least_5.Country)

fig, ax = plt.subplots(figsize=(10, 8))
ax.bar(least_label, df_least_5['Total'], label=least_label, color='red')
plt.title('Immigration Trend of Top 5 least Countries')
plt.xlabel('Year')
plt.ylabel('Number of immigirants')
plt.show()

# Histogram
df_country = df_can.groupby(['Country'])['2013'].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
count = ax.hist (df_country['2013'])
ax.hist(df_country['2013'])
plt.title('New Immigirants in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of immigirants')
plt.xticks(list(map(int, count[1])))
plt.legend(['Immigirants'])
plt.show()

# We can also plot multiple histograms on the same plot. 
# For example, let's try to answer the following questions using a histogram.
# What is the immigration distribution
# for Denmark, Norway, and Sweden for years 1980 - 2013?
df = df_can.groupby(['Country'])[years].sum()
df_dns = df.loc[['Denmark', 'Norway', 'Sweden'], years]
df_dns = df_dns.T

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df_dns)
plt.title('Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigirants')
plt.legend(['Denmark', 'Norway', 'Sweden'])
plt.show()

df_I_C = df_can.groupby(['Country'])[years].sum()
y = list(map(str, range(2000, 2014)))
df_IC = df_I_C.loc[['India', 'China'], y]
df_IC = df_IC.T

fig, ax = plt.subplots(figsize = (10, 4))
ax.hist(df_IC)
plt.title('Immigration from China and India from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigirants')
plt.legend(['China', 'India'])
plt.show()

# Pie Chart
fig, ax = plt.subplots(figsize= (10, 8))
ax.pie(total_immigirants[0:5], labels=years[0:5],
       colors = ['gold', 'blue', 'lightgreen', 'coral', 'cyan'],
       autopct='%1.1f%%',
       explode = [0, 0, 0, 0, 0.1]
       )
ax.set_aspect('equal')
plt.title('Distribution of immigirants from 1980 to 1985')
plt.legend(years[0:5])
plt.show()

# Question: 
# Create a pie chart representing # the total immigrants proportion for each continent
df_continent = df_can.groupby(['Continent'])['Total'].sum().reset_index()
labels = list(df_continent.Continent)
labels[3] = 'LAC'
labels[4] = 'NA'

fig,ax=plt.subplots(figsize=(10, 4))
#Pie on immigrants
ax.pie(df_continent['Total'], colors = ['gold','blue','lightgreen','coral','cyan','red'],
           autopct='%1.1f%%', pctdistance=1.25, explode=[0, 0, 0, 0, 0.1, 0.2])
ax.set_aspect('equal')  # Ensure pie is drawn as a circle
plt.title('Continent-wise distribution of immigrants')
ax.legend(label,bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()

# Sub plots
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].plot(total_immigirants)
ax[0].set_title('Line plot on immigirants')

ax[1].plot(total_immigirants.index, total_immigirants)
ax[1].set_title('Scatter Plot on immigirants')

ax[0].set_ylabel('Number of immigirants')

fig.suptitle('Subplotting Example', fontsize = 15)

fig.tight_layout()
plt.show()

# 2
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(total_immigirants)
ax1.set_title('Line plot on immigirants')

ax2 = fig.add_subplot(1, 2, 2)
ax2.barh(total_immigirants.index, total_immigirants)
ax2.set_title('Bar plot on immigirants')

fig.suptitle('Supplotting Example', fontsize=15)
fig.tight_layout()
plt.show()


# Question: 
# Choose any four plots, which you have developed in this lab, 
# with subplotting display them in a 2x2 display
fig = plt.figure(figsize=(10, 10))
# plot  number 1
ax1 = fig.add_subplot(2, 2, 1)
ax1.pie(df_continent['Total'], colors = ['gold','blue','lightgreen','coral','cyan','red'],
           autopct='%1.1f%%', pctdistance=1.25, explode=[0, 0, 0, 0, 0.1, 0.2])
ax1.set_aspect('equal')  # Ensure pie is drawn as a circle
ax1.set_title('Continent-wise distribution of immigrants')
ax1.legend(label,bbox_to_anchor=(1, 0, 0.5, 1))

# plot number 2
ax2 = fig.add_subplot(2, 2, 2)
ax2.hist(df_IC)
ax2.set_title('Immigration from China and India from 1980 - 2013')
ax2.set_ylabel('Number of Years')
ax2.set_xlabel('Number of Immigirants')
ax2.grid(True)
ax2.legend(['China', 'India'])

# plot number 3
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(total_immigirants,
        marker='s',
        markersize=5,
        color='green',
        linestyle='dotted'
        )
ax3.set_title('Immigrants between 1980 to 2013')
ax3.set_ylabel('Total immigirants')
ax3.set_xlabel('Years')
ax3.grid(True)
ax3.set_xlim(1975, 2015)
ax3.legend(['Immigirants'])

# plot number 4
ax4 = fig.add_subplot(2, 2, 4)
ax4.scatter(total_immigirants.index, total_immigirants, marker = 'o', s=20, color='darkblue')
ax4.set_title('Immigrants between 1980 to 2013')
ax4.set_ylabel('Total Immigirants')
ax4.legend(['Immigirants'], loc='upper center')
ax4.set_xlabel('Year')
ax4.set_xlim(1975, 2015)
ax4.grid(True)

fig.suptitle('Four plots', fontsize=15)
fig.tight_layout()
plt.show()