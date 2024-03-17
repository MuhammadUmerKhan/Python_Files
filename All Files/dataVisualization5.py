from operator import index
from tkinter import font
from PIL import Image
import requests
import urllib
from wordcloud import WordCloud, STOPWORDS
from matplotlib import legend
from pywaffle import Waffle
import matplotlib.patches as mpatches
from asyncio import Handle
from turtle import title
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv'
df_can = pd.read_csv(url)
df_can.head() 
df_can.shape
df_can.set_index('Country', inplace=True)

# Waffle Chart
df_dsn = df_can.loc[['Denmark', 'Sweden', 'Norway'], :]

total_values = df_dsn['Total'].sum()
category_proportion = df_dsn['Total'] / total_values
pd.DataFrame({'Category Proportion: ': category_proportion})

width = 40
height = 10
total_num_tiles = width * height
print(f'The Total Number of titles is: {total_num_tiles}')

tiles_per_category = (category_proportion * total_num_tiles).round().astype(int)
pd.DataFrame({'Number of titles is:': tiles_per_category})

# initialize the waffle chart as an empty matrix
waffle_chart = np.zeros((height, width), dtype = np.uint)

# define indices to loop through waffle chart
category_index = 0
tile_index = 0

# populate the waffle chart
for col in range(width):
    for row in range(height):
        tile_index += 1

        # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...
        if tile_index > sum(tiles_per_category[0:category_index]):
            # ...proceed to the next category
            category_index += 1       
            
        # set the class value to an integer, which increases with class
        waffle_chart[row, col] = category_index


# instantiate a new figure object
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()

# get the axis
ax = plt.gca()

# set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
# add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])
values_cumsum = np.cumsum(df_dsn['Total'])
total_values = values_cumsum[len(values_cumsum) - 1]
legend_handle = []

for i, category in enumerate(df_dsn.index.values):
    label_str = category + ' (' + str(df_dsn['Total'][i]) + ')'
    color_val = colormap(float(values_cumsum[i])/total_values)
    legend_handle.append(mpatches.Patch(color = color_val, label = label_str)) 

plt.legend(handles = legend_handle, loc='lower center', ncol=len(df_dsn.index.values),
           bbox_to_anchor = (0, -0.2, 0.95, .1)
           )    
plt.show()  


def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):

    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height # total number of tiles
    print ('Total number of tiles is', total_num_tiles)
    
    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print (df_dsn.index.values[i] + ': ' + str(tiles))
    
    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1

            # if the number of tiles populated for the current category 
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1       
            
            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index
    
    # instantiate a new figure object
    fig = plt.figure()

    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
            
        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )
    plt.show()
width = 40 # width of chart
height = 10 # height of chart

categories = df_dsn.index.values # categories
values = df_dsn['Total'] # correponding values of categories

colormap = plt.cm.coolwarm # color map class
create_waffle_chart(categories, values, height, width, colormap)

fig = plt.figure(
    FigureClass= Waffle,
    rows = 20, columns = 30,
    values = df_dsn['Total'],
    cmap_name = 'tab20',
    legend = {'labels': [f"{k} ({v})" for k, v in zip(df_dsn.index.values,df_dsn.Total)],
                            'loc': 'lower left', 'bbox_to_anchor':(0,-0.1),'ncol': 3} 
    )
plt.show()

# Question: Create a Waffle chart to dispaly the proportiona of China 
# and Inida total immigrant contribution.
df_IC = df_can.loc[['India', 'China'], :]
total_values_IC = df_IC['Total'].sum()
category_proportion_ic = df_IC['Total'] / total_values

fig = plt.figure(
    FigureClass= Waffle, rows=20, columns = 30, 
    values = df_IC['Total'],
    cmap_name = 'tab20',
    legend = {'labels': [f"{k} ({v})" for k, v in zip(df_IC.index.values, df_IC.Total)],
              'loc':'lower left', 'bbox_to_anchor':(0, -0.1), 'ncol': 2
              }
)

alice_novel = urllib.request.urlopen( 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/alice_novel.txt').read().decode("utf-8")
stopwords = set(STOPWORDS)
alice_wc = WordCloud()
alice_wc.generate(alice_novel)

fig = plt.figure(figsize= (14, 18))
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

alice_mask = np.array(Image.open(urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/labs/Module%204/images/alice_mask.png')))
fig = plt.figure(figsize= (14, 18))
plt.imshow(alice_mask, interpolation='bilinear')
plt.axis('off')
plt.show()

alice_wc = WordCloud(background_color='white', max_words = 2000, mask=alice_mask, stopwords=stopwords)
alice_wc.generate(alice_novel)
fig = plt.figure(figsize=(14, 18))
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

total_immigration  = df_can['Total'].sum()
max_words = 90
word_string = ''
for country in df_can.index.values:
     # check if country's name is a single-word name
    if country.count(" ") == 0:
        repeat_num_times = int(df_can.loc[country, 'Total'] / total_immigration * max_words)
        word_string = word_string + ((country + ' ') * repeat_num_times)

wordcloud = WordCloud(background_color='white').generate(word_string)
fig = plt.figure(figsize=(14, 18))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Plotting with seaborn
df_can['Continent'].unique()
# countplot
sns.countplot(x='Continent', data=df_can, palette='viridis')

df_can1 = df_can.replace('Latin America and the Caribbean', 'L-America')
df_can1 = df_can1.replace('Northern America', 'N-America')

plt.figure(figsize=(15, 10))
sns.countplot(x='Continent', data=df_can1, palette='viridis')

# Bar plot
plt.figure(figsize=(15, 10))
sns.barplot(x='Continent', y='Total', data=df_can1)

df_Can1 = df_can1.groupby('Continent')['Total'].mean()

# Regression Plot
years = list(map(str, range(1980, 2014)))
df_tot = pd.DataFrame(df_can[years].sum(axis=0))

df_tot.index = map(float, df_tot.index)
df_tot.reset_index(inplace=True)
df_tot.columns = ['Years', 'Total']
df_tot.head()

plt.figure(figsize=(15, 10))
sns.regplot(x='Years', y='Total', data=df_tot, color='green', marker='+', scatter_kws={'s':200})
plt.show()

plt.figure(figsize=(15, 10))
sns.set(font_scale = 1.5)
sns.set_style('whitegrid')  # change background to white background 'ticks'
ax = sns.regplot(x='Years', y='Total', data=df_tot, color='green', marker='+', scatter_kws={'s':200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()

df_dsn = df_can.loc[['Denmark', 'Sweden', 'Norway'], years].T
df_dsn_total = pd.DataFrame(df_dsn.sum(axis=1))
df_dsn_total.index = map(float, df_dsn_total.index)
df_dsn_total.reset_index(inplace=True)
df_dsn_total.columns = ['Years', 'Total']
df_dsn_total.head()

plt.figure(figsize=(15, 10))
sns.set(font_scale = 1.5)
sns.set_style('whitegrid')
ax = sns.regplot(x='Years', y='Total', data=df_dsn_total, color='green', marker='+', scatter_kws={'s':200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration of Denmark, Sweden, Norway Canada from 1980 - 2013')
plt.show()

# Regression plot of total immigiration from India, Pakistan and Afghanistan ton Canada
df_PIA = df_can.loc[['Pakistan', 'India', 'Afghanistan'], years].T
df_PTA_total = pd.DataFrame(df_PIA.sum(axis=1))
df_PTA_total.index = map(float, df_PTA_total.index)
df_PTA_total.reset_index(inplace=True)
df_PTA_total.columns = ['Year', 'Total']

plt.figure(figsize=(15, 10))
sns.set(font_scale = 1.5)
sns.set_style('whitegrid')
ax = sns.regplot(x='Year', y='Total', data=df_PTA_total, marker='+', color='red', scatter_kws={'s':200})
ax.set(xlabel='Year', ylabel='Total Immigiration')
ax.set_title('Total Number of immigiration from India, Pakistana dn afghanistan to Canada')
plt.show()