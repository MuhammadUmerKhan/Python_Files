import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

read_data = pd.read_csv("D:\Marketing Raw Data_seaaBorn.csv")
# print(read_data.head())

# plot types
#   1.  Line plot

# sns.lineplot(x='Date',y='Revenue',data=read_data)


# sns.lineplot(x='Date',y='Revenue',data=read_data,hue='Promo')


# sns.lineplot(x='Week_ID',y='Revenue',data=read_data,style='Promo')


# sns.set(rc={'figure.figsize':(12,10)})
# sns.lineplot(x='Week_ID', y='Revenue', hue = 'Promo', style = 'Promo', data = read_data, ci=False,markers=True)


#sns.lineplot(x='Week_ID', y='Revenue', hue = 'Promo', style = 'Day_Name', data = read_data, ci=False,markers=True)


#sns.lineplot(x='Year', y='Revenue', hue = 'Promo', style = 'Day_Name', data = read_data, ci=False,markers=True)


# Bar Plot
#sns.barplot(x='Month_ID', y='Revenue', data = read_data)
#read_data[['Month_ID','Revenue']].groupby('Month_ID').agg({'Revenue':'sum'})
#read_data[['Month_ID','Revenue']].groupby('Month_ID').agg({'Revenue':'mean'})


#sns.barplot(x='Month_ID', y='Revenue', data = read_data,ci=None)


#sns.barplot(x='Month_ID', y='Revenue', data = read_data,ci=None,hue='Promo')


#sns.barplot(x='Revenue', y='Month_ID', data = read_data,ci=None,hue='Promo',orient='h')


#sns.barplot(x='Revenue', y='Year', data = read_data,ci=None,hue='Promo',orient='h',color='#05CDB2')



# Histogram
# x = read_data['Revenue'].values
# sns.distplot(x,color = '#05CDB2')


# mean = read_data['Revenue'].mean()
# sns.distplot(x,color = '#05CDB2')
# plt.axvline(mean,0,1,color='red')

x = read_data['Visitors'].values
mean = read_data['Visitors'].mean()
sns.set(rc={'figure.figsize':(12,10)})
sns.distplot(x,color = '#05CDB2')
plt.axvline(mean,0,1,color='black')


plt.show()