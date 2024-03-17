import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


raw_data_read=pd.read_csv("D:\Marketing Raw Data_seaaBorn.csv")
# print(raw_data_read.head())
x=raw_data_read['Revenue'].values
# print(x)

# Box Plot
# sns.set(rc={'figure.figsize':(12,10)})
# sns.boxplot(x)


sns.set(rc={'figure.figsize':(12,10)})
# sns.boxplot(x='Day_Name', y='Revenue', data=raw_data_read,color='#5531E9')
# sns.boxplot(x='Day_Name', y='Revenue', data=raw_data_read,color='#D83F0F',hue='Promo')

# Swarm Plot

# sns.boxplot(x='Day_Name', y='Revenue', data=raw_data_read,color='#DBEF0C')
# sns.swarmplot(x='Day_Name', y='Revenue', data=raw_data_read,color='#D83F0F')
# plt.show()


# Scatter Plot
# sns.scatterplot(x='Marketing Spend', y='Revenue', data=raw_data_read,hue='Promo',style='Promo')
# sns.scatterplot(x='Marketing Spend', y='Visitors', data=raw_data_read,hue='Promo',style='Promo',size='Revenue',sizes=(20,200))
# plt.show()

# Implot
# sns.lmplot(x='Marketing Spend',y='Revenue',data=raw_data_read,height=10)
# sns.lmplot(x='Marketing Spend',y='Revenue',data=raw_data_read,height=10,hue='Promo',ci=None,markers=['o','s','+'])
# sns.lmplot(x='Marketing Spend',y='Revenue',data=raw_data_read,height=5,col='Promo',ci=None)
# sns.lmplot(x='Marketing Spend',y='Revenue',data=raw_data_read,height=5,col='Promo',ci=None,line_kws={'color':'red'},scatter_kws={'color':'orange'})
sns.lmplot(x='Marketing Spend',y='Revenue',data=raw_data_read,height=5,col='Day_Name',ci=None,line_kws={'color':'red'},scatter_kws={'color':'orange'},col_wrap=3)
plt.show()