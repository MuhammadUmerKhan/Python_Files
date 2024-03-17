import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
read_data = pd.read_csv("D:\Marketing Raw Data_seaaBorn.csv")
# print(read_data.columns)

# Subplots

# fig,axis=plt.subplots(2,2,figsize=(12,7))
# a = read_data['Revenue'].values
# b = read_data['Marketing Spend'].values
# c = read_data['Visitors'].values


# # plot 1
# sns.distplot(a,color = '#05CDB2',ax=axis[0,0])
# # plot 2
# sns.distplot(a,color = '#05CDB2',ax=axis[0,1])
# # plot 3
# sns.distplot(a,color = '#05CDB2',ax=axis[1,0])
# # plot 4
# sns.boxplot(x='Revenue', y='Day_Name', data=read_data,color='#5531E9')
# sns.scatterplot(x='Revenue', y='Day_Name', data=read_data,color='#D83F0F',hue='Promo')
# plt.show()

# Pair Plot
# sns.pairplot(read_data[['Visitors', 'Revenue', 'Marketing Spend','Promo']],plot_kws={'color':'#D83F0F'},height=5,hue='Promo',kind='reg')
# plt.show()

# Join Plot

# sns.jointplot(read_data[['Marketing Spend','Revenue']],color='green',height=10,kind='reg')
# plt.show()




pc=read_data[['Marketing Spend','Revenue']].corr()
# sns.heatmap(pc)
col=['Visitors_','Revenue_']
sns.heatmap(pc,annot=True,annot_kws={'size':10},xticklabels=col,yticklabels=col,cmap='Greens')
plt.show()