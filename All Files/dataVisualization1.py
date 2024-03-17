import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv'
df_can = pd.read_csv(url)
df_can.head()
df_can.set_index('Country', inplace=True)
df_can.head()
df_can.index.name = None
df_can.head()
years = list(map(str, range(1980, 2014)))

haiti = df_can.loc['Haiti', years]
haiti.head()
haiti.plot()

haiti.plot(kind='line')
plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')
plt.show() 


haiti.index = haiti.index.map(int) 
haiti.plot(kind='line', figsize=(14, 8))
plt.title('Immigration from Haiti')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')
plt.text(2000, 6000, '2010 Earthquake')
plt.show() 


df_can_IC = df_can.loc[['India','China'], years]
df_can_IC.plot(kind='line')

df_can_IC = df_can_IC.transpose()
df_can_IC.head()

df_can_IC.index =  df_can_IC.index.map(int)
df_can_IC.plot(kind='line', figsize=(14, 8))
plt.title('Imigiration from China and India')
plt.xlabel('Number of immigirants')
plt.ylabel('Years')
plt.show()  


df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)
df_can_top5 = df_can.head()
df_can_top5 = df_can_top5[years].transpose()

df_can_top5.index = df_can_top5.index.map(int)
df_can_top5.plot(kind='line', figsize=(14, 8))
plt.title('Immigirant Trend of Top 5 Countries')
plt.xlabel('Years')
plt.ylabel('Number of Countries')
plt.show()

df_can_lowest = df_can.sort_values(by='Total', ascending=True, axis=0, inplace=False)
df_can_lowest = df_can_lowest.head()
df_can_lowest = df_can_lowest[years].transpose()
df_can_lowest.index = df_can_lowest.index.map(int)
df_can_lowest.plot(kind='line', figsize=(14, 8))
plt.xlabel('Years')
plt.title("Immigirants from loawest 5 countries")
plt.ylabel('Number of Countries')
plt.show()