import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib import pyplot 
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv"

df = pd.read_csv(filepath)
df.info()

df[['Screen_Size_cm']] = np.round(df[['Screen_Size_cm']],2)
df['Screen_Size_cm']

df.replace('?',np.nan,inplace=True)
df.head()

missing_data = df.isnull()
missing_data.head()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
    
df['Weight_kg']

avg_Weight_kg = df['Weight_kg'].astype('float').mean(axis=0)
print(avg_Weight_kg)

df.dtypes

df[['Screen_Size_cm','Weight_kg']] = df[['Screen_Size_cm','Weight_kg']].astype('float')

df['Screen_Size_cm'] = df['Screen_Size_cm'] / 2.54
df.rename(columns={'Screen_Size_cm':'Screen_Size_inch'},inplace = True)

df['Weight_kg'] = df['Weight_kg'] * 2.205
df.rename(columns={'Weight_kg':'Weight_pounds'},inplace = True)

df['CPU_frequency'] = df['CPU_frequency']/df['CPU_frequency'].max()

bins = np.linspace(min(df['Price']),max(df['Price']),4)
group_names = ['Low','Medium','High']
df['Price-Bined'] = pd.cut(df['Price'],bins,labels=group_names,include_lowest = True)
df[['Price','Price-Bined']].head(60)
df['Price-Bined'].value_counts()

pyplot.bar(group_names,df['Price-Bined'].value_counts())
plt.pyplot.xlabel("Price")
plt.pyplot.ylabel("Count")
plt.pyplot.title("Price Bins")

dummmy_values_screen = pd.get_dummies(df['Screen'])
dummmy_values_screen.rename(columns={'IPS Panel':'Screen-IPS_panel','Full HD':"Screen-Full_HD"},inplace=True)
df = pd.concat([df,dummmy_values_screen],axis=1)
df.drop('Screen',axis=1,inplace=True)

df.to_csv("Loptop-Finalize.csv")
