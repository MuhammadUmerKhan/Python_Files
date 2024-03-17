from operator import index
from re import A
from statistics import correlation
from tkinter.tix import COLUMN
from scipy import stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
filename = "C:\DATA SCIENCE\python files\clean_df.csv"
df = pd.read_csv(filename)
df.head()

df['peak-rpm'].dtypes
df[['bore','stroke','compression-ratio','horsepower']].corr()

sns.regplot(x='engine-size',y='price',data = df)
plt.ylim(0,)

df[["engine-size", "price"]].corr()
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

df[['highway-L/100km', 'price']].corr()
sns.regplot(x="highway-L/100km", y="price", data=df)
plt.ylim(0,)

df[["stroke","price"]].corr()
sns.regplot(x="price", y="stroke", data=df)
plt.ylim(0,)


sns.boxplot(x="body-style", y="price", data=df)
sns.boxplot(x="engine-location", y="price", data=df)
sns.boxplot(x="drive-wheels", y="price", data=df)

df.describe(include=['object'])
df['drive-wheels'].value_counts()
df['drive-wheels'].value_counts().to_frame()


drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts


drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts


engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)


df['drive-wheels'].unique()
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot

df_avg_test = df[['price','body-style']]
group_df_avg_test = df_avg_test.groupby(['body-style'],as_index = False).mean()
group_df_avg_test

plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()


fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value ) 

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  

pearson_coef, p_value = stats.pearsonr(df['highway-L/100km'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value ) 



filepath="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
df1 = pd.read_csv(filepath)
df1.head()
# Task 1 - Visualize individual feature patterns¶
# Generate regression plots for each of the parameters "CPU_frequency", "Screen_Size_inch" and "Weight_pounds" against "Price". Also, print the value of correlation of each feature with "Price".
df1[["CPU_frequency", "Screen_Size_inch" , "Weight_pounds" , "Price"]].head()
# Write your code below and press Shift+Enter to execute
# CPU_frequency plot
sns.regplot(x="CPU_frequency",y="Price",data=df1)
plt.ylim(0,)
# Write your code below and press Shift+Enter to execute
# Screen_Size_inch plo
sns.regplot(x="Screen_Size_inch",y="Price",data=df1)
plt.ylim(0,)
# Write your code below and press Shift+Enter to execute
# Weight_pounds plot
sns.regplot(x="Weight_pounds",y="Price",data=df1)
plt.ylim(0,)
# Correlation values of the three attributes with Price
for param in ["CPU_frequency", "Screen_Size_inch" , "Weight_pounds"]:
    print(f"Correlation of Price and {param} is ", df[[param,"Price"]].corr())

# Generate Box plots for the different feature that hold categorical values. These features would be "Category", "GPU", "OS", "CPU_core", "RAM_GB", "Storage_GB_SSD"
sns.boxplot(x="Category",y="Price",data=df1)
sns.boxplot(x="GPU",y="Price",data=df1)
sns.boxplot(x="OS",y="Price",data=df1)
sns.boxplot(x="CPU_core",y="Price",data=df1)
sns.boxplot(x="RAM_GB",y="Price",data=df1)
sns.boxplot(x="Storage_GB_SSD",y="Price",data=df1)

# Task 2 - Descriptive Statistical Analysis¶
# Generate the statistical description of all the features being used in the data set. Include "object" data types as well.
df1.describe()
df1.describe(include=['object'])

# Task 3 - GroupBy and Pivot Tables¶
# Group the parameters "GPU", "CPU_core" and "Price" to make a pivot table and visualize this connection using the pcolor plot.
df1_gptest = df1[['GPU','CPU_core','Price']]
df1_gptest_groupby = df1_gptest.groupby(['GPU','CPU_core'],as_index=False).mean()
df1_gptest_groupby_pivot = df1_gptest_groupby.pivot(index='GPU',columns='CPU_core')

# Write your code below and press Shift+Enter to execute
# Create the Plot
fig, ax = plt.subplots()
im = ax.pcolor(df1_gptest_groupby_pivot, cmap='RdBu')

row_labels = df1_gptest_groupby_pivot.columns.levels[1]
col_labels = df1_gptest_groupby_pivot.index

ax.set_xticks(np.arange(df1_gptest_groupby_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(df1_gptest_groupby_pivot.shape[0]) + 0.5, minor=False)

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

for param in ['RAM_GB','CPU_frequency','Storage_GB_SSD','Screen_Size_inch','Weight_pounds','CPU_core','OS','GPU','Category']:
    pearson_coef, p_value = stats.pearsonr(df1[param], df1['Price'])
    print(param)
    print("The Pearson Correlation Coefficient for ",param," is", pearson_coef, " with a P-value of P =", p_value)