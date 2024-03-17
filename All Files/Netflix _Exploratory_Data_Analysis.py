import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import express as px
from plotly import graph_objects as go

df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\IPL Auction 2023\iplauction2023.csv")
df.head()

df.columns
df.shape
df.isnull()
df.isnull().sum()
df.dtypes
df.head()
# Replacing Null Values
df['final price (in lacs)'].fillna(0,inplace=True)
df['base price (in lacs)'].fillna(0,inplace=True)
df['franchise'].fillna('No Franchise',inplace=True)

# Detecting Outlier
df.describe()
_, mena, std_dev, *_ = df['final price (in lacs)'].describe()
mena = df['final price (in lacs)'].mean()
std_dev = df['final price (in lacs)'].std()
def getz_score(mean, value, std):
    z_score = (value-mean)/std
    return z_score
getz_score(mena, df['final price (in lacs)'], std_dev)

df['Z_Score'] = df['final price (in lacs)'].apply(lambda x: getz_score(mena, x, std_dev))

df[df['Z_Score'] > 3]

def MAD(x):
    median = np.median(x)
    absolute = abs(x - median)
    mad = np.median(absolute)
    return mad

get_mad = MAD(df['final price (in lacs)'])

def mod_z_score(x, median, mad):
    return (0.6745 * (x - median) / mad)
mod_z_score(df['final price (in lacs)'], np.median(df['final price (in lacs)']), get_mad)
median_value = np.median(df['final price (in lacs)'])
mad_value = MAD(df['final price (in lacs)'])
df['mod_z_score'] = df['final price (in lacs)'].apply(lambda x: mod_z_score(x, median_value, mad_value))
# Outliers
df[df['mod_z_score']>3.5]
df[df['mod_z_score']>3.5].shape

df.shape