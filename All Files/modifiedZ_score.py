import pandas as pd
import numpy as np

df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\movie_revenues.csv")
df.shape
df.describe()
df.head()

mean = df[['revenue']].mean()
std_dev = df[['revenue']].std()
# MAD (Median absolute deviation)
#  MAD = median(|x - median(x|)


# Modified Z_score
#  0.6745 * x - median(x)/ MAD

df.head()
# Converting to million
df['revenue_Milion'] = df['revenue'].apply(lambda x:x/1000000)
df[['revenue_Milion']]
df.head()

df.revenue_Milion.describe()
_, mean, std_dev, *_ = df.revenue_Milion.describe()
print(mean, std_dev)


# Outlier Detection using z_score
def get_z_score(value, mean, std):
    return (value - mean)/std

df['Z_Score'] = df.revenue_Milion.apply(lambda x: get_z_score(x, mean, std_dev))
df.head()

z_score_outlier = df[df['Z_Score'] > 3]



# Outlier detection using modified Z score
#  MAD = median(|x - median(x|)
#  0.6745 * x - median(x)/ MAD
def MAD(x):
    median = np.median(x)
    absolute = abs(x - median)
    mad = np.median(absolute)
    return mad
get_mad = MAD(df.revenue_Milion)
print(get_mad)

rev_median = np.median(df.revenue_Milion)
rev_median

def mod_z_score(x, median, mad):
    return (0.6745 * (x - median) / mad)
mod_z_score(df.revenue_Milion, rev_median, get_mad)

df.head()
df['mod_z_score'] = df.revenue_Milion.apply(lambda x: mod_z_score(x, rev_median, get_mad))

df.head()

outliers_mod_z_score = df[df['mod_z_score'] > 3.5]
z_score_outlier.shape
outliers_mod_z_score.shape