from scipy.stats import norm
from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "C:\DATA SCIENCE\Python-git-files\dataset\weight-height\weight-height.csv"
df = pd.read_csv(url)
df.head()

df.shape

df.describe(include='all')

sns.histplot(df.Height, kde=True)

mean = df.Height.mean()
std = df.Height.std()

mean - (3*std)
mean + (3*std)

df[(df.Height < 54.82) | (df.Height > 77.91)]
df_no_outliers = df[(df.Height > 54.82) & (df.Height < 77.91)]
df_no_outliers.shape

# Z-score
# z = (x-mean)/std
df['Z-score'] = (df.Height - df.Height.mean())/df.Height.std()
df.head()

df[df['Z-score'] > 3]
df[df['Z-score'] < -3]
df_no_outliers_1 = df[(df['Z-score']< 3) & (df['Z-score'] > -3)]
df_no_outliers_1.shape
# -----------------------------------------------------------------------#
url_ = "bhp.csv"
df_bhp = pd.read_csv(url_)
df_bhp.head()
df_bhp.shape
df_bhp.describe()
df_bhp.dtypes

plt.hist(df_bhp.price_per_sqft, bins=20, rwidth=0.8)
plt.xlabel('Price per square ft')
plt.ylabel('Count')
plt.show()


plt.hist(df_bhp.price_per_sqft, bins=20, rwidth=0.8)
plt.xlabel('Price per square ft')
plt.ylabel('Count')
plt.yscale('log')
plt.show()
# (1) Remove outliers using percentile technique first. Use [0.001, 0.999] for lower and upper bound percentiles
min_threshold, max_threshold = df_bhp.price_per_sqft.quantile([0.001, 0.999])
print(min_threshold, max_threshold)
# (2) After removing outliers in step 1, you get a new dataframe.
df_bhp2 = df_bhp[(df_bhp.price_per_sqft>min_threshold) & (df_bhp.price_per_sqft<max_threshold)]
df_bhp2.shape
df_bhp.shape

# (3) On step(2) dataframe, use 4 standard deviation to remove outliers
mean_bhp = df_bhp2.price_per_sqft.mean()
std_bhp = df_bhp2.price_per_sqft.std()
mean_bhp, std_bhp

mean_bhp - 4*std_bhp
mean_bhp + 4*std_bhp

df_bhp2[(df_bhp2.price_per_sqft < -9900.42) | (df_bhp2.price_per_sqft > 23227.73)]
df_bhp2_no_outlier = df_bhp2[(df_bhp2.price_per_sqft > -9900.42) & (df_bhp2.price_per_sqft < 23227.73)]
df_bhp2_no_outlier.shape

plt.hist(df_bhp2_no_outlier.price_per_sqft, bins=20, rwidth=0.8, density=True)
plt.xlabel('Height (inches)')
plt.ylabel('Count')

rng = np.arange(-5000, df_bhp2_no_outlier.price_per_sqft.max(), 100)
plt.plot(rng, norm.pdf(rng,df_bhp2_no_outlier.price_per_sqft.mean(),df_bhp2_no_outlier.price_per_sqft.std()))

# (4) Plot histogram for new dataframe that is generated after step (3). Also plot bell curve on same histogram
sns.histplot(df_bhp2_no_outlier.price_per_sqft, kde=True)
# (5) On step(2) dataframe, use zscore of 4 to remove outliers. This is quite similar to step (3) and you will get exact same result
df_bhp2.head()
df_bhp2['Z_score'] = (df_bhp2.price_per_sqft - df_bhp2.price_per_sqft.mean())/df_bhp2.price_per_sqft.std()
df_bhp2.head()

df_bhp2[df_bhp2['Z_score'] > 4]
df_bhp2[df_bhp2['Z_score'] < -4]
outlier_z_score = df_bhp2[(df_bhp2['Z_score'] > 4) | (df_bhp2['Z_score'] < -4)]
outlier_z_score.shape

df_bhp2_no_outlier_z_score = df_bhp2[(df_bhp2['Z_score'] < 4) & (df_bhp2['Z_score'] > -4)]
df_bhp2_no_outlier_z_score.shape