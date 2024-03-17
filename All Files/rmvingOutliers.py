import pandas as pd
import numpy as np

url = "C:\DATA SCIENCE\Python-git-files\dataset\AB_NYC_2019.csv\AB_NYC_2019.csv"
df = pd.read_csv(url)
df.head()

df.describe()
missData = df.isnull()
for col in missData.columns.values.tolist():
    print(col)
    print (missData[col].value_counts())
    print("")    
    
df.dtypes
df['price']
min_thresold, max_thresold = df.price.quantile([0.01,0.999])
min_thresold, max_thresold

df2 = df[(df.price>min_thresold)&(df.price<max_thresold)]
df2.shape
df2.sample(5)

df2.describe()
