from matplotlib import rc
from matplotlib.pyplot import xscale
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\income.csv")
df.head()
df.dtypes


sns.set(rc={'figure.figsize':(11.7,8.27)})
g = sns.barplot(x='income($)',y='count',data=df)
g.set_xticklabels(g.get_xticklabels(), 
                          rotation=45, 
                          horizontalalignment='right');

sns.set(rc={'figure.figsize':(11.7,8.27)})
g = sns.barplot(x='income($)',y='count',data=df)
g.set_xticklabels(g.get_xticklabels(), 
                          rotation=45, 
                          horizontalalignment='right');
g.set(xscale="log");