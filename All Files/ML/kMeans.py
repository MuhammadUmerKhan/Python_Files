from sklearn.datasets import load_iris
from cProfile import label
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import k_means, KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("income.csv")
df.head()
df.isnull().sum()
df.info()
df.describe()

df.columns

sns.scatterplot(x=df['Age'], y=df['Income($)'], data=df, markers='.')
plt.legend()

km = KMeans(n_clusters=3)
predicted_cluster = km.fit_predict(df[['Age', 'Income($)']])
df['Cluster'] = predicted_cluster
df.head()

km.cluster_centers_
df1 = df[df['Cluster']==0]
df2 = df[df['Cluster']==1]
df3 = df[df['Cluster']==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()

# Preprocessing using min max scaler
scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
df.head()

plt.scatter(df['Age'], df['Income($)'])

km = KMeans(n_clusters=3)
predicted_cluster = km.fit_predict(df[['Age', 'Income($)']])
df['Cluster'] = predicted_cluster

df.head()


km.cluster_centers_
df1 = df[df['Cluster']==0]
df2 = df[df['Cluster']==1]
df3 = df[df['Cluster']==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()

sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)
    
# the point at which line bend or look like hand elbow is best for value for k
plt.xlabel("K")
plt.ylabel("Sum of Squared Error")
plt.plot(k_rng, sse)

# -----------------
iris = load_iris()
dir(iris)
df_iris = pd.DataFrame(iris.data,columns = iris.feature_names)
df_iris.head()
df_iris.drop(columns=['sepal length (cm)',	'sepal width (cm)'], inplace=True)
df_iris.dtypes

plt.scatter(df_iris['petal length (cm)'], df_iris['petal width (cm)'])

km = KMeans(n_clusters=3)
predicted_cluster_iris = km.fit_predict(df_iris[['petal length (cm)', 'petal width (cm)']])
df_iris['Cluster'] = predicted_cluster_iris

df_iris0 = df_iris[df_iris['Cluster'] == 0]
df_iris1 = df_iris[df_iris['Cluster'] == 1]
df_iris2 = df_iris[df_iris['Cluster'] == 2]
plt.scatter(df_iris0['petal length (cm)'], df_iris0['petal width (cm)'],color='green')
plt.scatter(df_iris1['petal length (cm)'], df_iris1['petal width (cm)'],color='red')
plt.scatter(df_iris2['petal length (cm)'], df_iris2['petal width (cm)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend()


sse_iris = []
k_rng_iris = range(1, 10)
for i in k_rng_iris:
    km_iris = KMeans(n_clusters=i)
    km_iris.fit(df_iris[['petal length (cm)', 'petal width (cm)']])
    sse_iris.append(km_iris.inertia_)
    
plt.xlabel("K")
plt.ylabel("Sum of Squared Error")
plt.plot(k_rng_iris, sse_iris)