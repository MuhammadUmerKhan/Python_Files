from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
from matplotlib import figure 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans, k_means 
from sklearn.datasets import make_blobs 
# %matplotlib inline

np.random.seed(0)
# -------- Input -------
# n_samples: The total number of points equally divided among clusters.
# Value will be: 5000
# centers: The number of centers to generate, or the fixed center locations.
# Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
# cluster_std: The standard deviation of the clusters.
# Value will be: 0.9

# ------ Output ---------
# X: Array of shape [n_samples, n_features]. (Feature Matrix)
# The generated samples.
# y: Array of shape [n_samples]. (Response Vector)
# The integer labels for cluster membership of each sample.
x, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

plt.scatter(x[:, 0], x[:, 1], marker='.');
plt.show();

#  ----------- Setting up K-Means -----
# The KMeans class has many parameters that can be used, 
# but we will be using these three:

# init: Initialization method of the centroids.
# Value will be: "k-means++"
# k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
# n_clusters: The number of clusters to form as well as the number of centroids to generate.
# Value will be: 4 (since we have 4 centers)
# n_init: Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
# Value will be: 12
# Initialize KMeans with these parameters, where the output parameter is called k_means.
k_means = KMeans(init='k-means++', n_clusters=4, n_init=12)
k_means.fit(x)
k_means_labels = k_means.labels_
k_means_labels

k_means_cluster_center = k_means.cluster_centers_
k_means_cluster_center


# -------- Creating a visual plot -------
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1],])), colors):
    
    my_members = (k_means_labels == k)
    cluster_center = k_means_cluster_center[k]
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor = 'k', markersize = 6)
    
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.show()

k_means1 = KMeans(init='k-means++', n_clusters=3, n_init=12)
k_means1.fit(x)
# k_means1_label = k_means1.labels_
# k_means1_label
# k_means_cluster_center1 = k_means1.cluster_centers_
# k_means_cluster_center1

k_means3 = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means3.fit(x)


fig = plt.figure(figsize=(6, 4))
colors1 = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors1):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(x[my_members, 0], x[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
plt.show()

cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()


df = cust_df.drop(columns='Address', axis=1)
df.head()

x = df.values[:, 1:]
x = np.nan_to_num(x)
clus_dataset = StandardScaler().fit_transform(x)
clus_dataset

k_means2 = KMeans(init='k-means++', n_clusters=3, n_init=12)
k_means2.fit(x)
labels = k_means2.labels_

df['Clus_km'] = labels
df.head(); 

df.groupby('Clus_km').mean()
area = np.pi * ( x[:, 1])**2  
plt.scatter(x[:, 0], x[:, 3], s=area, c=labels.astype(float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()


fig2 = plt.figure(1, figsize=(8, 6))
plt.clf()
ax2 = Axes3D(fig2, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax2.set_xlabel('Education')
ax2.set_ylabel('Age')
ax2.set_zlabel('Income')

ax2.scatter(x[:, 1], x[:, 0], x[:, 3], c= labels.astype(float))
plt.show()