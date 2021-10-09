#!/usr/bin/env python
# coding: utf-8

# # 4. K-Means Variation

# Import required libraries again.

# In[1]:


import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# Import our data

# In[65]:


cleaned95 = pd.read_csv('data/cleaned95.csv', index_col='Unnamed: 0')
cleaned100 = pd.read_csv('data/cleaned100.csv', index_col='Unnamed: 0')
cleaned150 = pd.read_csv('data/cleaned150.csv', index_col='Unnamed: 0')
joined = pd.read_csv('data/cleaned_all.csv', index_col='Unnamed: 0')
standard = pd.read_csv('data/standardized_all.csv', index_col='Unnamed: 0')


# In[66]:


standard.head()


# ## Methodology
# We are going to attempt to follow the methods stated in {cite}`LinPP` in his attempt at constructing a privacy preserving clustering technique based on the k-means algorithm. This involves a 2 step process which is as follows:
# 
# 1. Data Protection Phase and
# 2. Data Recovery Phase
# 
# The first phase involving the data protection phase involves 4 key steps. Firstly, we apply the K-means algorithm on our dataset and then we select one of the clusters from the result. In our cluster let's say A, we select the furthest data point away from the centroid of A. We generate the set of noises by the using the following equation:
# 
# $$
#  n_i^u = d^{u} + \alpha \times (distance(c,d)) \tag{1}
# $$
# 
# We then use the following equation:
# 
# $$
#  p_i = |D| \times Rand(s) \tag{2}
# $$
# 
# This is to obtain the position of the noise from eq(1) in dataset D. This leads us on to our data recovery phase. Our first step in the phase is to use eq(2) and obtain p_i for position of noise in D' and commence removals. Then we delete all the noises and the original dataset D can be recovered immediately. The end result should be a dataset that shares cluster information but protects the privacy of the individuals at hand.

# In[69]:


kmeans_margin_joined = KMeans(n_clusters=3).fit(standard[["Margin", "Most Common Choice Picked"]])
centroids_betas_joined = kmeans_margin_joined.cluster_centers_


# In[70]:


plt.figure(figsize=(16,8))
plt.scatter(standard['Margin'], standard['Most Common Choice Picked'], c= kmeans_margin_joined.labels_, cmap = "Set1", alpha=0.5)
plt.scatter(centroids_betas_joined[:, 0], centroids_betas_joined[:, 1], c='blue', marker='x')
plt.title('K-Means cluster for all Subjects - Most Common Choice Picked')
plt.xlabel('Margin')
plt.ylabel('Times Most Common Choice Picked')
plt.show()


# In[105]:


centroids_betas_joined


# We can tell from our above cluster centres that cluster 0 is in red, cluster 1 is the centroid of the orange cluster and cluster 2 is in grey.

# In[110]:


standard['cluster'] = kmeans_margin_joined.labels_.tolist()
standard.head()


# In[139]:


cluster2 = standard[standard.cluster==2]
cluster2.head()


# In[ ]:




