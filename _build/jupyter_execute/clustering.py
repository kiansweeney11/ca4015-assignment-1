#!/usr/bin/env python
# coding: utf-8

# # 3. Clustering
# ## After our analysis of the data we must now cluster the data accordingly
# Importing approriate libraries

# In[1]:


import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ### First, we read in our data

# In[42]:


index95 = pd.read_csv('data/index_95.csv')
index100 = pd.read_csv('data/index_100.csv')
index150 = pd.read_csv('data/index_150.csv')
win95 = pd.read_csv('data/wi_95.csv')
win100 = pd.read_csv('data/wi_100.csv')
win150 = pd.read_csv('data/wi_150.csv')
loss95 = pd.read_csv('data/lo_95.csv')
loss100 = pd.read_csv('data/lo_100.csv')
loss150 = pd.read_csv('data/lo_150.csv')
choice95 = pd.read_csv('data/choice_95.csv')
choice100 = pd.read_csv('data/choice_100.csv')
choice150 = pd.read_csv('data/choice_150.csv')


# Cleaned data from processing

# In[43]:


cleaned95 = pd.read_csv('data/cleaned95.csv', index_col='Unnamed: 0')
cleaned100 = pd.read_csv('data/cleaned100.csv', index_col='Unnamed: 0')
cleaned150 = pd.read_csv('data/cleaned150.csv', index_col='Unnamed: 0')


# In[44]:


cleaned95.head()


# Initially, I decided to cluster based on the profit/loss margin for each subject and their most common choice. However, the most common choice would limit the clusters greatly I felt. There would only be 4 possible values (1,2,3 or 4) and this would limit what we could learn from the approriate cluster analysis. This lead me to going back to my data processing and creating the average choice column to add to my data. This was the sum of all the subjects selection divided by the number of trials and I felt this would provide better cluster analysis as a result as there would be far more variety in the range of values.

# In[123]:


kmeans_margin_choice = KMeans(n_clusters=3).fit(cleaned95[["Margin", "Average Choice"]])
centroids_betas = kmeans_margin_choice.cluster_centers_


# In[124]:


plt.scatter(cleaned95['Margin'], cleaned95['Average Choice'], c= kmeans_margin_choice.labels_, cmap = "Set1", alpha=0.5)
plt.scatter(centroids_betas[:, 0], centroids_betas[:, 1], c='blue', marker='x')
plt.title('K-Means cluster for 95 Trial Subjects')
plt.xlabel('Profit/Loss Margin')
plt.ylabel('Average Choice')
plt.show()


# From our cluster analysis here of the 95 trial experiments, it is interesting to note in the Fridberg study the cluster which had the highest average choice also made the most money by a significant distance. K = 3 is certainly the optimal number of clusters here as they are very much pre-defined and easy to distinguish by looking at the scatter plot. We will now try to look at respective studies in a scatter plot and try to compare them with our K-means clusters to see if we can detect trends with regards to profitable studies or unprofitable studies. To do this we may need to create another column in our dataframes. We will have to number the studies accordingly with 0 being the Fridberg study and iterate through them. This way we can get some interesting comparisons between the K-means clustering and actual scatter plots.

# In[158]:


replacements = {
  r'Fridberg': 0
}

cleaned95['StudyNumber'] = cleaned95.Study.replace(replacements, regex=True)
plt.scatter(cleaned95['Margin'], cleaned95['Average Choice'], c=cleaned95['StudyNumber'])
plt.title("Study Clusters")
plt.xlabel('Margin')
plt.ylabel('Average Choice')


# In[170]:


kmeans_margin_choice100 = KMeans(n_clusters=3).fit(cleaned100[["Margin", "Average Choice"]])
centroids_betas = kmeans_margin_choice100.cluster_centers_


# In[171]:


plt.figure(figsize=(16,8))
plt.scatter(cleaned100['Margin'], cleaned100['Average Choice'], c= kmeans_margin_choice100.labels_, cmap = "Set1", alpha=0.5)
plt.scatter(centroids_betas[:, 0], centroids_betas[:, 1], c='blue', marker='x')
plt.title('K-Means cluster for 100 Trial Subjects')
plt.xlabel('Margin')
plt.ylabel('Average Choice')
plt.show()


# This graph is slightly harder to tell how many clusters is ideal with far more subjects to choose from. K=3 seems a reasonable selection once again however.

# In[159]:


replacements100 = {
  r'Horstmann': 1,
  r'Kjome': 2,
  r'Maia': 3,
  r'SteingroverInPrep': 4,
  r'Premkumar': 5,
  r'Wood': 6,
  r'Worthy': 7  
}

cleaned100['StudyNumber'] = cleaned100.Study.replace(replacements100, regex=True)
plt.figure(figsize=(16,8))
plt.scatter(cleaned100['Margin'], cleaned100['Average Choice'], c=cleaned100['StudyNumber'])
plt.title("Study Clusters")
plt.xlabel('Margin')
plt.ylabel('Average Choice')


# In[167]:


kmeans_margin_choice150 = KMeans(n_clusters=3).fit(cleaned150[["Margin", "Average Choice"]])
centroids_betas = kmeans_margin_choice150.cluster_centers_


# In[168]:


plt.figure(figsize=(16,8))
plt.scatter(cleaned150['Margin'], cleaned150['Average Choice'], c= kmeans_margin_choice150.labels_, cmap = "Set1", alpha=0.5)
plt.scatter(centroids_betas[:, 0], centroids_betas[:, 1], c='blue', marker='x')
plt.title('K-Means cluster for 150 Trial Subjects')
plt.xlabel('Margin')
plt.ylabel('Choice')
plt.show()


# In[169]:


replacements150 = {
  r'Steingroever2011': 8,
  r'Wetzels': 9, 
}

cleaned150['StudyNumber'] = cleaned150.Study.replace(replacements150, regex=True)
plt.figure(figsize=(16,8))
plt.scatter(cleaned150['Margin'], cleaned150['Average Choice'], c=cleaned150['StudyNumber'])
plt.title("Study Clusters")
plt.xlabel('Margin')
plt.ylabel('Average Choice')


# In[94]:


distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(cleaned100[['Margin', 'Average Choice', 'Most Common Choice']])
    distortions.append(kmeanModel.inertia_)


# In[95]:


plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k for 100 Trial Dataset')
plt.show()


# From the elbow method graph above, we can see that the optimal number of clusters is K=3 for the 100 trial dataset. 

# In[119]:


sse_values = []
Z = range(1,10)
for k in Z:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(cleaned150[['Margin', 'Average Choice', 'Most Common Choice']])
    sse_values.append(kmeanModel.inertia_)


# In[120]:


plt.figure(figsize=(16,8))
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('The Elbow Method showing the optimal k for 150 Trial Dataset')
plt.plot(N, sse)


# Again, we can see that the optimal number of clusters for our 150 trial dataset is K=3. 

# In[ ]:




