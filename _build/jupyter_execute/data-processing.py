#!/usr/bin/env python
# coding: utf-8

# # 2. Data preparation for Clustering
# For our clustering analysis we need to prepare the data accordingly. For the purpose of this we want to try to cluster on the profit margin of participants and also cluster on the studies participants were a part of also. To do this we need to create appropriate CSV files that we can then use for clustering. 

# In[1]:


import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# In[2]:


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


# ### Creating margin csv files

# In[4]:


columnnames95 = [f'Trial{num}' for num in range(1,96)]
wins95 = win95
wins95 = wins95.set_axis(columnnames95, axis=1)
wins95.head()


# In[5]:


losses95 = loss95
losses95 = losses95.set_axis(columnnames95, axis=1)
losses95.head()


# In[9]:


df95_sum = wins95.add(losses95, fill_value=0)
df95_sum.head()


# In[23]:


profit95 = df95_sum.sum(axis=1)
profit95df = pd.DataFrame(data=profit95)
profit95df.rename(columns={0: 'Margin'}, inplace=True)
profit95df.head()


# In[12]:


choice95.head()


# In[24]:


mode95 = choice95.mode(axis=1)
mode95.rename(columns={0: 'Most Common Choice'}, inplace=True)
mode95.head()


# In[30]:


profit95df['Most Common Choice'] = mode95['Most Common Choice'].values


# In[72]:


profit95df['Study'] = index95['Study'].values
profit95df.head()


# In[73]:


mean95 = choice95.mean(axis=1)
mean95df = pd.DataFrame(data=mean95)
mean95df.rename(columns={0: 'Average Choice'}, inplace=True)
profit95df['Average Choice'] = mean95df['Average Choice'].values
profit95df.head()


# In[74]:


profit95df.to_csv('Data/cleaned95.csv')


# ## We now do this for the 100 trial and 150 trial experiments

# In[34]:


columnnames100 = [f'Trial{num}' for num in range(1,101)]
wins100 = win100
wins100 = wins100.set_axis(columnnames100, axis=1)
wins100.head()


# In[36]:


losses100 = loss100
losses100 = losses100.set_axis(columnnames100, axis=1)
losses100.head()


# In[37]:


df100_sum = wins100.add(losses100, fill_value=0)
df100_sum.head()


# In[38]:


profit100 = df100_sum.sum(axis=1)
profit100df = pd.DataFrame(data=profit100)
profit100df.rename(columns={0: 'Margin'}, inplace=True)
profit100df.head()


# In[39]:


profit100df['Study'] = index100['Study'].values
profit100df


# In[41]:


mode100 = choice100.mode(axis=1)
mode100.rename(columns={0: 'Most Common Choice'}, inplace=True)
profit100df['Most Common Choice'] = mode100['Most Common Choice'].values
profit100df.head()


# In[42]:


profit100df['Most Common Choice'].value_counts()


# In[43]:


profit100df['Most Common Choice'] = profit100df['Most Common Choice'].astype('int64')
profit100df.head()


# In[70]:


mean100 = choice100.mean(axis=1)
mean100df = pd.DataFrame(data=mean100)
mean100df.rename(columns={0: 'Average Choice'}, inplace=True)
profit100df['Average Choice'] = mean100df['Average Choice'].values
profit100df.head()


# In[71]:


profit100df.to_csv('Data/cleaned100.csv')


# Lastly, we take the 150 trial data.

# In[46]:


columnnames150 = [f'Trial{num}' for num in range(1,151)]
wins150 = win150
wins150 = wins150.set_axis(columnnames150, axis=1)
wins150.head()


# In[47]:


losses150 = loss150
losses150 = losses150.set_axis(columnnames150, axis=1)
losses150.head()


# In[48]:


df150_sum = wins150.add(losses150, fill_value=0)
df150_sum.head()


# In[49]:


profit150 = df150_sum.sum(axis=1)
profit150df = pd.DataFrame(data=profit150)
profit150df.rename(columns={0: 'Margin'}, inplace=True)
profit150df.head()


# In[50]:


profit150df['Study'] = index150['Study'].values
profit150df


# In[ ]:


mode150 = choice150.mode(axis=1)
mode150.rename(columns={0: 'Most Common Choice'}, inplace=True)
profit150df['Most Common Choice'] = mode150['Most Common Choice'].values


# In[67]:


mean150 = choice150.mean(axis=1)
mean150df = pd.DataFrame(data=mode150)
mean150df.rename(columns={0: 'Average Choice'}, inplace=True)
profit150df['Average Choice'] = mean150df['Average Choice'].values
profit150df.head()


# In[68]:


profit150df['Most Common Choice'] = profit150df['Most Common Choice'].astype('int64')
profit150df.head()


# In[69]:


profit150df.to_csv('Data/cleaned150.csv')


# In[ ]:




