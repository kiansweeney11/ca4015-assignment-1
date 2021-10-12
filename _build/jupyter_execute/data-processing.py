#!/usr/bin/env python
# coding: utf-8

# # Data preparation for Clustering
# For our clustering analysis we need to prepare the data accordingly. Following on from our data analysis we want to try to cluster on the profit margin of participants against the number of times the subjects picked their most common deck choice or their average choice. We will then combine this with a scatter plot showing the study each subject was a part of and see what information we can gather from this. We will be looking at age demographies more so but also look to combine this with the amount of cards that pay out in each study and gender breakdowns also. To do this we need to create appropriate CSV files that we can then use for clustering. 

# In[1]:


import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


# In[74]:


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

# In[75]:


columnnames95 = [f'Trial{num}' for num in range(1,96)]
wins95 = win95
wins95 = wins95.set_axis(columnnames95, axis=1)
wins95.head()


# In[76]:


losses95 = loss95
losses95 = losses95.set_axis(columnnames95, axis=1)
losses95.head()


# In[77]:


df95_sum = wins95.add(losses95, fill_value=0)
df95_sum.head()


# In[78]:


profit95 = df95_sum.sum(axis=1)
profit95df = pd.DataFrame(data=profit95)
profit95df.rename(columns={0: 'Margin'}, inplace=True)
profit95df.head()


# In[79]:


choice95.head()


# In[80]:


most_common_count = choice95.apply(pd.Series.value_counts, axis=1)
most_common_count = most_common_count.max(axis=1)
profit95df['Most Common Choice Picked'] = most_common_count
profit95df.head()


# In[81]:


mode95 = choice95.mode(axis=1)
mode95.rename(columns={0: 'Most Common Choice'}, inplace=True)


# In[82]:


profit95df['Most Common Choice'] = mode95['Most Common Choice'].values


# In[83]:


profit95df['Study'] = index95['Study'].values
profit95df.head()


# In[84]:


mean95 = choice95.mean(axis=1)
mean95df = pd.DataFrame(data=mean95)
mean95df.rename(columns={0: 'Average Choice'}, inplace=True)
profit95df['Average Choice'] = mean95df['Average Choice'].values
profit95df.head()


# In[85]:


profit95df.to_csv('Data/cleaned95.csv')


# ## We now do this for the 100 trial and 150 trial experiments

# In[86]:


columnnames100 = [f'Trial{num}' for num in range(1,101)]
wins100 = win100
wins100 = wins100.set_axis(columnnames100, axis=1)


# In[87]:


losses100 = loss100
losses100 = losses100.set_axis(columnnames100, axis=1)


# In[88]:


df100_sum = wins100.add(losses100, fill_value=0)


# In[89]:


profit100 = df100_sum.sum(axis=1)
profit100df = pd.DataFrame(data=profit100)
profit100df.rename(columns={0: 'Margin'}, inplace=True)


# In[90]:


profit100df['Study'] = index100['Study'].values


# In[91]:


mode100 = choice100.mode(axis=1)
mode100.rename(columns={0: 'Most Common Choice'}, inplace=True)
profit100df['Most Common Choice'] = mode100['Most Common Choice'].values


# In[92]:


profit100df['Most Common Choice'].value_counts()


# In[93]:


profit100df['Most Common Choice'] = profit100df['Most Common Choice'].astype('int64')


# In[94]:


most_common_count100 = choice100.apply(pd.Series.value_counts, axis=1)
most_common_count100 = most_common_count100.max(axis=1)
profit100df['Most Common Choice Picked'] = most_common_count100
profit100df.head()


# In[95]:


mean100 = choice100.mean(axis=1)
mean100df = pd.DataFrame(data=mean100)
mean100df.rename(columns={0: 'Average Choice'}, inplace=True)
profit100df['Average Choice'] = mean100df['Average Choice'].values


# In[96]:


profit100df.to_csv('Data/cleaned100.csv')


# Lastly, we take the 150 trial data.

# In[97]:


columnnames150 = [f'Trial{num}' for num in range(1,151)]
wins150 = win150
wins150 = wins150.set_axis(columnnames150, axis=1)


# In[98]:


losses150 = loss150
losses150 = losses150.set_axis(columnnames150, axis=1)


# In[99]:


df150_sum = wins150.add(losses150, fill_value=0)


# In[100]:


profit150 = df150_sum.sum(axis=1)
profit150df = pd.DataFrame(data=profit150)
profit150df.rename(columns={0: 'Margin'}, inplace=True)


# In[101]:


profit150df['Study'] = index150['Study'].values


# In[102]:


mode150 = choice150.mode(axis=1)
mode150.rename(columns={0: 'Most Common Choice'}, inplace=True)
profit150df['Most Common Choice'] = mode150['Most Common Choice'].values


# In[103]:


most_common_count150 = choice150.apply(pd.Series.value_counts, axis=1)
most_common_count150 = most_common_count150.max(axis=1)
most_common_count150 = most_common_count150.astype('int64')
profit150df['Most Common Choice Picked'] = most_common_count150
profit150df.head()


# In[104]:


mean150 = choice150.mean(axis=1)
mean150df = pd.DataFrame(data=mean150)
mean150df.rename(columns={0: 'Average Choice'}, inplace=True)
profit150df['Average Choice'] = mean150df['Average Choice'].values


# In[105]:


profit150df.to_csv('Data/cleaned150.csv')


# In[106]:


merged95_150 = pd.concat([profit95df, profit150df])


# In[114]:


mergedall = pd.concat([merged95_150, profit100df])
mergedall['Most Common Choice'] = mergedall['Most Common Choice'].astype('int64')
mergedall


# For some of our comparisons we may want to draw on in our k-means clusters we need to change our study values from strings to integers. After plotting our k-means algorithm this will allow us to plot a scatter plot comprising of the different studies which will be colour coded based on their numbers here.

# In[117]:


replacements_study = {
  r'Fridberg': 0,  
  r'Horstmann': 1,
  r'Kjome': 2,
  r'Maia': 3,
  r'SteingroverInPrep': 4,
  r'Premkumar': 5,
  r'Wood': 6,
  r'Worthy': 7,
  r'Steingroever2011': 8,
  r'Wetzels': 9,  
}

mergedall['StudyNumber'] = mergedall.Study.replace(replacements_study, regex=True)
mergedall = mergedall.drop(columns=['Study'])
mergedall


# In[118]:


mergedall.to_csv('Data/cleaned_all.csv')


# ## Standardize our Data
# To work best with our k-means algorithm we choose to standardize our values in our joined dataset. This is because the k-means algorithm is a distance based algorithm, calculating the similarity between points based on distance. This gives the data a mean of 0 and standard deviation of 1 and gives common ground between features which would use different values such as our margin and average choice columns.

# In[129]:


scaler = preprocessing.StandardScaler().fit(mergedall)
X_scaled = scaler.transform(mergedall)
X_scaled.std(axis=0)


# In[130]:


standard_all = pd.DataFrame(X_scaled)


# In[131]:


standard_all = standard_all.rename(columns={0:'Margin', 1: 'Most Common Choice Picked', 2: 'Most Common Choice', 3: 'Average Choice',
                                           4: 'StudyNumber'})


# In[132]:


standard_all


# In[134]:


standard_all.to_csv('data/standardized_all.csv')

