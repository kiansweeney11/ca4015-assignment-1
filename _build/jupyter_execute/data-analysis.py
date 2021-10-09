#!/usr/bin/env python
# coding: utf-8

# # 1. Data Analysis and Experiments
# ## Read in data

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


df = pd.read_csv('data/choice_100.csv')
df.head()


# In[3]:


index95 = pd.read_csv('data/index_95.csv')
index100 = pd.read_csv('data/index_100.csv')
index150 = pd.read_csv('data/index_150.csv')


# Let's just confirm our python version we are using on our scripts

# In[55]:


from platform import python_version
print(python_version())


# # Initial data exploration - How many per study made profit?

# Here we check the study that used 95 trials of the experiment and see how many of the 15 subjects made profit.

# In[4]:


win95 = pd.read_csv('data/wi_95.csv')
loss95 = pd.read_csv('data/lo_95.csv')
totalloss95 = loss95.sum(axis=1)
totalloss95.head()


# In[5]:


totalwin95 = win95.sum(axis=1)
totalwin95.head()


# In[6]:


margin95 = totalwin95 + totalloss95
margin95.head()


# In[7]:


columns = ['Margin']

margin95df = pd.DataFrame(margin95, columns=columns)
margin95df
ax1 = margin95df.plot.kde(title='Density plot of Profit/Loss margin for 95 people experiments', color='turquoise')
ax1.axvline(x=0, linestyle='--', color='red')


# In[8]:


sum(margin95df.select_dtypes(np.number).gt(0).sum(axis=1))


# Under half (7/15) of the participants made profit in the 95 trial experiment. We now do the same for both the 100 trial and 150 trial experiments

# In[9]:


win100 = pd.read_csv('data/wi_100.csv')
loss100 = pd.read_csv('data/lo_100.csv')
totalloss100 = loss100.sum(axis=1)
totalloss100.head()


# In[10]:


totalwin100 = win100.sum(axis=1)
totalwin100.head()


# In[11]:


margin100 = totalwin100 + totalloss100
margin100.head()


# In[12]:


columns = ['Margin']

margin100df = pd.DataFrame(margin100, columns=columns)
margin100df
ax2 = margin100df.plot.kde(title='Density plot of Profit/Loss margin for 100 people experiments', color='turquoise')
ax2.axvline(x=0, linestyle='--', color='red')


# In[13]:


sum(margin100df.select_dtypes(np.number).gt(0).sum(axis=1))


# Only 41% of participants in the 100 trial experiment made money!

# In[14]:


win150 = pd.read_csv('data/wi_150.csv')
loss150 = pd.read_csv('data/lo_150.csv')
totalloss150 = loss150.sum(axis=1)
totalloss150.head()


# In[15]:


totalwin150 = win150.sum(axis=1)
totalwin150.head()


# In[16]:


margin150 = totalwin150 + totalloss150
margin150.head()


# In[17]:


columns = ['Margin']

margin150df = pd.DataFrame(margin150, columns=columns)
margin150df
ax = margin150df.plot.kde(title='Density plot of Profit/Loss margin for 150 people experiments', color='turquoise')
ax.axvline(x=0, linestyle='--', color='red')


# In[18]:


sum(margin150df.select_dtypes(np.number).gt(0).sum(axis=1))


# There is a much higher % of people making profit in the 150 trial experiment (63%). There is far less people taking part in this experiment than the 100 trial experiment but it may be worth drilling down further into the data to check the differences between the different groups undertaking the experiments here. It might be interesting to see if there is any trends surrounding the age profiles surveyed. We will now try to combine the study groups with our "margin" results.

# ### Adding in Study undertakers to participant data

# In[21]:


margin95df['Study'] = index95['Study'].values


# In[24]:


margin150df['Study'] = index150['Study'].values


# In[26]:


margin100df['Study'] = index100['Study'].values


# Here we will investigate the Study groups in the 150 person experiment. We will try to see does one group achieve better results in the task than the other.

# In[28]:


margin150df['Study'].value_counts()


# In[29]:


print("Profit for the 150 trial Wetzels study")
margin150df.loc[margin150df['Study'] == 'Wetzels', 'Margin'].sum()


# In[30]:


print("Profit for the 150 trial Steingroever2011 study")
margin150df.loc[margin150df['Study'] == 'Steingroever2011', 'Margin'].sum()


# ### Assessment of 150 margin results for each study
# This is interesting to note these values. Both studies make a large cumulative profit over the course of the 150 trials undertaken. There is a net profit between the two studies of 36,550 dollars, almost an average profit of 375 dollars per participant. It is interesting to note that the 41 students in "Wetzels" study were exclusively students, while in Steingroever2011's study there was a young average age (19.9) but no specific mention of if the participants were students or not. Although they made money it was almost half of the other group studied. We will measure this against the other datasets to see if age is a factor between the decision making of the groups.

# In[31]:


margin100df['Study'].value_counts()


# In[32]:


print("Loss for the 100 trial Horstmann study")
margin100df.loc[margin100df['Study'] == 'Horstmann', 'Margin'].sum()


# In[33]:


print("Loss for the 100 trial Wood study")
margin100df.loc[margin100df['Study'] == 'Wood', 'Margin'].sum()


# In[34]:


print("Loss for the 100 trial SteingroverInPrep study")
margin100df.loc[margin100df['Study'] == 'SteingroverInPrep', 'Margin'].sum()


# In[35]:


print("Profit for the 100 trial Maia study")
margin100df.loc[margin100df['Study'] == 'Maia', 'Margin'].sum()


# In[36]:


print("Loss for the 100 trial Worthy study")
margin100df.loc[margin100df['Study'] == 'Worthy', 'Margin'].sum()


# In[37]:


print("Profit for the 100 trial Premkumar study")
margin100df.loc[margin100df['Study'] == 'Premkumar', 'Margin'].sum()


# In[38]:


print("Loss for the 100 trial Kjome study")
margin100df.loc[margin100df['Study'] == 'Kjome', 'Margin'].sum()


# ### Assessment of margin for 100 trial experiments
# The results here are in stark contrast to the 150 trial experiments. Despite less trials there is some significant losses accumalated by participants. Although the study conducted by Wood has a large number of participants in 153, the loss of 119,410 is certainly a major outlier. This equates to an average loss of roughly 780 dollars per person. It is particularly interesting to note [here](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.ak/) in table 1, we see this group has the oldest average age of any group in the study by some distance. The next oldest average age specified actually makes a profit (Premkumar). Again we see students with strong results in the Maia study as undergraduate students here make a strong profit, similar to the groups in the 150 trial experiments. We now check the 95 trial study as our last part of our margin analysis. 

# In[39]:


margin95df.head()


# In[40]:


print("Profit for the 95 trial Fridberg study")
margin95df.loc[margin95df['Study'] == 'Fridberg', 'Margin'].sum()


# ### Assessment of margin for the 95 trial study
# In this trial as mentioned previously less than half of the participants made profit, but the 15 participant group made 1,250 over the course of the task. This obviously points to some bigger wins mitigating a collection of smaller losses in the group. The group who took part in this study were slightly older than some of the groups who made bigger profits (mean age of 29.6 years old). One thing consistent through this analysis of profit/loss margins has been student groups making more money than older groups. Age appears to be a factor but it may be interesting to look at the flow of each study too. By this, we mean looking at how participants profit/loss fluctuated on each turn and see did a series of wins lead them to change strategy and go for broke for example? Or did a series of losses at the start of the game potentially set the tone for some participants? To do this we will need to combine win and loss dataframes together and graph our results accordingly.

# ## Analysis of participants selection flow

# We will start by looking at the participants in the 95 trial experiment. To merge our dataframes together we will need to change the column names so that they share common names.

# In[41]:


columnnames95 = [f'Trial{num}' for num in range(1,96)]
wins95test = win95
wins95test = wins95test.set_axis(columnnames95, axis=1)


# In[42]:


loss95test = loss95
loss95test = loss95test.set_axis(columnnames95, axis=1)


# In[43]:


df95_added = wins95test.add(loss95test, fill_value=0)


# This study was all part of the Fridberg study so we don't need to worry about comparing other studies here.

# In[44]:


per_trial_95 = df95_added.sum(axis=0)


# In[45]:


per_trial_95.plot(title='Fridberg Study Win/Loss per round', color='green')


# ### Next we move onto the 150 trial experiments and see how these studies flowed over the course of their trials

# In[46]:


columnnames150 = [f'Trial{num}' for num in range(1,151)]
#columnnames95
wins150test = win150
#wins95test.head()
wins150test = wins150test.set_axis(columnnames150, axis=1)


# In[47]:


loss150test = loss150
loss150test = loss150test.set_axis(columnnames150, axis=1)


# In[48]:


loss150test.head()


# In[49]:


df150_added = wins150test.add(loss150test, fill_value=0)


# In[50]:


df150_added['Study'] = index150['Study'].values


# In[51]:


teststein = df150_added.loc[df150_added['Study'] == 'Steingroever2011']
del teststein['Study']
per_trial_150_stein = teststein.sum(axis=0)


# In[52]:


per_trial_150_stein.plot(title="Winnings per round for Steingroever2011's study", color="green")


# In[53]:


testWetzels = df150_added.loc[df150_added['Study'] == 'Wetzels']
del testWetzels['Study']
per_trial_150_wetzels = testWetzels.sum(axis=0)


# In[54]:


per_trial_150_wetzels.plot(title="Winnings per round for Wetzels study", color="green")


# After assessing the participants winnings in each study and how their respective win / losses flowed per trial I decided to delve deeper into why these studies appeared more profitable than others. It is easy to point to the varying amounts of cards that pay out on each study and the number of respective trials each study undertook or the number of participants each study had. I felt it would be interesting to build on the win / loss flow per trials mentioned just above and try to pick out patterns accordingly. This would be something like if studies lost a lot of money is this down to a lower average choice of deck or constant changing of choosen cards and see if we could link this to research and studies on the IGT. This study and research could be related to gender based decision making or age demographies and see if these related studies findings hold true to our sample data.
