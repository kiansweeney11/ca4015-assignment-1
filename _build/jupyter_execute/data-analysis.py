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

# In[4]:


from platform import python_version
print(python_version())


# # Initial data exploration - How many per study made profit?

# Here we check the study that used 95 trials of the experiment and see how many of the 15 subjects made profit.

# In[5]:


win95 = pd.read_csv('data/wi_95.csv')
loss95 = pd.read_csv('data/lo_95.csv')
totalloss95 = loss95.sum(axis=1)
totalloss95.head()


# In[6]:


totalwin95 = win95.sum(axis=1)
totalwin95.head()


# In[7]:


margin95 = totalwin95 + totalloss95
margin95.head()


# In[8]:


columns = ['Margin']

margin95df = pd.DataFrame(margin95, columns=columns)
margin95df
ax1 = margin95df.plot.kde(title='Density plot of Profit/Loss margin for 95 people experiments', color='turquoise')
ax1.axvline(x=0, linestyle='--', color='red')


# In[9]:


sum(margin95df.select_dtypes(np.number).gt(0).sum(axis=1))


# Under half (7/15) of the participants made profit in the 95 trial experiment. We now do the same for both the 100 trial and 150 trial experiments

# In[10]:


win100 = pd.read_csv('data/wi_100.csv')
loss100 = pd.read_csv('data/lo_100.csv')
totalloss100 = loss100.sum(axis=1)
totalloss100.head()


# In[11]:


totalwin100 = win100.sum(axis=1)
totalwin100.head()


# In[12]:


margin100 = totalwin100 + totalloss100
margin100.head()


# In[13]:


columns = ['Margin']

margin100df = pd.DataFrame(margin100, columns=columns)
margin100df
ax2 = margin100df.plot.kde(title='Density plot of Profit/Loss margin for 100 people experiments', color='turquoise')
ax2.axvline(x=0, linestyle='--', color='red')


# In[14]:


sum(margin100df.select_dtypes(np.number).gt(0).sum(axis=1))


# Only 41% of participants in the 100 trial experiment made money!

# In[15]:


win150 = pd.read_csv('data/wi_150.csv')
loss150 = pd.read_csv('data/lo_150.csv')
totalloss150 = loss150.sum(axis=1)
totalloss150.head()


# In[16]:


totalwin150 = win150.sum(axis=1)
totalwin150.head()


# In[17]:


margin150 = totalwin150 + totalloss150
margin150.head()


# In[18]:


columns = ['Margin']

margin150df = pd.DataFrame(margin150, columns=columns)
margin150df
ax = margin150df.plot.kde(title='Density plot of Profit/Loss margin for 150 people experiments', color='turquoise')
ax.axvline(x=0, linestyle='--', color='red')


# In[19]:


sum(margin150df.select_dtypes(np.number).gt(0).sum(axis=1))


# There is a much higher % of people making profit in the 150 trial experiment (63%). There is far less people taking part in this experiment than the 100 trial experiment but it may be worth drilling down further into the data to check the differences between the different groups undertaking the experiments here. It might be interesting to see if there is any trends surrounding the age or the gender of the subjects or the number of cards that paid out in the study in question. We will now try to combine the study groups with our "margin" results.

# ### Adding in Study undertakers to participant data

# In[20]:


margin95df['Study'] = index95['Study'].values


# In[21]:


margin150df['Study'] = index150['Study'].values


# In[22]:


margin100df['Study'] = index100['Study'].values


# Here we will investigate the Study groups in the 150 person experiment. We will try to see does one group achieve better results in the task than the other.

# In[23]:


margin150df['Study'].value_counts()


# In[24]:


print("Profit for the 150 trial Wetzels study")
margin150df.loc[margin150df['Study'] == 'Wetzels', 'Margin'].sum()


# In[25]:


print("Profit for the 150 trial Steingroever2011 study")
margin150df.loc[margin150df['Study'] == 'Steingroever2011', 'Margin'].sum()


# ### Assessment of 150 margin results for each study
# This is interesting to note these values. Both studies make a large cumulative profit over the course of the 150 trials undertaken. There is a net profit between the two studies of 36,550 dollars, almost an average profit of 375 dollars per participant. It is interesting to note that the 41 students in "Wetzels" study were exclusively students, while in Steingroever2011's study there was a young average age (19.9) but no specific mention of if the participants were students or not. Although they made money it was almost half of the other group studied. We will now measure this against other datasets.

# In[26]:


margin100df['Study'].value_counts()


# In[27]:


print("Loss for the 100 trial Horstmann study")
margin100df.loc[margin100df['Study'] == 'Horstmann', 'Margin'].sum()


# In[28]:


print("Loss for the 100 trial Wood study")
margin100df.loc[margin100df['Study'] == 'Wood', 'Margin'].sum()


# In[29]:


print("Loss for the 100 trial SteingroverInPrep study")
margin100df.loc[margin100df['Study'] == 'SteingroverInPrep', 'Margin'].sum()


# In[30]:


print("Profit for the 100 trial Maia study")
margin100df.loc[margin100df['Study'] == 'Maia', 'Margin'].sum()


# In[31]:


print("Loss for the 100 trial Worthy study")
margin100df.loc[margin100df['Study'] == 'Worthy', 'Margin'].sum()


# In[32]:


print("Profit for the 100 trial Premkumar study")
margin100df.loc[margin100df['Study'] == 'Premkumar', 'Margin'].sum()


# In[33]:


print("Loss for the 100 trial Kjome study")
margin100df.loc[margin100df['Study'] == 'Kjome', 'Margin'].sum()


# ### Assessment of margin for 100 trial experiments
# The results here are in stark contrast to the 150 trial experiments. Despite less trials there is some significant losses accumalated by participants. Although the study conducted by Wood has a large number of participants in 153, the loss of 119,410 is certainly a major outlier. This equates to an average loss of roughly 780 dollars per person. It is particularly interesting to note [here](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.ak/) in table 1, we see this group has the oldest average age of any group in the study by some distance. However, there is a significant cohort of this group (the first 90) who are between 18-40 years old. The next oldest average age specified actually makes a profit (Premkumar). This potentially suggests age might not be a major factor. Again we see students with strong results in the Maia study as undergraduate students here make a strong profit, similar to the groups in the 150 trial experiments. We now check the 95 trial study as our last part of our margin analysis. 

# In[34]:


margin95df.head()


# In[35]:


print("Profit for the 95 trial Fridberg study")
margin95df.loc[margin95df['Study'] == 'Fridberg', 'Margin'].sum()


# ### Assessment of margin for the 95 trial study
# In this trial as mentioned previously less than half of the participants made profit, but the 15 participant group made 1,250 over the course of the task. This obviously points to some bigger wins mitigating a collection of smaller losses in the group. The group who took part in this study were slightly older than some of the groups who made bigger profits (mean age of 29.6 years old). One thing consistent through this analysis of profit/loss margins has been student groups making more money than older groups. Age appears to be a factor but it may be interesting to look at the flow of each study too. By this, we mean looking at how participants profit/loss fluctuated on each turn and see did a series of wins lead them to change strategy and go for broke for example? Or did a series of losses at the start of the game potentially set the tone for some participants? To do this we will need to combine win and loss dataframes together and graph our results accordingly.

# ## Analysis of participants selection flow

# We will start by looking at the participants in the 95 trial experiment. To merge our dataframes together we will need to change the column names so that they share common names.

# In[36]:


columnnames95 = [f'Trial{num}' for num in range(1,96)]
wins95test = win95
wins95test = wins95test.set_axis(columnnames95, axis=1)


# In[37]:


loss95test = loss95
loss95test = loss95test.set_axis(columnnames95, axis=1)


# In[38]:


df95_added = wins95test.add(loss95test, fill_value=0)


# This study was all part of the Fridberg study so we don't need to worry about comparing other studies here.

# In[39]:


per_trial_95 = df95_added.sum(axis=0)


# In[62]:


per_trial_95.plot(title='Fridberg Study Win/Loss per round', color='green', figsize=(14,8))


# ### Next we move onto the 150 trial experiments and see how these studies flowed over the course of their trials

# In[41]:


columnnames150 = [f'Trial{num}' for num in range(1,151)]
wins150test = win150
wins150test = wins150test.set_axis(columnnames150, axis=1)


# In[42]:


loss150test = loss150
loss150test = loss150test.set_axis(columnnames150, axis=1)


# In[43]:


loss150test.head()


# In[44]:


df150_added = wins150test.add(loss150test, fill_value=0)


# In[45]:


df150_added['Study'] = index150['Study'].values


# In[46]:


teststein = df150_added.loc[df150_added['Study'] == 'Steingroever2011']
del teststein['Study']
per_trial_150_stein = teststein.sum(axis=0)


# In[61]:


per_trial_150_stein.plot(title="Winnings per round for Steingroever2011's study", color="green", figsize=(14,8))


# In[48]:


testWetzels = df150_added.loc[df150_added['Study'] == 'Wetzels']
del testWetzels['Study']
per_trial_150_wetzels = testWetzels.sum(axis=0)


# In[60]:


per_trial_150_wetzels.plot(title="Winnings per round for Wetzels study", color="green", figsize=(14,8))


# After assessing the participants winnings in each study and how their respective win / losses flowed per trial I decided to delve deeper into why these studies appeared more profitable than others. It is easy to point to the varying amounts of cards that pay out on each study and the number of respective trials each study undertook or the number of participants each study had. I felt it would be interesting to build on the win / loss flow per trials mentioned just above and try to pick out patterns accordingly. This would be something like if studies lost a lot of money is this down to a lower average choice of deck or constant changing of choosen cards and see if we could link this to research and studies on the IGT. This study and research could be related to gender based decision making or age demographies and see if these related studies findings hold true to our sample data.

# ### Lastly, we look at our 100 Trial studies selection flow

# In[50]:


columnnames100 = [f'Trial{num}' for num in range(1,101)]
wins100test = win100
wins100test = wins100test.set_axis(columnnames100, axis=1)
loss100test = loss100
loss100test = loss100test.set_axis(columnnames100, axis=1)
df100_added = wins100test.add(loss100test, fill_value=0)
df100_added['Study'] = index100['Study'].values


# In[59]:


testin = df100_added.loc[df100_added['Study'] == 'SteingroverInPrep']
del testin['Study']
per_trial_100_in = testin.sum(axis=0)
per_trial_100_in.plot(title="Winnings per round for SteingroverInPrep's study", color="green", figsize=(14,8))


# In[63]:


testworth = df100_added.loc[df100_added['Study'] == 'Worthy']
del testworth['Study']
per_trial_100_wor = testworth.sum(axis=0)
per_trial_100_wor.plot(title="Winnings per round for Worthy's study",color="green", figsize=(14,8))


# In[64]:


testwood = df100_added.loc[df100_added['Study'] == 'Wood']
del testwood['Study']
per_trial_100_wood = testwood.sum(axis=0)
per_trial_100_wood.plot(title="Winnings per round for Wood's study",color="green", figsize=(14,8))


# In[67]:


testmaia = df100_added.loc[df100_added['Study'] == 'Maia']
del testmaia['Study']
per_trial_100_maia = testmaia.sum(axis=0)
per_trial_100_maia.plot(title="Winnings per round for Maia's study",color="green", figsize=(14,8))


# In[68]:


testhorstmann = df100_added.loc[df100_added['Study'] == 'Horstmann']
del testhorstmann['Study']
per_trial_100_horstmann = testhorstmann.sum(axis=0)
per_trial_100_horstmann.plot(title="Winnings per round for Horstmann's study",color="green", figsize=(14,8))


# In[70]:


testprekumar = df100_added.loc[df100_added['Study'] == 'Premkumar']
del testprekumar['Study']
per_trial_100_prekumar = testprekumar.sum(axis=0)
per_trial_100_prekumar.plot(title="Winnings per round for Prekumar's study",color="green", figsize=(14,8))


# In[71]:


testkjome = df100_added.loc[df100_added['Study'] == 'Kjome']
del testkjome['Study']
per_trial_100_kjome = testkjome.sum(axis=0)
per_trial_100_kjome.plot(title="Winnings per round for Kjome's study",color="green", figsize=(14,8))


# Looking at our selection flow graphs we can see studies that lose significant amounts of money such as the Wood and Worthy study start well for the initial trials, all in good profit but big dips around the 21st trial mark for both mark a downturn in results. This points to a varied selection policy in that even after exploring the decks at the start and seeing that they win the subjects deviate from these and pay the price with losses building up. The more profitable Maia and Premkumar studies are noticeable in that whenever they encounter a large dip they normally recover quickly and don't lose as big for many trials again. This points to a more informed knowledge of the decks and not being unsettled by such losses. We will now use our information gathered here to break down the studies and detect trends in age, gender number of trials / participants and so forth. To do this we will need to process the given data accordingly.
