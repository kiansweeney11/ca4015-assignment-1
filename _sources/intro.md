# CA4015 Assignment 1

## Introduction

### Iowa Gambling Task¶
The [Iowa Gambling Task](https://www.youtube.com/watch?v=A6SsQIyJMhs)(IGT) is a pyscological card game thought to portray real world decision making and help understand how and why people make decisions. Typically, the particpiants start with a loan of $2000 and are typically given a deck of four cards to pick from. The deck either rewards or punishes the particpiant each time and the end goal of the game is to make more money than what they started with. Some of the cards are more favourable than others, providing a steady winning return over the long term whereas others can be typically destructive but sporadically produce high winnings. 

![](https://media.imotions.com/images/20190528132235/The-Iowa-Gambling-Task-%E2%80%93-No-Dice-All-Science.jpg)

### Dataset
For the purpose of this assignment there is varying trial lengths for each task. Typically, the task is run in 100 trials but in this instance we have a 95 trial, 100 trial and 150 trial experiments. We were given a variety of data for the trials. There were 617 particpiants across all the studies and all were reported as "healthy". We had each subjects choice on their respective trial, their winnings on the respective rounds combined with their losses for each round. We also had an "index" file which told us which study the subject was part of. An overview of our data and other information such as the particpiant demographics can be seen [here](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.ak/).

### Methodology
Initially, we start by breaking down the data and seeing any interesting trends. We try to see which studies produced the highest winnings/losses and how subjects decision making flowed over time (i.e. did they consistently win small or try change strategy by going for broke and another set of cards). We try to combine this with the background demographic information provided to see can we tell anything from the studies. We then cluster the data accordingly by winnings and studys to detect any trends and try tell a story from the data.