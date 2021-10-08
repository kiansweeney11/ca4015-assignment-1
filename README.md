# CA4015 First Assignment
## Kian Sweeney
## Student ID: 18306226

Iowa Gambling Task assignment for CA4015 Advanced Machine Learning. Here we look at a dataset of 617 healthy participants from 10 different studies each with differing amounts of cards that payout and types of groups participating in the studies. Here we cluster on the average choice versus profit margin and the amount of times a subjects most common choice is picked vs profit margin again to analyse subject behaviour.

### Requirements
- jupyter-book v0.11.3
- matplotlib
- numpy
- pandas

Python v3.9.7 was used for this project also.

### Building the Book
To build the book after new code has been merged we must firstly build the html of the book running the command:
```
jupyter-book build ca4015-assignment-1
```

We run this from just outside the directory where the source code comes from. Then inside the directory we run the command to build the pages:
```
ghp-import -n -p -f _build/html
```

### Links
- Here is a link to my jupyter [book](https://kiansweeney11.github.io/ca4015-assignment-1/introduction.html)
- Link to github repo https://github.com/kiansweeney11/ca4015-assignment-1