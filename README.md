# CA4015 First Assignment
## Kian Sweeney
## Student ID: 18306226

Iowa Gambling Task assignment for CA4015 Advanced Machine Learning.

### Requirements
- jupyter-book version 0.11.3
- matplotlib
- numpy
- pandas

### Building the Book
To build the book after new code has been merged we must firstly build the html of the book running the command:

'''
jupyter-book build ca4015-assignment-1
'''

We run this from outside the directory where the source code comes from. Then inside the directory we run the command to build the pages:

'''
ghp-import -n -p -f _build/html
'''

Here is a link to my jupyter [book](https://kiansweeney11.github.io/ca4015-assignment-1/introduction.html)