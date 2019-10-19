# Reddit Classification

This mini-project implements models to analyze text from the website *Reddit* (https://www.reddit.com/), a popular social
 media forum where users post and comment on content in different themed communities, or subreddits. The goal of this
 project is to develop a supervised classification model that can predict what community a comment came from.
 
To run the project the following libraries must be installed:
   * numpy
   * nltk
   * sklearn
   * pandas
   
To run the project using sklearn models use the following command:

```
python3 main.py
```

or

```
./main.py
```

To run the project using our in-house Bernoulli Naive Bayes use the following command:

```
python3 bernoulli_naive_bayes.py
```

NOTE: due to the large number of features used, the main.py script seem to not work properly on Linux platfroms. Running it on Mac should resolve the issue.
