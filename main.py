#!/usr/bin/env python3

import numpy as np
from data_reader import read_data
from data_preprocessor import preprocess_data
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold

REDDIT_TRAIN_PATH = 'data_sources/reddit_train.csv'
REDDIT_TEST_PATH = 'data_sources/reddit_test.csv'
COMMENTS_INDEX = 1
CATEGORIES_INDEX = 2


def split_dataset(features, targets, pct):
    np.random.shuffle(features)
    train_pct_index = int(pct * len(features))
    X_train, X_val = features[:train_pct_index, :], features[train_pct_index:, :]
    Y_train, Y_val = targets[:train_pct_index], targets[train_pct_index:]
    return X_train, X_val, Y_train, Y_val

def main():
    reddit_data = read_data(REDDIT_TRAIN_PATH)
    X, y = reddit_data[:, COMMENTS_INDEX], reddit_data[:, CATEGORIES_INDEX]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    X = preprocess_data(X)
    # examples of comments
    print('2334 => {0}\n'.format(X[2334]))
    print('752 => {0}\n'.format(X[752]))
    print('89 => {0}\n'.format(X[89]))
    print('1545 => {0}\n'.format(X[1545]))
    print('2667 => {0}\n'.format(X[2667]))
    print('12916 => {0}\n'.format(X[12916]))
    print('7294 => {0}\n'.format(X[7294]))

    # ngram_range=(1, 2)
    vectorizer = CountVectorizer(max_features=5000, min_df=2, max_df=0.95, stop_words=stopwords.words('english'))
    X = vectorizer.fit(X)


    # tfidfconverter = TfidfVectorizer(max_features = 2000, min_df=2, max_df=0.95, stop_words=stopwords.words('english'))
    # X = tfidfconverter.fit_transform(X).toarray()

    # print(X.shape)
    print(X.get_feature_names())
    print('train_test_split')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # X_train, X_test, y_train, y_test = split_dataset(X, y, 0.8)

    print('LogisticRegression')
    # classifier = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial')
    classifier = tree.DecisionTreeClassifier()

    # k-folds
    # kf = KFold(n_splits=5)
    # for train_index, test_index in kf.split(X):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     classifier.fit(X_train, y_train)
    #     y_pred = classifier.predict(X_test)
    #     print('accuracy:{0}'.format(np.mean(y_test == y_pred)))

    # # print('Decision trees')
    # # classifier = tree.DecisionTreeClassifier()
    # print('classifier.fit')
    # classifier.fit(X_train, y_train)
    # print('classifier.pred')
    # y_pred = classifier.predict(X_test)
    #
    # print('accuracy')
    # print('accuracy:{0}'.format(np.mean(y_test == y_pred)))
    # print('sklearn accuracy score:{0}'.format(accuracy_score(y_test, y_pred)))


if __name__ == '__main__':
    main()
