#!/usr/bin/env python3

import numpy as np
from data_reader import read_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from data_preprocessor import preprocess_comment_simple
from sklearn import model_selection
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction import stop_words
import pandas as pd

REDDIT_TRAIN_PATH = 'data_sources/reddit_train.csv'
REDDIT_TEST_PATH = 'data_sources/reddit_test.csv'
COMMENTS_INDEX = 1
CATEGORIES_INDEX = 2

def main():
    reddit_data = read_data(REDDIT_TRAIN_PATH)
    X, y = reddit_data[:, COMMENTS_INDEX], reddit_data[:, CATEGORIES_INDEX]
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    tfidfconverter = TfidfVectorizer(stop_words=stop_words.ENGLISH_STOP_WORDS, preprocessor=preprocess_comment_simple)
    X = tfidfconverter.fit_transform(X).toarray()
    X = csr_matrix(X)

    skfold = model_selection.StratifiedKFold(n_splits=20)
    print('Ensemble of LR, NB, MNB')
    models = [
        ('LR', LogisticRegression(random_state=0, solver='saga', multi_class='multinomial')),
        ('CNB', ComplementNB(alpha=4.0, class_prior=None, fit_prior=True, norm=False)),
        ('MNB', MultinomialNB(alpha=0.4, fit_prior=True, class_prior=None))
    ]
    ensemble = VotingClassifier(models)
    ensemble.fit(X, y)
    results = model_selection.cross_val_score(ensemble, X, y, cv=skfold, scoring='accuracy', n_jobs=4)
    print(results.mean())
    
    # predicting on test dataset
    reddit_data_test = read_data(REDDIT_TEST_PATH)
    X_test = reddit_data_test[:, COMMENTS_INDEX]
    X_test = tfidfconverter.transform(X_test).toarray()
    X_test = csr_matrix(X_test)
    X_test = preprocessing.normalize(X_test)
    y_pred = ensemble.predict(X_test)
    y_pred = le.inverse_transform(y_pred)
    dataset_pred = reddit_data_test[:, 0:2]
    dataset_pred = np.hstack((dataset_pred, y_pred.reshape(-1, 1)))
    pd.DataFrame(dataset_pred).to_csv("submission.csv", header=['id', 'comments', 'subreddits'])

if __name__ == '__main__':
    main()
