#!/usr/bin/env python3

import numpy as np
from data_reader import read_data
from data_preprocessor import preprocess_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import decomposition
from scipy.sparse import csr_matrix
from sklearn import svm
from sklearn.decomposition import TruncatedSVD

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

    stopword_list = stop_words.ENGLISH_STOP_WORDS.union(stopwords.words('english'))
    tfidfconverter = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_df=0.75,
                                     stop_words=stopword_list)
    X = tfidfconverter.fit_transform(X).toarray()
    with open('csvfile.txt', 'w') as f:
        for item in tfidfconverter.get_feature_names():
            f.write("%s\n" % item)
    # print(tfidfconverter.get_feature_names())
    print(X.shape)

    X = csr_matrix(X)
    print('Normalized features')
    X = preprocessing.normalize(X)
    print(X.shape)
    # print('Logistic regression, whole dataset')
    # skfold = model_selection.StratifiedKFold(n_splits=5)
    # model = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial')
    # cv_result = model_selection.cross_val_score(model, X, y, cv=skfold, scoring='accuracy')
    # msg = "%s: %f" % ('Logistic regression', cv_result.mean())
    # print(msg)

    print('------------')
    results = []
    names = []
    scoring = 'accuracy'
    models = [('LR', LogisticRegression(random_state=0, solver='saga', multi_class='multinomial')),
              ('DTC', tree.DecisionTreeClassifier())]
    results = []
    names = []
    # #TruncatedSVD
    # print('TruncatedSVD n_components={0}'.format(700))
    # svd = TruncatedSVD(n_components=700)
    # X_truncated = svd.fit_transform(X)
    skfold = model_selection.StratifiedKFold(n_splits=5)
    for name, model in models:
        cv_results = model_selection.cross_val_score(model, X, y, cv=skfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        # X = csr_matrix(X)
        # print(X.shape)

    # models = [('LR', LogisticRegression(random_state=0, solver='saga', multi_class='multinomial')),
    #           ('DTC', tree.DecisionTreeClassifier()),
    #           ('SVM', svm.SVC(kernel='linear'))]
    #
    # # evaluate each model in turn
    # results = []
    # names = []
    # scoring = 'accuracy'
    #
    # for name, model in models:
    #     kfold = model_selection.KFold(n_splits=5)
    #     cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    #     results.append(cv_results)
    #     names.append(name)
    #     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #     print(msg)


if __name__ == '__main__':
    main()
