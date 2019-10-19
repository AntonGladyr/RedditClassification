import numpy as np
from sklearn import preprocessing

class BernoulliNaiveBayes:
    def __init__(self):
        self.le = preprocessing.LabelEncoder()
        self.feature_names = cv.get_feature_names()
        self.marginal_prob = None
        self.conditional_prob = None

    def fit(self, features, targets):
        # encode classes [0, 19]
        targetscv = self.le.fit_transform(targets)

        # step1: marginal probability of each class P(y=k)
        # get number of docs(examples) per class (i.e. doc_count(y=k))
        classes, doc_count = np.unique(targetscv, return_counts=True)

        # get total number of docs(examples)
        total_doc_count = np.sum(doc_count)

        # P(y=k) = doc_count(y=k)/total_docs_count
        self.marginal_prob = np.divide(doc_count, total_doc_count)

        # step2: conditional probability of each class P(x_j=1 | y=k)
        # get word count per class
        class_word_count_list = []

        for class_ in classes:
            # indices where target is specified class
            class_indices = np.where(targetscv == class_)[0]
            # slice of all docs(examples) of specified class
            class_features = features[class_indices, :]

            # number of times each feature(word) appears in specified class (i.e. doc_count(x_j=1, y=k))
            class_word_count_list.append(class_features.sum(axis=0) + 1)
        class_word_count = np.concatenate(class_word_count_list, axis=0)

        # P(x_j=1 | y=k) =  doc_count(x_j=1, y=k)/doc_count(y=k)
        self.conditional_prob = np.divide(class_word_count, doc_count[:, None] + 2)

    def predict(self, features):       
        predictions = []
        # foreach example
        for i in range(np.size(features, axis=0)):
            class_prob = []
            # foreach class
            for class_ in range(np.size(self.marginal_prob)):
                feature_likelihood = 0
                # foreach feature
                for j in range(np.size(features, axis=1)):
                    feature_likelihood += features[i, j]*np.log(self.conditional_prob[class_, j]) + \
                    (1 - features[i, j])*np.log(1 - self.conditional_prob[class_, j])
                class_prob.append(feature_likelihood + np.log(self.marginal_prob[class_]))
            predictions.append(np.argmax(class_prob))
        return self.le.inverse_transform(predictions)

if __name__ == "__main__":
    import nltk
    import pandas as pd

    from data_preprocessor import preprocess_comment_simple
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction import stop_words
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import BernoulliNB
    nltk.download('popular')

    # data pre-processing
    REDDIT_TRAIN_DATA_PATH = 'data_sources/reddit_train.csv'
    df = pd.read_csv(REDDIT_TRAIN_DATA_PATH, quotechar='"', delimiter=',', skipinitialspace=True)

    train_dataset = df.to_numpy()

    features = train_dataset[:, 1]
    targets = train_dataset[:, 2]

    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=4)

    cv = CountVectorizer(binary=True, stop_words=stop_words.ENGLISH_STOP_WORDS, preprocessor=preprocess_comment_simple, 
                            ngram_range=(1, 2), max_features=100)
    x_traincv = cv.fit_transform(x_train)

    BNB = BernoulliNaiveBayes()
    BNB.fit(x_traincv, y_train)

    clf = BernoulliNB(alpha=1.0)
    clf.fit(x_traincv, y_train)

    x_testcv = cv.transform(x_test)

    print('accuracy of Bernoulli Naive Bayes (mine)     = {}\n'.format(np.mean(BNB.predict(x_testcv) == y_test)))
    print('accuracy of Bernoulli Naive Bayes (sklearn)  = {}\n'.format(np.mean(clf.predict(x_testcv) == y_test)))
