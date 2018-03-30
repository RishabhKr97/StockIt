"""
    implement multinomial naive bayes for sentiment analysis
"""

import os
import load_data
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

class NaiveBayes:

    @classmethod
    def train_classifier(cls):
        dataFrameTraining = load_data.LoadData.get_labelled_data(type='training')

        # make a pipeline for transforms
        tweet_classifier = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), strip_accents= 'unicode', stop_words='english')), ('tfidf', TfidfTransformer(use_idf=False)), ('clf', MultinomialNB(alpha=0.01))])
        tweet_classifier.fit(dataFrameTraining['message'].values, dataFrameTraining['sentiment'].values)

        # grid search for best params
        # parameters ={'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf__alpha': (0,1,1e-2,1e-3,0.5), 'clf__fit_prior': (True, False)}
        # gridsearch = GridSearchCV(tweet_classifier, parameters, n_jobs=-1)
        # gridsearch = gridsearch.fit(dataFrameTraining['message'].values, dataFrameTraining['sentiment'].values)
        # print(gridsearch.best_score_)
        # print(gridsearch.best_params_)
        # print(gridsearch.cv_results_)

        # save the trained classifier
        file_location = 'naive_bayes_classifier.pkl'
        try:
            os.remove(file_location)
        except OSError:
            pass
        joblib.dump(tweet_classifier, file_location)

    @classmethod
    def test_classifier(cls):
        dataFrameTest = load_data.LoadData.get_labelled_data(type='training')

        # load the saved classifier
        file_location = 'naive_bayes_classifier.pkl'
        if os.path.isfile(file_location) is False:
            NaiveBayes.train_classifier()

        tweet_classifier = joblib.load(file_location)
        predicted = tweet_classifier.predict(dataFrameTest['message'].values)
        # print(np.mean(predicted == dataFrameTest['sentiment'].values))
        print(metrics.accuracy_score(dataFrameTest['sentiment'].values, predicted))
        print(metrics.classification_report(dataFrameTest['sentiment'].values, predicted))
        print(metrics.confusion_matrix(dataFrameTest['sentiment'].values, predicted))

    @classmethod
    def show_most_informative_features(cls):
        dataFrameTraining = load_data.LoadData.get_labelled_data(type='training')
        vect = CountVectorizer(ngram_range=(1,2), strip_accents= 'unicode',stop_words='english')
        x_train_counts = vect.fit_transform(dataFrameTraining['message'].values)
        tf_transformer = TfidfTransformer(use_idf=False)
        x_train_tf = tf_transformer.fit_transform(x_train_counts)
        clf = MultinomialNB().fit(x_train_tf, dataFrameTraining['sentiment'].values)

        """Prints features with the highest coefficient values, per class"""
        feature_names = vect.get_feature_names()
        n=20
        feature_names = vect.get_feature_names()
        coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
