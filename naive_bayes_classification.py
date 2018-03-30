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

class NaiveBayes:

    @classmethod
    def train_classifier(cls):
        dataFrameTraining = load_data.LoadData.get_labelled_data(type='training')

        # make a pipeline for transforms
        # tweet_classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
        tweet_classifier = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))), ('clf', MultinomialNB(alpha=0.01))])
        tweet_classifier.fit(dataFrameTraining['message'].values, dataFrameTraining['sentiment'].values)

        # grid search for best params
        # parameters ={'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3)}
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
        dataFrameTest = load_data.LoadData.get_labelled_data(type='test')

        # load the saved classifier
        file_location = 'naive_bayes_classifier.pkl'
        if os.path.isfile(file_location) is False:
            NaiveBayes.train_classifier()

        tweet_classifier = joblib.load(file_location)
        predicted = tweet_classifier.predict(dataFrameTest['message'].values)
        print(np.mean(predicted == dataFrameTest['sentiment'].values))
