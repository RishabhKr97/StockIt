"""
    implement multinomial naive bayes for sentiment analysis
"""

import os
import load_data
import sentiment_analysis
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import FunctionTransformer

SENTI_CACHE_DICT = {}

class NaiveBayes:

    @classmethod
    def setiwordnet_scorer(cls, messages):
        scores = []
        for x in messages:
            if x not in SENTI_CACHE_DICT:
                SENTI_CACHE_DICT[x] = sentiment_analysis.SentimentAnalysis.get_sentiword_score(x)

            scores.append(SENTI_CACHE_DICT[x])

        return scores

    @classmethod
    def train_classifier(cls):
        dataFrameTraining = load_data.LoadData.get_labelled_data(type='training')

        # make a pipeline for transforms
        tweet_classifier = Pipeline([
            ('feats', FeatureUnion([
                ('text', Pipeline([
                    ('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer())
                ])),
                ('sentiscore', FunctionTransformer(NaiveBayes.setiwordnet_scorer, validate=False))
            ])),
            ('clf', MultinomialNB(alpha=0.01))
        ])

        # grid search for best params
        parameters ={'feats__text__vect__ngram_range': [(1, 1), (1, 2)], 'feats__text__vect__strip_accents': ('unicode', None), 'feats__text__vect__stop_words': ('english', None), 'feats__text__tfidf__use_idf': (True, False), 'clf__alpha': (0.7,1,1e-2,1e-3,0.5,0.3), 'clf__fit_prior': (True, False)}
        gridsearch = GridSearchCV(tweet_classifier, parameters, n_jobs=-1)
        gridsearch = gridsearch.fit(dataFrameTraining['message'].values, dataFrameTraining['sentiment'].values)
        print(gridsearch.best_score_)
        print(gridsearch.best_params_)
        # print(gridsearch.cv_results_)

        # save the trained classifier
        file_location = 'naive_bayes_classifier.pkl'
        try:
            os.remove(file_location)
        except OSError:
            pass
        joblib.dump(gridsearch, file_location)

    @classmethod
    def test_classifier(cls):
        dataFrameTest = load_data.LoadData.get_labelled_data(type='test')

        # load the saved classifier
        file_location = 'naive_bayes_classifier.pkl'
        if os.path.isfile(file_location) is False:
            NaiveBayes.train_classifier()

        tweet_classifier = joblib.load(file_location)
        predicted = tweet_classifier.predict(dataFrameTest['message'].values)
        print(metrics.accuracy_score(dataFrameTest['sentiment'].values, predicted))
        print(metrics.classification_report(dataFrameTest['sentiment'].values, predicted))
        print(metrics.confusion_matrix(dataFrameTest['sentiment'].values, predicted))
