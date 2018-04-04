"""
    Multilayer perceptron classifier for classifying stocktwits data
"""

import os
import load_data
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import numpy as np

class MLP:

    @classmethod
    def get_scaling_cols(cls, df):
        """
            In test and training files column no.
                cash_volume = 0,
                label = 1,
                sentiment_actual_previous = 2,
                sentiment_calculated_bearish = 3,
                sentiment_calculated_bullish = 4,
                tweet_volume_change = 5
        """

        return df[:,[0,3,4,5]]

    @classmethod
    def get_non_scaling_cols(cls, df):
        """
            In test and training files column no.
                cash_volume = 0,
                label = 1,
                sentiment_actual_previous = 2,
                sentiment_calculated_bearish = 3,
                sentiment_calculated_bullish = 4,
                tweet_volume_change = 5
        """

        return df[:,[2]]

    @classmethod
    def train_nn(cls, symbol='ALL'):
        """
            Either train the neural network for one symbol or on all.
        """

        dataFrameTraining = load_data.LoadData.get_stock_prediction_data(symbol=symbol, type='training')

        Classifier = Pipeline([
            ('feats', FeatureUnion([
                ('scaler', Pipeline([
                    ('col_get_scale', FunctionTransformer(MLP.get_scaling_cols)),
                    ('scale', StandardScaler())
                ])),
                ('non_scaler', FunctionTransformer(MLP.get_non_scaling_cols))
            ])),
            ('mlp', MLPClassifier())
        ])

        # grid search for best params
        parameters = {
            'feats__scaler__scale__with_mean':(True,False),
            'feats__scaler__scale__with_std':(True,False),
            'mlp__hidden_layer_sizes':((2), (3), (2,3), (3,2), (2,2), (3,3), (3,3,3)),
            'mlp__solver':('lbfgs', 'adam'),
            'mlp__alpha': 10.0 ** -np.arange(1, 7)
        }
        gridsearch = GridSearchCV(Classifier, parameters, n_jobs=-1)
        gridsearch = gridsearch.fit(dataFrameTraining, dataFrameTraining['label'].values)

        print(gridsearch.best_score_)
        print(gridsearch.best_params_)

        file_location = 'NN_classifier_'+symbol+'.pkl'
        try:
            os.remove(file_location)
        except OSError:
            pass
        joblib.dump(gridsearch, file_location)

    @classmethod
    def test_nn(cls, symbol='ALL'):

        dataFrameTest = load_data.LoadData.get_stock_prediction_data(symbol=symbol, type='test')

        # load the saved classifier
        file_location = 'NN_classifier_'+symbol+'.pkl'
        if os.path.isfile(file_location) is False:
            MLP.train_nn(symbol=symbol)

        classifier = joblib.load(file_location)
        predicted = classifier.predict(dataFrameTest.values)
        print(metrics.accuracy_score(dataFrameTest['label'].values, predicted))
        print(metrics.classification_report(dataFrameTest['label'].values, predicted))
        print(metrics.confusion_matrix(dataFrameTest['label'].values, predicted))
