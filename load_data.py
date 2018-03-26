"""
    handle preprocessing and loading of data.
"""

import html
import os.path
import pandas as pd
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

class LoadData:

    @classmethod
    def preprocess_stocktwits_data(cls, file_location, columns=['datetime', 'message']):
        """
            preprocess the data in file location and saves it as a csv file (appending
            '_preprocessed' before '.csv). The preprocessing us in following ways:
            1) extract message and datetime columns.
            2) sort according to datetime in descending order (newest first)
            3) remove links, @ references, extra whitespaces
            4) decode html entities
            5) convert everything to lower case
        """

        if 'datetime' in columns:
            dataFrame = pd.read_csv(file_location, usecols=columns, parse_dates=['datetime'], infer_datetime_format=True)
            dataFrame.sort_values(by='datetime', ascending=False)
        else:
            dataFrame = pd.read_csv(file_location, usecols=columns)

        dataFrame['message'] = dataFrame['message'].apply(lambda x: re.sub('(www\.|https?://).*?(\s|$)|@.*?(\s|$)|\s+', ' ', x))
        dataFrame['message'] = dataFrame['message'].apply(lambda x: re.sub('^\s*|\s*$', '', x))
        dataFrame['message'] = dataFrame['message'].apply(lambda x: html.unescape(x))
        dataFrame['message'] = dataFrame['message'].apply(lambda x: x.lower())

        dataFrame.to_csv(file_location[:-4]+'_preprocessed.csv', index=False)

    # @classmethod
    # def labelled_data_analysis(cls):
    #     """
    #         extract keywords from labelled stocktwits data for improved accuracy of scoring
    #     """

    #     dataFrame = LoadData.get_labelled_data()
    #     bullish_keywords = set()
    #     bearish_keywords = set()
    #     stop_words = set(stopwords.words('english'))
    #     for index, row in dataFrame.iterrows():
    #         tokens = word_tokenize(row['message'])



    @classmethod
    def get_stocktwits_data(cls, symbol):
        """
            get_data loads the preprocessed data of 'symbol' from data-extractor
            and returns a pandas dataframe with columns [message(object), datetime(datetime64[ns])].
        """

        file_location = 'data-extractor/stocktwits_'+symbol+'_preprocessed.csv'
        if os.path.isfile(file_location) is False:
            LoadData.preprocess_stocktwits_data('data-extractor/stocktwits_'+symbol+'.csv')

        dataFrame = pd.read_csv(file_location)
        return dataFrame

    @classmethod
    def get_labelled_data(cls):
        """
            get_labelled_data loads the preprocessed labelled data of stocktwits from data-extractor
            and returns a pandas dataframe with columns [sentiment(object), message(object)].
        """

        file_location = 'data-extractor/labelled_data_preprocessed.csv'
        if os.path.isfile(file_location) is False:
            LoadData.preprocess_stocktwits_data('data-extractor/labelled_data.csv', columns=['sentiment', 'message'])

        dataFrame = pd.read_csv(file_location)
        return dataFrame
