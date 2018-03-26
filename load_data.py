"""
    handle preprocessing and loading of data.
"""

import html
import os.path
import pandas as pd
import re
from nltk import word_tokenize, pos_tag
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
            3) remove links, @ references, extra whitespaces, extra '.', digits, slashes, hyphons
            4) decode html entities
            5) convert everything to lower case
        """

        if 'datetime' in columns:
            dataFrame = pd.read_csv(file_location, usecols=columns, parse_dates=['datetime'], infer_datetime_format=True)
            dataFrame.sort_values(by='datetime', ascending=False)
        else:
            dataFrame = pd.read_csv(file_location, usecols=columns)

        dataFrame['message'] = dataFrame['message'].apply(lambda x: html.unescape(x))
        dataFrame['message'] = dataFrame['message'].apply(lambda x: re.sub(r'(www\.|https?://).*?(\s|$)|@.*?(\s|$)|\d|\%|\\|/|-|_', ' ', x))
        dataFrame['message'] = dataFrame['message'].apply(lambda x: re.sub(r'\.+', '. ', x))
        dataFrame['message'] = dataFrame['message'].apply(lambda x: re.sub(r'\,+', ', ', x))
        dataFrame['message'] = dataFrame['message'].apply(lambda x: re.sub(r'\?+', '? ', x))
        dataFrame['message'] = dataFrame['message'].apply(lambda x: re.sub(r'\s+', ' ', x))
        dataFrame['message'] = dataFrame['message'].apply(lambda x: x.lower())

        dataFrame.to_csv(file_location[:-4]+'_preprocessed.csv', index=False)

    @classmethod
    def labelled_data_lexicon_analysis(cls):
        """
            extract keywords from labelled stocktwits data for improved accuracy in scoring
            for each labelled message do
            1) tokenize the message
            2) perform POS tagging
            3) lemmatize the words
            4) remove stop words
            remove intersections from the two lists before saving
        """

        dataFrame = LoadData.get_labelled_data()
        bullish_keywords = set()
        bearish_keywords = set()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        for index, row in dataFrame.iterrows():
            tokens = word_tokenize(row['message'])
            pos = pos_tag(tokens)
            selected_tags = set()

            for i in range(len(pos)):
                if pos[i][1].startswith('J'):
                    selected_tags.add(lemmatizer.lemmatize(pos[i][0], 'a'))
                elif pos[i][1].startswith('V'):
                    selected_tags.add(lemmatizer.lemmatize(pos[i][0], 'v'))
                elif pos[i][1].startswith('N'):
                    selected_tags.add(lemmatizer.lemmatize(pos[i][0], 'n'))
                elif pos[i][1].startswith('R'):
                    selected_tags.add(lemmatizer.lemmatize(pos[i][0], 'r'))
            selected_tags -= stop_words

            if row['sentiment'] == 'Bullish':
                bullish_keywords = bullish_keywords.union(selected_tags)
            elif row['sentiment'] == 'Bearish':
                bearish_keywords = bearish_keywords.union(selected_tags)

        updated_bullish_keywords = bullish_keywords - bearish_keywords
        updated_bearish_keywords = bearish_keywords - bullish_keywords
        with open('data-extractor/lexicon_bullish_words.txt', 'a') as file:
            for word in updated_bullish_keywords:
                file.write(word+"\n")
        with open('data-extractor/lexicon_bearish_words.txt', 'a') as file:
            for word in updated_bearish_keywords:
                file.write(word+"\n")

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
