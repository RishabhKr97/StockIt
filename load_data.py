"""
    handle preprocessing and loading of data.
"""

import html
import os.path
import pandas as pd
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer

class LoadData:

    @classmethod
    def preprocess_stocktwits_data(cls, file_location, columns=['datetime', 'message']):
        """
            preprocess the data in file location and saves it as a csv file (appending
            '_preprocessed' before '.csv). The preprocessing us in following ways:
            1) extract message and datetime columns.
            2) sort according to datetime in descending order (newest first)
            3) remove links, @ and $ references, extra whitespaces, extra '.', digits, slashes,
            hyphons
            4) decode html entities
            5) convert everything to lower case
        """

        if 'datetime' in columns:
            dataFrame = pd.read_csv(file_location, usecols=columns, parse_dates=['datetime'], infer_datetime_format=True)
            dataFrame.sort_values(by='datetime', ascending=False)
        else:
            dataFrame = pd.read_csv(file_location, usecols=columns)

        dataFrame['message'] = dataFrame['message'].apply(lambda x: html.unescape(x))
        dataFrame['message'] = dataFrame['message'].apply(lambda x: re.sub(r'(www\.|https?://).*?(\s|$)|@.*?(\s|$)|\$.*?(\s|$)|\d|\%|\\|/|-|_', ' ', x))
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
            3) if a sense is present in wordnet then, lemmatize the word and remove stop words else ignore the word
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
                if len(wordnet.synsets(pos[i][0])):
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
    def get_price_data(cls, symbol):
        """
            loads the price data of 'symbol' from data-extractor
            and returns a pandas dataframe with columns [Date(datetime64[ns]), Opening Price(float64), Closing Price(float64), Volume(float64)].
        """

        file_location = 'data-extractor/stock_prices_'+symbol+'.csv'
        dataFrame = pd.read_csv(file_location, usecols=['Date', 'Opening Price', 'Closing Price', 'Volume'], parse_dates=['Date'], infer_datetime_format=True)
        return dataFrame

    @classmethod
    def get_labelled_data(cls, type='complete'):
        """
            get_labelled_data loads the preprocessed labelled data of stocktwits from data-extractor
            and returns a pandas dataframe with columns [sentiment(object), message(object)].
        """

        if type == 'complete':
            file_location = 'data-extractor/labelled_data_complete_preprocessed.csv'
            if os.path.isfile(file_location) is False:
                LoadData.preprocess_stocktwits_data('data-extractor/labelled_data_complete.csv', columns=['sentiment', 'message'])
        elif type == 'training':
            file_location = 'data-extractor/labelled_data_training_preprocessed.csv'
            if os.path.isfile(file_location) is False:
                LoadData.get_training_data()
        elif type == 'test':
            file_location = 'data-extractor/labelled_data_test_preprocessed.csv'
            if os.path.isfile(file_location) is False:
                LoadData.preprocess_stocktwits_data('data-extractor/labelled_data_test.csv', columns=['sentiment', 'message'])

        dataFrame = pd.read_csv(file_location)
        return dataFrame

    @classmethod
    def get_custom_lexicon(cls):
        """
            get custom lexicon of bearish and bullish words respectively
        """

        file_location1 = 'data-extractor/lexicon_bearish_words.txt'
        file_location2 = 'data-extractor/lexicon_bullish_words.txt'
        if os.path.isfile(file_location1) is False or os.path.isfile(file_location2) is False:
            LoadData.labelled_data_lexicon_analysis()

        dataFrameBearish = pd.read_csv(file_location1, header=None, names=['word'])
        dataFrameBullish = pd.read_csv(file_location2, header=None, names=['word'])
        return dataFrameBearish, dataFrameBullish

    @classmethod
    def get_training_data(cls):
        """
            get labelled training data with equal bearish and bullish messages
        """
        try:
            os.remove('data-extractor/labelled_data_training.csv')
        except OSError:
            pass

        dataFrame = LoadData.get_labelled_data(type='complete')
        dataFrameBearish = dataFrame[dataFrame['sentiment']=='Bearish']
        dataFrameBullish = dataFrame[dataFrame['sentiment']=='Bullish']
        dataFrameBearishTraining = dataFrameBearish
        dataFrameBullishTraining = dataFrameBullish[:len(dataFrameBearish)]

        dataFrameTraining = dataFrameBearishTraining.append(dataFrameBullishTraining, ignore_index=True).sample(frac=1).reset_index(drop=True)
        dataFrameTraining.to_csv('data-extractor/labelled_data_training_preprocessed.csv', index=False)

    @classmethod
    def combine_price_and_sentiment(cls, sentimentFrame, priceFrame):
        from datetime import timedelta

        """
            recieve sentimentFrame as (date, sentiment, message) indexed by date and setiment
            and priceFrame as (Date, Opening Price, Closing Price, Volume) and return a combined
            frame as (sentiment_calculated_bullish, sentiment_calculated_bearish,
            sentiment_actual_previous, tweet_volume_change, cash_volume, label)
        """

        dataFrame = pd.DataFrame()
        for date, df in sentimentFrame.groupby(level=0, sort=False):

            price_current = priceFrame[priceFrame['Date'] == date]
            if price_current.empty or date-timedelta(days=1) not in sentimentFrame.index:
                continue
            tweet_minus1 = sentimentFrame.loc[date-timedelta(days=1)]
            days = 1
            price_plus1 = priceFrame[priceFrame['Date'] == date+timedelta(days=days)]
            while price_plus1.empty:
                days += 1
                price_plus1 = priceFrame[priceFrame['Date'] == date+timedelta(days=days)]
            days = 1
            price_minus1 = priceFrame[priceFrame['Date'] == date-timedelta(days=days)]
            while price_minus1.empty:
                days += 1
                price_minus1 = priceFrame[priceFrame['Date'] == date-timedelta(days=days)]

            new_row = {}
            new_row['date'] = date
            new_row['sentiment_calculated_bullish'] = df.loc[(date, 'Bullish')]['message']
            new_row['sentiment_calculated_bearish'] = df.loc[(date, 'Bearish')]['message']
            new_row['sentiment_actual_previous'] = 1 if ((price_minus1.iloc[0]['Closing Price'] - price_minus1.iloc[0]['Opening Price']) >= 0) else -1
            new_row['tweet_volume_change'] = df['message'].sum() - tweet_minus1['message'].sum()
            new_row['cash_volume'] = price_current['Volume'].iloc[0]
            new_row['label'] = 1 if ((price_plus1.iloc[0]['Closing Price'] - price_current.iloc[0]['Closing Price']) >= 0) else -1
            print(new_row)
            dataFrame = dataFrame.append(new_row, ignore_index=True)

        return dataFrame

    @classmethod
    def aggregate_stock_price_data(cls):
        """
            compile stocktwits data for stock prediction analysis in the following form
            (date, sentiment_calculated_bullish, sentiment_calculated_bearish, sentiment_actual_previous, tweet_volume_change, cash_volume, label)

            we have choice to take previous n days sentiment_calculated and using label of next nth day

            returns dataframes for AAPL, AMZN, GOOGL respectively
        """

        if not (os.path.isfile('data-extractor/stocktwits_AAPL_sharedata.csv') and os.path.isfile('data-extractor/stocktwits_AMZN_sharedata.csv') and os.path.isfile('data-extractor/stocktwits_GOOGL_sharedata.csv')):

            from sklearn.externals import joblib
            file_location = 'naive_bayes_classifier.pkl'

            priceAAPL = LoadData.get_price_data('AAPL')
            priceAMZN = LoadData.get_price_data('AMZN')
            priceGOOGL = LoadData.get_price_data('GOOGL')

            sentimented_file = 'data-extractor/stocktwits_AAPL_withsentiment.csv'
            if os.path.isfile(sentimented_file) is False:
                tweet_classifier = joblib.load(file_location)
                dataAAPL = LoadData.get_stocktwits_data('AAPL')
                dataAAPL['sentiment'] = dataAAPL['message'].apply(lambda x: tweet_classifier.predict([x])[0])
                dataAAPL['datetime'] = dataAAPL['datetime'].apply(lambda x: x.date())
                dataAAPL.rename(columns={'datetime':'date'}, inplace=True)
                dataAAPL.to_csv('data-extractor/stocktwits_AAPL_withsentiment.csv', index=False)
            sentimented_file = 'data-extractor/stocktwits_AMZN_withsentiment.csv'
            if os.path.isfile(sentimented_file) is False:
                tweet_classifier = joblib.load(file_location)
                dataAMZN = LoadData.get_stocktwits_data('AMZN')
                dataAMZN['sentiment'] = dataAMZN['message'].apply(lambda x: tweet_classifier.predict([x])[0])
                dataAMZN['datetime'] = dataAMZN['datetime'].apply(lambda x: x.date())
                dataAMZN.rename(columns={'datetime':'date'}, inplace=True)
                dataAMZN.to_csv('data-extractor/stocktwits_AMZN_withsentiment.csv', index=False)
            sentimented_file = 'data-extractor/stocktwits_GOOGL_withsentiment.csv'
            if os.path.isfile(sentimented_file) is False:
                tweet_classifier = joblib.load(file_location)
                dataGOOGL = LoadData.get_stocktwits_data('GOOGL')
                dataGOOGL['sentiment'] = dataGOOGL['message'].apply(lambda x: tweet_classifier.predict([x])[0])
                dataGOOGL['datetime'] = dataGOOGL['datetime'].apply(lambda x: x.date())
                dataGOOGL.rename(columns={'datetime':'date'}, inplace=True)
                dataGOOGL.to_csv('data-extractor/stocktwits_GOOGL_withsentiment.csv', index=False)

            dataAAPL = pd.read_csv('data-extractor/stocktwits_AAPL_withsentiment.csv', parse_dates=['date'], infer_datetime_format=True)
            dataAMZN = pd.read_csv('data-extractor/stocktwits_AMZN_withsentiment.csv', parse_dates=['date'], infer_datetime_format=True)
            dataGOOGL = pd.read_csv('data-extractor/stocktwits_GOOGL_withsentiment.csv', parse_dates=['date'], infer_datetime_format=True)
            dataAAPL = dataAAPL.groupby(['date','sentiment'], sort=False).count()
            dataAMZN = dataAMZN.groupby(['date','sentiment'], sort=False).count()
            dataGOOGL = dataGOOGL.groupby(['date','sentiment'], sort=False).count()
            dataAAPL = LoadData.combine_price_and_sentiment(dataAAPL, priceAAPL)
            dataAMZN = LoadData.combine_price_and_sentiment(dataAMZN, priceAMZN)
            dataGOOGL = LoadData.combine_price_and_sentiment(dataGOOGL, priceGOOGL)
            dataAAPL.to_csv('data-extractor/stocktwits_AAPL_sharedata.csv', index=False)
            dataAMZN.to_csv('data-extractor/stocktwits_AMZN_sharedata.csv', index=False)
            dataGOOGL.to_csv('data-extractor/stocktwits_GOOGL_sharedata.csv', index=False)

        dataAAPL = pd.read_csv('data-extractor/stocktwits_AAPL_sharedata.csv', parse_dates=['date'], infer_datetime_format=True)
        dataAMZN = pd.read_csv('data-extractor/stocktwits_AMZN_sharedata.csv', parse_dates=['date'], infer_datetime_format=True)
        dataGOOGL = pd.read_csv('data-extractor/stocktwits_GOOGL_sharedata.csv', parse_dates=['date'], infer_datetime_format=True)

        return dataAAPL, dataAMZN, dataGOOGL

    @classmethod
    def get_stock_prediction_data(cls, symbol='ALL', type='training'):

        """
            get the training and test data for stock prediction in format
            (sentiment_calculated_bullish, sentiment_calculated_bearish, sentiment_actual_previous,
            tweet_volume_change, cash_volume, label)

            Standardize the data before using.
        """

        file_location = 'data-extractor/stockdata_'+symbol+'_'+type+'.csv'
        if not os.path.isfile(file_location):
            import numpy as np

            dataAAPL, dataAMZN, dataGOOGL = LoadData.aggregate_stock_price_data()
            combined_data = dataAAPL.append([dataAMZN, dataGOOGL], ignore_index=True)
            combined_data.sort_values('date')
            combined_data.drop(columns='date', inplace=True)
            combined_training, combined_test = np.split(combined_data.sample(frac=1), [int(.9*len(combined_data))])
            combined_training.to_csv('data-extractor/stockdata_ALL_training.csv', index=False)
            combined_test.to_csv('data-extractor/stockdata_ALL_test.csv', index=False)

            dataAAPL.sort_values('date')
            dataAAPL.drop(columns='date', inplace=True)
            AAPL_training, AAPL_test = np.split(dataAAPL.sample(frac=1), [int(.9*len(dataAAPL))])
            AAPL_training.to_csv('data-extractor/stockdata_AAPL_training.csv', index=False)
            AAPL_test.to_csv('data-extractor/stockdata_AAPL_test.csv', index=False)

            dataAMZN.sort_values('date')
            dataAMZN.drop(columns='date', inplace=True)
            AMZN_training, AMZN_test = np.split(dataAMZN.sample(frac=1), [int(.9*len(dataAMZN))])
            AMZN_training.to_csv('data-extractor/stockdata_AMZN_training.csv', index=False)
            AMZN_test.to_csv('data-extractor/stockdata_AMZN_test.csv', index=False)

            dataGOOGL.sort_values('date')
            dataGOOGL.drop(columns='date', inplace=True)
            GOOGL_training, GOOGL_test = np.split(dataGOOGL.sample(frac=1), [int(.9*len(dataGOOGL))])
            GOOGL_training.to_csv('data-extractor/stockdata_GOOGL_training.csv', index=False)
            GOOGL_test.to_csv('data-extractor/stockdata_GOOGL_test.csv', index=False)

        data = pd.read_csv(file_location)
        return data
