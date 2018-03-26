"""
    perform sentiment analysis of stocktwits data
"""

import load_data
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk
from nltk.stem.wordnet import WordNetLemmatizer

class SentimentAnalysis:

    @classmethod
    def get_sentiword_score(cls, message):
        """
            takes a message and performs following operations:
            1) tokenize
            2) POS tagging
            3) reduce text to nouns, verbs, adjectives, adverbs
            4) lemmatize the words

            for each selected tag, if more than one sense exists, performs word sense disambiguation
            using lesk algorithm and finally returns positivity score, negativity score from
            sentiwordnet lexicon
        """

        tokens = word_tokenize(message)
        pos = pos_tag(tokens)
        lemmatizer = WordNetLemmatizer()
        selected_tags = list()
        scores = list()

        for i in range(len(pos)):
            if pos[i][1].startswith('J'):
                selected_tags.append((lemmatizer.lemmatize(pos[i][0]), 'a'))
            elif pos[i][1].startswith('V'):
                selected_tags.append((lemmatizer.lemmatize(pos[i][0]), 'v'))
            elif pos[i][1].startswith('N'):
                selected_tags.append((lemmatizer.lemmatize(pos[i][0]), 'n'))
            elif pos[i][1].startswith('R'):
                selected_tags.append((lemmatizer.lemmatize(pos[i][0]), 'r'))

        # score list: [(sense name, pos score, neg score)]
        for i in range(len(selected_tags)):
            senses = list(swn.senti_synsets(selected_tags[i][0], selected_tags[i][1]))
            if len(senses) == 1:
                scores.append((senses[0].synset.name(), senses[0].pos_score(), senses[0].neg_score()))
            elif len(senses) > 1:
                sense = lesk(tokens, selected_tags[i][0], selected_tags[i][1])
                if sense is None:
                    # take average score of all original senses
                    pos_score = 0
                    neg_score = 0
                    for i in senses:
                        pos_score += i.pos_score()
                        neg_score += i.neg_score()
                    scores.append((senses[0].synset.name(), pos_score/len(senses), neg_score/len(senses)))
                else:
                    sense = swn.senti_synset(sense.name())
                    scores.append((sense.synset.name(), sense.pos_score(), sense.neg_score()))

        """
            there are a number of ways for aggregating sentiment scores
            1) sum up all scores
            2) average all scores (or only for non zero scores)
            3) (1) or (2) but only for adjectives
            4) if pos score greater than neg score +1 vote else -1 vote
            here we are using the following approach:
            for each calculated score, if pos score is greater than neg score add (counter*score)
            to 'final score' else if neg score is greater than pos score subtract (counter*score).
            counter is initially 1. whenever a negation_word is encountered do counter = counter*-1.
            Ignore score of 'not'.
        """

        # collected from word stat financial dictionary
        negation_words = list(open('data-extractor/lexicon_negation_words.txt').read().split())

        final_score = 0
        counter = 1
        for score in scores:
            if any(score[0].startswith(x) for x in negation_words):
                counter *= -1
            else:
                if score[1] > score[2]:
                    final_score += counter*score[1]
                elif score[1] < score[2]:
                    final_score -= counter*score[2]

        print(final_score)
        return final_score

    @classmethod
    def sentiword_data_analysis(cls, symbol):
        file_location = 'data-extractor/stocktwits_'+symbol+'_sentiwordnet_scored.csv'
        if os.path.isfile(file_location) is False:
            dataFrame = load_data.LoadData.get_stocktwits_data(symbol)
            dataFrame['sentiwordnet_score'] = dataFrame.apply(lambda x: SentimentAnalysis.get_sentiword_score(x['message']), axis = 1)
            dataFrame.to_csv(file_location, index=False)

        dataFrame = pd.read_csv(file_location)
        plt.hist(dataFrame['sentiwordnet_score'], bins = np.arange(-3.5, 4, 0.1))
        plt.show()

    @classmethod
    def labelled_data_sentiwordnet_analysis(cls):
        file_location = 'data-extractor/labelled_data_sentiwordnet_scored.csv'
        if os.path.isfile(file_location) is False:
            dataFrame = load_data.LoadData.get_labelled_data()
            dataFrame['sentiwordnet_score'] = dataFrame.apply(lambda x: SentimentAnalysis.get_sentiword_score(x['message']), axis = 1)
            dataFrame.to_csv(file_location, index=False)

        dataFrame = pd.read_csv(file_location)
        plt.hist(dataFrame[dataFrame['sentiment']=='Bullish']['sentiwordnet_score'], bins = np.arange(-3.5, 4, 0.1), label='Bullish', alpha=0.5)
        plt.hist(dataFrame[dataFrame['sentiment']=='Bearish']['sentiwordnet_score'], bins = np.arange(-3.5, 4, 0.1), label='Bearish', alpha=0.5)
        plt.legend(loc='upper right')
        plt.show()
