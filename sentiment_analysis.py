"""
    perform sentiment analysis of stocktwits data
"""

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
            if len(senses) == 0:
                scores.append((None, 0, 0))
            elif len(senses) == 1:
                scores.append((senses[0].synset.name(), senses[0].pos_score(), senses[0].neg_score()))
            else:
                # another approach is to take average score of all senses

                sense = lesk(tokens, selected_tags[i][0], selected_tags[i][1])
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
            counter is initially 1. whenever a 'not' is encountered do counter = counter*-1.
            Ignore score of 'not'.
        """

        final_score = 0
        counter = 1
        for score in scores:
            if score[0] is not None:
                if score[0].startswith('not'):
                    counter *= -1
                else:
                    if score[1] > score[2]:
                        final_score += counter*score[1]
                    elif score[1] < score[2]:
                        final_score -= counter*score[2]

        return final_score
