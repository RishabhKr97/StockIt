import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sys
import random
import unicodedata
import csv
import re
import numpy as np
import tensorflow as tf
import tflearn


# version 2
bullish = []
bearish = []
words = []
docs = []
training = []
output = []
dataset = []


tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith(('P', 'S')))


# method to remove punctuations from sentences.
def remove_punctuation(text):
    return text.translate(tbl)


stemmer = PorterStemmer()
stop_words = list(stopwords.words("english"))
for i in range(0, len(stop_words)) :
    stop_words[i] = remove_punctuation(stop_words[i])


# remove links, digits and ticker symbols for S&P500 companies 
def clean_and_tokenize(sentence):

    sentence = remove_punctuation(sentence)
    sentence = re.sub("http\S+", "", sentence)
    sentence = re.sub("\d+", "", sentence)

    with open("s&p500.txt", "r") as file:   

        contents = file.readlines()
        for i in range(0, len(contents)):
            contents[i] = contents[i][:-1]
    
    filtered_words = []
    for word in nltk.word_tokenize(sentence):
        word = word.lower()
        if word not in contents :
            if word not in stop_words:
                filtered_words.append(word)

    stemmed_words = set(stemmer.stem(word.lower()) for word in filtered_words)

    return stemmed_words


def get_data(limit):
    global bullish, bearish
    # limit is no. of rows to  be considered
    with open("classified_tweets.csv") as csvfile:  

        reader = csv.reader(csvfile)
        count = 0

        for row in reader:
            if row[0] == "Bullish":
                bullish.append(row[1])
            elif row[0] == "Bearish":
                bearish.append(row[1])
            if count > limit :
                break
            count = count + 1

    max_length = min(len(bearish), len(bullish))
    bearish = bearish[:max_length]
    bullish = bullish[:max_length]


print("stop words :")
print(stop_words)


# one issue remains that while dividing the datasets, bullish and bearish, may not come in equal proportions in training set
def prepare_data():
    global words

    # consider 0 is bearish and 1 is bullish    
    for sentence in bearish:

        stemmed_words = clean_and_tokenize(sentence)        
        words.extend(stemmed_words)
        docs.append((stemmed_words, 0))

    for sentence in bullish:
        
        stemmed_words = clean_and_tokenize(sentence)
        words.extend(stemmed_words)
        docs.append((stemmed_words, 1))


    words = sorted(list(set(words)))    
    print(words)
    print(len(words))

    for doc in docs:

        bow = []
        token_words = doc[0]

        for w in words:
            if w in token_words:
                bow.append(1)
            else:
                bow.append(0)

        output = [0, 0]
        if doc[1] == 1:
            output[1] = 1
        else:
            output[0] = 1

        dataset.append([bow, output])

    random.shuffle(dataset)


def divide_data(ratio):

    train_length = (len(dataset)*ratio)//(1+ratio)

    train_set = dataset[:train_length]
    test_set = dataset[train_length+1:]

    train_x = list((np.array(train_set))[:, 0])
    train_y = list((np.array(train_set))[:, 1])

    print(train_length, len(dataset))
    print(len(test_set), len(train_set))

    return (train_set, test_set, train_x, train_y)


def train_model(train_x, train_y, epoch, batchsize):

    tf.reset_default_graph()

    net = tflearn.input_data(shape = [None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation = 'softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net, tensorboard_dir = 'tflearn_logs')
    model.fit(train_x, train_y, n_epoch=epoch, batch_size=batchsize, show_metric=True)
    
    return model


def get_features(sentence):
    
    stemmed_words = clean_and_tokenize(sentence)
    
    # bag of words
    bow = [0]*len(words)
    # print(stemmed_words)
    for s in stemmed_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))


def determine_accuracy(dataset, model):

    correct = 0
    total = 0
    for row in dataset:
        values = model.predict([np.array(row[0])]).tolist()
        if values[0][1] >=0.5 and row[1][1] == 1:
            correct = correct + 1
        elif values[0][0] >=0.5 and row[1][0] == 1:
            correct = correct + 1
        total = total + 1
    return ((correct/total)*100)


def test_examples(model):

    sent = [""]*7
    sent[0] = "Sell all. "
    sent[1] = "Strongly Bullish "
    sent[2] = "Recesssion is coming soon."
    sent[3] = "what are you talking about ?"    
    sent[4] = "Amazon is high today"
    sent[5] = "Hold on to google. Bound to increase."
    sent[6] = "Be cautious of google. It can fall."

    print(sent)
    for sentence in sent:
        
        print(get_features(sentence).shape)
        print(model.predict([get_features(sentence)]))


def test(model, limit):

    with open("classified_tweets.csv", "r") as csvfile:

        total = 0
        correct = 0
        reader = csv.reader(csvfile)
        for row in reader:

            if len(row) >= 2:
                values = model.predict([get_features(row[1])]).tolist()
                if values[0][1] >=0.5 and row[0] == "Bullish":
                    correct = correct + 1
                elif values[0][0] >=0.5 and row[0] == "Bearish":
                    correct = correct + 1
            total = total + 1

            if total >= limit:
                break

    return ((correct/total)*100)

"""
Note for 20k tweets, trained on 1000 tweets :
accuracy between 60 - 65 on 1000 epochs
and around 65 on 3000 epochs

all the changes actually reduced accuracy 
:(

try to improve by considering only more frequent words


"""

if __name__ == "__main__":


    get_data(3000)
    prepare_data()
    train_set, test_set, train_x, train_y = divide_data(5)
    model = train_model(train_x, train_y, 3000, 8)
    print("accuracy in training set is : ", determine_accuracy(train_set, model))
    print("accuracy in test set is : ", determine_accuracy(test_set, model))
    print("accuracy on first x tweets is : ", test(model, 20000))
    # test_examples()












