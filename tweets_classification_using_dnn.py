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


# cont = 0;
# with open("classified_tweets.csv") as csvfile, open("bullish_tweets.csv", 'w', newline='') as b1, open("bearish_tweets.csv", 'w', newline='') as b2:

# 	writer1 = csv.writer(b1)
# 	writer2 = csv.writer(b2, dialect = 'excel')
# 	reader = csv.reader(csvfile)
# 	bullish = []
# 	bearish = []

# 	for row in reader:
# 		if row[0] == "Bullish":
# 			bullish.append(row[1])
# 		elif row[0] == "Bearish":
# 			bearish.append(row[1])

# 		if cont > 20:
# 			break

# 		cont = cont + 1
	
# 	for row in bearish:
# 		b2.write(row)

# 	print(cont)

tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith(('P', 'S')))


# method to remove punctuations from sentences.
def remove_punctuation(text):
    return text.translate(tbl)


with open("classified_tweets.csv") as csvfile:

	reader = csv.reader(csvfile)
	bullish = []
	bearish = []
	count = 0

	for row in reader:
		if row[0] == "Bullish":
			bullish.append(row[1])
		elif row[0] == "Bearish":
			bearish.append(row[1])
		if count > 500 :
			break
		count = count + 1


	print(bearish)

	print(len(bullish), len(bearish))

bearish = bearish[:min(1000, len(bearish))]
bullish = bullish[:min(1000, len(bullish))]

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
words = []
docs = []

# remove tickers and links 
for sentence in bearish:
	
	filtered_words = []
	
	sentence = remove_punctuation(sentence)
	w = nltk.word_tokenize(sentence)	
	w = [stemmer.stem(word.lower()) for word in w]

	for each_word in w:
		if each_word not in stop_words:
			filtered_words.append(each_word)

	print(filtered_words)
	words.extend(w)
	docs.append((w, 0))

# consider 1 is bullish
for sentence in bullish:
	
	filtered_words = []
	
	sentence = remove_punctuation(sentence)
	w = nltk.word_tokenize(sentence)	
	w = [stemmer.stem(word.lower()) for word in w]

	for each_word in w:
		if each_word not in stop_words:
			filtered_words.append(each_word)

	print(filtered_words)
	words.extend(w)
	docs.append((w, 1))


words = sorted(list(set(words)))
print(words)

training = []
output = []


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

	training.append([bow, output])


random.shuffle(training)
training = np.array(training)


train_x = list(training[:, 0])
train_y = list(training[:, 1])

print(train_x)
print(train_y)

tf.reset_default_graph()

net = tflearn.input_data(shape = [None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation = 'softmax')
net = tflearn.regression(net)


model = tflearn.DNN(net, tensorboard_dir = 'tflearn_logs')

model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

print(len(train_x[0]),len(words)) 


def get_features(sentence):
    
    # tokenize the pattern
    sentence = remove_punctuation(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    print(sentence_words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))


def test():

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
		
		print(model.predict([get_features(sentence)]))


test()

























