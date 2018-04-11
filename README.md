# StockIt
<h3>Predicting bullish or bearish sentiment of stocks (Apple, Google, Amazon) using analysis of Stocktwits tweets.</h3>
<p>Stock Market is one of the most dynamic and volatile places which is
highly dominated by general sentiment of traders. It is characterized by
high uncertainty factor and fast-paced changes in trends. Building a
system that can predict the movement of the stock market has posed a
major challenge for researchers since such a system has to deal with a
high noise to signal ratio. Moreover, the movement of the stock market is
mainly determined by the sentiment of traders which is not easily
captured. These traders are influenced by a large number of factors like
monetary reports, news, general opinion about the company as well as
the opinion of the fellow traders. Targeting the events that do have an
effect on the prices of stocks and predicting the exact effect they cause
has largely remained unsolved till date.</p>

<p>In this project, we aim to predict the bullish (increasing) or bearish
(decreasing) nature of stocks of three companies namely Apple, Amazon
and Google by performing sentiment analysis on the stocktwits tweets
for their stocks and gather the prevailing trend for them and using the
result, along with other factors like previous actual sentiment, change in
tweet volume and cash flow to predict their bullish or bearish nature.
We applied various approaches of sentiment analysis ranging from the
lexicon-based approach of SentiWordNet to supervised learning based
approach of RNN, DNN and Naive Bayes classifiers to a labelled corpus
of 1.5 lakh tweets, collected from stocktwits website. Ultimately we
combined the lexicon-based approach with supervised learning
approach for best results. We achieved an accuracy of 73% for
sentiment analysis on our test data.</p>
<p>However, the actual prediction of share market movement is so
uncertain and comprises of a large number of variables that a 50%
accuracy is considered satisfactory while an accuracy greater than 60%
is considered significant in this domain. Using our approach we were
able to achieve an accuracy of 57% in predicting the actual trend of the
market.</p>
