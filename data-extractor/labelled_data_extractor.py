"""
    EXTRACT LABELLED DATA FROM STOCKTWITS API
"""
import csv
import json
import os
import time
from http_request_randomizer.requests.proxy.requestProxy import RequestProxy

FIELDS = ['symbol', 'sentiment', 'message', 'message_id']
SYMBOL = ['AAPL', 'GOOGL', 'FB', 'AMZN']
SYMBOL_DICT = {'AMZN': 3, 'FB': 2, 'GOOGL': 1, 'AAPL': 0}
SYMBOL_DICT_REV = {3: 'AMZN', 2: 'FB', 1: 'GOOGL', 0: 'AAPL'}
FILE_NAME = 'sentiment_db.csv'
token = 0
access_token = ['', 'access_token=32a3552d31b92be5d2a3d282ca3a864f96e95818&',
                'access_token=44ae93a5279092f7804a0ee04753252cbf2ddfee&',
                'access_token=990183ef04060336a46a80aa287f774a9d604f9c&']

file = open(FILE_NAME, 'a', newline='', encoding='utf-8')
# DETERMINE WHERE TO START IF RESUMING SCRIPT
if os.stat(FILE_NAME).st_size == 0:
    # OPEN FILE IN APPEND MODE AND WRITE HEADERS TO FILE
    last_message_id = [None, None, None, None]
    csvfile = csv.DictWriter(file, FIELDS)
    csvfile.writeheader()
else:
    # FIRST EXTRACT LAST MESSAGE ID THEN OPEN FILE IN APPEND MODE WITHOUT WRITING HEADERS
    # FILE MUST HAVE ALL THE STOCKS PRESENT INITIALLY
    file = open(FILE_NAME, 'r', newline='', encoding='utf-8')
    csvfile = csv.DictReader((line.replace('\0', '') for line in file))
    data = list(csvfile)
    data_last = data[-1]
    symbol = data_last['symbol']
    last_message_id = [None, None, None, None]
    index = SYMBOL_DICT[symbol]
    last_message_id[index] = data_last['message_id']
    i = (index - 1) % (len(SYMBOL))
    data_index = -2
    while i is not index:
        while data[data_index]['symbol'] == symbol:
            data_index -= 1
        symbol = data[data_index]['symbol']
        data_last = data[data_index]
        last_message_id[i] = data[data_index]['message_id']
        i = (i - 1) % (len(SYMBOL))

    file.close()
    file = open(FILE_NAME, 'a', newline='', encoding='utf-8')
    csvfile = csv.DictWriter(file, FIELDS)

req_proxy = RequestProxy()

stocktwit_url = "https://api.stocktwits.com/api/2/streams/symbol/" + SYMBOL[token] + ".json?" + access_token[token]
if last_message_id[token] is not None:
    stocktwit_url += "max=" + str(last_message_id[token])

api_hits = 0
while True:
    response = req_proxy.generate_proxied_request(stocktwit_url)

    if response is not None:

        if response.status_code == 429:
            print("###############")
            print("REQUEST IP RATE LIMITED FOR {} seconds !!!".format(
                int(response.headers['X-RateLimit-Reset']) - int(time.time())))

        if not response.status_code == 200:
            token = (token + 1) % (len(access_token))
            stocktwit_url = "https://api.stocktwits.com/api/2/streams/symbol/" + SYMBOL[token] + ".json?" + \
                            access_token[token] + "max=" + str(
                last_message_id[token])

            continue

        api_hits += 1
        response = json.loads(response.text)
        last_message_id[token] = response['cursor']['max']
        null_sentiment_count = 0
        # WRITE DATA TO CSV FILE
        for message in response['messages']:
            # PREPARE OBJECT TO WRITE IN CSV FILE


            temp = message['entities']['sentiment']
            if temp is not None and temp['basic']== 'Bearish':
                obj = {}
                obj['symbol'] = SYMBOL[token]
                obj['message'] = message['body']
                obj['sentiment'] = temp['basic']
                obj['message_id'] = message['id']
                csvfile.writerow(obj)
                file.flush()

        print("API HITS TILL NOW = {}".format(api_hits))




    # ADD MAX ARGUMENT TO GET OLDER MESSAGES
    token = (token + 1) % (len(access_token))
    stocktwit_url = "https://api.stocktwits.com/api/2/streams/symbol/" + SYMBOL[token] + ".json?" + access_token[
        token] + "max=" + str(last_message_id[token])


file.close()
