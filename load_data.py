"""
    get data loads the data of 'symbol' from data-extractor
    and returns a pandas dataframe with columns [message(object), datetime(datetime64[ns])]
    sorted by datetime in descending order (newest first).
"""
import pandas as pd

class LoadData:

    @classmethod
    def get_data(cls, symbol):

        dataFrame = pd.read_csv('data-extractor/stocktwits_'+symbol+'.csv', usecols=['datetime', 'message'], parse_dates=['datetime'], infer_datetime_format=True)
        dataFrame.sort_values(by = 'datetime', ascending=False)

        return dataFrame
