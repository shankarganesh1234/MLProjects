import nltk
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ssl
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# nltk.download()

# initialize sentiment analyzer


def get_sentiment_by_date() -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()

    # read data
    news_df = pd.read_csv('/Users/shankarganesh/files/bitcoin/newsfile.csv', parse_dates=['Date'],
                          encoding='ISO-8859-1')

    # lose the timestamp
    news_df['Date'] = pd.to_datetime(news_df['Date'].dt.date)

    # initialize empty array to store sentiment value - values range between -1 and 1
    sentiment_arr = np.empty(news_df.size, dtype=float)

    # iterate rows of df and calculate sentiment and store in array
    for index, row in news_df.iterrows():
        ss = sia.polarity_scores(row.News)
        sentiment_arr[index] = ss['compound']

    # add sentiment column to original df
    news_df = news_df.assign(sentiment=pd.Series(sentiment_arr))

    # select relevant columns
    cols = ['Date', 'sentiment']
    news_df = news_df[cols]

    # dedupe the data since multiple values present for the same date, using mean()
    news_df = news_df.groupby(['Date'], as_index=False).mean()
    news_df = news_df.rename(columns={'Date': 'sentiment_date'})
    return news_df

