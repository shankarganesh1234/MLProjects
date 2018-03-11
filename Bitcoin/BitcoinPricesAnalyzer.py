import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

import BitcoinNewsProcessor as bitcoin_news

# read data
path = "/Users/shankarganesh/files/bitcoin/bitcoin_dataset.csv"
data = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

# get the news dataframe - shape : sentiment_date, sentiment
news_df = bitcoin_news.get_sentiment_by_date()

# merge with left join - left on index, right on sentiment_date
merge_df = pd.DataFrame(pd.merge(data, news_df, how='left', left_index=True, right_on='sentiment_date'))

# sentiment not available for all dates. Replacing missing sentiments with neutral 0
merge_df['sentiment'].fillna(0, inplace=True)

# init imputer for handling missing values
imputer = Imputer()

# prediction target
y = pd.DataFrame(merge_df.btc_market_price)
y = imputer.fit_transform(y)

# predictors - removed sentiment date since it's already present in the index of original data
X = merge_df[merge_df.columns.difference(['btc_market_price', 'sentiment_date'])]
X = imputer.fit_transform(X)

# split into train and test data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


def evaluate_performance(train_X, val_X, train_y, val_y):

    model = linear_model.LinearRegression()
    # model = RandomForestRegressor()
    model.fit(train_X, train_y.ravel())
    print('Accuracy = {}'.format(100 * model.score(val_X, val_y)))
    predictions = model.predict(val_X)
    print('MAE = {}'.format(mean_absolute_error(val_y, predictions)))
    print('Explained Variance Score = {}'.format(explained_variance_score(val_y, predictions)))


# print evaluation metrics
evaluate_performance(train_X, val_X, train_y, val_y)

