import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

path = "/Users/shankarganesh/files/bitcoin/bitcoin_dataset.csv"

data = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
data["btc_trade_volume"].fillna(method="ffill", inplace=True)
data.sort_index(inplace=True)

x_cols = [col for col in data.columns if col not in ['Date', 'btc_market_price'] if data[col].dtype == 'float64']

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(data[col].values, data.btc_market_price.values)[0, 1])
corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
corr_df = corr_df.sort_values(by='corr_values')

ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12, 40))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
# autolabel(rects)

top_features = corr_df.tail(10)["col_labels"]

t = np.append(top_features, "btc_market_price")

# top_features

feature_data = data[t].copy()

feature_data["lagprice1"] = feature_data["btc_market_price"].shift(1)
feature_data["lagprice"] = feature_data["btc_market_price"].shift(-1)

# del feature_data["btc_market_price"]

feature_data.dropna(axis=0, how='any', inplace=True)

feature_data.tail(5)


# In[8]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} .'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print(mean_absolute_error(test_labels, predictions))
    return accuracy


train = feature_data.head(2300)
test = feature_data.tail(718)

# train, test = train_test_split(data,test_size=0.25)

y_train = train.iloc[:, -1]
x_train = train.iloc[:, :-1]

# x_train.head(5)

y_test = test.iloc[:, -1]
x_test = test.iloc[:, :-1]

regressor = linear_model.LinearRegression()

regressor.fit(x_train, y_train)

print
evaluate(regressor, x_test, y_test)

y_train_pred = regressor.predict(x_train)
y_pred = regressor.predict(x_test)
print
r2_score(y_test, y_pred)

print
"MSE Test", mean_squared_error(y_test, y_pred)
print
"MSE Train", mean_squared_error(y_train, y_train_pred)

# data.info()
plt.plot(x_test.index, y_test, '.',
         x_test.index, y_pred, '-')

plt.show()
