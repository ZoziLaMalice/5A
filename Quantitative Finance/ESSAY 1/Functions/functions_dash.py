import pandas as pd
from scipy.stats import linregress, norm
import numpy as np
import yfinance as yf

def bbands(price, window_size=10, num_of_std=5):
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std  = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean, upper_band, lower_band


def trading(df, portfolio, buy_, sell_):
    buys = []
    sells = []
    portfolio = portfolio
    evolution = df.loc[df.index[0], 'Log_Returns']
    buy = False

    for i in range(df.index[1], len(df)):
        evolution += df.loc[i, 'Log_Returns']
        if evolution  <= -buy_ and not buy:
            buys.append([df.loc[i, 'Close'], df.loc[i, 'Datetime']])
            portfolio -= df.loc[i, 'Close']
            evolution = 0
            buy = True

        elif evolution <= -sell_ and buy:
            sells.append([df.loc[i, 'Close'], df.loc[i, 'Datetime']])
            portfolio += df.loc[i, 'Close']
            evolution = 0
            buy = False

        elif evolution >= sell_ and buy:
            sells.append([df.loc[i, 'Close'], df.loc[i, 'Datetime']])
            evolution = 0
            portfolio += df.loc[i, 'Close']
            buy = False

    return portfolio, buys, sells


def stats_dfs(stocks, dfs, market):

    stats = pd.DataFrame(
        {
            'Stock': [name for _, name in stocks.items()],
            'Std': [dfs[df].Returns.std() for df in dfs],
            'Annual Std': [dfs[df].Returns.std()* np.sqrt(252) for df in dfs],
            'Mean': [dfs[df].Returns.mean() for df in dfs],
            'Median': [np.median(dfs[df].Returns) for df in dfs],
            'Min': [dfs[df].Returns.min() for df in dfs],
            'Max': [dfs[df].Returns.max() for df in dfs],
            'Kurtosis': [dfs[df].Returns.kurtosis() for df in dfs],
            'Skewness': [dfs[df].Returns.skew() for df in dfs],
            'Alpha': [linregress(dfs[df].Returns, market[market.Date >= dfs[df].Date.min()].Returns).intercept for df in dfs],
            'Beta': [linregress(dfs[df].Returns, market[market.Date >= dfs[df].Date.min()].Returns).slope for df in dfs],
            'VaR 95% HS': [dfs[df].Returns.sort_values(ascending=True).quantile(0.05) for df in dfs],
            'VaR 95% DN': [norm.ppf(1-0.95, dfs[df].Returns.mean(), dfs[df].Returns.std()) for df in dfs],
            'Systemic Risk': [linregress(dfs[df].Returns, market[market.Date >= dfs[df].Date.min()].Returns).slope**2 * market[market.Date >= dfs[df].Date.min()].Returns.var() for df in dfs]
        },
        index=[df for _, df in stocks.items()]
    ).round(6)

    return stats


def stats_market(market):

    stats = pd.DataFrame(
        {
            'Stock': ['S&P500'],
            'Std': [market.Returns.std()],
            'Annual Std': [market.Returns.std()* np.sqrt(252)],
            'Mean': [market.Returns.mean()],
            'Median': [np.median(market.Returns.std())],
            'Min': [market.Returns.min()],
            'Max': [market.Returns.max()],
            'Kurtosis': [market.Returns.kurtosis()],
            'Skewness': [market.Returns.skew()],
            'Alpha': [linregress(market.Returns, market.Returns).intercept],
            'Beta': [linregress(market.Returns, market.Returns).slope],
            'VaR 95% HS': [market.Returns.sort_values(ascending=True).quantile(0.05)],
            'VaR 95% DN': [norm.ppf(1-0.95, market.Returns.mean(), market.Returns.std())],
            'Systemic Risk': [linregress(market.Returns, market.Returns).slope**2 * market.Returns.var()]
        },
        index=[0]
    ).round(6)

    return stats


def create_dfs(stocks):
    dfs = {}
    for stock, _ in stocks.items():
        dfs[stock] = yf.Ticker(stock).history(period="2y")
        dfs[stock]['Returns'] = dfs[stock].Close.pct_change()
        dfs[stock] = dfs[stock].iloc[1:]
        dfs[stock].reset_index(inplace=True)
        dfs[stock]["Color"] = np.where(dfs[stock]['Returns'] < 0, 'red', 'green')

    return dfs


def create_df(stock):
    df = yf.Ticker(stock).history(period="2y")
    df['Returns'] = df.Close.pct_change()
    df = df.iloc[1:]
    df.reset_index(inplace=True)
    df["Color"] = np.where(df['Returns'] < 0, 'red', 'green')

    return df


def stats_df(df, market, the_label):
    stats = pd.DataFrame(
        {
            'Stock': [the_label[0]],
            'Std': [df.Returns.std()],
            'Annual Std': [df.Returns.std()* np.sqrt(252)],
            'Mean': [df.Returns.mean()],
            'Median': [np.median(df.Returns.std())],
            'Min': [df.Returns.min()],
            'Max': [df.Returns.max()],
            'Kurtosis': [df.Returns.kurtosis()],
            'Skewness': [df.Returns.skew()],
            'Alpha': [linregress(df.Returns, market[market.Date >= df.Date.min()].Returns).intercept],
            'Beta': [linregress(df.Returns, market[market.Date >= df.Date.min()].Returns).slope],
            'VaR 95% HS': [df.Returns.sort_values(ascending=True).quantile(0.05)],
            'VaR 95% DN': [norm.ppf(1-0.95, df.Returns.mean(), df.Returns.std())],
            'Systemic Risk': [linregress(df.Returns, market[market.Date >= df.Date.min()].Returns).slope**2 * market[market.Date >= df.Date.min()].Returns.var()]
        },
        index=[0]
    ).round(6)

    return stats