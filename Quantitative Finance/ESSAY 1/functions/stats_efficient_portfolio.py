import pandas as pd
import numpy as np
from scipy.stats import linregress, norm

import yfinance as yf

import plotly.graph_objects as go


def get_data(dfs, date_min, weights_max_sharpe, log=False):
    market_date_min = yf.Ticker('^GSPC').history(period="2y")
    market_date_min['Returns'] = market_date_min.Close.pct_change()
    market_date_min = market_date_min.iloc[1:]
    market_date_min.reset_index(inplace=True)
    market_date_min = market_date_min[market_date_min.Date >= date_min]
    market_date_min.reset_index(inplace=True)

    full_returns = pd.concat([dfs[df].Returns for df in dfs], axis=1)
    full_returns.columns = [df for df in dfs]

    full_close = pd.concat([dfs[df].Close for df in dfs], axis=1)
    full_close.columns = [df for df in dfs]
    log_ret = np.log(full_close/full_close.shift(1))

    portfolio_returns = (full_returns * weights_max_sharpe).sum(axis=1)
    portfolio_log_returns = (log_ret * weights_max_sharpe).sum(axis=1)

    slope, intercept, r, p, std_err = linregress(portfolio_returns, market_date_min.Returns)

    x = np.linspace(np.amin(market_date_min.Returns), np.amax(portfolio_returns))
    y = slope * x + intercept

    if log:
        return portfolio_log_returns
    else:
        return market_date_min, portfolio_returns, x, y

def portfolio_regression(dfs, date_min, weights_max_sharpe, app_color):
    market_date_min, portfolio_returns, x, y = get_data(dfs, date_min, weights_max_sharpe)

    regression = go.Figure(go.Scatter(
        x=market_date_min.Returns,
        y=portfolio_returns,
        mode="markers",
        marker={'size': 5, 'color': '#468de2'},
        name="Returns"
    ))

    regression.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        marker={'color': '#e34029'},
        name="Linear Regression"
    ))
    regression.update_layout(title_text='Single Index Model', plot_bgcolor=app_color["graph_bg"], paper_bgcolor=app_color["graph_bg"],)

    return regression


def plot_monte_carlo(dfs, date_min, weights_max_sharpe, app_color):
    portfolio_log_returns = get_data(dfs, date_min, weights_max_sharpe, True)
    #Setting up drift and random component in relatoin to asset data
    u = portfolio_log_returns.mean()
    var = portfolio_log_returns.var()
    drift = u - (0.5 * var)
    stdev = portfolio_log_returns.std()

    daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(25, 30)))

    #Takes last data point as startpoint point for simulation
    S0 = portfolio_log_returns.iloc[-1]
    price_list = np.zeros_like(daily_returns)

    price_list[0] = S0

    #Applies Monte Carlo simulation in asset
    for t in range(1, 25):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    monte_carlo = go.Figure()

    for i in range(len(price_list)):
        monte_carlo.add_trace(go.Scatter(
            x=[i for i in range(31)],
            y=price_list[i],
        ))

    monte_carlo.update_layout(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
    )

    return monte_carlo