import pandas as pd
import numpy as np
from scipy.stats import norm

import plotly.graph_objects as go

def get_returns(dfs):
    full_returns = pd.concat([dfs[df].Returns for df in dfs], axis=1)
    full_returns.columns = [df for df in dfs]

    full_close = pd.concat([dfs[df].Close for df in dfs], axis=1)
    full_close.columns = [df for df in dfs]
    log_ret = np.log(full_close/full_close.shift(1))

    return full_returns, full_close, log_ret


def get_equal_weighted(dfs, stocks):
    full_returns, full_close, log_ret = get_returns(dfs)

    np.random.seed(42)
    all_weights = np.zeros((1, len(full_close.columns)))
    ret_arr = np.zeros(1)
    vol_arr = np.zeros(1)
    sharpe_arr = np.zeros(1)


    # Weights
    weights = np.array(np.random.random(len(stocks))) # Problem here
    weights = weights/np.sum(weights)

    # Expected return
    ret_arr = np.sum( (log_ret.mean() * weights * 252))

    # Expected volatility
    vol_arr = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))

    data = pd.DataFrame(
        {
            'Stock': [stock for _, stock in stocks.items()],
            'Weight': [100/len(stocks)] * len(stocks)
        })

    return ret_arr, vol_arr, data


def efficient_portfolio(dfs, stocks, app_color):
    full_returns, full_close, log_ret = get_returns(dfs)

    np.random.seed(42)
    num_ports = 6000
    all_weights = np.zeros((num_ports, len(full_close.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    try:
        for x in range(num_ports):
            # Weights
            weights = np.array(np.random.random(len(stocks))) # Problem here
            weights = weights/np.sum(weights)

            # Save weights
            all_weights[x,:] = weights

            # Expected return
            ret_arr[x] = np.sum( (log_ret.mean() * weights * 252))

            # Expected volatility
            vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))

            # Sharpe Ratio
            sharpe_arr[x] = ret_arr[x]/vol_arr[x]
    except ValueError:
        pass

    max_sr_ret = ret_arr[sharpe_arr.argmax()]
    max_sr_vol = vol_arr[sharpe_arr.argmax()]

    weights_max_sharpe = list(all_weights[sharpe_arr.argmax()])

    efficient_frontier = go.Figure(go.Scatter(
        x=vol_arr,
        y=ret_arr,
        marker=dict(
            size=5,
            color=sharpe_arr,
            colorbar=dict(
                title="Colorbar"
            ),
            colorscale="Viridis"
        ),
        mode="markers",
        name="Portfolios (6000)"))

    efficient_frontier.add_trace(go.Scatter(
        x=[max_sr_vol],
        y=[max_sr_ret],
        marker={'color':'red'},
        mode='markers',
        name='Efficient Portfolio'
    ))

    efficient_frontier.update_layout(
        height=600,
        width=1400,
        legend=dict(
            yanchor="top",
            y=1.2,
            xanchor="left",
            x=0.01
            ),
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
    )

    children = '''
                Your max sharpe ratio portfolio has a return of {0:.2f}%, with a volatility of {1:.2f}%
                '''.format(max_sr_ret*100, max_sr_vol*100)

    data = pd.DataFrame(
        {
            'Stock': [stock for _, stock in stocks.items()],
            'Weight': list(all_weights[sharpe_arr.argmax()])
        })

    return efficient_frontier, children, data, weights_max_sharpe


def get_min_var_portfolio(dfs, investment, stocks, app_color):
    full_returns, full_close, log_ret = get_returns(dfs)
    cov_matrix = log_ret.cov()

    np.random.seed(42)
    num_ports = 6000
    all_weights = np.zeros((num_ports, len(full_close.columns)))
    avg_rets = np.zeros(num_ports)
    port_mean = np.zeros(num_ports)
    port_stdev = np.zeros(num_ports)
    var = np.zeros(num_ports)
    mean_investment = np.zeros(num_ports)
    stdev_investment = np.zeros(num_ports)
    cutoff = np.zeros(num_ports)

    initial_investment = investment

    try:
        for x in range(num_ports):
            # Weights
            weights = np.array(np.random.random(len(stocks))) # Problem here
            weights = weights/np.sum(weights)

            # Save weights
            all_weights[x,:] = weights

            # Calculate mean returns for portfolio overall,
            port_mean[x] = log_ret.mean().dot(weights)

            # Calculate portfolio standard deviation
            port_stdev[x] = np.sqrt(weights.T.dot(cov_matrix).dot(weights))

            # Calculate mean of investment
            mean_investment[x] = (1+port_mean[x]) * initial_investment

            # Calculate standard deviation of investmnet
            stdev_investment[x] = initial_investment * port_stdev[x]

            # VaR 99%
            cutoff[x] = norm.ppf(1-0.99, mean_investment[x], stdev_investment[x])
            var[x] = initial_investment - cutoff[x]
    except (ValueError, TypeError):
        pass

    min_var_ret = port_mean[var.argmin()]
    min_var_vol = port_stdev[var.argmin()]
    min_var = var[var.argmin()]

    data = pd.DataFrame(
        {
            'Stock': [stock for _, stock in stocks.items()],
            'Weight': list(all_weights[var.argmin()])
        })

    return min_var_ret, min_var_vol, min_var, data