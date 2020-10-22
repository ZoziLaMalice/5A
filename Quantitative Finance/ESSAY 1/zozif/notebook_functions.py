import pandas as pd
import numpy as np
from scipy.stats import linregress, norm


# Compute Statistics
def get_statistics(stocks_data, market_data, stocks_name, percentage=True):

    if isinstance(stocks_data, pd.DataFrame):
        nb_cols = len(stocks_data.columns)
        alpha = [linregress(stocks_data.iloc[:, i], market_data).intercept for i in range(nb_cols)]
        beta = [linregress(stocks_data.iloc[:, i], market_data).slope for i in range(nb_cols)]
        sys_risk = [linregress(stocks_data.iloc[:, i], market_data).slope**2 * market_data.var() for i in range(nb_cols)]
        var_hs = [stocks_data.iloc[:, i].sort_values(ascending=True).quantile(0.05) for i in range(nb_cols)]
    else:
        alpha = linregress(stocks_data, market_data).intercept
        beta = linregress(stocks_data, market_data).slope
        sys_risk = linregress(stocks_data, market_data).slope**2 * market_data.var()
        var_hs = stocks_data.sort_values(ascending=True).quantile(0.05)



    stats = pd.DataFrame(
        {
            'Std': stocks_data.std(),
            'Annual Std': stocks_data.std()*np.sqrt(252),
            'Mean': stocks_data.mean(),
            'Geometric Mean': (1 + stocks_data).prod() ** (252/stocks_data.count())-1,
            'Median': np.median(stocks_data),
            'Min': stocks_data.min(),
            'Max': stocks_data.max(),
            'Kurtosis': stocks_data.kurtosis(),
            'Skewness': stocks_data.skew(),
            'Alpha': alpha,
            'Beta': beta,
            'VaR 95% HS': var_hs,
            'VaR 95% DN': norm.ppf(1-0.95, stocks_data.mean(), stocks_data.std()),
            'Systemic Risk': sys_risk,
        },
        index=[name for name in stocks_name]
    ).round(6)

    if percentage:
        stats = stats.multiply([100]*7+[1]*4+[100]*3, axis=1)

    return stats

def get_ret_vol_sr(returns, weights, geometric=True):
    # Get Return, Std, SHarpe Ratio from optimization
    weights = np.array(weights)
    if geometric:
        mean = (1 + returns).prod() ** (252/returns.count())-1
        ret = np.sum( (mean * weights))
    else:
        ret = np.sum(returns.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])


def neg_sharpe(weights, returns):
    # the number 2 is the sharpe ratio index from the get_ret_vol_sr
    return get_ret_vol_sr(returns, weights)[2] * -1


def check_sum(weights):
    # Check if sum of weights equal to 1
    return np.sum(weights)-1


def monte_carlo(returns, num_ports, geometric=False):

    np.random.seed(11041997)
    all_weights = np.zeros((num_ports, len(returns.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for x in range(num_ports):
        # Weights
        weights = np.array(np.random.random(len(returns.columns)))
        weights = weights/np.sum(weights)

        # Save weights
        all_weights[x,:] = weights

        # Expected return
        if geometric:
            mean = (1 + returns).prod() ** (252/returns.count())-1
            ret_arr[x] = np.sum( (mean * weights))
        else:
            ret_arr[x] = np.sum( (returns.mean() * weights * 252))

        # Expected volatility
        vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))

        # Sharpe Ratio
        sharpe_arr[x] = ret_arr[x]/vol_arr[x]

    simulation = {
        'weights': all_weights,
        'returns': ret_arr,
        'volatility': vol_arr,
        'sharpe': sharpe_arr
    }

    return simulation


def get_allocation(all_weights, ret_arr, vol_arr, sharpe_arr, returns):
    max_sr_ret, max_sr_vol = ret_arr[sharpe_arr.argmax()], vol_arr[sharpe_arr.argmax()]
    min_vol_ret, min_vol_vol = ret_arr[vol_arr.argmin()], vol_arr[vol_arr.argmin()]

    weights_max_sharpe = list(all_weights[sharpe_arr.argmax()])
    weights_min_vol = list(all_weights[vol_arr.argmin()])

    max_sharpe_allocation = pd.DataFrame([i*100 for i in weights_max_sharpe],index=returns.columns,columns=['Max Sharpe Allocation'])
    min_vol_allocation = pd.DataFrame([i*100 for i in weights_min_vol],index=returns.columns,columns=['Min Volatility Allocation'])

    allocation = max_sharpe_allocation.T.append(min_vol_allocation.T)

    allocation['Returns'] = [max_sr_ret*100, min_vol_ret*100]
    allocation['Volatility'] = [max_sr_vol*100, min_vol_vol*100]
    allocation['Sharpe Ratio'] = [sharpe_arr[sharpe_arr.argmax()], sharpe_arr[vol_arr.argmin()]]

    allocation = allocation.round(2)
    return allocation