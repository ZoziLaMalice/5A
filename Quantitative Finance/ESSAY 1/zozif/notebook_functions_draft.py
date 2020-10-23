import pandas as pd
import numpy as np
from scipy.stats import linregress, norm
from scipy.optimize import minimize
import statsmodels.api as sm



class StocksData:
    def __init__(self, stocks, market):
        # Initialize data
        self.complete_data = stocks
        self.market = market

        # Create all variable necessary
        # First about stocks
        self.complete_data.dropna(how="all", inplace=True)
        self.stocks_tickers = self.complete_data.columns.levels[0]
        self.prices = pd.concat([self.complete_data[name]['Adj Close'] for name in self.stocks_tickers], axis=1, keys=self.stocks_tickers)
        self.returns = self.prices.pct_change().dropna(how="all")
        self.logs = np.log(1 + self.prices.pct_change()).dropna(how="all")
        self.prices = self.prices.iloc[1:, :]

        # Second about market
        self.market.dropna(how='all', inplace=True)
        self.market_prices = self.market['Adj Close'].rename('Market Prices')
        self.market_returns = self.market_prices.pct_change().dropna(how='all').rename('Market Returns')
        self.market_logs = np.log(1 + self.market_prices.pct_change()).dropna(how="all").rename('Market Logs')
        self.market_prices = self.market_prices.iloc[1:]
        self.market.dropna(inplace=True)

    def get_something(self, something):
        if something == 'logs':
            to_return = self.logs
        elif something == 'market_logs':
            to_return = self.market_logs
        elif something == 'returns':
            to_return = self.returns
        elif something == 'market_returns':
            to_return = self.market_returns
        elif something == 'complete_data':
            to_return = self.complete_data
        elif something == 'market':
            to_return = self.market
        elif something == 'prices':
            to_return = self.prices
        else:
            to_return = self.market_prices

        return to_return


    # Compute Statistics
    def compute_statistics(self, stocks_data, market_data):
        if isinstance(stocks_data, pd.DataFrame):
            nb_cols = len(stocks_data.columns)
            alpha = [linregress(stocks_data.iloc[:, i], market_data).intercept for i in range(nb_cols)]
            beta = [linregress(stocks_data.iloc[:, i], market_data).slope for i in range(nb_cols)]
            sys_risk = [linregress(stocks_data.iloc[:, i], market_data).slope**2 * market_data.var() for i in range(nb_cols)]
            var_hs = [stocks_data.iloc[:, i].sort_values(ascending=True).quantile(0.05) for i in range(nb_cols)]
            columns = stocks_data.columns
        else:
            nb_cols = 1
            alpha = linregress(stocks_data, market_data).intercept
            beta = linregress(stocks_data, market_data).slope
            sys_risk = linregress(stocks_data, market_data).slope**2 * market_data.var()
            var_hs = stocks_data.sort_values(ascending=True).quantile(0.05)
            columns = stocks_data.name

        stats = pd.DataFrame(
            {
                'Std': np.array(stocks_data.std()),
                'Annual Std': np.array(stocks_data.std()*np.sqrt(252)),
                'Mean': np.array(stocks_data.mean()),
                'Geometric Mean': np.array((1 + stocks_data).prod() ** (252/stocks_data.count())-1),
                'Median': np.median(stocks_data),
                'Min': np.array(stocks_data.min()),
                'Max': np.array(stocks_data.max()),
                'Kurtosis': np.array(stocks_data.kurtosis()),
                'Skewness': np.array(stocks_data.skew()),
                'Alpha': alpha,
                'Beta': beta,
                'VaR 95% HS': var_hs,
                'VaR 95% DN': norm.ppf(1-0.95, stocks_data.mean(), stocks_data.std()),
                'Systemic Risk': sys_risk,
            },
            index=[columns]
        ).round(6)

        return stats


    def get_statistics(self, market=True, percentage=True, logs=False):
        if logs and market:
            df = pd.concat([self.logs, self.market_logs.rename('Market')], axis=1)
            stats = self.compute_statistics(df, self.market_logs)
        elif market and not logs:
            df = pd.concat([self.returns, self.market_returns.rename('Market')], axis=1)
            stats = self.compute_statistics(df, self.market_returns)
        elif logs and not market:
            stats = self.compute_statistics(self.logs, self.market_logs)
        elif not logs and not market:
            stats = self.compute_statistics(self.returns, self.market_returns)
        else:
            stats = pd.DataFrame()

        if percentage:
            stats = stats.multiply([100]*7+[1]*4+[100]*3, axis=1)

        return stats


    def get_equal_weighted_portfolio_returns(self, logs=False):
        if logs:
            weights = [1/len(self.returns.columns)]*len(self.returns.columns)
            equal_weighted_portfolio_returns = (self.logs*weights).sum(axis=1).rename('Eq W Logs')
        else:
            weights = [1/len(self.returns.columns)]*len(self.returns.columns)
            equal_weighted_portfolio_returns = (self.returns*weights).sum(axis=1).rename('Eq W Returns')

        return equal_weighted_portfolio_returns


    def get_statistics_eq_w_portfolio(self, logs=False, percentage=True):
        portfolio_returns = self.get_equal_weighted_portfolio_returns(logs=logs)

        if logs:
            stats = self.compute_statistics(portfolio_returns, self.market_returns)
        else:
            stats = self.compute_statistics(portfolio_returns, self.market_logs)

        if percentage:
            stats = stats.multiply([100]*7+[1]*4+[100]*3, axis=1)

        return stats


    def monte_carlo(self, selection, num_ports, logs=False, geometric=False, debug=False):

        if logs:
            returns = self.logs[selection]
        else:
            returns = self.returns[selection]

        np.random.seed(11041997)
        all_weights = np.zeros((num_ports, len(selection)))
        ret_arr = np.zeros(num_ports)
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)

        for x in range(num_ports):
            # Weights
            weights = np.array(np.random.random(len(selection)))
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

        if debug:
            return simulation, allocation
        else:
            return allocation



    def get_ret_vol_sr(self, returns, weights, geometric=False):
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


    def neg_sharpe(self, weights, returns, geometric=False):
        # the number 2 is the sharpe ratio index from the get_ret_vol_sr
        return self.get_ret_vol_sr(returns, weights, geometric=geometric)[2] * -1


    def check_sum(self, weights):
        # Check if sum of weights equal to 1
        return np.sum(weights)-1


    def optimize_sharpe_ratio(self, selection, returns, geometric=False):
        # create constraint variable
        cons = ({'type':'eq','fun':self.check_sum})

        # create weight boundaries
        bounds = tuple((0,1) for i in range(len(selection)))

        # initial guess
        init_guess = [1/len(selection)]*len(selection)

        opt_results = minimize(self.neg_sharpe, init_guess, args=(returns, geometric),
                    method='SLSQP', bounds=bounds, constraints=cons)

        return opt_results


    def maximize_sharpe_ratio(self, selection, logs=False, geometric=False):
        if logs:
            returns = self.logs[selection]
        else:
            returns = self.returns[selection]

        results = self.optimize_sharpe_ratio(selection, returns)

        stats = self.get_ret_vol_sr(returns, results.x, geometric=geometric)
        return stats, results.x


    def compute_returns_portfolio(self, selection, weights, logs=False):
        if logs:
            returns = self.logs[selection]
        else:
            returns = self.returns[selection]

        portfolio_returns = (returns*weights).sum(axis=1).rename('Portfolio Returns')

        return portfolio_returns


    def portfolio_vs_market_CAPM(self, selection, weights, logs=False):
        # split dependent and independent variable
        if logs:
            X = self.market_logs
        else:
            X = self.market_returns

        returns = self.compute_returns_portfolio(selection, weights, logs=logs)

        y = returns.rename('Max Sharpe Portfolio')

        # Add a constant to the independent value
        X1 = sm.add_constant(X)

        # make regression model
        model = sm.OLS(y, X1)

        # fit model and print results
        results = model.fit()
        print(results.summary())