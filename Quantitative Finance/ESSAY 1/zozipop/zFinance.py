import pandas as pd
import numpy as np
from scipy.stats import linregress, norm
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from itertools import combinations

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

    def get(self, something):
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
                'Median': np.array(stocks_data.median(axis=0)),
                'Min': np.array(stocks_data.min()),
                'Max': np.array(stocks_data.max()),
                'Kurtosis': np.array(stocks_data.kurtosis()),
                'Skewness': np.array(stocks_data.skew()),
                'Alpha': alpha,
                'Beta': beta,
                'VaR 95% HS': var_hs,
                'VaR 95% DN': norm.ppf(1-0.95, stocks_data.mean(), stocks_data.std()),
                'Systematic Risk': sys_risk,
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
            'sharpe': sharpe_arr,
            'num_ports': num_ports
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

    def find_optimal_portfolio(self, min_stocks, max_stocks, stocks_names):
        comb = {}

        sr_keys = [i for i in range(min_stocks, max_stocks)]
        sr_values = [[] for i in range(min_stocks, max_stocks)]
        sr = dict(zip(sr_keys, sr_values))

        sr_max = {}
        comb_max = {}
        comb_max_stocks = {}

        for i in range(min_stocks, max_stocks):
            comb[i] = list(combinations(stocks_names, i))
            for combs in comb[i]:
                sr[i].append(self.maximize_sharpe_ratio(list(combs))[0][2])
            sr_max[i] = np.array(sr[i]).argmax()
            comb_max[i] = sr[i][sr_max[i]]
            comb_max_stocks[i] = comb[i][sr_max[i]]

        results = {}
        for i in range(min_stocks, max_stocks):
            results[f'{i} Stocks'] = {
                'Sharpe Ratio': comb_max[i], 'Stocks': list(comb_max_stocks[i])}

        return results

    def compute_returns_portfolio(self, selection, weights, logs=False):
        if logs:
            returns = self.logs[selection]
        else:
            returns = self.returns[selection]

        portfolio_returns = (returns*weights).sum(axis=1).rename('Portfolio Returns')

        return portfolio_returns

    def get_statistics_portfolio(self, selection, weights, percentage=True, logs=False):
        portfolio_returns = self.compute_returns_portfolio(
            selection, weights, logs)

        if logs:
            stats = self.compute_statistics(
                portfolio_returns, self.market_logs)
        else:
            stats = self.compute_statistics(
                portfolio_returns, self.market_returns)

        if percentage:
            stats = stats.multiply([100]*7+[1]*4+[100]*3, axis=1)

        return stats

    def question1(self, pairs, logs=False):
        if logs:
            returns = self.logs
        else:
            returns = self.returns

        flat_pairs = [', '.join(pair) for pair in pairs]

        std = {}
        ret = {}
        for flat_pair, pair in zip(flat_pairs, pairs):
            std[flat_pair] = [returns[pair[0]].std(), returns[pair[1]].std()]
            ret[flat_pair] = [returns[pair[0]].mean(), returns[pair[1]].mean()]

        iterables = [flat_pairs, ['Returns', 'Std']]
        columns = pd.MultiIndex.from_product(iterables)
        df = pd.DataFrame(
            index=[np.arange(-1, 1.2, 0.2).round(2)], columns=columns)

        for col in df.columns.levels[0]:
            df[col, 'Returns'] = ret[col][0]*0.5+ret[col][1]*0.
            for index, _ in df[col, 'Std'].iteritems():
                df.loc[index, (col, 'Std')] = np.sqrt(
                                (0.5**2)*(std[col][0]**2) + \
                                (0.5**2) * (std[col][1]**2) + \
                                2*0.5*0.5*std[col][0]*std[col][1]*index[0])

        return df*100 # In percentage

    def regression(self, portfolio_ret, ff3_factors, include_ff3):
        # Cleaning DataFrame
        ff3_factors.index = pd.to_datetime(ff3_factors.index, format='%Y%m%d')
        ff3_factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)
        # Convert in percentile
        ff3_factors = ff3_factors.apply(lambda x: x/100)
        # Filter
        ff3_factors = ff3_factors[ff3_factors.index > "2018-06-30"]

        # Merging the stock and factor returns dataframes together
        df_stock_factor = pd.merge(
            portfolio_ret, ff3_factors, left_index=True, right_index=True)


        df_stock_factor['XsRet'] = df_stock_factor['Returns'] - \
            df_stock_factor['RF']  # Calculating excess returns

        # Running CAPM and FF3 models.
        CAPM = smf.ols(formula='XsRet ~ MKT', data=df_stock_factor).fit(
            cov_type='HAC', cov_kwds={'maxlags': 1})

        FF3 = smf.ols(formula='XsRet ~ MKT + SMB + HML',
                    data=df_stock_factor).fit(cov_type='HAC',
                    cov_kwds={'maxlags': 1})

        # t-Stats
        CAPMtstat = CAPM.tvalues
        FF3tstat = FF3.tvalues

        # Coeffs
        CAPMcoeff = CAPM.params
        FF3coeff = FF3.params

        if include_ff3:
            # DataFrame with coefficients and t-stats
            results_df = pd.DataFrame({'CAPMcoeff': CAPMcoeff,
                                    'CAPMtstat': CAPMtstat,
                                    'FF3coeff': FF3coeff,
                                    'FF3tstat': FF3tstat},
                                    index=['Intercept', 'MKT', 'SMB', 'HML'])


            dfoutput = summary_col([CAPM, FF3], stars=True, float_format='%0.4f',
                        model_names=['CAPM', 'FF3'],
                        info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                        'Adjusted R2': lambda x: "{:.4f}".format(x.rsquared_adj)},
                        regressor_order=['Intercept', 'MKT', 'SMB', 'HML'])
            print(dfoutput)
            return {
                'DataFrame':{'Portfolio_Factors':df_stock_factor,
                            'Results':results_df},
                'Factors':{'Fama-French':FF3,
                            'CAPM':CAPM}
            }
        else:
            # DataFrame with coefficients and t-stats
            results_df = pd.DataFrame({'CAPMcoeff': CAPMcoeff,
                                    'CAPMtstat': CAPMtstat},
                                      index=['Intercept', 'MKT'])

            dfoutput = summary_col([CAPM], stars=True, float_format='%0.4f',
                        model_names=['CAPM'],
                        info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                        'Adjusted R2': lambda x: "{:.4f}".format(x.rsquared_adj)},
                        regressor_order=['Intercept', 'MKT'])
            print(dfoutput)
            return {
                'DataFrame': {'Portfolio_Factors': df_stock_factor,
                              'Results': results_df},
                'Factors': {'CAPM': CAPM}
            }

