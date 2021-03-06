import pandas as pd
import numpy as np
from scipy.stats import linregress, norm
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from itertools import combinations

class PortfolioData:
    def __init__(self, green, bad, market):
        # Initialize data
        self.green = green.set_index('Date')
        self.bad = bad.set_index('Date')
        self.market = market.set_index('Date')['Adj Close'].rename('S&P500')

        # Create all variable necessary
        # First about green funds
        self.green_tickers = self.green.columns
        self.green_returns = self.green.pct_change().dropna(how="all")

        # Second about bad funds
        self.bad_tickers = self.bad.columns
        self.bad_returns = self.bad.pct_change().dropna(how="all")

        # Third about benchmark
        self.market_ticker = self.market.name
        self.market_returns = self.market.pct_change().dropna(how="all")

    def get(self, something):
        if something == 'green prices':
            to_return = self.green
        elif something == 'green returns':
            to_return = self.green_returns
        elif something == 'bad prices':
            to_return = self.bad
        elif something == 'bad returns':
            to_return = self.bad_returns
        elif something == 'market prices':
            to_return = self.market
        elif something == 'market returns':
            to_return = self.market_returns
        else:
            to_return = ''

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

    def get_statistics(self, portfolio='green', percentage=True):
        if portfolio == 'green':
            df = pd.concat(
                [self.green_returns,
                 self.green_returns.mean(axis=1).rename('Equal W Portfolio'),
                 self.market_returns], axis=1)
            stats = self.compute_statistics(df, self.market_returns)
        else:
            df = pd.concat(
                [self.bad_returns,
                 self.bad_returns.mean(axis=1).rename('Equal W Portfolio'),
                 self.market_returns], axis=1)
            stats = self.compute_statistics(df, self.market_returns)

        if percentage:
            stats = stats.multiply([100]*7+[1]*4+[100]*3, axis=1)

        return stats

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

    def optimize_sharpe_ratio(self, returns, constraint, geometric=False):
        # create constraint variable
        cons = ({'type':'eq','fun':self.check_sum})

        # create weight boundaries
        bounds = tuple((constraint[0], constraint[1])
                       for i in range(returns.shape[1]))

        # initial guess
        init_guess = [1/returns.shape[1]]*returns.shape[1]

        opt_results = minimize(self.neg_sharpe, init_guess, args=(returns, geometric),
                    method='SLSQP', bounds=bounds, constraints=cons)

        return opt_results

    def maximize_sharpe_ratio(self, portfolio='green', constraint=[0.01, 0.4], geometric=False):
        if portfolio == 'green':
            returns = self.green_returns
        else:
            returns = self.bad_returns

        results = self.optimize_sharpe_ratio(returns, constraint)

        stats = self.get_ret_vol_sr(returns, results.x, geometric=geometric)


        df_stats = pd.DataFrame(data=[i.round(2) for i in stats*[100, 100, 1]],
                                index=['Returns', 'Volatility', 'Sharpe Ratio'],
                                columns=['Stats']).T

        df_alloc = pd.DataFrame(data=[i.round(2) for i in results.x*100],
                                index=returns.columns,
                                columns=['Allocation']).T
        print(df_stats)
        print('\n---------------------------------------------\n')
        print(df_alloc)

        portfolio_returns = (
            returns*results.x).sum(axis=1).rename('Portfolio Returns')

        return portfolio_returns, [stats, results.x]

    def get_statistics_portfolio(self, portfolio_returns, percentage=True):
        stats = self.compute_statistics(
                portfolio_returns, self.market_returns)

        if percentage:
            stats = stats.multiply([100]*7+[1]*4+[100]*3, axis=1)

        return stats

    def regression(self, portfolio_ret, ff3_factors, umd_factor):
        # Cleaning DataFrame
        ff3_factors.index = pd.to_datetime(ff3_factors.index, format='%Y%m%d')
        umd_factor.index = pd.to_datetime(umd_factor.index, format='%Y%m%d')
        ff3_factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)
        factors = pd.concat([ff3_factors, umd_factor], axis=1)
        # Convert in percentile
        factors = factors.apply(lambda x: x/100)
        # Filter
        factors = factors[factors.index > "2014-01-01"]

        # Merging the stock and factor returns dataframes together
        df_stock_factor = pd.merge(
            portfolio_ret, factors, left_index=True, right_index=True)


        df_stock_factor['XsRet'] = df_stock_factor['Portfolio Returns'] - \
            df_stock_factor['RF']  # Calculating excess returns

        # Running CAPM and FF3 models.
        CAPM = smf.ols(formula='XsRet ~ MKT', data=df_stock_factor).fit(
            cov_type='HAC', cov_kwds={'maxlags': 1})

        FF3 = smf.ols(formula='XsRet ~ MKT + SMB + HML',
                    data=df_stock_factor).fit(cov_type='HAC',
                    cov_kwds={'maxlags': 1})

        UMD = smf.ols(formula='XsRet ~ MKT + SMB + HML + WML',
                    data=df_stock_factor).fit(cov_type='HAC',
                    cov_kwds={'maxlags': 1})

        # t-Stats
        CAPMtstat = CAPM.tvalues
        FF3tstat = FF3.tvalues
        UMDtstat = UMD.tvalues

        # Coeffs
        CAPMcoeff = CAPM.params
        FF3coeff = FF3.params
        UMDcoeff = UMD.params

        # DataFrame with coefficients and t-stats
        results_df = pd.DataFrame({'CAPMcoeff': CAPMcoeff,
                                    'CAPMtstat': CAPMtstat,
                                    'FF3coeff': FF3coeff,
                                    'FF3tstat': FF3tstat,
                                    'UMDcoeff': UMDcoeff,
                                    'UMDtstat': UMDtstat},
                                index=['Intercept', 'MKT', 'SMB', 'HML', 'UMD'])


        dfoutput = summary_col([CAPM, FF3, UMD], stars=True, float_format='%0.4f',
                    model_names=['CAPM', 'FF3', 'UMD'],
                    info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                    'Adjusted R2': lambda x: "{:.4f}".format(x.rsquared_adj)},
                    regressor_order=['Intercept', 'MKT', 'SMB', 'HML', 'UMD'])
        print(dfoutput)
        return {
            'DataFrame':{'Portfolio_Factors':df_stock_factor,
                        'Results':results_df},
            'Factors':{'Fama-French':FF3,
                        'CAPM':CAPM,
                        'UMD':UMD}
        }

