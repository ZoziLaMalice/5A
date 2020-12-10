import pandas as pd
import numpy as np
from scipy.stats import linregress, norm
import statsmodels.api as sm

class FundsData:
    def __init__(self, green, bad, benchmark, oil):
        # Initialize data
        self.green = green
        self.bad = bad
        self.benchmark = benchmark
        self.oil = oil

        # Create all variable necessary
        # First about green funds
        self.green_tickers = self.green.columns
        self.green_returns = self.green.pct_change().dropna(how="all")

        # Second about bad funds
        self.bad_tickers = self.bad.columns
        self.bad_returns = self.bad.pct_change().dropna(how="all")

        # Third about benchmark
        self.benchmark_ticker = self.benchmark.name
        self.benchmark_returns = self.benchmark.pct_change().dropna(how="all")

        # Fourth about oil
        self.oil_ticker = self.oil.name
        self.oil_returns = self.oil.pct_change().dropna(how="all")

    def get(self, something):
        if something == 'green prices':
            to_return = self.green
        elif something == 'green returns':
            to_return = self.green_returns
        elif something == 'bad prices':
            to_return = self.bad
        elif something == 'bad returns':
            to_return = self.bad_returns
        elif something == 'benchmark prices':
            to_return = self.benchmark
        elif something == 'benchmark returns':
            to_return = self.benchmark_returns
        elif something == 'oil prices':
            to_return = self.oil
        elif something == 'oil returns':
            to_return = self.oil_returns
        else:
            to_return = ''

        return to_return

    # Compute Statistics
    def compute_statistics(self, funds_data, benchmark):
        if isinstance(funds_data, pd.DataFrame):
            nb_cols = len(funds_data.columns)
            alpha = [linregress(funds_data.iloc[:, i],
                                benchmark).intercept for i in range(nb_cols)]
            beta = [linregress(funds_data.iloc[:, i],
                               benchmark).slope for i in range(nb_cols)]
            sys_risk = [linregress(funds_data.iloc[:, i], benchmark).slope **
                        2 * benchmark.var() for i in range(nb_cols)]
            var_hs = [funds_data.iloc[:, i].sort_values(
                ascending=True).quantile(0.05) for i in range(nb_cols)]
            columns = funds_data.columns
        else:
            nb_cols = 1
            alpha = linregress(funds_data, benchmark).intercept
            beta = linregress(funds_data, benchmark).slope
            sys_risk = linregress(
                funds_data, benchmark).slope**2 * benchmark.var()
            var_hs = funds_data.sort_values(ascending=True).quantile(0.05)
            columns = funds_data.name

        stats = pd.DataFrame(
            {
                'Std': np.array(funds_data.std()),
                'Annual Std': np.array(funds_data.std()*np.sqrt(252)),
                'Mean': np.array(funds_data.mean()),
                'Geometric Mean': np.array((1 + funds_data).prod() ** (252/funds_data.count())-1),
                'Median': np.array(funds_data.median(axis=0)),
                'Min': np.array(funds_data.min()),
                'Max': np.array(funds_data.max()),
                'Kurtosis': np.array(funds_data.kurtosis()),
                'Skewness': np.array(funds_data.skew()),
                'Alpha': alpha,
                'Beta': beta,
                'VaR 95% HS': var_hs,
                'VaR 95% DN': norm.ppf(1-0.95, funds_data.mean(), funds_data.std()),
                'Systemic Risk': sys_risk,
            },
            index=[columns]
        ).round(6)

        return stats

    def get_statistics(self, funds='green', on='benchmark', percentage=True):
        if funds=='green' and on=='benchmark':
            df = pd.concat(
                [self.green_returns,
                 self.green_returns.mean(axis=1).rename('10 Funds Mean'),
                self.benchmark_returns], axis=1)
            stats = self.compute_statistics(df, self.benchmark_returns)
        elif funds == 'bad' and on == 'benchmark':
            df = pd.concat(
                [self.bad_returns,
                 self.bad_returns.mean(axis=1).rename('10 Funds Mean'),
                 self.benchmark_returns], axis=1)
            stats = self.compute_statistics(df, self.benchmark_returns)
        elif funds=='green' and on=='oil':
            df = pd.concat(
                [self.green_returns,
                 self.green_returns.mean(axis=1).rename('10 Funds Mean'),
                self.oil_returns], axis=1)
            stats = self.compute_statistics(df, self.oil_returns)
        else:
            df = pd.concat(
                [self.bad_returns,
                 self.bad_returns.mean(axis=1).rename('10 Funds Mean'),
                 self.oil_returns], axis=1)
            stats = self.compute_statistics(df, self.oil_returns)

        if percentage:
            stats = stats.multiply([100]*7+[1]*4+[100]*3, axis=1)

        return stats

    def regression(self, x, y):
        X = sm.add_constant(x)
        model = sm.OLS(y, X)
        results = model.fit()
        print(results.summary())



