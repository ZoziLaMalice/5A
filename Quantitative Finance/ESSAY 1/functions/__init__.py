from .practicals_functions import bbands, trading, stats_df, stats_dfs, stats_market, create_df, create_dfs
from .twitter_data import get_tweets_by_query
from .sentiments_graphs import count_words, preprocess_nltk, plot_sentiment, plot_word_count
from .basic_graphs import plot_first_graph, plot_second_graph, plot_third_graph
from .stocks_stats import plot_VaR_log, plot_VaR_returns, plot_stocks_stats, stocks_regression
from .differents_portfolios import get_min_var_portfolio, get_returns, efficient_portfolio, get_equal_weighted
from .stats_efficient_portfolio import portfolio_regression, plot_monte_carlo, get_data
from .paulo_investment import get_paulo_return, plot_paulo_investment