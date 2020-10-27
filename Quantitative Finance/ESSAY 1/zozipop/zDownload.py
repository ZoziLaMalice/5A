import yfinance as yf

class zoziDl():
    """
    Download data from Yahoo Finance
    """
    def __init__(self, start):
        self.stocks = {
            'Materials': {'Name': 'Newmont Corporation', 'Ticker': 'NEM'},
            'Communication Services': {'Name': 'Alphabet Inc.', 'Ticker': 'GOOGL'},
            'Consumer Discretionary': {'Name': 'Amazon.com Inc.', 'Ticker': 'AMZN'},
            'Consumer Staples': {'Name': 'PepsiCo Inc.', 'Ticker': 'PEP'},
            'Energy': {'Name': 'National Oilwell Varco Inc.', 'Ticker': 'NOV'},
            'Financial Services': {'Name': 'Bank of America Corp', 'Ticker': 'BAC'},
            'Healthcare': {'Name': 'HCA Healthcare', 'Ticker': 'HCA'},
            'Industrials': {'Name': 'Boeing Company', 'Ticker': 'BA'},
            'Real Estate': {'Name': 'Host Hotels & Resorts', 'Ticker': 'HST'},
            'Information Technology': {'Name': 'Apple Inc.', 'Ticker': 'AAPL'},
            'Utilities': {'Name': 'American Electric Power', 'Ticker': 'AEP'}
        }
        self.start = start


    def get_stocks_data(self):
        complete_data = yf.download(
            tickers=[self.stocks[stock]['Ticker'] for stock in self.stocks],
            group_by='ticker',
            start=self.start
        )
        return complete_data


    def get_market_data(self):
        market = yf.download(
            tickers="^GSPC",
            start=self.start
        )
        return market