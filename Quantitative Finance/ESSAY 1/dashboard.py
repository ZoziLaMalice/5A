import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table
import yfinance as yf
import pandas as pd

from zozipop import StocksData, zoziDash, zoziDl

zdl = zoziDl('2018-06-30')
complete_data = zdl.get_stocks_data()
market = zdl.get_market_data()

stocks_object = StocksData(complete_data, market)
zp = zoziDash(500)
lilzp = zoziDash(250)

prices = stocks_object.get('prices')['AAPL']

app = dash.Dash(__name__, meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}])

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1('The S&P500 during the COVID crisis'),
        ], className='app__header__title'),
        html.Div([
            html.P(
                'This dashboard shows some financials charts about S&P500 stocks, especially during the COVID'
                ),
        ], className='app__comment')
    ], className='app__header'),
    html.Div([
        html.Div([
            html.H3('Here my stocks selection')
        ]),
        html.Div([
            html.Div([
                html.H4('Materials'),
                html.H5('Newmont Corporation'),
                html.P('This stocks represent....')
            ], className='three columns stock__info'),
            html.Div([
                html.H4('Communication Services'),
                html.H5('Alphabet Inc.'),
                html.P('This stocks represent....')
            ], className='three columns stock__info'),
            html.Div([
                html.H4('Consumer Discretionary'),
                html.H5('Amazon.com Inc.'),
                html.P('This stocks represent....')
            ], className='three columns stock__info'),
            html.Div([
                html.H4('Consumer Staples'),
                html.H5('PepsiCo Inc.'),
                html.P('This stocks represent....')
            ], className='three columns stock__info'),
        ], className='card__container'),
        html.Div([
            html.Div([
                html.H4('Energy'),
                html.H5('National Oilwell Varco Inc.'),
                html.P('This stocks represent....')
            ], className='three columns stock__info'),
            html.Div([
                html.H4('Financial Services'),
                html.H5('Bank of America Corp'),
                html.P('This stocks represent....')
            ], className='three columns stock__info'),
            html.Div([
                html.H4('Healthcare'),
                html.H5('HCA Healthcare'),
                html.P('This stocks represent....')
            ], className='three columns stock__info'),
            html.Div([
                html.H4('Industrials'),
                html.H5('Boeing Company'),
                html.P('This stocks represent....')
            ], className='three columns stock__info'),
        ], className='card__container'),
        html.Div([
            html.Div([
                html.H4('Real Estate'),
                html.H5(' Hotels & Resorts'),
                html.P('This stocks represent....')
            ], className='three columns stock__info'),
            html.Div([
                html.H4('Information Technology'),
                html.H5('Apple Inc.'),
                html.P('This stocks represent....')
            ], className='three columns stock__info'),
            html.Div([
                html.H4('Utilities'),
                html.H5('American Electric Power'),
                html.P('This stocks represent....')
            ], className='three columns stock__info'),
        ], className='card__container')
    ], className='app__infos'),
    html.Div([
        html.Div([
            dcc.Graph(
                id='Test',
                figure=zp.plot_candles(complete_data['AAPL'], 'AAPL'),
                config={
                    'displayModeBar': False
                }
            )
        ], className='two-thirds column'),
        html.Div([
            dcc.Graph(
                id='Test-2',
                # Focus on the COVID Period
                figure=lilzp.plot_prices(prices),
                config={
                    'displayModeBar': False
                }
            )
        ], className='one-third column')
    ], className='app__content')
], className='app__container')

if __name__ == '__main__':
    app.run_server(debug=True)
