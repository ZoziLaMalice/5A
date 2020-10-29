#!/bin/python3.8

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table
import yfinance as yf
import pandas as pd
import json

from zozipop import StocksData, zoziDash

complete_data = pd.read_csv(
    'data/complete_data.csv', header=[0, 1], index_col=[0])
market = pd.read_csv('data/market.csv', index_col=[0])

with open('data/stocks.json', 'r') as fp:
    stocks = json.load(fp)

stocks_object = StocksData(complete_data, market)
zp = zoziDash(525)
lilzp = zoziDash(250)

prices = stocks_object.get('prices')['NEM']
returns = stocks_object.get('returns')['NEM']

corr = stocks_object.get('returns').corr()
cov = stocks_object.get('returns').cov()

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
            html.Div([
                html.Div([
                    html.H6('Choose your stock: ')
                ]),
                html.Div([
                    dcc.Dropdown(
                        id='switch-stocks',
                        options=[{'label': str(stocks[stock]['Name']
                                            + f' ({stock})'),
                                'value': stocks[stock]['Ticker']}
                                for stock in stocks],
                        value='NEM',
                    ),
                ], className='dropdown'),
            ], className='graph__dropdown'),
            html.Div([
                html.Div([
                    html.H6('Prices & Returns')
                ], className='graph__title'),
                dcc.Graph(
                    id='prices-returns',
                    figure=zp.plot_prices_returns(prices, returns),
                    config={
                        'displayModeBar': False
                    },
                )
            ], className='prices_returns')
        ], className='two-thirds column'),
        html.Div([
            html.Div([
                html.Div([
                    html.H6('Prices')
                ], className='graph__title'),
                dcc.Graph(
                    id='prices',
                    # Focus on the COVID Period
                    figure=lilzp.plot_prices(prices),
                    config={
                        'displayModeBar': False
                    }
                )
            ], className='graph__container first'),
            html.Div([
                html.Div([
                    html.H6('Candles')
                ], className='graph__title'),
                dcc.Graph(
                    id='candles',
                    figure=lilzp.plot_candles(complete_data['NEM']),
                    config={
                        'displayModeBar': False
                    }
                )
            ], className='graph__container second')
        ], className='one-third column')
    ], className='app__content'),
    html.Div([
        html.Div([
            html.Button(
                id='correlation',
                n_clicks=0,
                children='Correlation Matrix'
            ),
        ], className='three columns'),
        html.Div([
            html.Button(
                id='covariance',
                n_clicks=0,
                children='Covariance Matrix'
            )
        ], className='three columns')
    ], className='graph__button'),
    html.Div([
        html.Div([
            dcc.Graph(
                id='matrix',
                figure=zp.plot_heatmap(corr, 2, 10),
                config={
                    'displayModeBar': False
                }
            )
        ], className='two-thirds column offset-by-two matrix')
    ], className='app__content')
], className='app__container')


@app.callback(
    Output('prices-returns', 'figure'),
    Output('prices', 'figure'),
    Output('candles', 'figure'),
    [Input('switch-stocks', 'value')],
    [State("switch-stocks", "options")]
)
def update_prices_returns_graph(stock, opt):
    the_label = [x['label'] for x in opt if x['value'] == stock]

    prices = stocks_object.get('prices')[stock]
    returns = stocks_object.get('returns')[stock]

    fig = zp.plot_prices_returns(prices, returns)
    fig2 = lilzp.plot_prices(prices)
    fig3 = lilzp.plot_candles(complete_data[stock])

    return fig, fig2, fig3


@app.callback(
    Output('matrix', 'figure'),
    [Input('correlation', 'n_clicks'),
     Input('covariance', 'n_clicks')]
)
def update_matrix(btn1, btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'correlation' in changed_id:
        fig = zp.plot_heatmap(corr, 2, 10)
        return fig
    elif 'covariance' in changed_id:
        fig = zp.plot_heatmap(cov, 4, 5)
        return fig
    else:
        return zp.plot_heatmap(corr, 2, 10)

if __name__ == '__main__':
    app.run_server(debug=True)
