import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table
import yfinance as yf
import pandas as pd
import csv
from tqdm.notebook import tqdm
import html5lib
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
import re
from scipy.stats import linregress, norm

def bbands(price, window_size=10, num_of_std=5):
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std  = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean, upper_band, lower_band


def trading(df, portfolio, buy_, sell_):
    buys = []
    sells = []
    portfolio = portfolio
    evolution = df.loc[df.index[0], 'Log_Returns']
    buy = False

    for i in range(df.index[1], len(df)):
        evolution += df.loc[i, 'Log_Returns']
        if evolution  <= -buy_ and not buy:
            buys.append([df.loc[i, 'Close'], df.loc[i, 'Datetime']])
            portfolio -= df.loc[i, 'Close']
            evolution = 0
            buy = True

        elif evolution <= -sell_ and buy:
            sells.append([df.loc[i, 'Close'], df.loc[i, 'Datetime']])
            portfolio += df.loc[i, 'Close']
            evolution = 0
            buy = False

        elif evolution >= sell_ and buy:
            sells.append([df.loc[i, 'Close'], df.loc[i, 'Datetime']])
            evolution = 0
            portfolio += df.loc[i, 'Close']
            buy = False

    return portfolio, buys, sells

buys = {}
sells = {}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

global ALL_STOCKS
ALL_STOCKS = {'^GSPC': 'S&P500'}

weights_max_sharpe = []

with open('sp500_sectors.csv', newline='') as f:
    reader = csv.reader(f)
    sp500_s = list(reader)

clean_sp500 = {
 'Basic Materials': [],
 'Communication Services': [],
 'Consumer Cyclical': [],
 'Consumer Defensive': [],
 'Energy': [],
 'Financial Services': [],
 'Healthcare': [],
 'Industrials': [],
 'Market': [],
 'Real Estate': [],
 'Technology': [],
 'Utilities': [],
 'No Information': []
}

for sector, value in clean_sp500.items():
    for row in sp500_s:
        if sector == row[1]:
            clean_sp500[row[1]] += [row[0], row[2]]

global market
market = yf.Ticker('^gspc').history(period="2y")
market['Returns'] = market.Close.pct_change()
market = market.iloc[1:]
market.reset_index(inplace=True)

global covid
covid = pd.read_csv('covid_USA.csv')
covid.Date = pd.to_datetime(covid.Date)

stats = pd.DataFrame(
    {
        'Stock': ['S&P500'],
        'Std': [market.Returns.std()],
        'Annual Std': [market.Returns.std()* np.sqrt(252)],
        'Mean': [market.Returns.mean()],
        'Median': [np.median(market.Returns.std())],
        'Min': [market.Returns.min()],
        'Max': [market.Returns.max()],
        'Kurtosis': [market.Returns.kurtosis()],
        'Skewness': [market.Returns.skew()],
        'Alpha': [linregress(market.Returns, market.Returns).intercept],
        'Beta': [linregress(market.Returns, market.Returns).slope],
        'VaR 95% HS': [market.Returns.sort_values(ascending=True).quantile(0.05)],
        'VaR 95% DN': [norm.ppf(1-0.95, market.Returns.mean(), market.Returns.std())],
        'Systemic Risk': [linregress(market.Returns, market.Returns).slope**2 * market.Returns.var()]
    },
    index=[0]
).round(6)

correl = pd.DataFrame([market.Returns], columns=['S&P500'], index=['S&P500'])

#First Tab Charts
# Market Chart
market_chart = make_subplots(specs=[[{"secondary_y": True}]])
market_chart.add_trace(
    go.Scatter(
        x = market.Date,
        y = market.Close,
        name = 'S&P500',
        yaxis='y'),
    secondary_y=False
)

market_chart.add_trace(go.Scatter(x= covid.Date, y=covid.Case, name='COVID', yaxis='y1'), secondary_y=True)

market_chart.update_layout(
    updatemenus=[dict(
        x=1.1,
        y=0.8,
        active=0,
        type='buttons',
        direction='down',
        buttons=list(
            [dict(label = 'Show COVID',
                method = 'update',
                args = [{'visible': [True, True]}]),
            dict(label = 'Hide COVID',
                method = 'update',
                args = [{'visible': [True, False]}]),
            ])
        )
    ])

market_chart.update_layout(
    width=1400,
    height=600,
    xaxis = dict(
        rangeslider = {'visible': False},
    ),
    yaxis_title='Stocks',
    shapes = [dict(
        x0='2020-02-15', x1='2020-02-15', y0=0, y1=1, xref='x', yref='paper',
        line_width=2)],
    annotations=[dict(
        x='2020-02-17', y=0.95, xref='x', yref='paper',
        showarrow=False, xanchor='left', text='COVID Begins')],
    yaxis=dict(
    ticksuffix=' $'
    ),
)

# First Chart
first = go.Figure()


# Second Analysis
second = go.Figure()


# Third Analysis
third = go.Figure()

# Second Tab Charts
# Fourth Chart
fourth = go.Figure()

# Regression
regression = go.Figure()

# Heatmap
heatmap = go.Figure()

# Variance CoVariance Matrix
covariance = go.Figure()

# Efficient Frontier
efficient_frontier = go.Figure()

# Paulo Portfolio
paulo_portfolio = go.Figure()

app.layout = html.Div([
        html.Div([
            html.H1('The S&P500 during the COVID crisis'),
            html.H4('This dashboard shows some financials charts about S&P500 stocks, especially during the COVID'),
        ]),
        html.Div([
            dcc.Graph(
                id='market-chart',
                figure=market_chart
            )
        ]),
        html.Div([
            html.H3('Choose some stocks and start to build your portfolio !'),
        ]),
        html.Div([
            dcc.Dropdown(
                id='sectors-drop',
                options=[{'label': k, 'value': k} for k in clean_sp500.keys()],
                value='Consumer Cyclical'
            ),
        ], style={'padding-top': 15}),

        html.Hr(),

        html.Div([
            html.Div([
                dcc.Dropdown(id='stock-drop', value='AAP', clearable=False),
            ], style={'display': 'table-cell', 'width': '65%'}),

            html.Div([
                html.Button(id='add-stock', n_clicks=0, children='Add Stock'),
            ], style={'display': 'table-cell', 'width': '10%', 'padding-left': 25}),

            html.Div([
                html.Button(id='remove-stock', n_clicks=0, children='Remove Stock'),
            ], style={'display': 'table-cell', 'padding-left': 25}),

            html.Div([
                html.Button(id='remove-market', n_clicks=0, children='Remove Market'),
            ], style={'display': 'table-cell', 'padding-left': 25}),

            html.Div(id='hidden-div', style={'display':'none'})

        ], style={'display': 'table'}),

        html.Hr(),

        html.Div([
            html.H5('Your portfolio:'),
        ]),

        html.Div([
            dash_table.DataTable(
                id='selected',
                columns=[{"name": i, "id": i} for i in stats.columns],
                data=stats.to_dict('records'),
            ),
        ]),

        dcc.Tabs(id='tabs', children=[
            dcc.Tab(label='Basic Charts', id='tab-1', children=[
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='first-graph',
                            figure=first
                        ),
                    ]),
                ]),

                html.Div([
                    dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in stats.columns],
                        data=stats.to_dict('records'),
                    ),
                ]),

                html.Div([
                    dcc.Graph(
                        id='second-graph',
                        figure=second
                    ),
                ]),

                html.Div([
                    dcc.Graph(
                        id='third-graph',
                        figure=third
                    ),
                ]),
            ]),

            dcc.Tab(label='Stocks Stats', id='tab-2',  children=[
                html.Div([
                    html.Button(id='load-stocks', n_clicks=0, children='Load Stocks',
                    style={'display': 'table-cell'})
                ], style={'padding': 40, 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'table'}),

                html.Div(id='drop-portfolio', children=[
                    dcc.Dropdown(
                        id='portfolio-stocks',
                        options=[{'label': item, 'value': key} for key, item in ALL_STOCKS.items()],
                    ),
                ], style={'padding-top': 10}),

                html.Div([
                    dcc.Graph(
                        id='regression',
                        figure=regression
                    )
                ]),

                html.Hr(),

                html.Div([
                    html.Div([
                        html.Button(id='var-normal', n_clicks=0, children='VaR Returns',
                        ),
                    ], style={'display': 'table-cell'}),
                    html.Div([
                        html.Button(id='var-log', n_clicks=0, children='VaR Log Returns',
                        ),
                    ], style={'display': 'table-cell', 'padding-left': 20}),
                ], style={'display': 'table', 'margin-left': 'auto', 'margin-right': 'auto'}),

                html.Div([
                    dcc.Graph(
                        id='VaR-HS',
                        figure=fourth
                    )
                ]),

                html.Hr(),

                html.Div([
                    dcc.Graph(
                        id='correl',
                        figure=heatmap
                    )
                ]),

                html.Hr(),

                html.Div([
                    dcc.Graph(
                        id='cov',
                        figure=covariance
                    )
                ]),
            ]),

            dcc.Tab(label='Differents Portfolios', id='tab-3', children=[

                html.Div([
                    html.Button(id='equal-weighted', n_clicks=0, children='Generate Equal Weighted Portfolio',
                    style={'display':'table-cell'})
                ], style={'padding': 40, 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'table'}),

                html.Div([
                    dash_table.DataTable(
                        id='weights-equal',
                        columns=[{"name": 'Stock', "id": 'Stock'}, {'name': 'Weight', 'id': 'Weight'}],
                        data=[0, 0],
                    ),
                ], style={'width': '30%', 'padding': 25, 'left': '33.5%', 'position': 'relative'}),

                html.Div(id='equal-weighted-portfolio', children=[

                ], style={'padding-top': 15, 'font-size': 20, 'text-align': 'center'}),

                html.Hr(),

                html.Div([
                    html.Button(id='load-portfolio', n_clicks=0, children='Generate Efficient Frontier',
                    style={'display': 'table-cell'})
                ], style={'padding': 40, 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'table'}),

                html.Div([
                    dash_table.DataTable(
                        id='weights-sharpe',
                        columns=[{"name": 'Stock', "id": 'Stock'}, {'name': 'Weight', 'id': 'Weight'}],
                        data=[0, 0],
                    ),
                ], style={'width': '30%', 'padding': 25, 'left': '33.5%', 'position': 'relative'}),

                html.Div([
                    dcc.Graph(
                        id='efficient-frontier',
                        figure=efficient_frontier
                    )
                ]),

                html.Div(id='max-sharpe-text', children=[

                ], style={'padding-top': 15, 'font-size': 20, 'text-align': 'center'}),

                html.Hr(),

                html.Div([
                    html.Div(dcc.Input(id='investment', type='number'), style={'display': 'table-cell', 'padding-right': 20}),
                    html.Button(id='submit-investment', n_clicks=0, children='Generate Min VaR Portfolio',
                    style={'display': 'table-cell'})
                ], style={'margin-left': 'auto', 'margin-right': 'auto', 'display': 'table'}),

                html.Div([
                    dash_table.DataTable(
                        id='weights-min-var',
                        columns=[{"name": 'Stock', "id": 'Stock'}, {'name': 'Weight', 'id': 'Weight'}],
                        data=[0, 0],
                    ),
                ], style={'width': '30%', 'padding': 25, 'left': '33.5%', 'position': 'relative'}),

                html.Div(id='min-var-portfolio', children=[

                ], style={'padding-top': 15, 'font-size': 20, 'text-align': 'center'}),

                html.Hr(),

                html.Div([
                    html.Div(dcc.Input(id='paulo', type='number'), style={'display': 'table-cell', 'padding-right': 20}),
                    html.Button(id='submit-paulo', n_clicks=0, children='Paulo Investment',
                    style={'display': 'table-cell'})
                ], style={'margin-left': 'auto', 'margin-right': 'auto', 'display': 'table'}),

                html.Div(id='drop-paulo', children=[
                    dcc.Dropdown(
                        id='paulo-stocks',
                        options=[{'label': item, 'value': key} for key, item in ALL_STOCKS.items()],
                    ),
                ], style={'padding-top': 10}),

                html.Div([
                    dcc.Graph(
                        id='paulo-portfolio',
                        figure=paulo_portfolio
                    )
                ]),

                html.Div(id='paulo-portfolio-text', children=[

                ], style={'padding-top': 15, 'font-size': 20, 'text-align': 'center'}),

                html.Hr(),

            ]),

            dcc.Tab(label='Stats on Efficient Portfolio', id='tab-4', children=[

                html.Div([
                    html.Button(id='load-stocks-2', n_clicks=0, children='Load Stocks',
                    style={'display': 'table-cell'})
                ], style={'padding': 40, 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'table'}),

                html.Div([
                    dcc.Graph(
                        id='regression-portfolio',
                        figure=go.Figure()
                    )
                ]),

                html.Hr(),

                html.Div([
                    dcc.Graph(
                        id='monte-carlo-portfolio',
                        figure=go.Figure()
                    )
                ]),

            ]),
        ], style={'padding-top': 30}),
])


@app.callback(
    Output('stock-drop', 'options'),
    [Input('sectors-drop', 'value')])
def set_stocks_options(selected_sector):
    return [{'label': clean_sp500[selected_sector][i+1], 'value': clean_sp500[selected_sector][i]} for i in range(len(clean_sp500[selected_sector])) if i % 2 ==0]


@app.callback(
    Output('stock-drop', 'value'),
    [Input('stock-drop', 'options')])
def set_stocks_value(available_options):
    return available_options[0]['value']


@app.callback(
    Output('selected', 'data'),
    [Input('add-stock', 'n_clicks'),
    Input('remove-stock', 'n_clicks'),
    Input('remove-market', 'n_clicks')],
    [State('stock-drop', 'value'),
    State("stock-drop","options")])
def set_stocks_value(btn1, btn2, btn3, stock, opt):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    stats = pd.DataFrame(
        {
            'Stock': ['S&P500'],
            'Std': [market.Returns.std()],
            'Annual Std': [market.Returns.std()* np.sqrt(252)],
            'Mean': [market.Returns.mean()],
            'Median': [np.median(market.Returns.std())],
            'Min': [market.Returns.min()],
            'Max': [market.Returns.max()],
            'Kurtosis': [market.Returns.kurtosis()],
            'Skewness': [market.Returns.skew()],
            'Alpha': [linregress(market.Returns, market.Returns).intercept],
            'Beta': [linregress(market.Returns, market.Returns).slope],
            'VaR 95% HS': [market.Returns.sort_values(ascending=True).quantile(0.05)],
            'VaR 95% DN': [norm.ppf(1-0.95, market.Returns.mean(), market.Returns.std())],
            'Systemic Risk': [linregress(market.Returns, market.Returns).slope**2 * market.Returns.var()]
        },
        index=[0]
    ).round(6)


    if 'add-stock' in changed_id:
        ALL_STOCKS.update({stock: [x['label'] for x in opt if x['value'] == stock][0]})

        dfs = {}
        for stock, _ in ALL_STOCKS.items():
            dfs[stock] = yf.Ticker(stock).history(period="2y")
            dfs[stock]['Returns'] = dfs[stock].Close.pct_change()
            dfs[stock] = dfs[stock].iloc[1:]
            dfs[stock].reset_index(inplace=True)
            dfs[stock]["Color"] = np.where(dfs[stock]['Returns'] < 0, 'red', 'green')

        stats = pd.DataFrame(
            {
                'Stock': [name for _, name in ALL_STOCKS.items()],
                'Std': [dfs[df].Returns.std() for df in dfs],
                'Annual Std': [dfs[df].Returns.std()* np.sqrt(252) for df in dfs],
                'Mean': [dfs[df].Returns.mean() for df in dfs],
                'Median': [np.median(dfs[df].Returns) for df in dfs],
                'Min': [dfs[df].Returns.min() for df in dfs],
                'Max': [dfs[df].Returns.max() for df in dfs],
                'Kurtosis': [dfs[df].Returns.kurtosis() for df in dfs],
                'Skewness': [dfs[df].Returns.skew() for df in dfs],
                'Alpha': [linregress(dfs[df].Returns, market[market.Date >= dfs[df].Date.min()].Returns).intercept for df in dfs],
                'Beta': [linregress(dfs[df].Returns, market[market.Date >= dfs[df].Date.min()].Returns).slope for df in dfs],
                'VaR 95% HS': [dfs[df].Returns.sort_values(ascending=True).quantile(0.05) for df in dfs],
                'VaR 95% DN': [norm.ppf(1-0.95, dfs[df].Returns.mean(), dfs[df].Returns.std()) for df in dfs],
                'Systemic Risk': [linregress(dfs[df].Returns, market[market.Date >= dfs[df].Date.min()].Returns).slope**2 * market[market.Date >= dfs[df].Date.min()].Returns.var() for df in dfs]
            },
            index=[df for _, df in ALL_STOCKS.items()]
        ).round(6)

    elif 'remove-stock' in changed_id:
        try:
            del ALL_STOCKS[stock]
        except KeyError:
            pass

        dfs = {}
        for stock, _ in ALL_STOCKS.items():
            dfs[stock] = yf.Ticker(stock).history(period="2y")
            dfs[stock]['Returns'] = dfs[stock].Close.pct_change()
            dfs[stock] = dfs[stock].iloc[1:]
            dfs[stock].reset_index(inplace=True)
            dfs[stock]["Color"] = np.where(dfs[stock]['Returns'] < 0, 'red', 'green')

        stats = pd.DataFrame(
            {
                'Stock': [name for _, name in ALL_STOCKS.items()],
                'Std': [dfs[df].Returns.std() for df in dfs],
                'Annual Std': [dfs[df].Returns.std()* np.sqrt(252) for df in dfs],
                'Mean': [dfs[df].Returns.mean() for df in dfs],
                'Median': [np.median(dfs[df].Returns) for df in dfs],
                'Min': [dfs[df].Returns.min() for df in dfs],
                'Max': [dfs[df].Returns.max() for df in dfs],
                'Kurtosis': [dfs[df].Returns.kurtosis() for df in dfs],
                'Skewness': [dfs[df].Returns.skew() for df in dfs],
                'Alpha': [linregress(dfs[df].Returns, market[market.Date >= dfs[df].Date.min()].Returns).intercept for df in dfs],
                'Beta': [linregress(dfs[df].Returns, market[market.Date >= dfs[df].Date.min()].Returns).slope for df in dfs],
                'VaR 95% HS': [dfs[df].Returns.sort_values(ascending=True).quantile(0.05) for df in dfs],
                'VaR 95% DN': [norm.ppf(1-0.95, dfs[df].Returns.mean(), dfs[df].Returns.std()) for df in dfs],
                'Systemic Risk': [linregress(dfs[df].Returns, market[market.Date >= dfs[df].Date.min()].Returns).slope**2 * market[market.Date >= dfs[df].Date.min()].Returns.var() for df in dfs]
            },
            index=[df for _, df in ALL_STOCKS.items()]
        ).round(6)

    elif 'remove-market' in changed_id:
        try:
            del ALL_STOCKS['^GSPC']
        except KeyError:
            pass

        dfs = {}
        for stock, _ in ALL_STOCKS.items():
            dfs[stock] = yf.Ticker(stock).history(period="2y")
            dfs[stock]['Returns'] = dfs[stock].Close.pct_change()
            dfs[stock] = dfs[stock].iloc[1:]
            dfs[stock].reset_index(inplace=True)
            dfs[stock]["Color"] = np.where(dfs[stock]['Returns'] < 0, 'red', 'green')

        stats = pd.DataFrame(
            {
                'Stock': [name for _, name in ALL_STOCKS.items()],
                'Std': [dfs[df].Returns.std() for df in dfs],
                'Annual Std': [dfs[df].Returns.std()* np.sqrt(252) for df in dfs],
                'Mean': [dfs[df].Returns.mean() for df in dfs],
                'Median': [np.median(dfs[df].Returns) for df in dfs],
                'Min': [dfs[df].Returns.min() for df in dfs],
                'Max': [dfs[df].Returns.max() for df in dfs],
                'Kurtosis': [dfs[df].Returns.kurtosis() for df in dfs],
                'Skewness': [dfs[df].Returns.skew() for df in dfs],
                'Alpha': [linregress(dfs[df].Returns, market[market.Date >= dfs[df].Date.min()].Returns).intercept for df in dfs],
                'Beta': [linregress(dfs[df].Returns, market[market.Date >= dfs[df].Date.min()].Returns).slope for df in dfs],
                'VaR 95% HS': [dfs[df].Returns.sort_values(ascending=True).quantile(0.05) for df in dfs],
                'VaR 95% DN': [norm.ppf(1-0.95, dfs[df].Returns.mean(), dfs[df].Returns.std()) for df in dfs],
                'Systemic Risk': [linregress(dfs[df].Returns, market[market.Date >= dfs[df].Date.min()].Returns).slope**2 * market[market.Date >= dfs[df].Date.min()].Returns.var() for df in dfs]
            },
            index=[df for _, df in ALL_STOCKS.items()]
        )

    return stats.to_dict('records')


@app.callback(
    Output('first-graph', 'figure'),
    Output('second-graph', 'figure'),
    Output('third-graph', 'figure'),
    Output('table', 'data'),
    [Input('stock-drop', 'value')],
    [State("stock-drop","options")]
)
def update_output_div(stock, opt):

    the_label = [x['label'] for x in opt if x['value'] == stock]

    df = yf.Ticker(stock).history(period="2y")
    df['Returns'] = df.Close.pct_change()
    df = df.iloc[1:]
    df.reset_index(inplace=True)
    df["Color"] = np.where(df['Returns'] < 0, 'red', 'green')

    stats = pd.DataFrame(
        {
            'Stock': [the_label[0]],
            'Std': [df.Returns.std()],
            'Annual Std': [df.Returns.std()* np.sqrt(252)],
            'Mean': [df.Returns.mean()],
            'Median': [np.median(df.Returns.std())],
            'Min': [df.Returns.min()],
            'Max': [df.Returns.max()],
            'Kurtosis': [df.Returns.kurtosis()],
            'Skewness': [df.Returns.skew()],
            'Alpha': [linregress(df.Returns, market[market.Date >= df.Date.min()].Returns).intercept],
            'Beta': [linregress(df.Returns, market[market.Date >= df.Date.min()].Returns).slope],
            'VaR 95% HS': [df.Returns.sort_values(ascending=True).quantile(0.05)],
            'VaR 95% DN': [norm.ppf(1-0.95, df.Returns.mean(), df.Returns.std())],
            'Systemic Risk': [linregress(df.Returns, market[market.Date >= df.Date.min()].Returns).slope**2 * market[market.Date >= df.Date.min()].Returns.var()]
        },
        index=[0]
    ).round(6)

    df['log_ret'] = np.log(df.Close/df.Close.shift(1))
    df["log_Color"] = np.where(df['log_ret'] < 0, 'red', 'green')

    # First Chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x = df.Date,
            y = df.Close,
            name = stock,
            yaxis='y'),
        secondary_y=False
    )

    fig.add_trace(go.Scatter(x= covid.Date, y=covid.Case, name='COVID', yaxis='y1'), secondary_y=True)

    fig.update_layout(
        updatemenus=[dict(
            x=1.1,
            y=0.8,
            active=0,
            type='buttons',
            direction='down',
            buttons=list(
                [dict(label = 'Show COVID',
                    method = 'update',
                    args = [{'visible': [True, True]}]),
                dict(label = 'Hide COVID',
                    method = 'update',
                    args = [{'visible': [True, False]}]),
                ])
            )
        ])

    fig.update_layout(
        width=1400,
        height=600,
        xaxis = dict(
            rangeslider = {'visible': True},
        ),
        title=f'{the_label[0]} Analysis during the COVID',
        yaxis_title='Stocks',
        shapes = [dict(
            x0='2020-02-15', x1='2020-02-15', y0=0, y1=1, xref='x', yref='paper',
            line_width=2)],
        annotations=[dict(
            x='2020-02-17', y=0.95, xref='x', yref='paper',
            showarrow=False, xanchor='left', text='COVID Begins')],
        yaxis=dict(
        ticksuffix=' $'
        ),
    )

    # Second Chart
    row_width = [1]
    row_width.extend([0.25] * (2 - 1))

    fig2 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0,
        row_width=row_width[::-1],
    )

    bb_avg, bb_upper, bb_lower = bbands(df.Close)

    fig2.add_trace(go.Scatter(x=df.Date, y=bb_upper, yaxis='y2',
                            line = dict(width = 1),
                            line_color='#ccc', hoverinfo='none',
                            legendgroup='Bollinger Bands', name='Bollinger Bands',
                            mode='lines'),
                            row=1, col=1)

    fig2.add_trace(go.Scatter(x=df.Date, y=bb_lower, yaxis='y2',
                            line = dict(width = 1),
                            line_color='#ccc', hoverinfo='none',
                            legendgroup='Bollinger Bands', showlegend=False,
                            mode='lines', fill='tonexty'),
                            row=1, col=1)


    fig2.add_trace(go.Candlestick(x=df.Date,
                    open=df.Open,
                    high=df.High,
                    low=df.Low,
                    close=df.Close,
                    name='Candle Stick'), row=1, col=1)


    colors = []
    for i in range(len(df.Close)):
        if i != 0:
            if df.Close[i] > df.Close[i-1]:
                colors.append('green')
            else:
                colors.append('red')
        else:
            colors.append('red')


    fig2.add_trace(go.Bar(x=df.Date, y=df.Volume,
                        marker=dict(color=colors),
                        yaxis='y', name='Volume'),
                        row=2, col=1)

    fig2.update_layout(
        width=1400,
        height=800,
        title=f'{the_label[0]} Analysis during the COVID',
        yaxis_title='Stocks',
        shapes = [dict(
            x0='2020-02-15', x1='2020-02-15', y0=0, y1=1, xref='x', yref='paper',
            line_width=2)],
        annotations=[dict(
            x='2020-02-17', y=0.95, xref='x', yref='paper',
            showarrow=False, xanchor='left', text='COVID Begins')],
        yaxis=dict(
        ticksuffix=' $'
        ),
        xaxis = dict(
        rangeslider = {'visible': False},
        ),
    )


    # Third Chart
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig3.add_trace(
        go.Scatter(x=df.Date, y=df.Close, name="Closing Prices", yaxis="y"),
        secondary_y=False
    )

    fig3.add_trace(
        go.Bar(x=df.Date, y=(df.Returns * 100), name="Returns", marker_color=df.Color, yaxis="y1"),
        secondary_y=True
    )

    fig3.add_trace(
        go.Bar(x=df.Date, y=(df.log_ret * 100), name="Log Returns", marker_color=df.log_Color, yaxis="y1", visible=False),
        secondary_y=True
    )

    # Set x-axis title
    fig3.update_xaxes(title_text="Date")

    fig3.update_layout(
        width=1400,
        height=600,
        title=f'{the_label[0]} Analysis during the COVID',
        yaxis_title='Stocks',
        shapes = [dict(
            x0='2020-02-15', x1='2020-02-15', y0=0, y1=1, xref='x', yref='paper',
            line_width=2)],
        annotations=[dict(
            x='2020-02-17', y=0.95, xref='x', yref='paper',
            showarrow=False, xanchor='left', text='COVID Begins')]
    )

    fig3.update_layout(
        yaxis=dict(
        title=f"{the_label[0]}'s Closing Prices",
        ticksuffix=' $'
        ),
        yaxis2=dict(
            title=f"{the_label[0]}'s Returns",
            ticksuffix = '%'
        )
    )

    fig3.update_layout(
        updatemenus=[dict(
            x=1.1,
            y=0.8,
            active=0,
            type='buttons',
            direction='down',
            buttons=list(
                [dict(label = 'Returns',
                    method = 'update',
                    args = [{'visible': [True, True, False]}]),
                dict(label = 'Log Returns',
                    method = 'update',
                    args = [{'visible': [True, False, True]}]),
                ])
            )
        ])

    return fig, fig2, fig3, stats.to_dict('records')


@app.callback(
    Output('drop-portfolio', 'children'),
    Output('correl', 'figure'),
    Output('cov', 'figure'),
    [Input('load-stocks', 'n_clicks')]
)
def load_stocks(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    dfs = {}
    for stock, _ in ALL_STOCKS.items():
        dfs[stock] = yf.Ticker(stock).history(period="2y")
        dfs[stock]['Returns'] = dfs[stock].Close.pct_change()
        dfs[stock] = dfs[stock].iloc[1:]
        dfs[stock].reset_index(inplace=True)

    date_min = max([dfs[stock].Date.min() for stock in dfs])
    for df in dfs:
        dfs[df] = dfs[df][dfs[df].Date >= date_min]
        dfs[df].reset_index(inplace=True)


    full_returns = pd.concat([dfs[df].Returns for df in dfs], axis=1)
    full_returns.columns = [df for df in dfs]

    full_close = pd.concat([dfs[df].Close for df in dfs], axis=1)
    full_close.columns = [df for df in dfs]
    log_ret = np.log(full_close/full_close.shift(1))


    cov = pd.DataFrame.cov(full_returns)


    correl_matrix = np.corrcoef([dfs[stock].Returns for stock in dfs])

    correl = pd.DataFrame(correl_matrix, columns=[stock for stock in dfs],
                        index=[stock for stock in dfs])

    children = html.Div([
                dcc.Dropdown(
                    id='portfolio-stocks',
                    options=[{'label': label, 'value': value} for value, label in ALL_STOCKS.items()],
                ),
            ])

    heatmap = ff.create_annotated_heatmap(
        z=correl.values[::-1].round(2),
        x=[stock for stock in dfs],
        y=[stock for stock in dfs][::-1],
        xgap=10,
        ygap=10,
    )
    heatmap.update_layout(title_text='Correlation Matrix')

    covariance = ff.create_annotated_heatmap(
        z=cov.values[::-1].round(6),
        x=[stock for stock in dfs],
        y=[stock for stock in dfs][::-1],
        xgap=10,
        ygap=10,
    )
    covariance.update_layout(title_text='Variance - Covariance Matrix')

    if 'load-stocks' in changed_id:
        return children, heatmap, covariance
    else:
        return html.Div([' ']), go.Figure(), go.Figure()

@app.callback(
    Output('portfolio-stocks', 'value'),
    [Input('portfolio-stocks', 'options')])
def set_portfolio_stocks_value(available_options):
    try:
        return available_options[0]['value']
    except TypeError:
        pass

@app.callback(
    Output('VaR-HS', 'figure'),
    Output('var-normal', 'style'),
    Output('var-log', 'style'),
    Output('regression', 'figure'),
    [Input('portfolio-stocks', 'value'),
    Input('var-normal', 'n_clicks'),
    Input('var-log', 'n_clicks')],
    State('portfolio-stocks', 'options'))
def update_VaR_chart(stock, btn1, btn2, opt):
    the_label = [x['label'] for x in opt if x['value'] == stock]

    df = yf.Ticker(stock).history(period="2y")
    df['Returns'] = df.Close.pct_change()
    df = df.iloc[1:]
    df.reset_index(inplace=True)

    date_min = df.Date.min()

    market_date_min = yf.Ticker('^GSPC').history(period="2y")
    market_date_min['Returns'] = market_date_min.Close.pct_change()
    market_date_min = market_date_min.iloc[1:]
    market_date_min.reset_index(inplace=True)
    market_date_min = market_date_min[market_date_min.Date >= date_min]
    market_date_min.reset_index(inplace=True)

    log_ret = np.log(df.Close/df.Close.shift(1)).dropna()

    slope, intercept, r, p, std_err = linregress(df.Returns, market_date_min.Returns)

    x = np.linspace(np.amin(market_date_min.Returns), np.amax(df.Returns))
    y = slope * x + intercept

    regression = go.Figure(go.Scatter(
        x=market_date_min.Returns,
        y=df.Returns,
        mode="markers",
        marker={'size': 5, 'color': '#468de2'},
        name="Returns"
    ))

    regression.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        marker={'color': '#e34029'},
        name="Linear Regression"
    ))

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'var-normal' in changed_id:
        var = ff.create_distplot([df.Returns], ['Historical Simulation'], bin_size=.002, show_rug=False, colors=['#1669e9', '#e4ed1e'])

        var.add_trace(go.Scatter(
            mode= "markers+text",
            text="VaR",
            name="Value at Risk 95%",
            x=[df.Returns.sort_values(ascending=True).quantile(0.05)],
            y=[0],
            marker={"size": 20, 'color': "#ff9b00"},
            textposition= 'bottom center'
        ))

        var.add_trace(go.Scatter(
            mode= "markers+text",
            text="VaR",
            name="Value at Risk 99%",
            x=[df.Returns.sort_values(ascending=True).quantile(0.01)],
            y=[0],
            marker={"size": 20, 'color': "#ff0000"},
            textposition= 'bottom center',
            visible=False
        ))

        var.update_layout(
            width=1400,
            height=600,
        )

        var.update_layout(
            updatemenus=[dict(
                active=0,
                type='buttons',
                direction='down',
                buttons=list(
                    [dict(label = 'VaR 95%',
                        method = 'update',
                        args = [{'visible': [True, True, True, False]}]),
                    dict(label = 'VaR 99%',
                        method = 'update',
                        args = [{'visible': [True, True, False, True]}]),
                    ])
                )
            ])

        return var, {'display': 'table-cell', 'margin-left': 'auto', 'margin-right': 'auto', 'background-color': 'rgba(150, 220, 240, 0.5'}, {'display': 'table-cell', 'margin-left': 'auto', 'margin-right': 'auto'}, regression
    elif 'var-log' in changed_id:
        var = ff.create_distplot([log_ret], ['Historical Simulation'], bin_size=.002, show_rug=False, colors=['#1669e9', '#e4ed1e'])

        var.add_trace(go.Scatter(
            mode= "markers+text",
            text="VaR",
            name="Value at Risk 95%",
            x=[log_ret.sort_values(ascending=True).quantile(0.05)],
            y=[0],
            marker={"size": 20, 'color': "#ff9b00"},
            textposition= 'bottom center'
        ))

        var.add_trace(go.Scatter(
            mode= "markers+text",
            text="VaR",
            name="Value at Risk 99%",
            x=[log_ret.sort_values(ascending=True).quantile(0.01)],
            y=[0],
            marker={"size": 20, 'color': "#ff0000"},
            textposition= 'bottom center',
            visible=False
        ))

        var.update_layout(
            width=1400,
            height=600,
        )

        var.update_layout(
            updatemenus=[dict(
                active=0,
                type='buttons',
                direction='down',
                buttons=list(
                    [dict(label = 'VaR 95%',
                        method = 'update',
                        args = [{'visible': [True, True, True, False]}]),
                    dict(label = 'VaR 99%',
                        method = 'update',
                        args = [{'visible': [True, True, False, True]}]),
                    ])
                )
            ])
        return var, {'display': 'table-cell', 'margin-left': 'auto', 'margin-right': 'auto'}, {'display': 'table-cell', 'margin-left': 'auto', 'margin-right': 'auto', 'background-color': 'rgba(150, 220, 240, 0.5)'}, regression
    else:
        return go.Figure(), {'display': 'table-cell', 'margin-left': 'auto', 'margin-right': 'auto'}, {'display': 'table-cell', 'margin-left': 'auto', 'margin-right': 'auto'}, regression

@app.callback(
    Output('equal-weighted-portfolio', 'children'),
    Output('weights-equal', 'data'),
    [Input('equal-weighted', 'n_clicks')],
)
def equal_weighted(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'equal-weighted' in changed_id:
        dfs = {}
        for stock, _ in ALL_STOCKS.items():
            dfs[stock] = yf.Ticker(stock).history(period="2y")
            dfs[stock]['Returns'] = dfs[stock].Close.pct_change()
            dfs[stock] = dfs[stock].iloc[1:]
            dfs[stock].reset_index(inplace=True)

        date_min = max([dfs[stock].Date.min() for stock in dfs])
        for df in dfs:
            dfs[df] = dfs[df][dfs[df].Date >= date_min]
            dfs[df].reset_index(inplace=True)


        full_returns = pd.concat([dfs[df].Returns for df in dfs], axis=1)
        full_returns.columns = [df for df in dfs]

        full_close = pd.concat([dfs[df].Close for df in dfs], axis=1)
        full_close.columns = [df for df in dfs]
        log_ret = np.log(full_close/full_close.shift(1))

        np.random.seed(42)
        all_weights = np.zeros((1, len(full_close.columns)))
        ret_arr = np.zeros(1)
        vol_arr = np.zeros(1)
        sharpe_arr = np.zeros(1)


        # Weights
        weights = np.array(np.random.random(len(ALL_STOCKS))) # Problem here
        weights = weights/np.sum(weights)

        # Expected return
        ret_arr = np.sum( (log_ret.mean() * weights * 252))

        # Expected volatility
        vol_arr = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))

        data = pd.DataFrame(
            {
                'Stock': [stock for _, stock in ALL_STOCKS.items()],
                'Weight': [100/len(ALL_STOCKS)] * len(ALL_STOCKS)
            })

        return '''
        Your equal weighted portfolio has a return of {0:.2f}%, with a volatility of {1:.2f}%
        '''.format(ret_arr*100, vol_arr*100), data.to_dict('records')
    else:
        return '', []


@app.callback(
    Output('efficient-frontier', 'figure'),
    Output('max-sharpe-text', 'children'),
    Output('weights-sharpe', 'data'),
    [Input('load-portfolio', 'n_clicks')]
)
def load_portfolio(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'load-portfolio' in changed_id:
        dfs = {}
        for stock, _ in ALL_STOCKS.items():
            dfs[stock] = yf.Ticker(stock).history(period="2y")
            dfs[stock]['Returns'] = dfs[stock].Close.pct_change()
            dfs[stock] = dfs[stock].iloc[1:]
            dfs[stock].reset_index(inplace=True)

        date_min = max([dfs[stock].Date.min() for stock in dfs])
        for df in dfs:
            dfs[df] = dfs[df][dfs[df].Date >= date_min]
            dfs[df].reset_index(inplace=True)


        full_returns = pd.concat([dfs[df].Returns for df in dfs], axis=1)
        full_returns.columns = [df for df in dfs]

        full_close = pd.concat([dfs[df].Close for df in dfs], axis=1)
        full_close.columns = [df for df in dfs]
        log_ret = np.log(full_close/full_close.shift(1))

        np.random.seed(42)
        num_ports = 6000
        all_weights = np.zeros((num_ports, len(full_close.columns)))
        ret_arr = np.zeros(num_ports)
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)

        try:
            for x in range(num_ports):
                # Weights
                weights = np.array(np.random.random(len(ALL_STOCKS))) # Problem here
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

        global weights_max_sharpe
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
        ))

        children = '''
                    Your max sharpe ratio portfolio has a return of {0:.2f}%, with a volatility of {1:.2f}%
                    '''.format(max_sr_ret*100, max_sr_vol*100)

        data = pd.DataFrame(
            {
                'Stock': [stock for _, stock in ALL_STOCKS.items()],
                'Weight': list(all_weights[sharpe_arr.argmax()])
            })

        return efficient_frontier, children, data.to_dict('records')
    else:
        return go.Figure(), '', []

@app.callback(
    Output('min-var-portfolio', 'children'),
    Output('weights-min-var', 'data'),
    [Input('submit-investment', 'n_clicks')],
    [State('investment', 'value')]
)
def min_var_portfolio(n_clicks, investment):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-investment' in changed_id:
        dfs = {}
        for stock, _ in ALL_STOCKS.items():
            dfs[stock] = yf.Ticker(stock).history(period="2y")
            dfs[stock]['Returns'] = dfs[stock].Close.pct_change()
            dfs[stock] = dfs[stock].iloc[1:]
            dfs[stock].reset_index(inplace=True)

        date_min = max([dfs[stock].Date.min() for stock in dfs])
        for df in dfs:
            dfs[df] = dfs[df][dfs[df].Date >= date_min]
            dfs[df].reset_index(inplace=True)


        full_returns = pd.concat([dfs[df].Returns for df in dfs], axis=1)
        full_returns.columns = [df for df in dfs]

        full_close = pd.concat([dfs[df].Close for df in dfs], axis=1)
        full_close.columns = [df for df in dfs]
        log_ret = np.log(full_close/full_close.shift(1))
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
                weights = np.array(np.random.random(len(ALL_STOCKS))) # Problem here
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
                'Stock': [stock for _, stock in ALL_STOCKS.items()],
                'Weight': list(all_weights[var.argmin()])
            })

        return '''
        Your min VaR portfolio has a return of {0:.2f}%, with a volatility of {1:.2f}%.
        Here we are saying with 95% confidence that our portfolio of {2:d} USD will not exceed losses greater than {3:.2f} USD over a one day period.
        '''.format(min_var_ret*100, min_var_vol*100, investment, min_var), data.to_dict('records')
    else:
        return '', []


@app.callback(
    Output('paulo-stocks', 'value'),
    [Input('paulo-stocks', 'options')])
def set_paulo_stocks_value(available_options):
    try:
        return available_options[0]['value']
    except TypeError:
        pass


@app.callback(
    Output('paulo-portfolio-text', 'children'),
    Output('drop-paulo', 'children'),
    [Input('submit-paulo', 'n_clicks')],
    [State('paulo', 'value')]
)
def paulo_investment(n_clicks, investment):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-paulo' in changed_id:
        dfs = {}
        for stock, _ in ALL_STOCKS.items():
            dfs[stock] = yf.Ticker(stock).history(period="2y", interval='60m')
            dfs[stock].reset_index(inplace=True)

        date_min = max([dfs[stock].Datetime.min() for stock in dfs])
        for df in dfs:
            dfs[df] = dfs[df][dfs[df].Datetime >= date_min]
            dfs[df].reset_index(inplace=True)
            dfs[df] = dfs[df][['Datetime', 'Close']]
            dfs[df]['Log_Returns'] = np.log(dfs[df].Close/dfs[df].Close.shift(1))
            dfs[df] = dfs[df].dropna()

        stock_inv = [investment * i for i in weights_max_sharpe]
        result = 0

        global buys, sells

        for (df, inv) in zip(dfs, stock_inv):
            r, b, s = trading(dfs[df], inv, 0.04, 0.01)
            result += r
            buys[df] = b
            sells[df] = s

        children = html.Div([
                    dcc.Dropdown(
                        id='paulo-stocks',
                        options=[{'label': label, 'value': value} for value, label in ALL_STOCKS.items()],
                    ),
                ])

        return f'Your result is {result}', children
    else:
        return '', html.Div([' '])

@app.callback(
    Output('paulo-portfolio', 'figure'),
    [Input('paulo-stocks', 'value')],
    [State("paulo-stocks","options")]
)
def update_paulo_figure(stock, opt):

    df = yf.Ticker(stock).history(period="2y", interval='60m')
    df['Returns'] = df.Close.pct_change()
    df = df.iloc[1:]
    df.reset_index(inplace=True)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = df.Datetime,
            y = df.Close,
            name = stock,
        )
    )

    fig.add_trace(go.Scatter(
        mode= "markers",
        marker_symbol="triangle-up",
        x=pd.Series([buys[stock][i][1] for i in range(len(buys[stock]))]),
        y=pd.Series([buys[stock][i][0] for i in range(len(buys[stock]))]),
        marker={"size": 8, 'color': "#1df344"},
        name="Buys"
    ))

    fig.add_trace(go.Scatter(
        mode= "markers",
        marker_symbol="triangle-down",
        x=pd.Series([sells[stock][i][1] for i in range(len(sells[stock]))]),
        y=pd.Series([sells[stock][i][0] for i in range(len(sells[stock]))]),
        marker={"size": 8, 'color': "#f10f0f"},
        name='Sells'
    ))

    return fig


@app.callback(
    Output('regression-portfolio', 'figure'),
    Output('monte-carlo-portfolio', 'figure'),
    [Input('load-stocks-2', 'n_clicks')]
)
def load_stocks_2(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'load-stocks-2' in changed_id:
        dfs = {}
        for stock, _ in ALL_STOCKS.items():
            dfs[stock] = yf.Ticker(stock).history(period="2y")
            dfs[stock]['Returns'] = dfs[stock].Close.pct_change()
            dfs[stock] = dfs[stock].iloc[1:]
            dfs[stock].reset_index(inplace=True)

        date_min = max([dfs[stock].Date.min() for stock in dfs])
        for df in dfs:
            dfs[df] = dfs[df][dfs[df].Date >= date_min]
            dfs[df].reset_index(inplace=True)

        market_date_min = yf.Ticker('^GSPC').history(period="2y")
        market_date_min['Returns'] = market_date_min.Close.pct_change()
        market_date_min = market_date_min.iloc[1:]
        market_date_min.reset_index(inplace=True)
        market_date_min = market_date_min[market_date_min.Date >= date_min]
        market_date_min.reset_index(inplace=True)

        full_returns = pd.concat([dfs[df].Returns for df in dfs], axis=1)
        full_returns.columns = [df for df in dfs]

        full_close = pd.concat([dfs[df].Close for df in dfs], axis=1)
        full_close.columns = [df for df in dfs]
        log_ret = np.log(full_close/full_close.shift(1))

        portfolio_returns = (full_returns * weights_max_sharpe).sum(axis=1)
        portfolio_log_returns = (log_ret * weights_max_sharpe).sum(axis=1)

        slope, intercept, r, p, std_err = linregress(portfolio_returns, market_date_min.Returns)

        x = np.linspace(np.amin(market_date_min.Returns), np.amax(portfolio_returns))
        y = slope * x + intercept

        regression = go.Figure(go.Scatter(
            x=market_date_min.Returns,
            y=portfolio_returns,
            mode="markers",
            marker={'size': 5, 'color': '#468de2'},
            name="Returns"
        ))

        regression.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            marker={'color': '#e34029'},
            name="Linear Regression"
        ))
        regression.update_layout(title_text='Single Index Model')

        #Setting up drift and random component in relatoin to asset data
        u = portfolio_log_returns.mean()
        var = portfolio_log_returns.var()
        drift = u - (0.5 * var)
        stdev = portfolio_log_returns.std()

        daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(25, 30)))

        #Takes last data point as startpoint point for simulation
        S0 = portfolio_log_returns.iloc[-1]
        price_list = np.zeros_like(daily_returns)

        price_list[0] = S0

        #Applies Monte Carlo simulation in asset
        for t in range(1, 25):
            price_list[t] = price_list[t - 1] * daily_returns[t]

        monte_carlo = go.Figure()

        for i in range(len(price_list)):
            monte_carlo.add_trace(go.Scatter(
                x=[i for i in range(31)],
                y=price_list[i],
            ))


        return regression, monte_carlo
    else:
        return go.Figure(), go.Figure()


if __name__ == '__main__':
    app.run_server(debug=True)