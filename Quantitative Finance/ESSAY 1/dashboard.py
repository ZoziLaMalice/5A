import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table
import yfinance as yf
import pandas as pd
import csv
import html5lib
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
import re
from scipy.stats import linregress, norm

from functions.practicals_functions import bbands, trading, stats_df, stats_dfs, stats_market, create_df, create_dfs
from functions.twitter_data import get_tweets_by_query
from functions.sentiments_graphs import count_words, preprocess_nltk, plot_sentiment, plot_word_count
from functions.basic_graphs import plot_first_graph, plot_second_graph, plot_third_graph
from functions.stocks_stats import plot_VaR_log, plot_VaR_returns, plot_stocks_stats, stocks_regression
from functions.differents_portfolios import get_min_var_portfolio, get_returns, efficient_portfolio, get_equal_weighted
from functions.stats_efficient_portfolio import portfolio_regression, plot_monte_carlo, get_data
from functions.paulo_investment import get_paulo_return, plot_paulo_investment

buys = {}
sells = {}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

global ALL_STOCKS
ALL_STOCKS = {'^GSPC': 'S&P500'}

# colors for plots
global chart_colors
chart_colors = [
    '#664DFF',
    '#893BFF',
    '#3CC5E8',
    '#2C93E8',
    '#0BEBDD',
    '#0073FF',
    '#00BDFF',
    '#A5E82C',
    '#FFBD42',
    '#FFCA30'
]
# global color setting
global app_color
app_color = {
    "graph_bg": "rgb(221, 236, 255)",
    "graph_line": "rgb(8, 70, 151)",
    "graph_font":"rgb(2, 29, 65)"
}

weights_max_sharpe = []

with open('./sp500_sectors.csv', newline='') as f:
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
covid = pd.read_csv('./covid_USA.csv')
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
    plot_bgcolor=app_color["graph_bg"],
    paper_bgcolor=app_color["graph_bg"],
)

global BASIC_LAYOUT
BASIC_LAYOUT = go.Layout(plot_bgcolor=app_color["graph_bg"], paper_bgcolor=app_color["graph_bg"],)

# Layout
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
                            figure=go.Figure(layout=BASIC_LAYOUT)
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
                        figure=go.Figure(layout=BASIC_LAYOUT)
                    ),
                ]),

                html.Div([
                    dcc.Graph(
                        id='third-graph',
                        figure=go.Figure(layout=BASIC_LAYOUT)
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
                        figure=go.Figure(layout=BASIC_LAYOUT)
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
                        figure=go.Figure(layout=BASIC_LAYOUT)
                    )
                ]),

                html.Hr(),

                html.Div([
                    dcc.Graph(
                        id='correl',
                        figure=go.Figure(layout=BASIC_LAYOUT)
                    )
                ]),

                html.Hr(),

                html.Div([
                    dcc.Graph(
                        id='cov',
                        figure=go.Figure(layout=BASIC_LAYOUT)
                    )
                ]),
            ]),

            dcc.Tab(label='Stocks Sentiments Analysis', id='tab-3', children=[

                html.Div([
                    html.Button(id='load-twitter-data', n_clicks=0, children='Load Twitter Data',
                    style={'display': 'table-cell'})
                ], style={'padding': 40, 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'table'}),

                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='tweet-count',
                            figure=go.Figure(layout=BASIC_LAYOUT)
                        ),
                    ]),
                ]),

                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='word-count',
                            figure=go.Figure(layout=BASIC_LAYOUT)
                        ),
                    ], className="two-thirds column"),
                    html.Div([
                        dcc.Graph(
                            id='sentiment',
                            figure=go.Figure(layout=BASIC_LAYOUT)
                        ),
                    ], className="one-third column"),
                ]),

            ]),

            dcc.Tab(label='Differents Portfolios', id='tab-4', children=[

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
                        figure=go.Figure(layout=BASIC_LAYOUT)
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
                        figure=go.Figure(layout=BASIC_LAYOUT)
                    )
                ]),

                html.Div(id='paulo-portfolio-text', children=[

                ], style={'padding-top': 15, 'font-size': 20, 'text-align': 'center'}),

                html.Hr(),

            ]),

            dcc.Tab(label='Stats on Efficient Portfolio', id='tab-5', children=[

                html.Div([
                    html.Button(id='load-stocks-2', n_clicks=0, children='Load Stocks',
                    style={'display': 'table-cell'})
                ], style={'padding': 40, 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'table'}),

                html.Div([
                    dcc.Graph(
                        id='regression-portfolio',
                        figure=go.Figure(layout=BASIC_LAYOUT)
                    )
                ]),

                html.Hr(),

                html.Div([
                    dcc.Graph(
                        id='monte-carlo-portfolio',
                        figure=go.Figure(layout=BASIC_LAYOUT)
                    )
                ]),

            ]),
        ], style={'padding-top': 30}),
], style={'background-color': app_color["graph_bg"]})


# Callbacks
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

    stats = stats_market(market)


    if 'add-stock' in changed_id:
        ALL_STOCKS.update({stock: [x['label'] for x in opt if x['value'] == stock][0]})

        dfs = create_dfs(ALL_STOCKS)

        stats = stats_dfs(ALL_STOCKS, dfs, market)

    elif 'remove-stock' in changed_id:
        try:
            del ALL_STOCKS[stock]
        except KeyError:
            pass

        dfs = create_dfs(ALL_STOCKS)

        stats = stats_dfs(ALL_STOCKS, dfs, market)

    elif 'remove-market' in changed_id:
        try:
            del ALL_STOCKS['^GSPC']
        except KeyError:
            pass

        dfs = create_dfs(ALL_STOCKS)

        stats = stats_dfs(ALL_STOCKS, dfs, market)

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

    df = create_df(stock)

    stats = stats_df(df, market, the_label)

    df['log_ret'] = np.log(df.Close/df.Close.shift(1))
    df["log_Color"] = np.where(df['log_ret'] < 0, 'red', 'green')

    fig = plot_first_graph(df, stock, the_label, covid, app_color)
    fig2 = plot_second_graph(df, stock, the_label, app_color)
    fig3 = plot_third_graph(df, stock, the_label, app_color)

    return fig, fig2, fig3, stats.to_dict('records')


@app.callback(
    Output('drop-portfolio', 'children'),
    Output('correl', 'figure'),
    Output('cov', 'figure'),
    [Input('load-stocks', 'n_clicks')]
)
def load_stocks(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'load-stocks' in changed_id:
        dfs = create_dfs(ALL_STOCKS)

        date_min = max([dfs[stock].Date.min() for stock in dfs])
        for df in dfs:
            dfs[df] = dfs[df][dfs[df].Date >= date_min]
            dfs[df].reset_index(inplace=True)

        children, heatmap, covariance = plot_stocks_stats(dfs, ALL_STOCKS, app_color)

        return children, heatmap, covariance
    else:
        children = html.Div([
                    dcc.Dropdown(
                        id='portfolio-stocks',
                        options=[{'label': label, 'value': value} for value, label in ALL_STOCKS.items()],
                    ),
                ])
        return children, go.Figure(layout=BASIC_LAYOUT), go.Figure(layout=BASIC_LAYOUT)

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

    regression, log_ret = stocks_regression(df, app_color)

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'var-normal' in changed_id:
        var = plot_VaR_returns(df, app_color)

        return var, {'display': 'table-cell', 'margin-left': 'auto', 'margin-right': 'auto', 'background-color': 'rgba(150, 220, 240, 0.5'}, {'display': 'table-cell', 'margin-left': 'auto', 'margin-right': 'auto'}, regression
    elif 'var-log' in changed_id:
        var = plot_VaR_log(log_ret, app_color)
        return var, {'display': 'table-cell', 'margin-left': 'auto', 'margin-right': 'auto'}, {'display': 'table-cell', 'margin-left': 'auto', 'margin-right': 'auto', 'background-color': 'rgba(150, 220, 240, 0.5)'}, regression
    else:
        return go.Figure(layout=BASIC_LAYOUT), {'display': 'table-cell', 'margin-left': 'auto', 'margin-right': 'auto'}, {'display': 'table-cell', 'margin-left': 'auto', 'margin-right': 'auto'}, regression

@app.callback(
    Output('equal-weighted-portfolio', 'children'),
    Output('weights-equal', 'data'),
    [Input('equal-weighted', 'n_clicks')],
)
def equal_weighted(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'equal-weighted' in changed_id:
        dfs = create_dfs(ALL_STOCKS)

        date_min = max([dfs[stock].Date.min() for stock in dfs])
        for df in dfs:
            dfs[df] = dfs[df][dfs[df].Date >= date_min]
            dfs[df].reset_index(inplace=True)

        ret_arr, vol_arr, data = get_equal_weighted(dfs, ALL_STOCKS)

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
        dfs = create_dfs(ALL_STOCKS)

        date_min = max([dfs[stock].Date.min() for stock in dfs])
        for df in dfs:
            dfs[df] = dfs[df][dfs[df].Date >= date_min]
            dfs[df].reset_index(inplace=True)

        global weights_max_sharpe
        efficient_frontier, children, data, weights_max_sharpe = efficient_portfolio(dfs, ALL_STOCKS, app_color)

        return efficient_frontier, children, data.to_dict('records')
    else:
        return go.Figure(layout=BASIC_LAYOUT), '', []

@app.callback(
    Output('min-var-portfolio', 'children'),
    Output('weights-min-var', 'data'),
    [Input('submit-investment', 'n_clicks')],
    [State('investment', 'value')]
)
def min_var_portfolio(n_clicks, investment):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-investment' in changed_id:
        dfs = create_dfs(ALL_STOCKS)

        date_min = max([dfs[stock].Date.min() for stock in dfs])
        for df in dfs:
            dfs[df] = dfs[df][dfs[df].Date >= date_min]
            dfs[df].reset_index(inplace=True)

        min_var_ret, min_var_vol, min_var, data = get_min_var_portfolio(dfs, investment, ALL_STOCKS, app_color)

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

        global buys, sells

        result, children, buys, sells = get_paulo_return(dfs, ALL_STOCKS, investment, weights_max_sharpe)

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

    fig = plot_paulo_investment(df, buys, sells, stock, app_color)

    return fig


@app.callback(
    Output('regression-portfolio', 'figure'),
    Output('monte-carlo-portfolio', 'figure'),
    [Input('load-stocks-2', 'n_clicks')]
)
def load_stocks_2(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'load-stocks-2' in changed_id:
        dfs = create_dfs(ALL_STOCKS)

        date_min = max([dfs[stock].Date.min() for stock in dfs])
        for df in dfs:
            dfs[df] = dfs[df][dfs[df].Date >= date_min]
            dfs[df].reset_index(inplace=True)

        regression = portfolio_regression(dfs, date_min, weights_max_sharpe, app_color)
        monte_carlo = plot_monte_carlo(dfs, date_min, weights_max_sharpe, app_color)

        return regression, monte_carlo
    else:
        return go.Figure(layout=BASIC_LAYOUT), go.Figure(layout=BASIC_LAYOUT)


@app.callback(
    Output('tweet-count', 'figure'),
    Output('word-count', 'figure'),
    Output('sentiment', 'figure'),
    [Input('load-twitter-data', 'n_clicks')]
)
def load_twitter_data(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'load-twitter-data' in changed_id:
        twitter_data = {}
        tweet_count = go.Figure()
        for stock, name in ALL_STOCKS.items():
            twitter_data[stock] = get_tweets_by_query(stock, True)
            tweet_count.add_trace(
                go.Scatter(
                    x=twitter_data[stock].groupby('Date').count().index,
                    y=twitter_data[stock].groupby('Date').count().Text,
                    name=name,
                )
            )

        tweet_count.update_layout(
                plot_bgcolor=app_color["graph_bg"],
                paper_bgcolor=app_color["graph_bg"],
        )

        word_count = plot_word_count(ALL_STOCKS, twitter_data, chart_colors, app_color)

        sentiment = plot_sentiment(ALL_STOCKS, twitter_data, chart_colors, app_color)

        return tweet_count, word_count, sentiment
    else:
        return go.Figure(layout=BASIC_LAYOUT), go.Figure(layout=BASIC_LAYOUT), go.Figure(layout=BASIC_LAYOUT)


if __name__ == '__main__':
    app.run_server(debug=True)
