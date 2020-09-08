import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
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


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

global ALL_STOCKS
ALL_STOCKS = {'^GSPC': 'S&P500'}

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

# Heatmap
heatmap = go.Figure()

# Variance CoVariance Matrix
covariance = go.Figure()

# Efficient Frontier
efficient_frontier = go.Figure()

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
            dcc.Dropdown(id='stock-drop', value='AAP', clearable=False),
        ], style={'display': 'table-cell', 'width': '75%'}),

        html.Div([
            html.Button(id='add-stock', n_clicks=0, children='Add Stock'),
        ], style={'display': 'table-cell', 'width': '10%', 'padding-left': 25}),

        html.Div([
            html.Button(id='remove-stock', n_clicks=0, children='Remove Stock'),
        ], style={'display': 'table-cell', 'padding-left': 25}),

        html.Hr(),

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

            dcc.Tab(label='Advanced Charts', id='tab-2',  children=[
                html.Div([
                    html.Button(id='load-portfolio', n_clicks=0, children='Load Portfolio',
                    style={'margin': 0, 'position': 'absolute', 'left': '44%'})
                ], style={'padding': 40}),

                html.Div(id='drop-portfolio', children=[
                    dcc.Dropdown(
                        id='portfolio-stocks',
                        options=[{'label': item, 'value': key} for key, item in ALL_STOCKS.items()],
                    ),
                ], style={'padding-top': 10}),

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

                html.Hr(),

                html.Div([
                    dcc.Graph(
                        id='efficient-frontier',
                        figure=efficient_frontier
                    )
                ]),
            ])
        ], style={'padding-top': 30})
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
    Input('remove-stock', 'n_clicks')],
    [State('stock-drop', 'value'),
    State("stock-drop","options")])
def set_stocks_value(btn1, btn2, stock, opt):
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
        del ALL_STOCKS[stock]

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


    # Third Chartyubplots(specs=[[{"secondary_y": True}]])
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

    return fig, fig2, fig3, stats.to_dict('records')


@app.callback(
    Output('drop-portfolio', 'children'),
    Output('correl', 'figure'),
    Output('cov', 'figure'),
    Output('efficient-frontier', 'figure'),
    [Input('load-portfolio', 'n_clicks')]
)
def load_portfolio(n_clicks):
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
        z=correl.values[::-1],
        x=[stock for stock in dfs],
        y=[stock for stock in dfs][::-1],
        xgap=10,
        ygap=10
    )

    covariance = ff.create_annotated_heatmap(
        z=cov.values[::-1],
        x=[stock for stock in dfs],
        y=[stock for stock in dfs][::-1],
        xgap=10,
        ygap=10
    )

    np.random.seed(42)
    num_ports = 6000
    all_weights = np.zeros((num_ports, len(full_close.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

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

    max_sr_ret = ret_arr[sharpe_arr.argmax()]
    max_sr_vol = vol_arr[sharpe_arr.argmax()]

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

    efficient_frontier.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="left",
        x=0.01
    ))


    if 'load-portfolio' in changed_id:
        return children, heatmap, covariance, efficient_frontier
    else:
        return html.Div([' ']), go.Figure(), go.Figure(), go.Figure()

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
    [Input('portfolio-stocks', 'value')],
    State('portfolio-stocks', 'options'))
def update_VaR_chart(stock, opt):
    the_label = [x['label'] for x in opt if x['value'] == stock]

    df = yf.Ticker(stock).history(period="2y")
    df['Returns'] = df.Close.pct_change()
    df = df.iloc[1:]
    df.reset_index(inplace=True)

    var = ff.create_distplot([df.Returns], ['Historical Simulation'], bin_size=.002, show_rug=False, colors=['#1669e9', '#e4ed1e'])

    # var.add_trace(go.Scatter(x=[np.arange(df.Returns.sort_values(ascending=True).quantile(0.05), df.Returns.min(), -0.1)],
    # y=[norm.pdf(np.linspace(np.mean(df.Returns) - 3*np.std(df.Returns), np.mean(df.Returns) + 3* np.std(df.Returns), 100), np.mean(df.Returns), np.std(df.Returns))],
    # fill='tonexty'))

    # var.add_trace(go.Scatter(
    #     x=[-0.15, df.Returns.sort_values(ascending=True).quantile(0.05)],
    #     y=[0, 0],
    #     fill='tozeroy'
    # ))

    var.add_trace(go.Scatter(
        mode= "markers+text",
        text="VaR",
        name="Value at Risk 95%",
        x=[df.Returns.sort_values(ascending=True).quantile(0.05)],
        y=[0],
        marker={"size": 20, 'color': "#f12828"},
        textposition= 'bottom center'
    ))

    # var = go.Figure()

    # var.add_trace(go.Histogram(
    #     x=df.Returns,
    #     nbinsx=1,
    #     histnorm='probability density'
    #     ))

    # var.add_trace(go.Scatter(
    #     x=np.linspace(np.mean(df.Returns) - 3*np.std(df.Returns), np.mean(df.Returns) + 3* np.std(df.Returns), 100),
    #     y=norm.pdf(np.linspace(np.mean(df.Returns) - 3*np.std(df.Returns), np.mean(df.Returns) + 3* np.std(df.Returns), 100), np.mean(df.Returns), np.std(df.Returns)),
    # ))

    # var.add_trace(go.Scatter(
    #     x=[-0.15, df.Returns.sort_values(ascending=True).quantile(0.05)],
    #     y=[0, 0],
    #     fill='tonexty'
    # ))


    var.update_layout(
        width=1400,
        height=600,
    )

    return var


if __name__ == '__main__':
    app.run_server(debug=True)