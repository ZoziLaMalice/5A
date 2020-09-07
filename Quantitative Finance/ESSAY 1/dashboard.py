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
market['Returns'] = (market.Open - market.Open.shift(1)) / market.Open.shift(1)
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
        'VaR 95% HS': [np.percentile(market.Returns, 0.05)],
        'VaR 95% DN': [norm.cdf(norm.ppf(0.95)) * market.Returns.std()],
        'Systemic Risk': [linregress(market.Returns, market.Returns).slope**2 * market.Returns.var()]
    },
    index=[0]
).round(6)

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
                ])
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
            'VaR 95% HS': [np.percentile(market.Returns, 0.05)],
            'VaR 95% DN': [norm.cdf(norm.ppf(0.95)) * market.Returns.std()],
            'Systemic Risk': [linregress(market.Returns, market.Returns).slope**2 * market.Returns.var()]
        },
        index=[0]
    ).round(6)


    if 'add-stock' in changed_id:
        ALL_STOCKS.update({stock: [x['label'] for x in opt if x['value'] == stock][0]})

        dfs = {}
        for stock, _ in ALL_STOCKS.items():
            dfs[stock] = yf.Ticker(stock).history(period="2y")
            dfs[stock]['Returns'] = (dfs[stock].Open - dfs[stock].Open.shift(1)) / dfs[stock].Open.shift(1)
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
                'Alpha': [linregress(dfs[df].Returns, market.Returns).intercept for df in dfs],
                'Beta': [linregress(dfs[df].Returns, market.Returns).slope for df in dfs],
                'VaR 95% HS': [np.percentile(dfs[df].Returns, 0.05) for df in dfs],
                'VaR 95% DN': [norm.cdf(norm.ppf(0.95)) * dfs[df].Returns.std() for df in dfs],
                'Systemic Risk': [linregress(dfs[df].Returns, market.Returns).slope**2 * market.Returns.var() for df in dfs]
            },
            index=[df for _, df in ALL_STOCKS.items()]
        ).round(6)

    elif 'remove-stock' in changed_id:
        del ALL_STOCKS[stock]

        dfs = {}
        for stock, _ in ALL_STOCKS.items():
            dfs[stock] = yf.Ticker(stock).history(period="2y")
            dfs[stock]['Returns'] = (dfs[stock].Open - dfs[stock].Open.shift(1)) / dfs[stock].Open.shift(1)
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
                'Alpha': [linregress(dfs[df].Returns, market.Returns).intercept for df in dfs],
                'Beta': [linregress(dfs[df].Returns, market.Returns).slope for df in dfs],
                'VaR 95% HS': [np.percentile(dfs[df].Returns, 0.05) for df in dfs],
                'VaR 95% DN': [norm.cdf(norm.ppf(0.95)) * dfs[df].Returns.std() for df in dfs],
                'Systemic Risk': [linregress(dfs[df].Returns, market.Returns).slope**2 * market.Returns.var() for df in dfs]
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
    df['Returns'] = (df.Open - df.Open.shift(1)) / df.Open.shift(1)
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
            'Alpha': [linregress(df.Returns, market.Returns).intercept],
            'Beta': [linregress(df.Returns, market.Returns).slope],
            'VaR 95% HS': [np.percentile(df.Returns, 0.05)],
            'VaR 95% DN': [norm.cdf(norm.ppf(0.95)) * df.Returns.std()],
            'Systemic Risk': [linregress(df.Returns, market.Returns).slope**2 * market.Returns.var()]
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
    [Input('load-portfolio', 'n_clicks')]
)
def load_portfolio(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'load-portfolio' in changed_id:
        return html.Div([
            dcc.Dropdown(
                id='portfolio-stocks',
                options=[{'label': label, 'value': value} for value, label in ALL_STOCKS.items()],
            ),
        ])

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
    df['Returns'] = (df.Open - df.Open.shift(1)) / df.Open.shift(1)
    df = df.iloc[1:]
    df.reset_index(inplace=True)
    var = go.Figure()
    var.add_trace(go.Histogram(
            x=df.Returns*100
    ))

    var.add_trace(go.Scatter(
        mode= "lines+markers+text",
        text="VaR",
        name="Value at Risk",
        x=[np.percentile(df.Returns*100, 0.05)],
        y=[0],
        marker={"size": 20},
        textposition= 'bottom center'
    ))
    print(stock, the_label)
    return var
    # trace1 = {
    # "name": "Fitted normal distribution",
    # "type": "scatter",
    # "x": [-0.019559895712306395, -0.01903855131815702, -0.018517206924007643, -0.017995862529858266, -0.01747451813570889, -0.016953173741559517, -0.01643182934741014, -0.015910484953260765, -0.015389140559111388, -0.014867796164962012, -0.014346451770812636, -0.013825107376663261, -0.013303762982513885, -0.012782418588364508, -0.012261074194215134, -0.011739729800065758, -0.011218385405916381, -0.010697041011767005, -0.010175696617617629, -0.009654352223468254, -0.009133007829318878, -0.008611663435169501, -0.008090319041020127, -0.0075689746468707506, -0.007047630252721374, -0.006526285858571998, -0.006004941464422622, -0.005483597070273247, -0.004962252676123871, -0.0044409082819744945, -0.00391956388782512, -0.0033982194936757436, -0.0028768750995263673, -0.002355530705376991, -0.0018341863112276147, -0.0013128419170782384, -0.000791497522928862, -0.0002701531287794892, 0.0002511912653698871, 0.0007725356595192634, 0.0012938800536686397, 0.001815224447818016, 0.0023365688419673923, 0.0028579132361167686, 0.0033792576302661415, 0.0039006020244155178, 0.004421946418564894, 0.00494329081271427, 0.005464635206863647, 0.005985979601013023, 0.006507323995162399, 0.007028668389311776, 0.007550012783461152, 0.008071357177610525, 0.008592701571759901, 0.009114045965909277, 0.009635390360058654, 0.01015673475420803, 0.010678079148357406, 0.011199423542506783, 0.011720767936656155, 0.012242112330805535, 0.012763456724954908, 0.013284801119104288, 0.01380614551325366, 0.014327489907403033, 0.014848834301552413, 0.015370178695701786, 0.015891523089851166, 0.01641286748400054, 0.01693421187814992, 0.01745555627229929, 0.01797690066644867, 0.018498245060598044, 0.019019589454747417, 0.019540933848896797, 0.02006227824304617, 0.02058362263719555, 0.021104967031344922, 0.021626311425494302, 0.022147655819643675, 0.022669000213793047, 0.023190344607942427, 0.0237116890020918, 0.02423303339624118, 0.024754377790390553, 0.025275722184539932, 0.025797066578689305, 0.026318410972838678, 0.026839755366988058, 0.02736109976113743, 0.02788244415528681, 0.028403788549436183, 0.028925132943585563, 0.029446477337734936, 0.029967821731884316, 0.03048916612603369, 0.03101051052018306, 0.03153185491433244, 0.032053199308481814, 0.032574543702631194], 
    # "y": [0.1285998709974904, 0.17578526227475214, 0.23837318181769354, 0.3206751669941835, 0.4279630455863012, 0.5666047529764865, 0.7441957079092862, 0.9696770398275635, 1.2534299502515125, 1.607333613474029, 2.0447724734872765, 2.5805778011115827, 3.2308881824676416, 4.012914461232168, 4.944596771697046, 6.044144843507723, 7.329457814663846, 8.817426326796141, 10.523127528356918, 12.458932456061957, 14.633554624171367, 17.051077900799207, 19.710010149876805, 22.60241585709473, 25.713185222960764, 29.01949825089758, 32.490539589474594, 36.08751294285306, 39.76399267911117, 43.46663512653501, 47.1362536045912, 50.70924049313906, 54.11929790570787, 57.2994173198142, 60.1840294460824, 62.71123026508556, 64.82497891814933, 66.47715907501016, 67.62939814820672, 68.25454839451042, 68.33775009904335, 67.87701870573449, 66.88332350210283, 65.38015350229053, 63.402594517445976, 60.99596804785198, 58.2141057143248, 55.11735091056713, 51.77039107978278, 48.24002889218985, 44.59299857110171, 40.89392516341894, 37.203510621961534, 33.57701246925938, 30.06306008083197, 26.7028318803451, 23.529595542048366, 20.568594028926285, 17.83724406112596, 15.345601170681517, 13.097037251028336, 11.089072492183966, 9.3143035270822, 7.761372970202332, 6.415931598739888, 5.2615524068974535, 4.280564836782058, 3.454786887465066, 2.7661418633456614, 2.197154719041724, 1.7313299211933266, 1.3534182646174748, 1.04958408143547, 0.8074868182538972, 0.6162921719782214, 0.4666280754435366, 0.3505000489720698, 0.2611790299600404, 0.1930729934113766, 0.14159168798455676, 0.1030118038844448, 0.07434798957073578, 0.05323343246821949, 0.037812266914975204, 0.026644891951414205, 0.018626369580271626, 0.012917410827051983, 0.008887011298169382, 0.006065533368759336, 0.004106910741710634, 0.002758636866586088, 0.001838259467669355, 0.0012152121158017426, 0.0007969487139266641, 0.0005184915240935834, 0.0003346462419423725, 0.0002142709263188337, 0.0001361048326324508, 8.576633428420223e-05, 5.361584498352188e-05, 3.3250828054666355e-05]
    # }

# fig = Figure(data=data)

# trace2 = {
#   "mode": "lines+markers+text", 
#   "name": "Value at Risk", 
#   "text": "Value at Risk -0.012", 
#   "type": "scatter", 
#   "x": [-0.012261074194215134], 
#   "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], 
#   "marker": {"size": 20},
#   "textposition": "bottom"
# }


if __name__ == '__main__':
    app.run_server(debug=True)