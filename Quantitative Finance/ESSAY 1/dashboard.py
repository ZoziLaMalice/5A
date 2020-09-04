import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

with open('sp500_sectors.csv', newline='') as f:
    reader = csv.reader(f)
    sp500_s = dict(reader)

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
    for key, item in sp500_s.items():
        if sector == item:
            clean_sp500[sector] += [key]


stocks = [
    'AMZN',
    'GOOG',
    'TSLA'
]

dfs = {}
for stock in stocks:
    dfs[stock] = yf.Ticker(stock).history(period="2y")

for df in dfs:
    dfs[df]['Returns'] = (dfs[df].Open - dfs[df].Open.shift(1)) / dfs[df].Open.shift(1)
    dfs[df] = dfs[df].iloc[1:, [0,1,2,3,7]]
    dfs[df].reset_index(inplace=True)

# Google Analysis
goog = go.Figure(data=go.Scatter(x=dfs['GOOG'].Date, y=dfs['GOOG'].Close, name="GOOG's Closing Prices"))

# goog.update_xaxes(rangeslider_visible=True)
goog.update_layout(
    width=1200,
    height=600,
    xaxis = dict(
        rangeslider = {'visible': True},
    ),
    title='GOOG Analysis during the COVID',
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


# Amazon Analysis
amzn = go.Figure(data=[go.Candlestick(x=dfs['AMZN'].Date,
                open=dfs['AMZN'].Open,
                high=dfs['AMZN'].High,
                low=dfs['AMZN'].Low,
                close=dfs['AMZN'].Close)])

amzn.update_layout(
    width=1200,
    height=800,
    title='AMZN Analysis during the COVID',
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


# Tesla Analysis
dfs['TSLA']["Color"] = np.where(dfs['TSLA']['Returns'] < 0, 'red', 'green')

# Create figure with secondary y-axis
tsla = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
tsla.add_trace(
    go.Scatter(x=dfs['TSLA'].Date, y=dfs['TSLA'].Close, name="Close Prices", yaxis="y"),
    secondary_y=False
)

tsla.add_trace(
    go.Bar(x=dfs['TSLA'].Date, y=(dfs['TSLA'].Returns * 100), name="Returns", marker_color=dfs['TSLA'].Color, yaxis="y1"),
    secondary_y=True
)

# Set x-axis title
tsla.update_xaxes(title_text="Date")

tsla.update_layout(
    width=1200,
    height=600,
    title='TSLA Analysis during the COVID',
    yaxis_title='Stocks',
    shapes = [dict(
        x0='2020-02-15', x1='2020-02-15', y0=0, y1=1, xref='x', yref='paper',
        line_width=2)],
    annotations=[dict(
        x='2020-02-17', y=0.95, xref='x', yref='paper',
        showarrow=False, xanchor='left', text='COVID Begins')]
)

tsla.update_layout(
    yaxis=dict(
    title="TSLA Close's Prices",
    ticksuffix=' $'

    ),
    yaxis2=dict(
        title="TSLA Returns",
        ticksuffix = '%'
    )
)

stocks = go.Figure()

for df in dfs:
    stocks.add_trace(
        go.Scatter(
            x = dfs[df].Date,
            y = dfs[df].Close,
            name = df
        )
    )

stocks.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=list(
            [dict(label = 'All',
                  method = 'update',
                  args = [{'visible': [True, True, True]},
                          {'title': 'The COVID in USA',
                           'showlegend':True}]),
             dict(label = 'AMZN',
                  method = 'update',
                  args = [{'visible': [True, False, False]},
                          {'title': 'Amazon Stock during the COVID',
                           'showlegend':True}]),
             dict(label = 'GOOG',
                  method = 'update',
                  args = [{'visible': [False, True, False]},
                          {'title': 'Google Stock during the COVID',
                           'showlegend':True}]),
             dict(label = 'TSLA',
                  method = 'update',
                  args = [{'visible': [False, False, True]},
                          {'title': 'Tesla Stock during the COVID',
                           'showlegend':True}]),
            ])
        )
    ])

stocks.update_layout(
    width=1200,
    height=600,
    title='The COVID in USA (S&P500)',
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

app.layout = html.Div(children=[
    html.H1(children='The S&P500 during the COVID crisis'),

    html.Div(children='''
        The COVID crisis begins around the February 02 2020 in the world (economicly)
    '''),

    dcc.Dropdown(
        id='sectors-drop',
        options=[{'label': k, 'value': k} for k in clean_sp500.keys()],
        value='Utilities'
    ),

    html.Hr(),

    dcc.Dropdown(id='stock-drop'),

    dcc.Graph(
        id='goog-graph',
        figure=goog
    ),

    dcc.Graph(
        id='amzn-graph',
        figure=amzn
    ),

    dcc.Graph(
        id='tsla-graph',
        figure=tsla
    ),

    dcc.Graph(
        id='stocks-graph',
        figure=stocks
    )
])

@app.callback(
    Output('stock-drop', 'options'),
    [Input('sectors-drop', 'value')])
def set_cities_options(selected_country):
    return [{'label': i, 'value': i} for i in clean_sp500[selected_country]]


@app.callback(
    Output('stock-drop', 'value'),
    [Input('stock-drop', 'options')])
def set_cities_value(available_options):
    return available_options[0]['value']


@app.callback(
    Output('goog-graph', 'figure'),
    Output('amzn-graph', 'figure'),
    Output('tsla-graph', 'figure'),
    [Input('stock-drop', 'value')]
)
def update_output_div(stock):

    df = yf.Ticker(stock).history(period="2y")

    df['Returns'] = (df.Open - df.Open.shift(1)) / df.Open.shift(1)
    df = df.iloc[1:, [0,1,2,3,7]]
    df.reset_index(inplace=True)

    # First Chart
    fig = go.Figure(data=
        (go.Scatter(x=df.Date, y=df.Close, name=f"{stock}'s Closing Prices")
    ))

    fig.update_layout(
        width=1200,
        height=600,
        xaxis = dict(
            rangeslider = {'visible': True},
        ),
        title=f'{stock} Analysis during the COVID',
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
    fig2 = go.Figure(data=[go.Candlestick(x=df.Date,
                    open=df.Open,
                    high=df.High,
                    low=df.Low,
                    close=df.Close)])

    fig2.update_layout(
        width=1200,
        height=800,
        title=f'{stock} Analysis during the COVID',
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


    # Third Chart
    df["Color"] = np.where(df['Returns'] < 0, 'red', 'green')

    # Create figure with secondary y-axis
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig3.add_trace(
        go.Scatter(x=df.Date, y=df.Close, name="Close Prices", yaxis="y"),
        secondary_y=False
    )

    fig3.add_trace(
        go.Bar(x=df.Date, y=(df.Returns * 100), name="Returns", marker_color=df.Color, yaxis="y1"),
        secondary_y=True
    )

    # Set x-axis title
    fig3.update_xaxes(title_text="Date")

    fig3.update_layout(
        width=1200,
        height=600,
        title=f'{stock} Analysis during the COVID',
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
        title=f"{stock} Close's Prices",
        ticksuffix=' $'

        ),
        yaxis2=dict(
            title=f"{stock} Returns",
            ticksuffix = '%'
        )
    )

    return fig, fig2, fig3

if __name__ == '__main__':
    app.run_server(debug=True)