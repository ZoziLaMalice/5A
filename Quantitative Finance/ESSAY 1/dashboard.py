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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

global ALL_STOCKS
ALL_STOCKS = []

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


# stocks = ['AAP', 'AMZN']

# dfs = {}
# for stock in stocks:
#     dfs[stock] = yf.Ticker(stock).history(period="2y")

# for df in dfs:
#     dfs[df]['Returns'] = (dfs[df].Open - dfs[df].Open.shift(1)) / dfs[df].Open.shift(1)
#     dfs[df] = dfs[df].iloc[1:, [0,1,2,3,7]]
#     dfs[df].reset_index(inplace=True)
#     dfs[df]["Color"] = np.where(dfs[df]['Returns'] < 0, 'red', 'green')

# stats = pd.DataFrame(
#     {
#         'Stock': [df for df in dfs],
#         'Std': [dfs[df].Returns.std() for df in dfs],
#         'Mean': [dfs[df].Returns.mean() for df in dfs],
#         'Min': [dfs[df].Returns.min() for df in dfs],
#         'Max': [dfs[df].Returns.max() for df in dfs],
#         'Kurtosis': [dfs[df].Returns.kurtosis() for df in dfs],
#         'Skewness': [dfs[df].Returns.skew() for df in dfs],
#     },
#     index=[df for df in dfs]
# )

stats = pd.DataFrame(
    {
        'Stock': [0],
        'Std': [0],
        'Mean': [0],
        'Min': [0],
        'Max': [0],
        'Kurtosis': [0],
        'Skewness': [0],
    },
    index=[0]
)

# First Chart
first = go.Figure()

# for df in dfs:
#     first.add_trace(
#         go.Scatter(
#             x = dfs[df].Date,
#             y = dfs[df].Close,
#             name = df
#         )
#     )

# first.update_layout(
#     width=1200,
#     height=600,
#     xaxis = dict(
#         rangeslider = {'visible': True},
#     ),
#     title=f'{stock} Analysis during the COVID',
#     yaxis_title='Stocks',
#     shapes = [dict(
#         x0='2020-02-15', x1='2020-02-15', y0=0, y1=1, xref='x', yref='paper',
#         line_width=2)],
#     annotations=[dict(
#         x='2020-02-17', y=0.95, xref='x', yref='paper',
#         showarrow=False, xanchor='left', text='COVID Begins')],
#     yaxis=dict(
#     ticksuffix=' $'
#     ),
# )


# Second Analysis
second = go.Figure()
# second = go.Figure(data=[go.Candlestick(x=df.Date,
#                 open=df.Open,
#                 high=df.High,
#                 low=df.Low,
#                 close=df.Close)])

# second.update_layout(
#     width=1200,
#     height=800,
#     title='Amazon Analysis during the COVID',
#     yaxis_title='Stocks',
#     shapes = [dict(
#         x0='2020-02-15', x1='2020-02-15', y0=0, y1=1, xref='x', yref='paper',
#         line_width=2)],
#     annotations=[dict(
#         x='2020-02-17', y=0.95, xref='x', yref='paper',
#         showarrow=False, xanchor='left', text='COVID Begins')],
#     yaxis=dict(
#     ticksuffix=' $'
#     ),
#     xaxis = dict(
#         rangeslider = {'visible': False},
#     ),
# )


# Third Analysis
third = go.Figure()
# # Create figure with secondary y-axis
# third = make_subplots(specs=[[{"secondary_y": True}]])

# # Add traces
# third.add_trace(
#     go.Scatter(x=df.Date, y=df.Close, name="Closing Prices", yaxis="y"),
#     secondary_y=False
# )

# third.add_trace(
#     go.Bar(x=df.Date, y=(df.Returns * 100), name="Returns", marker_color=df.Color, yaxis="y1"),
#     secondary_y=True
# )

# # Set x-axis title
# third.update_xaxes(title_text="Date")

# third.update_layout(
#     width=1200,
#     height=600,
#     title='Amazon Analysis during the COVID',
#     yaxis_title='Stocks',
#     shapes = [dict(
#         x0='2020-02-15', x1='2020-02-15', y0=0, y1=1, xref='x', yref='paper',
#         line_width=2)],
#     annotations=[dict(
#         x='2020-02-17', y=0.95, xref='x', yref='paper',
#         showarrow=False, xanchor='left', text='COVID Begins')]
# )

# third.update_layout(
#     yaxis=dict(
#     title="Amazon's Closing Prices",
#     ticksuffix=' $'

#     ),
#     yaxis2=dict(
#         title="Amazon's Returns",
#         ticksuffix = '%'
#     )
# )

# stocks = go.Figure()

# for df in dfs:
#     stocks.add_trace(
#         go.Scatter(
#             x = dfs[df].Date,
#             y = dfs[df].Close,
#             name = df
#         )
#     )

# stocks.update_layout(
#     updatemenus=[go.layout.Updatemenu(
#         active=0,
#         buttons=list(
#             [dict(label = 'All',
#                   method = 'update',
#                   args = [{'visible': [True, True, True]},
#                           {'title': 'The COVID in USA',
#                            'showlegend':True}]),
#              dict(label = 'AMZN',
#                   method = 'update',
#                   args = [{'visible': [True, False, False]},
#                           {'title': 'Amazon Stock during the COVID',
#                            'showlegend':True}]),
#              dict(label = 'GOOG',
#                   method = 'update',
#                   args = [{'visible': [False, True, False]},
#                           {'title': 'Google Stock during the COVID',
#                            'showlegend':True}]),
#              dict(label = 'TSLA',
#                   method = 'update',
#                   args = [{'visible': [False, False, True]},
#                           {'title': 'Tesla Stock during the COVID',
#                            'showlegend':True}]),
#             ])
#         )
#     ])

# stocks.update_layout(
#     width=1200,
#     height=600,
#     title='The COVID in USA (S&P500)',
#     yaxis_title='Stocks',
#     shapes = [dict(
#         x0='2020-02-15', x1='2020-02-15', y0=0, y1=1, xref='x', yref='paper',
#         line_width=2)],
#     annotations=[dict(
#         x='2020-02-17', y=0.95, xref='x', yref='paper',
#         showarrow=False, xanchor='left', text='COVID Begins')],
#     yaxis=dict(
#     ticksuffix=' $'
#     ),
# )

app.layout = html.Div(children=[
    html.H1(children='The S&P500 during the COVID crisis'),

    html.Div(children='''
        The COVID crisis begins around the February 02 2020 in the world (economicly)
    '''),

    dcc.Dropdown(
        id='sectors-drop',
        options=[{'label': k, 'value': k} for k in clean_sp500.keys()],
        value='Consumer Cyclical'
    ),

    html.Hr(),

    dcc.Dropdown(id='stock-drop', value='AAP', clearable=False),

    html.Hr(),

    html.Button(id='add-stock', n_clicks=0, children='Add Stock'),

    html.Button(id='remove-stock', n_clicks=0, children='Remove Stock'),

    html.Div(id='output-stocks'),

    html.Hr(),

    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in stats.columns],
        data=stats.to_dict('records'),
    ),

    dcc.Graph(
        id='first-graph',
        figure=first
    ),

    dcc.Graph(
        id='second-graph',
        figure=second
    ),

    dcc.Graph(
        id='third-graph',
        figure=third
    ),

    # dcc.Graph(
    #     id='stocks-graph',
    #     figure=stocks
    # )
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
    Output('output-stocks', 'children'),
    [Input('add-stock', 'n_clicks'),
    Input('remove-stock', 'n_clicks'),
    Input('stock-drop', 'value')])
def set_stocks_value(btn1, btn2, selection):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'add-stock' in changed_id:
        ALL_STOCKS.append(selection)
    elif 'remove-stock' in changed_id:
        ALL_STOCKS.remove(selection)
    return html.Div(ALL_STOCKS)


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
    df = df.iloc[1:, [0,1,2,3,7]]
    df.reset_index(inplace=True)
    df["Color"] = np.where(df['Returns'] < 0, 'red', 'green')

    stats = pd.DataFrame(
        {
            'Stock': [the_label[0]],
            'Std': [df.Returns.std()],
            'Mean': [df.Returns.mean()],
            'Min': [df.Returns.min()],
            'Max': [df.Returns.max()],
            'Kurtosis': [df.Returns.kurtosis()],
            'Skewness': [df.Returns.skew()],
        },
        index=[0]
    )

    # First Chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = df.Date,
            y = df.Close,
            name = stock
        )
    )

    fig.update_layout(
        width=1200,
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
    fig2 = go.Figure(data=[go.Candlestick(x=df.Date,
                    open=df.Open,
                    high=df.High,
                    low=df.Low,
                    close=df.Close)])

    fig2.update_layout(
        width=1200,
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
    # Create figure with secondary y-axis
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
        width=1200,
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


if __name__ == '__main__':
    app.run_server(debug=True)