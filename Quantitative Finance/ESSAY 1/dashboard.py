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

def bbands(price, window_size=10, num_of_std=5):
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std  = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean, upper_band, lower_band


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

global ALL_STOCKS
ALL_STOCKS = []

global stocks_name
stocks_name = []


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

global covid
covid = pd.read_csv('covid_USA.csv')
covid.Date = pd.to_datetime(covid.Date)

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


# Second Analysis
second = go.Figure()


# Third Analysis
third = go.Figure()


app.layout = html.Div([
        html.Div([
            html.H1('The S&P500 during the COVID crisis'),
            html.Div('This dashboard shows some financials charts about S&P500 stocks, especially during the COVID'),
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


    if 'add-stock' in changed_id:
        ALL_STOCKS.append(stock)
        stocks_name.append([x['label'] for x in opt if x['value'] == stock][0])

        dfs = {}
        for stock in ALL_STOCKS:
            dfs[stock] = yf.Ticker(stock).history(period="2y")
            dfs[stock]['Returns'] = (dfs[stock].Open - dfs[stock].Open.shift(1)) / dfs[stock].Open.shift(1)
            dfs[stock] = dfs[stock].iloc[1:]
            dfs[stock].reset_index(inplace=True)
            dfs[stock]["Color"] = np.where(dfs[stock]['Returns'] < 0, 'red', 'green')

        stats = pd.DataFrame(
            {
                'Stock': [name for name in stocks_name],
                'Std': [dfs[df].Returns.std() for df in dfs],
                'Mean': [dfs[df].Returns.mean() for df in dfs],
                'Min': [dfs[df].Returns.min() for df in dfs],
                'Max': [dfs[df].Returns.max() for df in dfs],
                'Kurtosis': [dfs[df].Returns.kurtosis() for df in dfs],
                'Skewness': [dfs[df].Returns.skew() for df in dfs],
            },
            index=[df for df in ALL_STOCKS]
        )

    elif 'remove-stock' in changed_id:
        ALL_STOCKS.remove(stock)
        stocks_name.remove([x['label'] for x in opt if x['value'] == stock][0])

        dfs = {}
        for stock in ALL_STOCKS:
            dfs[stock] = yf.Ticker(stock).history(period="2y")
            dfs[stock]['Returns'] = (dfs[stock].Open - dfs[stock].Open.shift(1)) / dfs[stock].Open.shift(1)
            dfs[stock] = dfs[stock].iloc[1:]
            dfs[stock].reset_index(inplace=True)
            dfs[stock]["Color"] = np.where(dfs[stock]['Returns'] < 0, 'red', 'green')

        stats = pd.DataFrame(
            {
                'Stock': [name for name in stocks_name],
                'Std': [dfs[df].Returns.std() for df in dfs],
                'Mean': [dfs[df].Returns.mean() for df in dfs],
                'Min': [dfs[df].Returns.min() for df in dfs],
                'Max': [dfs[df].Returns.max() for df in dfs],
                'Kurtosis': [dfs[df].Returns.kurtosis() for df in dfs],
                'Skewness': [dfs[df].Returns.skew() for df in dfs],
            },
            index=[df for df in ALL_STOCKS]
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
    df['Returns'] = (df.Open - df.Open.shift(1)) / df.Open.shift(1)
    df = df.iloc[1:]
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


if __name__ == '__main__':
    app.run_server(debug=True)