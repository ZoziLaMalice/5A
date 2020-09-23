import numpy as np
import pandas as pd

import plotly.graph_objects as go
from .practicals_functions import trading

import dash
import dash_core_components as dcc
import dash_html_components as html


def get_paulo_return(dfs, stocks, investment, weights_max_sharpe):
    date_min = max([dfs[stock].Datetime.min() for stock in dfs])
    for df in dfs:
        dfs[df] = dfs[df][dfs[df].Datetime >= date_min]
        dfs[df].reset_index(inplace=True)
        dfs[df] = dfs[df][['Datetime', 'Close']]
        dfs[df]['Log_Returns'] = np.log(dfs[df].Close/dfs[df].Close.shift(1))
        dfs[df] = dfs[df].dropna()

    stock_inv = [investment * i for i in weights_max_sharpe]
    result = 0

    buys = {}
    sells = {}

    for (df, inv) in zip(dfs, stock_inv):
        r, b, s = trading(dfs[df], inv, 0.04, 0.01)
        result += r
        buys[df] = b
        sells[df] = s

    children = html.Div([
                dcc.Dropdown(
                    id='paulo-stocks',
                    options=[{'label': label, 'value': value} for value, label in stocks.items()],
                ),
                ])

    return result, children, buys, sells



def plot_paulo_investment(df, buys, sells, stock, app_color):
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

    fig.update_layout(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
    )

    return fig