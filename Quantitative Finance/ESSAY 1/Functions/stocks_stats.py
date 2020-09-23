import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
from scipy.stats import linregress

import yfinance as yf

import dash
import dash_core_components as dcc
import dash_html_components as html

def plot_stocks_stats(dfs, stocks, app_color):
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
                    options=[{'label': label, 'value': value} for value, label in stocks.items()],
                ),
            ])

    heatmap = ff.create_annotated_heatmap(
        z=correl.values[::-1].round(2),
        x=[stock for stock in dfs],
        y=[stock for stock in dfs][::-1],
        xgap=10,
        ygap=10,
    )
    heatmap.update_layout(
        title_text='Correlation Matrix',
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        )

    covariance = ff.create_annotated_heatmap(
        z=cov.values[::-1].round(6),
        x=[stock for stock in dfs],
        y=[stock for stock in dfs][::-1],
        xgap=10,
        ygap=10,
    )
    covariance.update_layout(
        title_text='Variance - Covariance Matrix',
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        )

    return children, heatmap, covariance


def stocks_regression(df, app_color):
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

    regression.update_layout(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
    )

    return regression, log_ret


def plot_VaR_returns(df, app_color):
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
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
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

    return var


def plot_VaR_log(log_ret, app_color):
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
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
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

    return var