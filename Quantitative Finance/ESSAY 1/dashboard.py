import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import csv
import html5lib
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

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

fig = px.line(dfs['GOOG'], x='Date', y='Close', title="Google Close Prices (2Y)")
fig.update_xaxes(rangeslider_visible=True)


app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True)  # Turn off reloader if inside Jupyter