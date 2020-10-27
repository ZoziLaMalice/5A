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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1('The S&P500 during the COVID crisis'),
            html.H4(
                '''This dashboard shows some financials
                charts about S&P500 stocks,
                especially during the COVID'''
                ),
        ]),
        html.Div([
            html.Div([
                html.H3('Here my stocks selection')
            ]),
            html.Div([
                html.H5('Sectors: Materials'),
                html.H6('Stocks: NME'),
                html.P('This stocks represent....')
            ])
        ])
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
