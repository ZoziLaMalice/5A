import pandas as pd
import numpy as np

from numpy.lib.arraysetops import isin
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyo

pyo.init_notebook_mode()

class ecoPlot:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def indexed_price(self, df):
        fig = go.Figure()

        colors = ['#00cc96', '#EF553B', '#636efa']

        df = df.apply(lambda x: x+1)
        df.iloc[0, :] = 100
        df = df.cumprod(axis=0)

        for name, color in zip(df.columns, colors):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[name],
                    name=name,
                    marker=dict(color=color)
                )
            )

        fig.update_layout(
            width=self.width,
            height=self.height,
            title='Green Vs. Bad Portfolio & S&P500',
            yaxis_title='Indexed prices',
            yaxis=dict(
                ticksuffix=' $'
            ),
        )

        return fig
