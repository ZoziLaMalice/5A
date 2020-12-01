import pandas as pd
import numpy as np

from numpy.lib.arraysetops import isin
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.offline as pyo

pyo.init_notebook_mode()
pio.templates.default = "plotly_dark"

class zoziDash:
    def __init__(self, height):
        self.height = height
        self.bg_color = 'rgb(221, 236, 255)'


    def plot_heatmap(self, data, round, gap):
        heatmap = ff.create_annotated_heatmap(
                    z=data.values[::-1].round(round),
                    x=list(data.columns),
                    y=list(data.columns)[::-1],
                    xgap=gap,
                    ygap=gap,
                    visible=True
                )
        heatmap.update_layout(
            margin=go.layout.Margin(
                l=10,
                r=10,
                b=30,
                t=30,
                pad=4
            ),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
                )

        return heatmap


    def plot_efficient_frontier(self, simulation, allocation):
        efficient_frontier = go.Figure(go.Scatter(
            x=simulation['volatility'],
            y=simulation['returns'],
            marker=dict(
                size=5,
                color=simulation['sharpe'],
                colorbar=dict(
                    title="Colorbar"
                ),
            ),
            mode="markers",
            name=f"Portfolios ({simulation['num_ports']})"))

        efficient_frontier.add_trace(go.Scatter(
            x=[allocation.loc['Max Sharpe Allocation', 'Volatility']/100],
            y=[allocation.loc['Max Sharpe Allocation', 'Returns']/100],
            marker={'color':'red'},
            mode='markers',
            name='Efficient Portfolio'
        ))

        efficient_frontier.add_trace(go.Scatter(
            x=[allocation.loc['Min Volatility Allocation', 'Volatility']/100],
            y=[allocation.loc['Min Volatility Allocation', 'Returns']/100],
            marker={'color':'green'},
            mode='markers',
            marker_symbol='x',
            name='Min Volatility Portfolio'
        ))

        efficient_frontier.update_layout(
            height=self.height,
            legend=dict(
                yanchor="top",
                y=1.2,
                xanchor="left",
                x=1
                ),
            title='Simulated Portfolio Optimization based on Efficient Frontier'
        )

        return efficient_frontier

    def plot_prices(self, prices):
        if isinstance(prices, pd.Series):
            fig = go.Figure(go.Scatter(x=prices.index, y=prices))

            stock_name = prices.name

            fig.update_layout(
                height=self.height,
                xaxis=dict(automargin=False,
                           rangeslider=dict(visible=False),
                           range=['2020-01-01', '2020-03-30']),
                shapes=[dict(
                    x0='2020-02-15', x1='2020-02-15', y0=0, y1=1, xref='x', yref='paper',
                    line_width=2)],
                annotations=[dict(
                    x='2020-02-17', y=0.95, xref='x', yref='paper',
                    showarrow=False, xanchor='left', text='COVID Begins')],
                yaxis=dict(
                    ticksuffix=' $',
                    range=[prices[(prices.index > '2020-01-01') &
                                (prices.index < '2020-03-30')].min(),
                            prices[(prices.index > '2020-01-01') &
                                (prices.index < '2020-03-30')].max()*1.05]
                ),
                autosize=True,
                margin=go.layout.Margin(
                    l=10,
                    r=10,
                    b=30,
                    t=30,
                    pad=4
                ),
                paper_bgcolor=self.bg_color,
                plot_bgcolor=self.bg_color,
            )

            return fig
        else:
            print('Please, pass a pandas.Series Object.')


    def plot_candles(self, complete_data):
        if isinstance(complete_data, pd.DataFrame):
            fig = go.Figure(data=[go.Candlestick(x=complete_data.index,
                            open=complete_data.Open,
                            high=complete_data.High,
                            low=complete_data.Low,
                            close=complete_data.Close)])

            fig.update_layout(
                height=self.height,
                xaxis=dict(automargin=False,
                           rangeslider=dict(visible=False),
                           range=['2020-01-01', '2020-03-30']),
                shapes = [dict(
                    x0='2020-02-15', x1='2020-02-15', y0=0, y1=1, xref='x', yref='paper',
                    line_width=2)],
                annotations=[dict(
                    x='2020-02-17', y=0.95, xref='x', yref='paper',
                    showarrow=False, xanchor='left', text='COVID Begins')],
                yaxis=dict(
                    ticksuffix=' $',
                    range=[complete_data[(complete_data.index > '2020-01-01') &
                                         (complete_data.index < '2020-03-30')].Open.min(),
                           complete_data[(complete_data.index > '2020-01-01') &
                                         (complete_data.index < '2020-03-30')].Open.max()*1.05]
                ),
                autosize=True,
                margin=go.layout.Margin(
                    l=10,
                    r=10,
                    b=30,
                    t=30,
                    pad=4
                ),
                paper_bgcolor=self.bg_color,
                plot_bgcolor=self.bg_color,
            )

            return fig
        else:
            print('Please, pass a pandas.DataFrame Object.')


    def plot_prices_returns(self, prices, returns):
        colors = np.where(returns < 0, 'red', 'green')

        stock_name = prices.name

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=prices.index,
                    y=prices,
                    name="Closing Prices",
                    yaxis="y"),
            secondary_y=False
        )

        fig.add_trace(
            go.Bar(x=returns.index,
                    y=(returns * 100),
                    name="Returns",
                    marker_color=colors,
                    marker_line_width=0.5,
                    yaxis="y1"),
            secondary_y=True
        )

        fig.update_layout(
            height=self.height,
            xaxis=dict(automargin=False,
                       rangeslider=dict(visible=False)),
            yaxis=dict(
                ticksuffix=' $',
            ),
            yaxis2=dict(
                ticksuffix = '%'
            ),
            shapes = [dict(
                x0='2020-02-15', x1='2020-02-15',
                y0=0, y1=1,
                xref='x', yref='paper',
                line_width=2)],
            annotations=[dict(
                x='2020-02-17', y=0.95,
                xref='x', yref='paper',
                showarrow=False, xanchor='left',
                text='COVID Begins')],
            autosize=True,
            margin=go.layout.Margin(
                l=10,
                r=10,
                b=30,
                t=30,
                pad=4
            ),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            showlegend=False
        )

        return fig


    def plot_CAPM(self, market_returns, portfolio_returns, regression):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=market_returns,
                y=portfolio_returns,
                name='Original Data',
                mode='markers',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=market_returns,
                y=regression.intercept + regression.slope*market_returns,
                name='Fitted Line',
                mode='lines',
            )
        )

        fig.update_layout(
            title_text='CAPM Regression',
            height=self.height,
            )

        return fig


    def plot_FF3(self, df, ff3):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df['MKT'],
                y=df['XsRet'],
                name='Original Data',
                mode='markers',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df['MKT'],
                y=ff3.params.Intercept + ff3.params.MKT*df['MKT'],
                name='Fitted Line',
                mode='lines',
            )
        )

        fig.update_layout(
            title_text='FF3 Regression',
            height=self.height,
        )

        return fig
