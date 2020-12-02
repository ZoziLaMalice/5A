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

class zoziPlot:
    def __init__(self, width, height):
        self.width = width
        self.height = height


    def plot_heatmap(self, data, title, round, gap):
        heatmap = ff.create_annotated_heatmap(
                    z=data.values[::-1].round(round),
                    x=list(data.columns),
                    y=list(data.columns)[::-1],
                    xgap=gap,
                    ygap=gap,
                    visible=True
                ).update_layout(
                    title_text=title,
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
            width=self.width,
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
            fig = px.line(x=prices.index, y=prices)
            fig.update_xaxes(rangeslider_visible=True)

            stock_name = prices.name

            fig.update_layout(
                width=self.width,
                height=self.height,
                title=f'{stock_name} Analysis during the COVID',
                yaxis_title='Adj Closing Price',
                xaxis_title='Date',
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

            return fig
        else:
            print('Please, pass a pandas.Series Object.')


    def plot_candles(self, complete_data, stock_name):
        if isinstance(complete_data, pd.DataFrame):
            fig = go.Figure(data=[go.Candlestick(x=complete_data.index,
                            open=complete_data.Open,
                            high=complete_data.High,
                            low=complete_data.Low,
                            close=complete_data.Close)])

            fig.update_layout(
                width=self.width,
                height=self.height,
                title=f'{stock_name} Analysis during the COVID',
                yaxis_title='Adj Closing Prices',
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
                yaxis="y1"),
            secondary_y=True
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Date")

        fig.update_layout(
            width=self.width,
            height=self.height,
            title=f'{stock_name} Analysis during the COVID',
            yaxis=dict(
            title=f"{stock_name} Adj Closing Prices",
            ticksuffix=' $'

            ),
            yaxis2=dict(
                title=f"{stock_name} Returns",
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
                text='COVID Begins')]
        )

        return fig


    def plot_comparative_prices(self, prices):
        fig = go.Figure()

        visible = [False] * len(prices.columns)
        visible[0] = True

        for i, name in enumerate(prices.columns):
            fig.add_trace(
                go.Scatter(
                    x = prices[name].index,
                    y = prices[name],
                    name = name,
                    visible=visible[i]
                )
            )

        buttons = []

        for i, name in enumerate(prices.columns):
            false_true = [False] * len(prices.columns)
            false_true[i] = True
            buttons.append(
                dict(label = name,
                        method = 'update',
                        args = [{'visible': false_true}])
            )

        fig.update_layout(

            updatemenus=[
                dict(buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                x=0.8,
                xanchor="left",
                y=1.2,
                yanchor="top",
                active=0,
                )],
        )


        fig.update_layout(
            width=self.width,
            height=self.height,
            title='Two years in S&P500',
            yaxis_title='Adj Closing Prices',
            shapes = [dict(
                x0='2020-02-15', x1='2020-02-15',
                y0=0, y1=1,
                xref='x', yref='paper',
                line_width=2)],

            annotations=[
                dict(x='2020-02-17', y=0.95,
                xref='x', yref='paper',
                showarrow=False, xanchor='left',
                text='COVID Begins'),

                dict(text="Choose Stocks:", showarrow=False,
                x=0.61, xanchor='left',
                y=1.17, yanchor="top",
                yref='paper', xref='paper',
                font=dict(size=18))
                ],

            yaxis=dict(
            ticksuffix=' $'
            ),
        )


        return fig

    def plot_regression(self, df, factors, name):
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
                y=factors.params.Intercept + factors.params.MKT*df['MKT'],
                name='Fitted Line',
                mode='lines',
            )
        )

        fig.update_layout(
            title_text=f'{name} Regression',
            width=self.width,
            height=self.height,
        )

        return fig


    def plot_comparative_prices_ret(self, prices, returns):
        colors = {}
        for name in returns.columns:
            colors[name] = np.where(returns[name] < 0, 'red', 'green')

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        visible = [False] * len(prices.columns)
        visible[0] = True

        for i, name in enumerate(prices.columns):
            fig.add_trace(
                go.Scatter(
                    x=prices[name].index,
                    y=prices[name],
                    name=name,
                    visible=visible[i],
                    yaxis='y'
                ),
                secondary_y=False
            )

        for i, name in enumerate(prices.columns):
            fig.add_trace(
                go.Bar(x=returns[name].index,
                    y=(returns[name] * 100),
                    marker_color=colors[name],
                    name=name,
                    visible=visible[i],
                    yaxis="y1"),
                secondary_y=True
            )


        buttons = []

        for i, name in enumerate(prices.columns):
            false_true = [False] * len(prices.columns)
            false_true[i] = True
            buttons.append(
                dict(label=name,
                    method='update',
                    args=[{'visible': false_true}])
            )

        fig.update_layout(

            updatemenus=[
                dict(buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    x=0.8,
                    xanchor="left",
                    y=1.2,
                    yanchor="top",
                    active=0,
                    )],
        )

        fig.update_layout(
            width=self.width,
            height=self.height,
            title='Two years in S&P500',
            yaxis_title='Adj Closing Prices',
            yaxis2_title='Returns',
            shapes=[dict(
                x0='2020-02-15', x1='2020-02-15',
                y0=0, y1=1,
                xref='x', yref='paper',
                line_width=2)],

            annotations=[
                dict(x='2020-02-17', y=0.95,
                    xref='x', yref='paper',
                    showarrow=False, xanchor='left',
                    text='COVID Begins'),

                dict(text="Choose Stocks:", showarrow=False,
                    x=0.61, xanchor='left',
                    y=1.17, yanchor="top",
                    yref='paper', xref='paper',
                    font=dict(size=18))
            ],

            yaxis=dict(
                ticksuffix=' $'
            ),
        )

        return fig


    def plot_comparative_candles(self, complete_data):
        fig = go.Figure()

        stocks_name = complete_data.columns.levels[0]

        visible = [False] * len(stocks_name)
        visible[0] = True

        for i, name in enumerate(stocks_name):
            fig.add_trace(
                go.Candlestick(x=complete_data[name].index,
                            open=complete_data[name].Open,
                            high=complete_data[name].High,
                            low=complete_data[name].Low,
                            close=complete_data[name].Close,
                            name=name,
                            visible=visible[i])
            )

        fig.update_layout(
            width=self.width,
            height=self.height,
            title='Empirical Analysis during the COVID',
            yaxis_title='Adj Closing Prices',
            shapes=[dict(
                x0='2020-02-15', x1='2020-02-15', y0=0, y1=1, xref='x', yref='paper',
                line_width=2)],
            annotations=[dict(
                x='2020-02-17', y=0.95, xref='x', yref='paper',
                showarrow=False, xanchor='left', text='COVID Begins'),
                dict(text="Choose Stocks:", showarrow=False,
                     x=0.61, xanchor='left',
                     y=1.17, yanchor="top",
                     yref='paper', xref='paper',
                     font=dict(size=18))],
            yaxis=dict(
                ticksuffix=' $'
            ),
        )

        buttons = []

        for i, name in enumerate(stocks_name):
            false_true = [False] * len(stocks_name)
            false_true[i] = True
            buttons.append(
                dict(label=name,
                    method='update',
                    args=[{'visible': false_true}])
            )

        fig.update_layout(

            updatemenus=[
                dict(buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    x=0.8,
                    xanchor="left",
                    y=1.2,
                    yanchor="top",
                    active=0,
                    )],
        )

        return fig

    def plot_corr_vs_std(self, df):
        fig = go.Figure()

        visible = [False] * len(df.columns.levels[0])
        visible[0] = True

        for i, name in enumerate(df.columns.levels[0]):
            fig.add_trace(
                go.Scatter(
                    x=df[name, 'Std'],
                    y=[v[0] for v in df.index.values],
                    name=name,
                    visible=visible[i]
                )
            )

        buttons = []

        for i, name in enumerate(df.columns.levels[0]):
            false_true = [False] * len(df.columns.levels[0])
            false_true[i] = True
            buttons.append(
                dict(label=name,
                     method='update',
                     args=[{'visible': false_true}])
            )

        fig.update_layout(

            updatemenus=[
                dict(buttons=buttons,
                     direction="down",
                     pad={"r": 10, "t": 10},
                     x=0.8,
                     xanchor="left",
                     y=1.2,
                     yanchor="top",
                     active=0,
                     )],
        )

        fig.update_layout(
            width=self.width,
            height=self.height,
            title='Two years in S&P500',
            yaxis_title='Correlation',
            xaxis_title='Standard Deviation',

            annotations=[
                dict(text="Choose Pair:", showarrow=False,
                     x=0.61, xanchor='left',
                     y=1.17, yanchor="top",
                     yref='paper', xref='paper',
                     font=dict(size=18))
            ],

            xaxis=dict(
                ticksuffix=' %'
            ),
        )

        return fig
