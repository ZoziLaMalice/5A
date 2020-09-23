import plotly.graph_objects as go
import os
from datetime import datetime
import nltk
import re
from collections import Counter
import pandas as pd
import numpy as np

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# download nltk dependencies
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# initialize a sentiment analyzer
sid = SentimentIntensityAnalyzer()

# stop words for the word-counts
stops = stopwords.words('english')
stops.append('https')

# the number of most frequently mentioned tags
num_tags_scatter = 5

# initalize a dictionary to store the number of tweets for each game
scatter_dict = {}
sentiment_dict = {}

def count_words(series):
    # merge the text from all the tweets into one document
    document = ' '.join([row for row in series])

    # lowercasing, tokenization, and keep only alphabetical tokens
    tokens = [word for word in word_tokenize(document.lower()) if word.isalpha()]

    # filtering out tokens that are not all alphabetical
    tokens = [word for word in re.findall(r'[A-Za-z]+', ' '.join(tokens))]

    # remove all stopwords
    no_stop = [word for word in tokens if word not in stops]

    return Counter(no_stop)


def preprocess_nltk(row):
    # lowercasing, tokenization, and keep only alphabetical tokens
    tokens = [word for word in word_tokenize(row.lower()) if word.isalpha()]

    # filtering out tokens that are not all alphabetical
    tokens = [word for word in re.findall(r'[A-Za-z]+', ' '.join(tokens))]

    # remove all stopwords
    no_stop = [word for word in tokens if word not in stops]

    return ' '.join(no_stop)

def plot_word_count(stocks, data, chart_colors, app_color):
    figure = go.Figure()

    visible = [False] * len(stocks)
    visible[0] = True

    for i, stock in enumerate(data):
        cnt = count_words(data[stock].Text)
        top_n = cnt.most_common(10)[::-1]

        # get the x and y values
        X = [cnt for word, cnt in top_n]
        Y = [word for word, cnt in top_n]

        # plot the bar chart
        figure.add_trace(go.Bar(
            x=X, y=Y,
            name='Word Counts',
            orientation='h',
            marker=dict(color=chart_colors[::-1]),
            visible=visible[i]
        ))

    # specify the layout
    figure.update_layout(
            xaxis={
                'type': 'log',
                'autorange': True,
                'title': 'Number of Words'
            },
            height=500,
            plot_bgcolor=app_color["graph_bg"],
            paper_bgcolor=app_color["graph_bg"],
            font={"color": app_color["graph_font"]},
            autosize=False,
        )

    buttons = []

    for i, (stock, name) in enumerate(stocks.items()):
        false_true = [False] * len(stocks)
        false_true[i] = True
        buttons.append(
            dict(label = name,
                    method = 'update',
                    args = [{'visible': false_true}])
        )


    figure.update_layout(
        updatemenus=[dict(
            x=0.1,
            y=1.25,
            active=0,
            type='buttons',
            direction='left',
            buttons=buttons
            )
        ])

    return figure


def plot_sentiment(stocks, data, chart_colors, app_color):
    # initialize a sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    figure = go.Figure()

    visible = [False] * len(stocks)
    visible[0] = True

    for stock, (_, name), v in zip(data, stocks.items(), visible):
        data[stock].Text = data[stock].Text.apply(preprocess_nltk)

        avg_sentiments = {}

        for date in data[stock].Date.unique():
            sub_df = data[stock][data[stock].Date == date]

            sentiments = []
            for row in sub_df['Text']:
                sentiments.append(sid.polarity_scores(row)['compound'])

            avg_sentiments[date] = [np.mean(sentiments), np.std(sentiments)]

        # plot the scatter plot
        figure.add_trace(go.Scatter(
            x=[time for time, score in avg_sentiments.items()],
            y=[score[0] for time, score in avg_sentiments.items()],
            error_y={
                "type": "data",
                "array": [score[1]/30 for time, score in avg_sentiments.items()],
                "thickness": 1.5,
                "width": 1,
                "color": "#000",
            },
            name=name,
            mode='markers',
            opacity=0.7,
            marker=dict(color=chart_colors[3], size=10),
            visible=v
        ))

    # specify the layout
    figure.update_layout(
            xaxis={
                'automargin': False,
                'nticks': len(data[stock].Date.unique()),
            },
            yaxis={
                'autorange': True,
                'title': 'Sentiment Score'
            },
            height=500,
            plot_bgcolor=app_color["graph_bg"],
            paper_bgcolor=app_color["graph_bg"],
            font={"color": app_color["graph_font"]},
            autosize=True,

        )

    buttons = []

    for i, (stock, name) in enumerate(stocks.items()):
        false_true = [False] * len(stocks)
        false_true[i] = True
        buttons.append(
            dict(label = name,
                    method = 'update',
                    args = [{'visible': false_true}])
        )

    figure.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active=0,
            buttons=buttons,
            x=0.7,
            y=1.25
            )
        ])

    return figure