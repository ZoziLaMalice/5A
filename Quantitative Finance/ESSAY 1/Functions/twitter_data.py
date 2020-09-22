import tweepy
import configparser as cp
import os
from datetime import datetime
import nltk
import re
from collections import Counter

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


# Config Parser for Twitter API authentification
config = cp.ConfigParser()
config.read('./ESSAY 1/config.ini')

# Twitter API credentials
consumer_key = config.get('AUTH', 'consumer_key')
consumer_secret = config.get('AUTH', 'consumer_secret')

access_key = config.get('AUTH', 'access_key')
access_secret = config.get('AUTH', 'access_secret')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


def get_tweets_by_query(stock, nb_tweets):
    date_until = datetime.today().strftime('%Y-%m-%d')

    # get tweets
    tweets = []
    query_tag = '#'+str(stock)

    for tweet in tweepy.Cursor(api.search, q=query_tag, lang="en", until=date_until, count=nb_tweets).items(nb_tweets):
        # create array of tweet information: created at, username, text
        tweets.append([tweet.created_at, tweet.user.screen_name, tweet.text])

    return tweets, len(tweets)


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