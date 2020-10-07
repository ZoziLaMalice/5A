import tweepy
import configparser as cp
import pandas as pd
from datetime import datetime

# Config Parser for Twitter API authentification
config = cp.ConfigParser()
config.read('./config.ini')

# Twitter API credentials
consumer_key = config.get('AUTH', 'consumer_key')
consumer_secret = config.get('AUTH', 'consumer_secret')

access_key = config.get('AUTH', 'access_key')
access_secret = config.get('AUTH', 'access_secret')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


def get_tweets_by_query(stock, debug=False):
    date_until = datetime.today().strftime('%Y-%m-%d')

    # get tweets
    tweets = []
    query_tag = '#'+str(stock)

    if debug:
        json_limit = api.rate_limit_status()
        print("Calls restants : " + str(json_limit['resources']['search']['/search/tweets']['remaining']) + '\n')

    for tweet in tweepy.Cursor(api.search, q=query_tag, lang="en", count=100).items():
        # create array of tweet information: created at, username, text
        tweets.append([tweet.created_at, tweet.user.screen_name, tweet.text])

    if debug:
        json_limit = api.rate_limit_status()
        print("Calls restants : " + str(json_limit['resources']['search']['/search/tweets']['remaining']) + '\n')

    df = pd.DataFrame(tweets, columns=['Date', 'Author', 'Text'])
    df.Date = df.Date.dt.strftime('%Y-%m-%d')

    if debug:
        print(df.head())

    return df
