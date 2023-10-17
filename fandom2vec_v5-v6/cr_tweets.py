# -*- coding: utf-8 -*-
"""
Created on Sat May  6 23:09:54 2023

@author: sy
"""



import json
import time
import pandas as pd
import networkx as nx
import pickle
import config

import twitter
import tweetkit
from tweetkit.auth import BearerTokenAuth
from tweetkit.client import TwitterClient
import datetime


def get_tweets(artist_name, start_time, end_time):
    auth = BearerTokenAuth(config.twitter_bearer_token)
    client = TwitterClient(auth=auth)

    
    paginator = client.tweets.tweets_fullarchive_search(
        '({}) -is:retweet -is:verified has:mentions lang:en'.format(artist_name),
        start_time=start_time,
        end_time=end_time,
        max_results=100,
        paginate=True,
    )
    
    tweets = []
    
    start_time = datetime.datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%SZ')
    end_time = datetime.datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%SZ')
    percentage_remaining = 0.0
    total_period = (end_time - start_time).total_seconds()
    for tweet in paginator.content:
        created_at = datetime.datetime.strptime(tweet['data']['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
        tweets.append(tweet)
        remaining_period = (end_time - created_at).total_seconds()
        percentage_remaining = round(remaining_period * 100 / total_period, 2)
        print('\rTweet Count: {:3.0f}%, {}'.format(percentage_remaining, len(tweets)), end='')
    if percentage_remaining != 100.00:
        print('\rTweet Count: {:3.0f}%, {}'.format(100.00, len(tweets)), end='')
    
    return tweets

def get_tweets_all():
    years, months, days = ['2022'], ['01','02','03','04','05','06','07','08','09','10','11','12'], ['01', '15']
    
    for year in years:
        for month in months:
            ONG = True
            m_tweets = []
            while(ONG):
                try:
                    for day in days:
                        m_tweets += get_tweets('enhypen', '{}-{}-{}T18:00:00Z'.format(year,month,day), '{}-{}-{}T23:59:59Z'.format(year,month,day))
                        print()
                    filename = 'tweets/enhypen_{}_{}.pickle'.format(year, month)
                    with open(filename, 'wb') as f_obj:
                        pickle.dump(m_tweets, f_obj)
                    print('Save {}'.format(filename))
                    ONG = False
                except Exception as e:
                    print(e)
                    ONG = True

def get_graph(tweets):
    graph_org = nx.Graph()
    graph_fan = nx.Graph()
    
    api = twitter.Api(consumer_key=config.twitter_consumer_key,
                          consumer_secret=config.twitter_consumer_secret, 
                          access_token_key=config.twitter_access_token, 
                          access_token_secret=config.twitter_access_secret,
                          sleep_on_rate_limit=True)
    
    for tweet in tweets:
        author_id = tweet.get('data').get('author_id')
        if tweet.get('data').get('entities'):
            if tweet.get('data').get('entities').get('mentions'):
                for mention in tweet.get('data').get('entities').get('mentions'):
                    mentioned_name = mention.get('username')
                    mentioned_id = mention.get('id')
                    graph_org.add_edge(author_id, mentioned_id)
                    if mentioned_name != 'ENHYPEN' and mentioned_name != 'ENHYPEN_members':
                        try:
                            mentioned_user = api.GetUser(screen_name=mentioned_name)
                        except Exception as e:
                            print(e)
                            continue
                        if not mentioned_user.verified:
                            following1 = api.ShowFriendship(source_user_id=author_id, target_screen_name='ENHYPEN_members')
                            following2 = api.ShowFriendship(source_screen_name=mentioned_name, target_screen_name='ENHYPEN_members')
                            if following1.get('relationship').get('source').get('following') and following2.get('relationship').get('source').get('following'):
                                graph_fan.add_edge(author_id, mentioned_id)
        print('\rTweet {}/{}'.format(tweets.index(tweet)+1, len(tweets)), end='')
    
    return graph_org, graph_fan
                                
    
def main():
    years, months = ['2022'], ['01','02','03','04','05','06','07','08','09','10','11','12']
    for year in years:
        for month in months:
            filename = 'tweets/enhypen_{}_{}.pickle'.format(year, month)
            with open(filename, 'rb') as f_obj:
                tweets = pickle.load(f_obj)
            g1, g2 = get_graph(tweets)
            nx.write_pajek(g1, 'networks/enhypen_{}_{}_org.net'.format(year, month))
            nx.write_pajek(g2, 'networks/enhypen_{}_{}_fan.net'.format(year, month))
            print('Save graph file of {}_{}'.format(year, month))
    
if __name__=='__main__':
    main()