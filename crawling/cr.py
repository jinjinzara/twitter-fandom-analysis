# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:10:35 2022

@author: rubby
"""

import twitter
import json
import time
import pandas as pd
import networkx as nx
import pickle
import random
import config
import os.path

twitter_api = twitter.Api(consumer_key=config.twitter_consumer_key,
                              consumer_secret=config.twitter_consumer_secret, 
                              access_token_key=config.twitter_access_token, 
                              access_token_secret=config.twitter_access_secret,
                              sleep_on_rate_limit=True)

twitter_api2 = twitter.Api(consumer_key=config.twitter_consumer_key,
                              consumer_secret=config.twitter_consumer_secret, 
                              access_token_key=config.twitter_access_token, 
                              access_token_secret=config.twitter_access_secret)

most_followed = pd.read_csv('top100.csv')

cursor = -1
followers_list = []
while(len(followers_list) <= 1000):
    next_cursor, prev_cursor, followers = twitter_api.GetFollowersPaged(screen_name='by_verivery', cursor=cursor)
    cursor = next_cursor
    followers_list.append(followers)