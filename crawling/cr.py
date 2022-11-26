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
import time
from datetime import datetime

def collecting_followers(account):
    cursor = -1
    followers_list = []
    
    while(len(followers_list) < 5000):
        next_cursor, prev_cursor, followers = twitter_api.GetFollowersPaged(screen_name=account, cursor=cursor)
        cursor = next_cursor
        followers_list += followers
        print('{}: {}/{}'.format(account, len(followers_list), 5000))
        
    return followers_list

twitter_api = twitter.Api(consumer_key=config.twitter_consumer_key,
                              consumer_secret=config.twitter_consumer_secret, 
                              access_token_key=config.twitter_access_token, 
                              access_token_secret=config.twitter_access_secret,
                              sleep_on_rate_limit=True)

twitter_api2 = twitter.Api(consumer_key=config2.twitter_consumer_key,
                              consumer_secret=config2.twitter_consumer_secret, 
                              access_token_key=config2.twitter_access_token, 
                              access_token_secret=config2.twitter_access_secret,
                              sleep_on_rate_limit=True)

most_followed = pd.read_csv('crawling/top100.csv')


def collecting_mention(artist_name, artist_account):
    filepath_n = ('data/network/{}.net'.format(artist_name)).replace('*', ' ').replace('!', '')
    filepath_f = ('data/followers_filtered/{}.p'.format(artist_name)).replace('*', ' ').replace('!', '')
    
    if not os.path.isfile(filepath_n):
        with open(filepath_f, 'rb') as f:
            followers = pickle.load(f)
        
        G = nx.Graph()
        
        for follower in followers:
            print('artist: {}, follower: {}/{}'.format(artist_name, followers.index(follower)+1, len(followers)))
            
            last_date = datetime.now()
            last_id = None
            
            try:
                # for test, initialization
                statuses = twitter_api.GetUserTimeline(screen_name=artist_account)
            except Exception as e:
                print(e)
            
            while last_date > datetime(2022,5,1) or statuses:
                try:
                    statuses = twitter_api.GetUserTimeline(user_id=follower.id, max_id=last_id, count=200)
                except Exception as e:
                    print('In GetUserTimeline: {}'.format(e))
                    statuses = []
                    time.sleep(5)
                if statuses:
                    last_status = statuses[-1]
                    last_date = datetime.strptime(last_status.created_at,'%a %b %d %H:%M:%S +0000 %Y')
                    last_id = last_status.id
                    
                    for status in statuses:
                        s_created_date = datetime.strptime(status.created_at,'%a %b %d %H:%M:%S +0000 %Y')
                        
                        if s_created_date < datetime(2022,11,1) and s_created_date > datetime(2022,5,1) \
                        and status.user_mentions:
                            for mentioned_user in status.user_mentions:
                                if mentioned_user.id != follower.id:
                                    try:
                                        following = twitter_api.ShowFriendship(source_user_id=mentioned_user.id, 
                                                                               target_screen_name=artist_account)
                                        
                                        if following.get('relationship').get('source').get('following'):
                                            G.add_edge(follower.id, mentioned_user.id)
                                            nx.write_pajek(G, filepath_n)
                                            print('add edge of {}. date: {}'.format(artist_name, status.created_at))
                                    except Exception as e:
                                        print('In ShowFriendship: {}'.format(e))
                if len(statuses) < 200:
                    break

    else:
        print('{} network exists'.format(artist_name))


for i in most_followed.index:
    name = most_followed['name'][i]
    acc = most_followed['account'][i]
    collecting_mention(name, acc)
    

for i in most_followed.index:
    name = most_followed['name'][i]
    acc = most_followed['account'][i]
    filepath = ('data/followers/{}.p'.format(name)).replace('*', ' ').replace('!', '')
    while not os.path.isfile(filepath):
        try:
            fs = collecting_followers(acc)
            with open(filepath, 'wb') as f:
                pickle.dump(fs, f)
        except Exception as e:
            print(e)
            time.sleep(5)
            print('-Retry-')
    print('{}/{}'.format(i+1, len(most_followed)))
    
    
for i in most_followed.index:
    name = most_followed['name'][i]
    filepath = ('data/followers_all/{}.p'.format(name)).replace('*', ' ').replace('!', '')
    filepath2 = ('data/followers_filtered/{}.p'.format(name)).replace('*', ' ').replace('!', '')
    
    with open(filepath, 'rb') as f:
        followers = pickle.load(f)
        
    filtered_fs = []
    for follower in followers:
        c_date = datetime.strptime(follower.created_at,'%a %b %d %H:%M:%S +0000 %Y')
        if c_date < datetime(2022,5,1) and not follower.verified:
            filtered_fs.append(follower)
    
    with open(filepath2, 'wb') as f2:
        pickle.dump(random.sample(filtered_fs, 100), f2)