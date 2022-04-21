# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 16:35:37 2022

@author: rubby
"""

import config
import twitter
import json
import time
import pandas as pd
import networkx as nx
    
#collecting filtered stream data
def collecting_tweets(account, search_keyword):
    start = time.time()
    followers = twitter_api.GetFollowerIDs(screen_name=account, cursor=-1, count=1000, total_count=1000, stringify_ids=True)

    output_file_name = "data/" + account + ".txt"
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        length = 1
        p_time = time.time() - start
        while length < 100 and p_time < 43200: #if data is not collected until 12 hrs, stop collecting
            try:
                stream_of_followers = twitter_api.GetStreamFilter(follow=followers, track=[search_keyword], languages=["en"])
                while length < 100:
                    for tweets_f in stream_of_followers:
                        if length > 100:
                            break
                        tweet_f = json.dumps(tweets_f, ensure_ascii=False)
                        print(tweet_f, file=output_file, flush=True)
                        length += str(tweets_f).count('\n') + 1
            except:
                time.sleep(5) #Rate limit exceeded. Wait 5 seconds.

def collecting_multimode(inputs):
    for ipt in inputs:
        account = ipt[0]
        search_keyword = ipt[1]
        collecting_tweets(account, search_keyword)
              
#collecting timeline (test)
def collecting_timeline(account):
    statuses = twitter_api.GetUserTimeline(screen_name=account, count=10, include_rts=True, exclude_replies=False)

    for status in statuses:
        print(status.text)
        
    followers = twitter_api.GetFollowerIDs(screen_name=account, cursor=-1, count=10, total_count=10)
    for follower in followers:
        try:
            statuses_of_follower = twitter_api.GetUserTimeline(user_id=follower, count=10, include_rts=False, exclude_replies=False)
            for status_f in statuses_of_follower:
                print(status_f.text)
        except:
            continue           

#get friendship between two users
def collecting_network():
    udf = pd.read_csv('user.csv', index_col=0)
    udf_sampled = udf.groupby('following_account').sample(10)
    user_ids = udf_sampled['id'].values.tolist()
    graph = nx.Graph()
    for i in range(0, len(user_ids)-1):
        for j in range(i+1, len(user_ids)):
            sid = user_ids[i]
            tid = user_ids[j]
            w = 0
            try:
                time.sleep(1)
                following_json = twitter_api.ShowFriendship(source_user_id=sid, target_user_id = tid)
                friendship = following_json['relationship']['target']
                if friendship['following'] == True:
                    w += 1
                elif friendship['followed_by'] == True:
                    w += 1
                else:
                    w = 0
                graph.add_edge(sid, tid, weight=w)
                if j % 10 == 0:
                    print('{} edge add'.format(j))
            except Exception as e:
                try:
                    if e.args[0][0]['code'] == 88:
                        print(e)
                        print('{}, {} failed.'.format(sid, tid))
                        time.sleep(900)
                    elif e.args[0][0]['code'] == 163:
                        break
                    else:
                        continue
                except TypeError as e:
                    print(e)
        print('{}/{} user complete'.format(i+1, len(user_ids)))
    return graph

if __name__ == '__main__':
    twitter_api = twitter.Api(consumer_key=config.twitter_consumer_key,
                              consumer_secret=config.twitter_consumer_secret, 
                              access_token_key=config.twitter_access_token, 
                              access_token_secret=config.twitter_access_secret)

    #most_followed = pd.read_csv('most_followed.csv')
    #most_followed.apply(lambda x: collecting_tweets(x['account'], x['name']), axis=1)
    graph = collecting_network()
    nx.write_shp(graph, 'usrNet.DiGraph')