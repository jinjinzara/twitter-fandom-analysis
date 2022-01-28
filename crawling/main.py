# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 16:35:37 2022

@author: rubby
"""

import config
import twitter
import json
import time

twitter_api = twitter.Api(consumer_key=config.twitter_consumer_key,
                          consumer_secret=config.twitter_consumer_secret, 
                          access_token_key=config.twitter_access_token, 
                          access_token_secret=config.twitter_access_secret)

account = "@ENHYPEN_members"
search_keyword = "enhypen"

account = "@NCTsmtown"
search_keyword = "nct"

account = "@ITZYofficial"
search_keyword = "itzy"
    
#collecting filtered stream data (test)
def collecting_tweets(account, search_keyword):
    followers = twitter_api.GetFollowerIDs(screen_name=account, cursor=-1, count=1000, total_count=1000, stringify_ids=True)

    for follower in followers:
        output_file_name = "data/" + account + "/id" + follower + ".txt"
        with open(output_file_name, "w", encoding="utf-8") as output_file:
            try:
                time.sleep(1)
                stream_of_followers = twitter_api.GetStreamFilter(follow=[follower], track=[search_keyword], languages=["en"])
                length = 1
                while length < 100:
                    for tweets_f in stream_of_followers:
                        if length > 100:
                            break
                        tweet_f = json.dumps(tweets_f, ensure_ascii=False)
                        print(tweet_f, file=output_file, flush=True)
                        length += str(tweets_f).count('\n') + 1
            except:
                continue

inputs = list()
inputs.append(['@ENHYPEN_members', 'enhypen'])
inputs.append(['@NCTsmtown', 'nct'])
inputs.append(['@ITZYofficial', 'itzy'])

def collecting_multimode(inputs):
    for ipt in inputs:
        account = ipt[0]
        search_keyword = ipt[1]
        collecting_tweets(account, search_keyword)
              
#collecting timeline (test)
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
