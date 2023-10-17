# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 20:21:06 2022

@author: rubby
"""

import pandas as pd
import os
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#json to dataframe
def create_tdf(filename):
    tweets = [json.loads(line) for line in open(filename, 'r', encoding='utf-8')]
    tdf = pd.DataFrame(tweets)
    tdf['tokens'] = ''
    tdf['following_account'] = filename[15:-4]
    for i in range(len(tdf)):
        entities = tdf['entities'][i]
        h_text = ''
        hts = entities.get('hashtags')
        for h in hts:
            h_text += ' ' + h.get('text')
        tdf['tokens'][i] = tdf['text'][i] + h_text
    tdf['tokens'] = tdf['tokens'].apply(nltk_tokenizer)
    tdf = tdf.drop_duplicates(['id'], keep='first', ignore_index=True)
    return tdf

def create_udf(tdf):
    udf = pd.DataFrame([tdf['user'][0]])
    for i in range(1, len(tdf)):
        new_data = tdf['user'][i]
        udf = udf.append(new_data, ignore_index=True)
    udf['following_account'] = tdf['following_account']
    udf = udf.drop_duplicates(['id'], keep='first', ignore_index=True)
    udf = udf[(udf['verified'] == False) & (udf['followers_count'] < 50000)]
    return udf

#tokenize & preprocessing
def nltk_tokenizer(_wd):
    custom_stopwords = open('stopwords.txt', 'r').read().split('\n')
    tokens = RegexpTokenizer('[A-Za-z]+').tokenize(_wd.lower()) #english only
    all_stopwords = custom_stopwords + stopwords.words('english')
    tokens_without_sw = [word for word in tokens if not word in all_stopwords and len(word) >= 2]
    return tokens_without_sw

def doc2vec(tdf):
    tdf = tdf[['tokens', 'following_account']]
    tdf_agg = tdf.groupby('following_account').sum()
    tdf_agg['following_account'] = tdf_agg.index
    tagged_data = [TaggedDocument(words=f, tags=[_d]) for f, _d in tdf_agg.values.tolist()]
    doc2vec = Doc2Vec(vector_size=300, alpha=0.0001, min_alpha=0.00001, seed=100)
    doc2vec.build_vocab(tagged_data)
    max_epoch = 10
    for epoch in range(max_epoch):
        print('epoch {0}'.format(epoch))
        doc2vec.train(tagged_data, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)
        doc2vec.alpha -= 0.00001
        doc2vec.min_alpha = doc2vec.alpha
    return doc2vec

def merge_data(dir_list):
    tdf = pd.DataFrame()
    udf = pd.DataFrame()
    i = 0
    for filedir in dir_list:
        for file in os.listdir(filedir):
            tdf_cur = create_tdf(os.path.join(filedir, file))
            udf_cur = create_udf(tdf_cur)
            tdf = pd.concat([tdf, tdf_cur])
            udf = pd.concat([udf, udf_cur])
            if i % 10 == 0:
                print('iter {} complete'.format(i))
            i += 1
    tdf.to_csv('tweet.csv', encoding='utf-8')
    udf.to_csv('user.csv', encoding='utf-8')
    return tdf, udf

if __name__ == '__main__':
    dir1 = 'data/data_v2.1/'
    dir2 = 'data/data_v2.2/'
    dir_list = [dir1, dir2]
    tdf, udf = merge_data(dir_list)