# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:04:34 2022

@author: rubby
"""

import pandas as pd
import os
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

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

def tagging(tdf):
    tdf = tdf[['following_account', 'tokens']]
    tdf_agg = tdf.groupby('following_account').sum()
    tdf_agg['following_account'] = tdf_agg.index
    tagged_data = [TaggedDocument(words=_d, tags=[f]) for _d, f in tdf_agg.values.tolist()]
    return tagged_data

#doc2vec training
def doc2vec(tdf):
    tagged_data = tagging(tdf)
    doc2vec = Doc2Vec(vector_size=300, alpha=0.0001, min_alpha=0.00001, seed=100)
    doc2vec.build_vocab(tagged_data)
    max_epoch = 10
    for epoch in range(max_epoch):
        print('epoch {0}'.format(epoch))
        doc2vec.train(tagged_data, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)
        doc2vec.alpha -= 0.00001
        doc2vec.min_alpha = doc2vec.alpha
    return doc2vec

#merge data v1 & v2
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
    #tdf.to_csv('tweet.csv', encoding='utf-8')
    #udf.to_csv('user.csv', encoding='utf-8')
    return tdf, udf

#make networkx of users
def usr_network(tweet, user):
    uids = user['id_str'].values.tolist()
    tweet = tweet.fillna(0)
    G = nx.Graph()
    for index, row in tweet.iterrows():
        for uid in uids:
            if row['in_reply_to_status_id_str'] == uid or row['in_reply_to_user_id_str'] == uid:
                G.add_edge(row['id_str'], uid)
    nx.write_pajek(G, 'usrNet_v2.net')
    return G

#node embedding
def node2vec(G):
    node2vec = Node2Vec(graph=G, # target graph
                    dimensions=128, # embedding dimension
                    walk_length=80, # number of nodes in each walks 
                    p = 1, # return hyper parameter
                    q = 1, # inout parameter, q값을 작게 하면 structural equivalence를 강조하는 형태로 학습됩니다. 
                    weight_key=None, # if weight_key in attrdict 
                    num_walks=10, 
                    workers=1)
    model = node2vec.fit(window=2)
    return model

def nodes_table(nodes, user, tweet):
    nodes_f = []
    for n in nodes:
        mat = user[user['id_str'] == n]
        fan = mat['following_account'].values.tolist()
        if len(fan) > 0:
            nodes_f.append(fan)
        else:
            mat2 = tweet[tweet['id_str'] == n]
            fan2 = mat2['following_account'].values.tolist()
            fan2 = list(dict.fromkeys(fan2))
            nodes_f.append(fan2)
    return nodes_f

def tf_idf(tdf):
    tfidf = TfidfVectorizer()
    svd = TruncatedSVD(n_components = 100)
    tdf['text_p'] = tdf['tokens'].apply(' '.join)
    tdf_agg = tdf[['text_p', 'following_account']].groupby('following_account').sum()
    tfidf.fit(tdf_agg['text_p'])
    tfidf_v = tfidf.transform(tdf_agg['text_p']).toarray()
    tfidf_v_svd = svd.fit(tfidf_v)
    tfidf_v_svd = svd.fit_transform(tfidf_v)
    return tfidf_v_svd


def print_plot(embedding_vectors):
    two_dim_vectors = TSNE(n_components=2, random_state=100).fit_transform(embedding_vectors)
    fig, ax = plt.subplots(figsize=(16,20))
    kmeans = KMeans(n_clusters=5).fit(embedding_vectors)
    clusters = kmeans.labels_
    sns.scatterplot(two_dim_vectors[:,0], two_dim_vectors[:,1], hue=clusters)
    return clusters

if __name__ == '__main__':
    dir1 = 'data/data_v2.1/'
    dir2 = 'data/data_v2.2/'
    dir_list = [dir1, dir2]
    tdf, udf = merge_data(dir_list)
    tdf_tagged = tagging(tdf)
    
    tdf = pd.read_csv('tweet.csv', encoding='utf-8', index_col=0)
    udf = pd.read_csv('user.csv', encoding='utf-8', index_col=0)
    
    graph = usr_network(tdf, udf)
    graph = nx.read_pajek('usrNet_v1.net')
    n2v = node2vec(graph)
    nt = nodes_table(list(graph.nodes()), udf, tdf)
    n2v_v = n2v.wv.vectors
    
    new_n2v_v = np.zeros((163,128))
    new_nt = [tagged.tags[0] for tagged in tdf_tagged]
    for i in range(len(new_nt)):
        for j in range(len(nt)):
            if new_nt[i] in nt[j] and len(nt[j]) <= 5:
                new_n2v_v[i] = (new_n2v_v[i] + n2v_v[j]) / 2
    
    n2v_v = new_n2v_v
    
    d2v = doc2vec(tdf)
    d2v_v = np.array([d2v.infer_vector(tagged.words) for tagged in tdf_tagged])
    
    tfidf_v = tf_idf(tdf)
    
    sc = MinMaxScaler()
    n2v_v_norm = sc.fit_transform(n2v_v)
    d2v_v_norm = sc.fit_transform(d2v_v)
    tfidf_v_norm = sc.fit_transform(tfidf_v)
    
    f2v_v = np.concatenate((n2v_v, d2v_v, tfidf_v), axis=1)
    f2v_37 = np.dot(0.3, n2v_v) + np.dot(0.7, d2v_v)
    f2v_46 = np.dot(0.4, n2v_v) + np.dot(0.6, d2v_v)
    f2v_55 = np.dot(0.5, n2v_v) + np.dot(0.5, d2v_v)
    f2v_64 = np.dot(0.6, n2v_v) + np.dot(0.4, d2v_v) #아니면 이거
    f2v_73 = np.dot(0.7, n2v_v) + np.dot(0.3, d2v_v)
    
    f2v_37_norm = np.dot(0.3, n2v_v_norm) + np.dot(0.7, d2v_v_norm)
    f2v_46_norm = np.dot(0.4, n2v_v_norm) + np.dot(0.6, d2v_v_norm)
    f2v_55_norm = np.dot(0.5, n2v_v_norm) + np.dot(0.5, d2v_v_norm)
    f2v_64_norm = np.dot(0.6, n2v_v_norm) + np.dot(0.4, d2v_v_norm)
    f2v_73_norm = np.dot(0.7, n2v_v_norm) + np.dot(0.3, d2v_v_norm) #지금은 이게 베스트
    