# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 22:23:37 2022

@author: rubby
"""

import os
import json
import hashlib
import pandas as pd
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

#json to dataframe
def create_tdf(filename):
    tweets = [json.loads(line) for line in open(filename, 'r', encoding='utf-8')]
    tdf = pd.DataFrame(tweets)
    tdf['following_account'] = filename[15:-4]
    tdf = tdf.drop_duplicates(['id'], keep='first', ignore_index=True)
    return tdf

def create_udf(tdf):
    udf = pd.DataFrame([tdf['user'][0]])
    for i in range(1, len(tdf)):
        new_data = tdf['user'][i]
        udf = udf.append(new_data, ignore_index=True)
    #udf['following_account'] = tdf['following_account']
    udf = udf.drop_duplicates(['id'], keep='first', ignore_index=True)
    #udf = udf[(udf['verified'] == False) & (udf['followers_count'] < 50000)]
    return udf

#make networkx of users
def usr_network(tweet, user):
    uids = user['id_str'].values.tolist()
    tweet = tweet.fillna(0)
    G = nx.Graph()
    for index, row in tweet.iterrows():
        G.add_node(row['id_str'])
        for uid in uids:
            if row['in_reply_to_status_id_str'] == uid or row['in_reply_to_user_id_str'] == uid:
                G.add_edge(row['id_str'], uid)
    nx.write_pajek(G, 'usrNet/usrNet_' + tweet['following_account'].iloc[0]+ '.net')
    return G

def do_a_recursion(g, name, features, extracted_features):
    new_features = {}
    for node in g.nodes:
        nebs = g.neighbors(node)
        degs = [features[neb] for neb in nebs]
        ft = [str(features[node])] + sorted([str(deg) for deg in degs])
        ft = "_".join(ft)
        hash_object = hashlib.md5(ft.encode())
        hashing = hash_object.hexdigest()
        new_features[node] = hashing
    extracted_features += list(new_features.values())
    return extracted_features

def do_recursions(graphs, gnames, iterations):
    extracted_features = []
    docs = []
    gn = 0
    for g in graphs:
        name = gnames[gn]
        features = nx.degree(g)
        features = {k: v for k, v in features}
        extracted_features = []
        for i in range(iterations):
            extracted_features = do_a_recursion(g, name, features, extracted_features)
        doc = TaggedDocument(words=extracted_features, tags=[name])
        docs.append(doc)
        gn += 1
    return docs

#merge data v1 & v2
def merge_data(dir_list):
    tdf = pd.DataFrame()
    i = 0
    for filedir in dir_list:
        for file in os.listdir(filedir):
            tdf_cur = create_tdf(os.path.join(filedir, file))
            tdf = pd.concat([tdf, tdf_cur])
            if i % 10 == 0:
                print('iter {} complete'.format(i))
            i += 1
    return tdf

#doc2vec training
def doc2vec(docs):
    doc2vec = Doc2Vec(vector_size=32, window=0, min_count=5, dm=0, workers=4, epochs=10, alpha=0.025, seed=10)
    doc2vec.build_vocab(docs)
    doc2vec.train(docs, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)
    return doc2vec

def print_plot(embedding_vectors):
    two_dim_vectors = TSNE(n_components=2, random_state=100).fit_transform(embedding_vectors)
    fig, ax = plt.subplots(figsize=(16,20))
    kmeans = KMeans(n_clusters=3).fit(embedding_vectors)
    clusters = kmeans.labels_
    sns.scatterplot(two_dim_vectors[:,0], two_dim_vectors[:,1], hue=clusters)
    return clusters

def draw_networks(artist, subgraphs):
    for i in range(len(artist)):
        subg = subgraphs[i]
        plt.figure(figsize=(30,15))
        title_font = {
            'fontsize': 64,
            'fontweight': 'bold'}
        plt.title(artist[i], fontdict=title_font, loc='left', pad=20)
        pos = nx.kamada_kawai_layout(subg, scale=2)
        over5 = dict(filter(lambda e: e[1] >= 5, dict(subg.degree).items()))
        nx.draw_networkx_nodes(subg, pos, node_size=500, node_color='#0096FF')
        nx.draw_networkx_nodes(subg, pos, nodelist=list(over5.keys()), node_size=1500, node_color='#0437F2')
        nx.draw_networkx_edges(subg, pos, edge_color='#0437F2')
        plt.savefig('plot/network/{}.png'.format(artist[i]))
        plt.show()

def ideal_G(n):
    Gi = nx.Graph()
    for node in range(1, n):
        Gi.add_edge(0, node)
    return Gi

def centralization_b(G):
    G = nx.Graph(G)
    node_c = list(nx.betweenness_centrality(G, normalized=False).values())
    i = 0
    for c in node_c:
        i += max(node_c) - c
    
    Gi = ideal_G(len(G))
    node_c = list(nx.betweenness_centrality(Gi, normalized=False).values())
    j = 0
    for c in node_c:
        j += max(node_c) - c
    
    return round(100*(i/j), 2)
    
def centralization_d(G):
    G = nx.Graph(G)
    node_c = list(nx.degree_centrality(G).values())
    i = 0
    for c in node_c:
        i += max(node_c) - c
    
    Gi = ideal_G(len(G))
    node_c = list(nx.degree_centrality(Gi).values())
    j = 0
    for c in node_c:
        j += max(node_c) - c
    
    return round(100*(i/j), 2)

def centralization_c(G):
    G = nx.Graph(G)
    node_c = list(nx.closeness_centrality(G).values())
    i = 0
    for c in node_c:
        i += max(node_c) - c
    
    Gi = ideal_G(len(G))
    node_c = list(nx.closeness_centrality(Gi).values())
    j = 0
    for c in node_c:
        j += max(node_c) - c
    
    return round(100*(i/j), 2)

def network_info(artist, subgraphs):
    metrics = pd.DataFrame()
    metrics['artist'] = artist
    
    metrics['size'] = [s.size() for s in subgraphs]
    
    avgs = []
    for i in range(len(metrics)):
        degree = dict(subgraphs[i].degree())
        avg = sum(degree.values()) / len(degree)
        avgs.append(round(avg, 2))
    metrics['avg_degree'] = avgs
    metrics['density'] = [round(nx.density(g),3) for g in subgraphs]
    metrics['centralization(degree)'] = [centralization_d(s) for s in subgraphs]
    metrics['centralization(betweenness)'] = [centralization_b(s) for s in subgraphs]
    metrics['centralization(closeness)'] = [centralization_c(s) for s in subgraphs]
    
    return metrics

if __name__ == '__main__':
    dir1 = 'data/data_v2.1/'
    dir2 = 'data/data_v2.2/'
    dir_list = [dir1, dir2]
    tdf = merge_data(dir_list)
    
    most_followed = pd.read_csv('most_followed_in.csv')
    accounts = most_followed['account'].values.tolist()
    gnames = most_followed['name'].values.tolist()
    
    tdf_group = tdf.groupby('following_account')
    tdf_groups = [tdf_group.get_group(x) for x in accounts[:150]]
    gnames = gnames[:150]
    
    udf_groups = []
    for tg in tdf_groups:
        tg = tg.reset_index()
        udf_groups.append(create_udf(tg))
    
    net_groups = []
    for i in range(len(tdf_groups)):
        net_groups.append(usr_network(tdf_groups[i], udf_groups[i]))
    
    docs = do_recursions(net_groups, gnames, 10)
    
    sc = MinMaxScaler()
    
    doc2vec = doc2vec(docs)
    g2v = doc2vec.dv.vectors
    g2v_norm = sc.fit_transform(g2v)
    clusters = print_plot(g2v_norm)
    
    cdf = pd.DataFrame()
    cdf['name'] = gnames
    cdf['followers_count'] = most_followed['followers_count'][:150]
    cdf['cluster'] = clusters

    nodes_count = []
    edges_count = []
    density = []
    for i in range(len(cdf)):
        nodes_count.append(len(net_groups[i].nodes))
        edges_count.append(len(net_groups[i].edges))
        density.append(nx.density(net_groups[i]))
    
    cdf['nodes_count'] = nodes_count
    cdf['edges_count'] = edges_count
    cdf['density'] = density
    
    with open('f_dict.p', 'rb') as f:
        f_dict = pickle.load(f)
    
    G = nx.read_pajek('usrNet_top30_sampled.net')
    nodes = list(G.nodes)
    nlabel = {}
    for n in nodes:
        for k in f_dict:
            if n in f_dict[k]:
                nlabel[n] = k
    
    keys = list(nlabel.keys())
    for n in nodes:
        if n not in keys:
            nei = G.neighbors(n)
            for ne in nei:
                if nlabel[ne]:
                    nlabel[n] = nlabel[ne]
                
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G_labeled = nx.relabel_nodes(G, nlabel)
    
    subgraphs = []
    for f in f_dict:
        subnodes = []
        for n in nlabel:
            if f in nlabel[n]:
                subnodes.append(nlabel[n])
        subgraphs.append(G_labeled.subgraph(subnodes))
    
    artist = most_followed['name'].iloc[:30].values.tolist()
    draw_networks(artist, subgraphs)