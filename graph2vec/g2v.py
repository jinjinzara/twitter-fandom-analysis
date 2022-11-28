# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:55:55 2022

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
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score
from datetime import datetime
import warnings
warnings.filterwarnings(action='ignore')

def plot_tsne(embedding_vectors, clusters, artist):
    two_dim_vectors = TSNE(n_components=2, 
                           random_state=1, 
                           perplexity=10).fit_transform(embedding_vectors)

    fig, ax = plt.subplots(figsize=(8,5))
    for i in range(len(embedding_vectors)):
        x, y, c = two_dim_vectors[i,0], two_dim_vectors[i,1], str(i+1)
        plt.text(x, y, c)

    sns.scatterplot(x=two_dim_vectors[:,0], y=two_dim_vectors[:,1], hue=clusters, palette='bright')
    plt.show()
    
def draw_networks(artist, subgraphs):
    for i in range(len(artist)):
        subg = subgraphs[i]
        plt.figure(figsize=(20,12))
        plt.axis('off')
        plt.grid(b=None)
        title_font = {
            'fontsize': 100,
            'fontweight': 'bold'}
        plt.title(artist[i], fontdict=title_font, loc='left', pad=20)
        pos = nx.kamada_kawai_layout(subg, scale=2)
        over5 = dict(filter(lambda e: e[1] >= 5, dict(subg.degree).items()))
        nx.draw_networkx_nodes(subg, pos, node_size=500, node_color='#0096FF')
        nx.draw_networkx_nodes(subg, pos, nodelist=list(over5.keys()), node_size=1500, node_color='#0437F2')
        nx.draw_networkx_edges(subg, pos, edge_color='#0437F2')
        try:
            plt.savefig('plot/network/{}.png'.format(artist[i]))
        except:
            plt.savefig('plot/network/{}.png'.format(artist[i].replace('*', '')))
        plt.show()

def feature_extractor(graphs, gnames, n_neighbor=3, hash_size=16):
    docs = []
    i = 0
    for g in graphs:
        features = nx.weisfeiler_lehman_subgraph_hashes(g, iterations=n_neighbor, digest_size=hash_size)
        features = list(features.values())
        features = ['_'.join(ft) for ft in features]
        doc = TaggedDocument(words=features, tags=[gnames[i]])
        docs.append(doc)
        i += 1
    return docs

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

def getCentralization(centrality, c_type):
	
	c_denominator = float(1)
	n_val = float(len(centrality))
	#print (str(len(centrality)) + "," +  c_type + "\n")
	
	if (c_type=="degree"):
		c_denominator = (n_val-1)*(n_val-2)
		
	if (c_type=="close"):
		c_top = (n_val-1)*(n_val-2)
		c_bottom = (2*n_val)-3	
		c_denominator = float(c_top/c_bottom)
		
	if (c_type=="between"):
		c_denominator = (n_val*n_val*(n_val-2))
		
	if (c_type=="eigen"):

		'''
		M = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(),weight='weight',dtype=float)
		eigenvalue, eigenvector = linalg.eigs(M.T, k=1, which='LR') 
		largest = eigenvector.flatten().real
		norm = sp.sign(largest.sum())*sp.linalg.norm(largest)
		centrality = dict(zip(G,map(float,largest)))
		'''
		
		c_denominator = np.sqrt(2)/2 * (n_val - 2)

	#start calculations
		
	c_node_max = max(centrality.values())
	c_sorted = sorted(centrality.values(),reverse=True)	
	#print ("max node " + str(c_node_max) + "\n")
	c_numerator = 0

	for value in c_sorted:		
		if c_type == "degree":
			#remove normalisation for each value
			c_numerator += (c_node_max*(n_val-1) - value*(n_val-1))
		else:
			c_numerator += (c_node_max - value)
	
	#print ('numerator:' + str(c_numerator)  + "\n")	
	#print ('denominator:' + str(c_denominator)  + "\n")	

	network_centrality = float(c_numerator/c_denominator)
	
	if c_type == "between":
		network_centrality = network_centrality * 2
		
	return network_centrality

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
    
    #metrics['centralization(degree)'] = [centralization_d(s) for s in subgraphs]
    #metrics['centralization(betweenness)'] = [centralization_b(s) for s in subgraphs]
    #metrics['centralization(closeness)'] = [centralization_c(s) for s in subgraphs]
    
    metrics['centralization(degree)'] = [getCentralization(nx.degree_centrality(s), 'degree') for s in subgraphs]
    metrics['centralization(betweenness)'] = [getCentralization(nx.betweenness_centrality(s), 'between') for s in subgraphs]
    metrics['centralization(closeness)'] = [getCentralization(nx.closeness_centrality(s), 'close') for s in subgraphs]
    #metrics['centralization(eigenvector)'] = [getCentralization(nx.eigenvector_centrality(s), 'eigen') for s in subgraphs]
    metrics['connectivity'] = [nx.average_node_connectivity(s) for s in subgraphs]
    metrics['efficiency'] = [nx.global_efficiency(s) for s in subgraphs]
    metrics['assortativity'] = [nx.degree_assortativity_coefficient(s) for s in subgraphs]
    #metrics['group_centrality(degree)'] = [nx.group_degree_centrality(G, s.nodes) for s in subgraphs]
    #metrics['group_centrality(betweenness)'] = [nx.group_betweenness_centrality(G, s.nodes) for s in subgraphs]
    #metrics['group_centrality(closeness)'] = [nx.group_closeness_centrality(G, s.nodes) for s in subgraphs]
    metrics['s_metric'] = [nx.s_metric(s, normalized=False) for s in subgraphs]
    #metrics['diameter'] = [nx.diameter(s) for s in subgraphs]
    
    return metrics

def posthoc_test(metric, metric_table):
    print('metric: {}'.format(metric))
    posthoc = pairwise_tukeyhsd(metrics[metric], metrics['cluster'], alpha=0.05)
    print(posthoc)
    print('\n')
    
def plot_elbow(v):
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(v)
        distortions.append(kmeanModel.inertia_)
        
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def plot_silhouette(data, param_init='random', param_n_init=10, param_max_iter=300):
    clusters_range = range(2,15)
    results = []

    for i in clusters_range:
        clusterer = KMeans(n_clusters=i, init=param_init, n_init=param_n_init, max_iter=param_max_iter, random_state=0)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        results.append([i, silhouette_avg])

    result = pd.DataFrame(results, columns=["n_clusters", "silhouette_score"])
    pivot_km = pd.pivot_table(result, index="n_clusters", values="silhouette_score")

    plt.figure()
    sns.heatmap(pivot_km, annot=True, linewidths=.5, fmt='.3f', cmap=sns.cm._rocket_lut)
    plt.tight_layout()
    plt.show()

def get_filtered_followers(most_followed):
    all_ffs = []
    for i in most_followed.index:
        name = most_followed['name'][i]
        filepath = ('data/followers_all/{}.p'.format(name)).replace('*', ' ').replace('!', '')
        with open(filepath, 'rb') as f:
            followers = pickle.load(f)
          
        filtered_fs = []
        for follower in followers:
            c_date = datetime.strptime(follower.created_at,'%a %b %d %H:%M:%S +0000 %Y')
            if c_date < datetime(2022,5,1) and not follower.verified:
                filtered_fs.append(follower)
        all_ffs.append(filtered_fs)
    return all_ffs

def get_ex_info():
    user_info_cols = ['n_custom_profile_of_fans', 'n_followers_of_fans', 'n_friends_of_fans', 'ff_ratio_of_fans','n_tweets_of_fans', 'n_core_fans']
    most_followed[user_info_cols] = 0
    
    all_filtered_fs = get_filtered_followers(most_followed)
    for i in most_followed.index:
        followers = all_filtered_fs[i]
        artist_name = most_followed['name'][i]
        n = len(followers)
        n_custom_profile, n_followers, n_friends, friends_followers_ratio, n_tweets, n_core = 0, 0, 0, 0, 0, 0
        for follower in followers:
            n_custom_profile += int(follower.default_profile_image)
            n_followers += follower.followers_count / n
            n_friends += follower.friends_count / n
            if follower.followers_count != 0:
                friends_followers_ratio += (follower.friends_count / follower.followers_count) / n
            n_tweets += follower.statuses_count / n
            if artist_name.lower() in follower.screen_name or \
                artist_name.lower() in follower.description.lower():
                    n_core += 1
        most_followed['n_custom_profile_of_fans'][i] = n_custom_profile
        most_followed['n_followers_of_fans'][i] = n_followers
        most_followed['n_friends_of_fans'][i] = n_friends
        most_followed['ff_ratio_of_fans'][i] = friends_followers_ratio
        most_followed['n_tweets_of_fans'][i] = n_tweets
        most_followed['n_core_fans'][i] = n_core
        
    metrics[user_info_cols] = most_followed[user_info_cols]
    
    metrics['n_followers_of_artist'] = most_followed['followers_count']
    metrics['gender'] = most_followed['gender_en']
    
if __name__ == '__main__':
    
    most_followed = pd.read_csv('crawling/top100.csv')
    most_followed = most_followed[:40]
    most_followed['gender_en'] = most_followed['gender'].map({'M':0, 'MIX':1, 'F':2})

    artists = most_followed['name'].values.tolist()
    accounts = most_followed['account'].values.tolist()
    
    graphs = []
    
    for artist in artists:
        filepath_n = ('data/network/{}.net'.format(artist)).replace('*', ' ').replace('!', '')
        
        G = nx.read_pajek(filepath_n)
        G.remove_edges_from(list(nx.selfloop_edges(nx.Graph(G))))
        graphs.append(nx.Graph(G))

    #draw_networks(artist_info, graphs)
    draw_networks(artists, graphs)
    
    metrics = network_info(artists, graphs)
    #metrics = pd.read_csv('metrics_all.csv')
    print('Network metric complete')
    
    sc = MinMaxScaler()
    
    # 현재 베스트
    docs = feature_extractor(graphs, artists, n_neighbor=5, hash_size=8)
    #doc2vec = Doc2Vec(docs, vector_size=16, window=0, min_count=5, dm=0, workers=4, epochs=10, alpha=0.025, seed=10)
    doc2vec = Doc2Vec(docs, vector_size=16)
    doc2vec.build_vocab(docs)
    
    epochs = 20
    for i in range(epochs):
        doc2vec.train(docs, total_examples=doc2vec.corpus_count, epochs=100)
    
    g2v = np.array([doc2vec.infer_vector(doc.words) for doc in docs])
    g2v_norm = sc.fit_transform(g2v)
    #g2v_norm = g2v
    
    plot_elbow(g2v_norm)
    plot_silhouette(g2v_norm)
    
    kmeans = KMeans(n_clusters=3).fit(g2v_norm)
    clusters = kmeans.labels_
    metrics['cluster'] = clusters
    sns.set_style("white")
    sns.countplot(x='cluster', data=metrics, palette='bright')
    plt.show()
    
    plot_tsne(g2v_norm, clusters, artists)
    
    posthoc_test('avg_degree', metrics)
    posthoc_test('density', metrics)
    posthoc_test('centralization(degree)', metrics)
    posthoc_test('centralization(betweenness)', metrics)
    posthoc_test('centralization(closeness)', metrics)
    posthoc_test('connectivity', metrics)
    posthoc_test('efficiency', metrics)
    posthoc_test('assortativity', metrics)
    posthoc_test('s_metric', metrics)
    posthoc_test('n_followers_of_artist', metrics)
    posthoc_test('n_custom_profile_of_fans', metrics)
    posthoc_test('n_followers_of_fans', metrics)
    posthoc_test('n_friends_of_fans', metrics)
    posthoc_test('ff_ratio_of_fans', metrics)
    posthoc_test('n_tweets_of_fans', metrics)
    posthoc_test('n_core_fans', metrics)
    posthoc_test('gender', metrics)
    
    