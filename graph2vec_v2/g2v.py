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

def print_plot(embedding_vectors, clusters, artist):
    two_dim_vectors = TSNE(n_components=2, 
                           random_state=1, 
                           perplexity=5).fit_transform(embedding_vectors)
    fig, ax = plt.subplots(figsize=(8,5))
    for i in range(len(embedding_vectors)):
        x, y, c = two_dim_vectors[i,0], two_dim_vectors[i,1], str(i+1)
        plt.text(x, y, c)
    #kmeans = KMeans(n_clusters=3).fit(embedding_vectors)
    #clusters = kmeans.labels_
    sns.scatterplot(two_dim_vectors[:,0], two_dim_vectors[:,1], hue=clusters, palette=['#1E2F97', '#2ECBE9', '#FFD200'], sizes=(100, 100))
    
def draw_networks(artist, subgraphs):
    for i in range(len(artist)):
        subg = subgraphs[i]
        plt.figure(figsize=(30,15))
        title_font = {
            'fontsize': 100,
            'fontweight': 'bold'}
        plt.title(artist[i], fontdict=title_font, loc='left', pad=20)
        pos = nx.kamada_kawai_layout(subg, scale=2)
        over5 = dict(filter(lambda e: e[1] >= 5, dict(subg.degree).items()))
        nx.draw_networkx_nodes(subg, pos, node_size=500, node_color='#0096FF')
        nx.draw_networkx_nodes(subg, pos, nodelist=list(over5.keys()), node_size=1500, node_color='#0437F2')
        nx.draw_networkx_edges(subg, pos, edge_color='#0437F2')
        plt.savefig('plot/network/{}.png'.format(artist[i]))
        plt.show()

def feature_extractor(graphs, gnames, iteration, hash_size):
    docs = []
    i = 0
    for g in graphs:
        features = nx.weisfeiler_lehman_subgraph_hashes(g, iterations=iteration, digest_size=hash_size)
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
    posthoc = pairwise_tukeyhsd(metrics[metric], metrics['cluster'], alpha=0.05)
    print(posthoc)

if __name__ == '__main__':
    
    most_followed = pd.read_csv('most_followed_in.csv')
    accounts = most_followed['account'].values.tolist()
    gnames = most_followed['name'].values.tolist()
    
    with open('f_dict.p', 'rb') as f:
        f_dict = pickle.load(f)

    
    artist = most_followed['name'].values.tolist()
    accounts = most_followed['account'].values.tolist()
    
    graphs = []
    for ac in accounts:
        G = nx.read_pajek('data/usrNet/{}.net'.format(ac))
        G_in = G.remove_edges_from(list(nx.selfloop_edges(nx.Graph(G.copy()))))
        graphs.append(G_in)
        
    artist_info = []
    for i in most_followed.index:
        artist_info.append('{}({}, {})'.format(most_followed['name'].loc[i], 
                                               int(most_followed['debut_year'].loc[i]), 
                                               int(most_followed['activity_period'].loc[i])))
   
    #draw_networks(artist_info, graphs)
    draw_networks(artist, graphs)
    
    metrics = network_info(artist, graphs)
    
    sc = MinMaxScaler()
    
    # 현재 베스트
    docs = feature_extractor(graphs, artist, 5, 8)
    doc2vec = Doc2Vec(docs, vector_size=32, window=0, min_count=5, dm=0, workers=4, epochs=10, alpha=0.025, seed=10)
    g2v = doc2vec.dv.vectors
    g2v_norm = sc.fit_transform(g2v)
    clusters = print_plot(g2v_norm)
    metrics['cluster'] = clusters
    
    posthoc_test('avg_degree', metrics)
    posthoc_test('density', metrics)
    posthoc_test('centralization(degree)', metrics)
    posthoc_test('centralization(betweenness)', metrics)
    posthoc_test('centralization(closeness)', metrics)
    posthoc_test('connectivity', metrics)
    posthoc_test('efficiency', metrics)
    posthoc_test('assortativity', metrics)
    posthoc_test('s_metric', metrics)