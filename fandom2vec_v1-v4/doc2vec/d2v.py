# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 17:06:04 2022

@author: rubby
"""

import json
import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


#json to dataframe
def to_df(filedir):
    df = pd.DataFrame(columns=['f_id', 'text'])
    for filename in os.listdir(filedir):
        tweets = [json.loads(line) for line in open(os.path.join(filedir, filename), 'r', encoding='utf-8')]
        t_text = ''
        f_id = filename[1:-4]
        for tweet in tweets:
            try:
                t_text += tweet['text'] + ' '
            except KeyError: #raise keyerror when there is no text attribute in tweet data
                print('No text in this tweet.')
        new_data = {'f_id' : f_id, 'text' : t_text}
        df = df.append(new_data, ignore_index=True)
    df['tokens'] = df['text'].apply(nltk_tokenizer)
    return df

#tokenize & preprocessing
def nltk_tokenizer(_wd):
    custom_stopwords = open('analysis/stopwords.txt', 'r').read().split('\n')
    tokens = RegexpTokenizer('[A-Za-z]+').tokenize(_wd.lower()) #english only
    for token in tokens:
        if token in stopwords.words('english') or token in custom_stopwords or len(token) < 2:
            tokens.remove(token)
    return tokens

#make TaggedDocument for Doc2vec
def tagging(df):
    doc_df = df[['f_id', 'tokens']].values.tolist()
    tagged_data = [TaggedDocument(words=_d, tags=[uid]) for uid, _d in doc_df]
    return tagged_data

#doc2vec training
def training(merge_tagged):
    model = Doc2Vec(alpha=0.025, min_alpha=0.0001, seed=100)
    model.build_vocab(merge_tagged)

    max_epoch = 10
    for epoch in range(max_epoch):
        print('epoch {0}'.format(epoch))
        model.train(merge_tagged, total_examples=model.corpus_count, epochs=model.epochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    return model

df = to_df('data')
df_tagged = tagging(df)

model = training(df_tagged)

model.save('analysis/doc2vec_v2.0.0.model')
model = Doc2Vec.load('analysis/doc2vec_v2.0.0.model')

#infer embedding vectors
embedding_vectors = [model.infer_vector(text) for text in df_tagged]
similarity_matrix = pd.DataFrame(cosine_similarity(embedding_vectors, embedding_vectors))

#print 2-dim t-sne plot for whole data
embedding_vectors = [model.infer_vector(tagged.words) for tagged in df_tagged]
two_dim_vectors = TSNE(n_components=2, random_state=100).fit_transform(embedding_vectors)
fig, ax = plt.subplots(figsize=(16,20))
kmeans = KMeans(n_clusters=5).fit(embedding_vectors)
clusters = kmeans.labels_
sns.scatterplot(two_dim_vectors[:,0], two_dim_vectors[:,1], hue=clusters)