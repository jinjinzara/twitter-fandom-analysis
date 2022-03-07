# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 17:06:04 2022

@author: rubby
"""

import json
import os
import pandas as pd
import random
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


nct_dir = 'data/@NCTsmtown'
itzy_dir= 'data/@ITZYofficial'
enhypen_dir = 'data/@ENHYPEN_members'

#json to dataframe
def to_df(filedir):
    df = pd.DataFrame(columns=['id', 'following_account', 'text'])
    for filename in os.listdir(filedir):
        tweets = [json.loads(line) for line in open(os.path.join(filedir, filename), 'r', encoding='utf-8')]
        t_text = ''
        t_id = filename[2:-4]
        for tweet in tweets:
            try:
                t_text += tweet['text'] + ' '
            except KeyError: #raise keyerror when there is no text attribute in tweet data
                print('No text in this tweet.')
        new_data = {'id' : t_id, 'following_account' : filedir[5:],'text' : t_text}
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
    doc_df = df[['id', 'tokens']].values.tolist()
    tagged_data = [TaggedDocument(words=_d, tags=[uid]) for uid, _d in doc_df]
    return tagged_data

nct_df, itzy_df, enhypen_df = to_df(nct_dir), to_df(itzy_dir), to_df(enhypen_dir)
merged_df = pd.concat([nct_df, itzy_df, enhypen_df])

nct_tagged, itzy_tagged, enhypen_tagged = tagging(nct_df), tagging(itzy_df), tagging(enhypen_df)
merge_tagged = nct_tagged + itzy_tagged + enhypen_tagged

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

model = training(merge_tagged)
model.save('analysis/doc2vec_v1.1.3.model')

model = Doc2Vec.load('analysis/doc2vec_v1.1.3.model')

#sampling for network
random.seed(10)
nct_tagged_sample = [tagged.words for tagged in random.sample(nct_tagged, 50)]
itzy_tagged_sample = [tagged.words for tagged in random.sample(itzy_tagged, 50)]
enhypen_tagged_sample = [tagged.words for tagged in random.sample(enhypen_tagged, 50)]

#infer embedding vectors of sample data
merge_tagged_sample = nct_tagged_sample + itzy_tagged_sample + enhypen_tagged_sample
embedding_vectors = [model.infer_vector(text) for text in merge_tagged_sample]
similarity_matrix = pd.DataFrame(cosine_similarity(embedding_vectors, embedding_vectors))

#print 2-dim t-sne plot for whole data
embedding_vectors = [model.infer_vector(tagged) for tagged in merge_tagged]
two_dim_vectors = TSNE(n_components=2, random_state=100).fit_transform(embedding_vectors)
fig, ax = plt.subplots(figsize=(16,20))
sns.scatterplot(two_dim_vectors[:,0], two_dim_vectors[:,1], hue=merged_df['following_account'].values)

#test for evaluation
txt_dir = 'data/@TXT_members'
twice_dir = 'data/@JYPETWICE'

txt_df, twice_df = to_df(txt_dir), to_df(twice_dir)
merged_df = pd.concat([merged_df, txt_df, twice_df])
txt_tagged, twice_tagged = tagging(txt_df), tagging(twice_df)
merge_tagged = merge_tagged + txt_tagged + twice_tagged

txt_tagged_words = [tagged.words for tagged in txt_tagged]
twice_tagged_words = [tagged.words for tagged in twice_tagged]

merge_tagged_sample += txt_tagged_words + twice_tagged_words