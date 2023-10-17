# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:38:33 2022

@author: rubby
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten
from tensorflow.keras.models import Model
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
import SOM

#json to dataframe
def create_tdf(filename):
    tweets = [json.loads(line) for line in open(filename, 'r', encoding='utf-8')]
    tdf = pd.DataFrame(tweets)
    tdf['tokens'] = ''
    tdf['following_account'] = filename[5:-4]
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
    tagged_data = [TaggedDocument(words=_d, tags=[f]) for f, _d in tdf_agg.values.tolist()]
    doc2vec = Doc2Vec(vector_size=300, alpha=0.0001, min_alpha=0.00001, seed=100)
    doc2vec.build_vocab(tagged_data)
    max_epoch = 10
    for epoch in range(max_epoch):
        print('epoch {0}'.format(epoch))
        doc2vec.train(tagged_data, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)
        doc2vec.alpha -= 0.00001
        doc2vec.min_alpha = doc2vec.alpha
    return doc2vec

def continuous_pp(udf):
    udf = udf[['default_profile', 'following_account']]
    udf_agg = udf.groupby('following_account').sum()
    cs = MinMaxScaler()
    scaled = cs.fit_transform(udf_agg[['default_profile']])
    return scaled
    
def training(tdf, udf):
    d2v_input = Input(shape=(300,))
    x = Embedding(input_dim=300, output_dim=30)(d2v_input)
    x = Flatten()(x)
    x = Dense(30, activation='relu')(x)
    x = Dense(30, activation='relu')(x)
    x = Model(inputs=d2v_input, outputs=x)
    
    mlp_input = Input(shape=(1,))
    y = Dense(3, activation='relu')(mlp_input)
    y = Dense(3, activation='linear')(y)
    y = Model(inputs=mlp_input, outputs=y)
    
    combined_input = concatenate([x.output, y.output])
    
    z = Dense(2, activation='relu')(combined_input)
    som = SOM(6, 6, 4, 0.5, 0.5, 100)
    som.train(z)
    
    model = Model(inputs=[x.input, y.input], outputs=z)
    
    opt = Adam()
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
    print('Training model...')
    trainX = doc2vec(tdf).dv.vectors
    trainY = np.array(continuous_pp(udf))
    
    testZ = tdf['following_account'].unique()
    
    som = SOM()
    
    model.fit([trainX, trainY], testZ, validation_split=0.1)
    
    return model
        
if __name__ == '__main__':
    filedir = 'data/'
    tdf = pd.DataFrame()
    udf = pd.DataFrame()
    i = 0
    for file in os.listdir(filedir):
        tdf_cur = create_tdf(os.path.join(filedir, file))
        udf_cur = create_udf(tdf_cur)
        tdf = pd.concat([tdf, tdf_cur])
        udf = pd.concat([udf, udf_cur])
        if i % 10 == 0:
            print('iter {} complete'.format(i))
        i += 1
    tdf.to_csv('tweet.csv')
    udf.to_csv('user.csv')
    
    model = training(tdf, udf)