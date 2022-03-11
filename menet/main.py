# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:38:33 2022

@author: rubby
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

#json to dataframe
def create_tdf(filename):
    tweets = [json.loads(line) for line in open(filename, 'r', encoding='utf-8')]
    tdf = pd.DataFrame(tweets)
    tdf['tokens'] = ''
    for i in range(len(tdf)):
        entities = tdf['entities'][i]
        h_text = ''
        hts = entities.get('hashtags')
        for h in hts:
            h_text += ' ' + h.get('text')
        tdf['tokens'][i] = tdf['text'][i] + h_text
    tdf['tokens'] = tdf['tokens'].apply(nltk_tokenizer)
    return tdf
    
def create_udf(tdf):
    udf = pd.DataFrame([tdf['user'][0]])
    for i in range(1, len(tdf)):
        new_data = tdf['user'][i]
        udf = udf.append(new_data, ignore_index=True)
    return udf

#tokenize & preprocessing
def nltk_tokenizer(_wd):
    custom_stopwords = open('stopwords.txt', 'r').read().split('\n')
    tokens = RegexpTokenizer('[A-Za-z]+').tokenize(_wd.lower()) #english only
    all_stopwords = custom_stopwords + stopwords.words('english')
    tokens_without_sw = [word for word in tokens if not word in all_stopwords and len(word) >= 2]
    return tokens_without_sw