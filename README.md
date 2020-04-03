# Comparing different Embeddings 


This repository takes an initiative in comparing the effect of different Embeddings on the performance of text classification Model. We'll begin with basic bag-of-words approach , tf-idf and NN-based Word2vec, Fasttext, DBOW, PV-DM. 

Data used for the purpose of analysis corresponds to news classification dataset from Kaggle. It is a classic text classification problem in which text has to be classified into one of the categories.

The python code is as follows:

#import modules
import os
path = 'C:\\Users\\ggund.GARTNER\\Desktop\\hackathon_ltfs\\bbc news classification'
os.chdir(path)
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy import spatial
import lightgbm
from sklearn.metrics import f1_score, accuracy_score
from string import punctuation
#import glove
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import pandarallel
from pandarallel import pandarallel
import swifter
from swifter import swifter
from statistics import mean
#from __future__ import division
stop= set(stopwords.words('english'))
tqdm.pandas()

#read data
data = pd.read_csv('bbc-text.csv')

data.columns

data.category.value_counts(normalize = True)

#train and test

train, test = train_test_split(data, test_size = 0.2)

#remove numbers

train['cleaned_text'] = train.text.str.replace('\d+', '')

#remove punctuation

train['cleaned_text'] = train.cleaned_text.apply(lambda x: [''.join(word) for word in x if word not in punctuation])

def str(x):
    x1= ''
    for i in x:
        x1+= i
        x1+=''
    return(x1)

train['cleaned_text'] = train.cleaned_text.apply(lambda x: str(x) )

#lowercase

train['cleaned_text'] = [word.lower() for word in train['cleaned_text']]

#remove extraspaces

train['cleaned_text'] = train.cleaned_text.str.replace('\s+', ' ')

#convert to snetences

train['sentences'] = [nltk.word_tokenize(i) for i in train['cleaned_text']]

#train['word_tok'] = [[sent.split(" ") for sent in para ] for para in  train['sentences'] ]
train['sentences'] = train.sentences.apply(lambda x: [word for word in x if word not in stop])

#remove punctuation

train['sentences'] = train.sentences.apply(lambda x : [word for word in x if word not in punctuation])

#remove alpha numeric values

train['sentences'] = train.sentences.apply(lambda x: [word for word in x if word.isalpha()])
