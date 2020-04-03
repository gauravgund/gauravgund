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

data = pd.read_csv('bbc-text.csv')

data.columns

data.category.value_counts(normalize = True)

# train and test

train, test = train_test_split(data, test_size = 0.2)

######################### train data pre-processing ####################
# remove numbers

train['cleaned_text'] = train.text.str.replace('\d+', '')

# remove punctuation

train['cleaned_text'] = train.cleaned_text.apply(lambda x: [''.join(word) for word in x if word not in punctuation])

def str(x):
    x1= ''
    for i in x:
        x1+= i
        x1+=''
    return(x1)

train['cleaned_text'] = train.cleaned_text.apply(lambda x: str(x) )

# lowercase

train['cleaned_text'] = [word.lower() for word in train['cleaned_text']]

# remove extraspaces

train['cleaned_text'] = train.cleaned_text.str.replace('\s+', ' ')

# convert to snetences

train['sentences'] = [nltk.word_tokenize(i) for i in train['cleaned_text']]

#train['word_tok'] = [[sent.split(" ") for sent in para ] for para in  train['sentences'] ]
train['sentences'] = train.sentences.apply(lambda x: [word for word in x if word not in stop])

# remove punctuation

train['sentences'] = train.sentences.apply(lambda x : [word for word in x if word not in punctuation])

# remove alpha numeric values

train['sentences'] = train.sentences.apply(lambda x: [word for word in x if word.isalpha()])

# Target train

y_train = train[['category']]

################################## test data preprocessing ###################

def preprocess(test):
    test['cleaned_text'] = test.text.str.replace('\d+', '')

    # remove punctuation

    test['cleaned_text'] = test.cleaned_text.apply(lambda x : [word for word in x if word not in punctuation])
 
    # convert to string
    test['cleaned_text'] = test.cleaned_text.apply(lambda x : str(x))

    # to lower case
    test['cleaned_text'] = [i.lower() for i in test['cleaned_text']]

    # remove whitespaces
    test['cleaned_text'] = test.text.str.replace('\s+', ' ')

    # tokenize
    test['cleaned_text'] = [nltk.word_tokenize(i) for i in test['cleaned_text']]

    # remove stopwords
    test['cleaned_text'] = test.cleaned_text.apply(lambda x: [word for word in x if word not in stop])

    # remove alpha numeric

    test['cleaned_text'] = test.cleaned_text.apply(lambda x: [word for word in x if word.isalpha()])
    test1 = test
    return(test1)
    
    
 ####################### Word2vec approach ############################


# get list of words

voc = {}

for sentence in tqdm(train.sentences):
    for word in sentence:
           try:
              voc[word]+= 1
           except:
               voc[word] = 1
# get the first value
next(iter(voc))

# length of vocabulary
len(voc)

# create word embeddings
wv = Word2Vec(train.sentences,window = 2, min_count= 1, negative = 5 )
wv.save('w2v.model')
vocab = wv.wv.vocab

# since word2vec doesn't deal with out of vocabulary words
# we should capture words which might be out of vocabulary
def oov_coverage(voc, wv):
    i= 0
    k= 0
    a= {}
    oov = {}


    for word in tqdm(voc):
        try:
            a[word] = wv[word]
            i += voc[word]
        except:
            oov[word] = voc[word]
            k+= voc[word]
    return(i*1.0/(k+i), oov)

m, ovv1 = oov_coverage(voc, wv)

# function to get embeddings for the sentences by taking the average of word emebddings
def wv_average(wv, doc):
    words =[word for word in doc if word in vocab]
    
    return(list(np.mean(wv[words], axis = 0)))

# get wordembeddings for training data

train['wv_avg'] = train.sentences.progress_apply(lambda x : wv_average(wv, x))  

# create word embeddings to columns
wv_cols  = ['wv'+f'{i}' for i in range(len(train.wv_avg.iloc[0]))]

# get embeddings in the dataframe format for each news snippet

x_train1= pd.DataFrame(train.wv_avg.tolist(), columns  = wv_cols)

# initialise a lgbm model

lgbm = lightgbm.LGBMClassifier()

# train the model
lgbm.fit(x_train1, y_train)

# test the model
# get embeddings for test data

test['wv_avg'] = test.cleaned_text.apply(lambda x: wv_average(wv, x))

x_test1 = pd.DataFrame(test.wv_avg.tolist(), columns = wv_cols)



## prediction

pred1 = lgbm.predict(x_test1)

# aaccuracy
accuracy_score(test['category'], pred1)
