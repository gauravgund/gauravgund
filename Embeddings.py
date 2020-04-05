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

###################### TF-idf Approach #######################################################################

tf = TfidfVectorizer()

def str2(x):
    x1= ''
    for i in x:
        x1+= i
        x1+=' '
    return(x1)

train['sentences1'] = train.sentences.apply(lambda x: ' '.join(x))
test['cleaned_text1'] = test.cleaned_text.apply(lambda x: ' '.join(x))

x_train2 = tf.fit_transform(train.sentences1)
lgbm_tfidf = lightgbm.LGBMClassifier()
lgbm_tfidf.fit(x_train2, y_train)

x_test2= tf.transform(test.cleaned_text1)

predict2 = lgbm_tfidf.predict(x_test2)
accuracy_score(test['category'], predict2)

########################## Fasttext ####################################

from gensim.models.fasttext import FastText 
import pickle as pkl
#ft_model= FastText(train.sentences)

# =============================================================================
# with open('ft_model.pkl', 'wb') as f:
#     pkl.dump(ft_model, f)
# 
# =============================================================================

with open('ft_model.pkl', 'rb') as f:
    ft_model = pkl.load(f)

#ft_model.wv['russia']
vocab = ft_model.wv.vocab    

def wv_average(ft_model, doc):
    words =[word for word in doc if word in vocab]
    
    return(list(np.mean(ft_model.wv[words], axis = 0)))  

train['wv_ft'] = train.sentences.progress_apply(lambda x: wv_average(ft_model, x))
wv_cols  = ['wv'+f'{i}' for i in range(len(train.wv_ft.iloc[0]))]
x_train4 = pd.DataFrame(train.wv_ft.to_list(), columns = wv_cols)
# train model

lgbm4 = lightgbm.LGBMClassifier( )
lgbm4.fit(x_train4, y_train)


def wv_average(ft_model, doc):
    words = [word for word in doc]
    
    return(list(np.mean(ft_model.wv[words], axis = 0)))

test['wv_ft'] = test.cleaned_text.progress_apply(lambda x: wv_average(ft_model, x))

x_test4 = pd.DataFrame(test.wv_ft.to_list(), columns = wv_cols)

predict4 = lgbm4.predict(x_test4)

accuracy_score(test['category'], predict4)

############################ DBOW #######################

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

model = Doc2Vec(dm= 0)

train['sentences1'] = [' '.join(i) for i in train['sentences']]

# tag the document

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train['sentences'])]
#doc_train = [documents[doc].words.split(' ') for doc in range(len(documents))]
model.build_vocab(documents)

# train model
model.train(documents, total_examples = model.corpus_count, epochs = 30)

x_train5 = [model.docvecs[i] for i in range(train['sentences1'].shape[0]) ]

lgbm5 = lightgbm.LGBMClassifier()
lgbm5.fit(x_train5,y_train )

# apply it on test data
#test['cleaned_text1'] = [' '.join(i) for i in test['cleaned_text']]
documents_test= [TaggedDocument(doc, [i]) for i,doc in enumerate(test['cleaned_text'])]
#doc_test= [documents_test[doc].words.split(' ') for i in range(len(documents_test))]
x_test5= [model.infer_vector(documents_test[doc].words, steps= 30) for doc in range(len(documents_test))] 

predict5= lgbm5.predict(x_test5)

#accuracy_score(y_train, lgbm5.predict(x_train))

accuracy_score(test['category'],predict5)

######################### PV-DM #####################################

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

model1 = Doc2Vec(dm= 1)

document_dm = [TaggedDocument(doc, [i]) for i,doc in enumerate(train['sentences'])]

model1.build_vocab(document_dm)

model1.train(document_dm, total_examples = model1.corpus_count, epochs = 30)

x_train6 = [model1.docvecs[i] for i in range(train['sentences'].shape[0])]

lgbm6= lightgbm.LGBMClassifier()

lgbm6.fit(x_train6, y_train)

document_dm_test= [TaggedDocument(doc, [i]) for i,doc in enumerate(test['cleaned_text'])]
x_test6 = [model1.infer_vector(document_dm_test[doc].words, steps = 30) for doc in range(len(document_dm_test))]

predict6 = lgbm6.predict(x_test6)

accuracy_score(test['category'], predict6)

####################### Combining DBOW & PV-DM #################################

x_train7 = [model.docvecs[i]+ model1.docvecs[i] for i in range(train['sentences'].shape[0])]

x_test7 = [model.infer_vector(documents_test[doc].words, steps= 30) + model1.infer_vector(document_dm_test[doc].words, steps = 30) for doc in range(len(document_dm_test))]

lgbm7 = lightgbm.LGBMClassifier()

lgbm7.fit(x_train7, y_train)

predict7 = lgbm7.predict(x_test7)

accuracy_score(test['category'], predict7)

