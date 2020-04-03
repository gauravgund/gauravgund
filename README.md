# Comparing different Embeddings 


This repository takes an initiative in comparing the effect of different Embeddings on the performance of text classification Model. We'll begin with basic bag-of-words approach , tf-idf and NN-based Word2vec, Fasttext, DBOW, PV-DM. 

Data used for the purpose of analysis corresponds to news classification dataset from Kaggle. It is a classic text classification problem in which text has to be classified into one of the categories.

Objective : To predict the class of news based on the news snippet

ML Model: Light GBM 

Software: Python3.7

Success Metric: Accuracy score is considered as a success metric since the number of observations from each class are good in representation.

Techniques tried so far:

Bag-of-words: 85% accuracy

Tf-idf : 95% accuracy

Word2vec : 80% accuracy

Fasttext: 96% accuracy

DBOW: 95% accuracy

PV-DM: 90% accuracy

Combination of DBOW & PV-DM: 91% accuracy




