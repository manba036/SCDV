
# coding: utf-8

# In[ ]:


# tfidf ... http://catindog.hatenablog.com/entry/2017/02/15/222915
# word2vec model ... https://github.com/shiroyagicorp/japanese-word2vec-model-builder


# In[ ]:


import time
import warnings
from gensim.models import Word2Vec
import pandas as pd
import time
from numpy import float32
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.svm import SVC, LinearSVC
import pickle
from math import *
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize


# In[ ]:


num_features = 200     # Word vector dimensionality
min_word_count = 20   # Minimum word count
num_workers = 40       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words


# In[ ]:


# Load the trained Word2Vec model.
model = Word2Vec.load('../japanese-dataset/data/word2vec.gensim.model')
# Get wordvectors for all words in vocabulary.
word_vectors = model.wv.syn0


# In[ ]:


# plain word2vec t-SNE Visualization
word2vec_model=model
 
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import numpy as np

skip=0
limit=241 

vocab = word2vec_model.wv.vocab
emb_tuple = tuple([word2vec_model[v] for v in vocab])
X = np.vstack(emb_tuple)
 
model = TSNE(n_components=2, random_state=0,verbose=2)
np.set_printoptions(suppress=True)
model.fit_transform(X) 

plt.figure(figsize=(40,40))
plt.scatter(model.embedding_[skip:limit, 0], model.embedding_[skip:limit, 1])
 
count = 0
for label, x, y in zip(vocab, model.embedding_[:, 0], model.embedding_[:, 1]):
    count +=1
    if(count<skip):continue
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    if(count==limit):break
plt.show()


# In[ ]:


def cluster_GMM(num_clusters, word_vectors):
    # Initalize a GMM object and use it for clustering.
    clf =  GaussianMixture(n_components=num_clusters,
                    covariance_type="tied", init_params='kmeans', max_iter=50)
    # Get cluster assignments.
    clf.fit(word_vectors)
    idx = clf.predict(word_vectors)
    print ("Clustering Done...")
    # Get probabilities of cluster assignments.
    idx_proba = clf.predict_proba(word_vectors)
#   # Dump cluster assignments and probability of cluster assignments. 
#   joblib.dump(idx, 'gmm_latestclusmodel_len2alldata.pkl')
#   print ("Cluster Assignments Saved...")

#   joblib.dump(idx_proba, 'gmm_prob_latestclusmodel_len2alldata.pkl')
#   print ("Probabilities of Cluster Assignments Saved...")
    return idx, idx_proba

def read_GMM(idx_name, idx_proba_name):
    # Loads cluster assignments and probability of cluster assignments. 
    idx = joblib.load(idx_name)
    idx_proba = joblib.load(idx_proba_name)
    print ("Cluster Model Loaded...")
    return (idx, idx_proba)


# In[ ]:


# Set number of clusters.
num_clusters = 60
# Uncomment below line for creating new clusters.
idx, idx_proba = cluster_GMM(num_clusters, word_vectors)


# In[ ]:


# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip( model.wv.index2word, idx ))
# Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
# list of probabilities of cluster assignments.
word_centroid_prob_map = dict(zip( model.wv.index2word, idx_proba ))


# In[ ]:


word_idf_dict = open('../japanese-dataset/data/words_idf.json', 'r')

