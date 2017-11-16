
# coding: utf-8

# In[ ]:


import pandas as pd
import glob
import os
from tqdm import tqdm_notebook as tqdm


# In[ ]:


#preprocessing
dirlist = ["dokujo-tsushin","it-life-hack","kaden-channel","livedoor-homme",
           "movie-enter","peachy","smax","sports-watch","topic-news"]


# In[ ]:


df = pd.DataFrame(columns=["class","news"])
for i in tqdm(dirlist):
    path = "../japanese-dataset/livedoor-news-corpus/"+i+"/*.txt"
    files = glob.glob(path)
    files.pop()
    for j in tqdm(files):
        f = open(j)
        data = f.read() 
        f.close()
        t = pd.Series([i,"".join(data.split("\n")[3:])],index = df.columns)
        df  = df.append(t,ignore_index=True)


# In[ ]:


df


# In[ ]:


## create word2vec
import logging
import numpy as np
from gensim.models import Word2Vec
import MeCab
import time
from sklearn.preprocessing import normalize
import sys
import re


# In[ ]:


start = time.time()
tokenizer =  MeCab.Tagger("-Owakati")  
sentences = []
print ("Parsing sentences from training set...")

# Loop over each news article.
for review in tqdm(df["news"]):
    try:
        # Split a review into parsed sentences.
        result = tokenizer.parse(review).replace("\u3000","").replace("\n","")
        result = re.sub(r'[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+', "", result)
        h = result.split(" ")
        h = list(filter(("").__ne__, h))
        sentences.append(h)
    except:
        continue

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

num_features = 200     # Word vector dimensionality
min_word_count = 20   # Minimum word count
num_workers = 40       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

print ("Training Word2Vec model...")
# Train Word2Vec model.
model = Word2Vec(sentences, workers=num_workers, hs = 0, sg = 1, negative = 10, iter = 25,            size=num_features, min_count = min_word_count,             window = context, sample = downsampling, seed=1)

model_name = str(num_features) + "features_" + str(min_word_count) + "minwords_" + str(context) + "context_len2alldata"
model.init_sims(replace=True)
# Save Word2Vec model.
print ("Saving Word2Vec model...")
model.save("../japanese-dataset/livedoor-news-corpus/model/"+model_name)
endmodeltime = time.time()

print ("time : ", endmodeltime-start)


# In[ ]:


model.wv.save_word2vec_format("/Users/01018534/Downloads/testw2v.txt",binary=False)


# In[ ]:


# plain word2vec t-SNE Visualization
word2vec_model=pickle.load(open("../japanese-dataset/livedoor-news-corpus/model/"+model_name,"rb"))
 
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import numpy as np

skip=0
limit=241 

vocab = word2vec_model.wv.vocab
emb_tuple = tuple([word2vec_model[v] for v in vocab])
X = np.vstack(emb_tuple)
 
tsne_model = TSNE(n_components=2, random_state=0,verbose=2)
np.set_printoptions(suppress=True)
tsne_model.fit_transform(X) 


# In[ ]:


pickle.dump(tsne_model,open("../japanese-dataset/livedoor-news-corpus/model/tsne_plain.pkl","wb"))


# In[ ]:


## create gwbowv
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[ ]:


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def cluster_GMM(num_clusters, word_vectors):
    # Initalize a GMM object and use it for clustering.
    clf =  GaussianMixture(n_components=num_clusters,
                    covariance_type="tied", init_params='kmeans', max_iter=50)
    # Get cluster assignments.
    clf.fit(word_vectors)
    idx = clf.predict(word_vectors)
    print ("Clustering Done...", time.time()-start, "seconds")
    # Get probabilities of cluster assignments.
    idx_proba = clf.predict_proba(word_vectors)
    # Dump cluster assignments and probability of cluster assignments. 
    pickle.dump(idx, open('../japanese-dataset/livedoor-news-corpus/model/gmm_latestclusmodel_len2alldata.pkl',"wb"))
    print ("Cluster Assignments Saved...")

    pickle.dump(idx_proba,open( '../japanese-dataset/livedoor-news-corpus/model/gmm_prob_latestclusmodel_len2alldata.pkl',"wb"))
    print ("Probabilities of Cluster Assignments Saved...")
    return (idx, idx_proba)

def read_GMM(idx_name, idx_proba_name):
    # Loads cluster assignments and probability of cluster assignments. 
    idx = pickle.load(open('../japanese-dataset/livedoor-news-corpus/model/gmm_latestclusmodel_len2alldata.pkl',"rb"))
    idx_proba = pickle.load(open( '../japanese-dataset/livedoor-news-corpus/model/gmm_prob_latestclusmodel_len2alldata.pkl',"rb"))
    print ("Cluster Model Loaded...")
    return (idx, idx_proba)

def get_probability_word_vectors(featurenames, model,word_centroid_map, num_clusters, word_idf_dict):
    # This function computes probability word-cluster vectors
    prob_wordvecs = {}
    for word in word_centroid_map:
        prob_wordvecs[word] = np.zeros( num_clusters * num_features, dtype="float32" )
        for index in range(0, num_clusters):
            try:
                prob_wordvecs[word][index*num_features:(index+1)*num_features] = model[word] * word_centroid_prob_map[word][index] * word_idf_dict[word]
            except:
                continue

    # prob_wordvecs_idf_len2alldata = {}
    # i = 0
    # for word in featurenames:
    #     i += 1
    #     if word in word_centroid_map:    
    #         prob_wordvecs_idf_len2alldata[word] = {}
    #         for index in range(0, num_clusters):
    #                 prob_wordvecs_idf_len2alldata[word][index] = model[word] * word_centroid_prob_map[word][index] * word_idf_dict[word] 

    

    # for word in prob_wordvecs_idf_len2alldata.keys():
    #     prob_wordvecs[word] = prob_wordvecs_idf_len2alldata[word][0]
    #     for index in prob_wordvecs_idf_len2alldata[word].keys():
    #         if index==0:
    #             continue
    #         prob_wordvecs[word] = np.concatenate((prob_wordvecs[word], prob_wordvecs_idf_len2alldata[word][index]))
    return prob_wordvecs

def create_cluster_vector_and_gwbowv(prob_wordvecs, wordlist, word_centroid_map, word_centroid_prob_map, dimension, word_idf_dict, featurenames, num_centroids, train=False):
    # This function computes SDV feature vectors.
    bag_of_centroids = np.zeros( num_centroids * dimension, dtype="float32" )
    global min_no
    global max_no

    for word in wordlist:
        try:
            temp = word_centroid_map[word]
        except:
            continue

        bag_of_centroids += prob_wordvecs[word]

    norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
    if(norm!=0):
        bag_of_centroids /= norm

    # To make feature vector sparse, make note of minimum and maximum values.
    if train:
        min_no += min(bag_of_centroids)
        max_no += max(bag_of_centroids)

    return bag_of_centroids


# In[ ]:


num_features = 200     # Word vector dimensionality
min_word_count = 20   # Minimum word count
num_workers = 40       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

model_name = str(num_features) + "features_" + str(min_word_count) + "minwords_" + str(context) + "context_len2alldata"
# Load the trained Word2Vec model.
model = Word2Vec.load("../japanese-dataset/livedoor-news-corpus/model/"+model_name)
# Get wordvectors for all words in vocabulary.
word_vectors = model.wv.syn0

# Load train data.
train,test = train_test_split(df,test_size=0.3,random_state=40)
all = df

# Set number of clusters.
num_clusters = 40
# Uncomment below line for creating new clusters.
idx, idx_proba = cluster_GMM(num_clusters, word_vectors)

# Uncomment below lines for loading saved cluster assignments and probabaility of cluster assignments.
# idx_name = "gmm_latestclusmodel_len2alldata.pkl"
# idx_proba_name = "gmm_prob_latestclusmodel_len2alldata.pkl"
# idx, idx_proba = read_GMM(idx_name, idx_proba_name)

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip( model.wv.index2word, idx ))
# Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
# list of probabilities of cluster assignments.
word_centroid_prob_map = dict(zip( model.wv.index2word, idx_proba ))


# In[ ]:


# color scatter map
plt.style.use('ggplot') 
plainw2v = pd.DataFrame(tsne_model.embedding_[skip:limit, 0],columns = ["x"])
plainw2v["y"] = pd.DataFrame(tsne_model.embedding_[skip:limit, 1])
plainw2v["word"] = list(vocab)[skip:limit]
plainw2v["cluster"] = idx[skip:limit]
plainw2v.plot.scatter(x="x",y="y",c="cluster",cmap="viridis",figsize=(8, 6),s=30)

# import plotly
# import plotly.plotly as py
# import plotly.graph_objs as go
# plotly.offline.init_notebook_mode(connected=False)
# # Create a trace
# trace = go.Scatter(
#     x =model.embedding_[skip:limit, 0],
#     y = model.embedding_[skip:limit, 1],
        
#     text=list(vocab.keys()),
#     mode = 'markers+text'
# )

# data = [trace]

# # Plot and embed in ipython notebook!
# plotly.offline.iplot(data)


# In[ ]:


# Computing tf-idf values.
traindata = []
for review in all["news"]:
    result = tokenizer.parse(review).replace("\u3000","").replace("\n","")
    result = re.sub(r'[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+', "", result)
    h = result.split(" ")
    h = filter(("").__ne__, h)
    traindata.append(" ".join(h))

tfv = TfidfVectorizer(dtype=np.float32)
tfidfmatrix_traindata = tfv.fit_transform(traindata)
featurenames = tfv.get_feature_names()
idf = tfv._tfidf.idf_

# Creating a dictionary with word mapped to its idf value 
print ("Creating word-idf dictionary for Training set...")

word_idf_dict = {}
for pair in zip(featurenames, idf):
    word_idf_dict[pair[0]] = pair[1]
    
# Pre-computing probability word-cluster vectors.
prob_wordvecs = get_probability_word_vectors(featurenames, model,word_centroid_map, num_clusters, word_idf_dict)


# In[ ]:


from gensim.models import KeyedVectors
## for gensim keyedvectors
file = open('../japanese-dataset/livedoor-news-corpus/model/temp2.txt', 'w')
string_list = [str(len(prob_wordvecs)),str(len(list(prob_wordvecs.values())[0]))]
string_list.append("\n")
file.writelines(" ".join(string_list))
for key,value in tqdm(prob_wordvecs.items()):
    string_list = []
    string_list.append(key)
    for i in value:
        string_list.append(str(i))
    string_list.append("\n")
    file.writelines(" ".join(string_list))
file.close()


# In[ ]:


word2vec_weighted = KeyedVectors.load_word2vec_format("../japanese-dataset/livedoor-news-corpus/model/temp2.txt",binary=False)
word2vec_weighted.save("../japanese-dataset/livedoor-news-corpus/model/vector-response-test/word2vec_weighted.model")


# In[ ]:


pickle.dump(prob_wordvecs,open("../japanese-dataset/livedoor-news-corpus/model/prob_wordvecs.pkl","wb"))


# In[ ]:


# gwbowv is a matrix which contains normalised document vectors.
gwbowv = np.zeros( (train["news"].size, num_clusters*(num_features)), dtype="float32")

counter = 0

min_no = 0
max_no = 0
for review in train["news"]:
    # Get the wordlist in each news article.
    result = tokenizer.parse(review).replace("\u3000","").replace("\n","")
    result = re.sub(r'[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+', "", result)
    h = result.split(" ")
    h = filter(("").__ne__, h)
    words = h
    gwbowv[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map, word_centroid_prob_map, num_features, word_idf_dict, featurenames, num_clusters, train=True)
    counter+=1
    if counter % 1000 == 0:
        print ("Train News Covered : ",counter)

gwbowv_name = "SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"

gwbowv_test = np.zeros( (test["news"].size, num_clusters*(num_features)), dtype="float32")

counter = 0

for review in test["news"]:
    # Get the wordlist in each news article.
    result = tokenizer.parse(review).replace("\u3000","").replace("\n","")
    result = re.sub(r'[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+', "", result)
    h = result.split(" ")
    h = filter(("").__ne__, h)
    words = h
    gwbowv_test[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map, word_centroid_prob_map, num_features, word_idf_dict, featurenames, num_clusters)
    counter+=1
    if counter % 1000 == 0:
        print ("Test News Covered : ",counter)

test_gwbowv_name = "TEST_SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"


# In[ ]:


len(gwbowv[0])


# In[ ]:


print ("Making sparse...")
# Set the threshold percentage for making it sparse. 
percentage = 0.04
min_no = min_no*1.0/len(train["news"])
max_no = max_no*1.0/len(train["news"])
print ("Average min: ", min_no)
print ("Average max: ", max_no)
thres = (abs(max_no) + abs(min_no))/2
thres = thres*percentage

# Make values of matrices which are less than threshold to zero.
temp = abs(gwbowv) < thres
gwbowv[temp] = 0

temp = abs(gwbowv_test) < thres
gwbowv_test[temp] = 0

#saving gwbowv train and test matrices
np.save("../japanese-dataset/livedoor-news-corpus/model/"+gwbowv_name, gwbowv)
np.save("../japanese-dataset/livedoor-news-corpus/model/"+test_gwbowv_name, gwbowv_test)


# In[ ]:


## plot modified word vector representation
skip=0
limit=241 

vocab = list(prob_wordvecs.keys())
tsne_target = []
for i in range(limit):
    tsne_target.append(prob_wordvecs[vocab[i]])
X = np.vstack(tsne_target)
 
tsne_model_scdv = TSNE(n_components=2, random_state=0,verbose=2)
np.set_printoptions(suppress=True)
tsne_model_scdv.fit_transform(X)
pickle.dump(tsne_model_scdv,open("../japanese-dataset/livedoor-news-corpus/model/tsne_scdv.pkl","wb"))


# In[ ]:


scdv_tsne = pd.DataFrame(tsne_model_scdv.embedding_[skip:limit, 0],columns = ["x"])
scdv_tsne["y"] = pd.DataFrame(tsne_model_scdv.embedding_[skip:limit, 1])
scdv_tsne["word"] = list(vocab)[skip:limit]
scdv_tsne["cluster"] = idx[skip:limit]
scdv_tsne.plot.scatter(x="x",y="y",c="cluster",cmap="viridis",figsize=(8, 6),s=30)


# In[ ]:


# import plotly
# import plotly.plotly as py
# import plotly.graph_objs as go
# plotly.offline.init_notebook_mode(connected=False)
# # Create a trace
# trace = go.Scatter(
#     x =tsne_model_scdv.embedding_[skip:limit, 0],
#     y = tsne_model_scdv.embedding_[skip:limit, 1],
#     text=list(vocab)[skip:limit],
#     mode = 'markers+text'
# )

# data = [trace]

# # Plot and embed in ipython notebook!
# plotly.offline.iplot(data)


# In[ ]:


## test lgb
from sklearn.metrics import classification_report
import lightgbm as lgb

start = time.time()
clf = lgb.LGBMClassifier(objective="multiclass")
clf.fit(gwbowv, train["class"])
Y_true, Y_pred  = test["class"], clf.predict(gwbowv_test)
print ("Report")
print (classification_report(Y_true, Y_pred, digits=6))
print ("Accuracy: ",clf.score(gwbowv_test,test["class"]))
print ("Time taken:", time.time() - start, "\n")


# In[ ]:


# res = []
# for review in all["news"]:
#     # Get the wordlist in each news article.
#     result = tokenizer.parse(review).replace("\u3000","").replace("\n","")
#     result = re.sub(r'[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+', "", result)
#     h = result.split(" ")
#     h = filter(("").__ne__, h)
#     words = list(h)
#     res.append(" ".join(words))
# corpus = " ".join(res)
# f = open('../japanese-dataset/livedoor-news-corpus/for-fasttext/corpus.txt', 'w') # 書き込みモードで開く
# f.write(corpus) # 引数の文字列をファイルに書き込む
# f.close() # ファイルを閉じる


# In[ ]:


# from gensim.models.wrappers import FastText
# fasttext_model = FastText.load_fasttext_format('../japanese-dataset/livedoor-news-corpus/for-fasttext/fasttext_model_200dim')


# In[ ]:


## fasttext
import fasttext
fasttext_model = fasttext.skipgram('../japanese-dataset/livedoor-news-corpus/for-fasttext/corpus.txt',
                          '../japanese-dataset/livedoor-news-corpus/for-fasttext/fasttext_model',
                            dim=200)


# In[ ]:


def plain_docvec(num_features, wordlist,model,train=False):
    # This function computes SDV feature vectors.
    bag_of_centroids = np.zeros( num_features, dtype="float32" )
    global min_no
    global max_no

    for word in wordlist:
        bag_of_centroids += model[word]

    norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
    if(norm!=0):
        bag_of_centroids /= norm

    # To make feature vector sparse, make note of minimum and maximum values.
    if train:
        min_no += min(bag_of_centroids)
        max_no += max(bag_of_centroids)

    return bag_of_centroids


# In[ ]:


# gwbowv is a matrix which contains normalised document vectors.
num_features = 2000
plain_fasttext = np.zeros( (train["news"].size, num_features), dtype="float32")

counter = 0

min_no = 0
max_no = 0
for review in tqdm(train["news"]):
    # Get the wordlist in each news article.
    result = tokenizer.parse(review).replace("\u3000","").replace("\n","")
    result = re.sub(r'[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+', "", result)
    h = result.split(" ")
    h = filter(("").__ne__, h)
    words = list(h)
    plain_fasttext[counter] = plain_docvec(num_features, words, model,train=True)
    counter+=1

plain_fasttext_test = np.zeros( (test["news"].size, num_features), dtype="float32")

counter = 0

for review in tqdm(test["news"]):
    # Get the wordlist in each news article.
    result = tokenizer.parse(review).replace("\u3000","").replace("\n","")
    result = re.sub(r'[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+', "", result)
    h = result.split(" ")
    h = filter(("").__ne__, h)
    words = list(h)
    plain_fasttext_test[counter] = plain_docvec(num_features, words, model,train=False)
    counter+=1


# In[ ]:


## test lgb fasttext-average
from sklearn.metrics import classification_report
import lightgbm as lgb

start = time.time()
clf = lgb.LGBMClassifier(objective="multiclass")
clf.fit(plain_fasttext, train["class"])
Y_true, Y_pred  = test["class"], clf.predict(plain_fasttext_test)
print ("Report")
print (classification_report(Y_true, Y_pred, digits=6))
print ("Accuracy: ",clf.score(plain_fasttext_test,test["class"]))
print ("Time taken:", time.time() - start, "\n")


# In[ ]:


## SCDV based fasttext
from gensim.models.wrappers.fasttext import FastText
fasttext_model_200 = FastText.load_fasttext_format('../japanese-dataset/livedoor-news-corpus/for-fasttext/fasttext_model_200dim')


# In[ ]:


# Get wordvectors for all words in vocabulary.
word_vectors = fasttext_model_200.wv.syn0

# Set number of clusters.
num_clusters = 60
# Uncomment below line for creating new clusters.
idx, idx_proba = cluster_GMM(num_clusters, word_vectors)

# Uncomment below lines for loading saved cluster assignments and probabaility of cluster assignments.
# idx_name = "gmm_latestclusmodel_len2alldata.pkl"
# idx_proba_name = "gmm_prob_latestclusmodel_len2alldata.pkl"
# idx, idx_proba = read_GMM(idx_name, idx_proba_name)

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip( fasttext_model_200.wv.index2word, idx ))
# Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
# list of probabilities of cluster assignments.
word_centroid_prob_map = dict(zip( fasttext_model_200.wv.index2word, idx_proba ))


# In[ ]:


num_features = 200


# In[ ]:


# Pre-computing probability word-cluster vectors.
prob_wordvecs = get_probability_word_vectors(featurenames, fasttext_model_200,word_centroid_map, num_clusters, word_idf_dict)


# In[ ]:


## for gensim keyedvectors
file = open('../japanese-dataset/livedoor-news-corpus/model/temp3.txt', 'w')
string_list = [str(len(prob_wordvecs)),str(len(list(prob_wordvecs.values())[0]))]
string_list.append("\n")
file.writelines(" ".join(string_list))
for key,value in tqdm(prob_wordvecs.items()):
    string_list = []
    string_list.append(key)
    for i in value:
        string_list.append(str(i))
    string_list.append("\n")
    file.writelines(" ".join(string_list))
file.close()


# In[ ]:


fasttext_weighted = KeyedVectors.load_word2vec_format("../japanese-dataset/livedoor-news-corpus/model/temp3.txt",binary=False)
fasttext_weighted.save("../japanese-dataset/livedoor-news-corpus/model/vector-response-test/fasttext_weighted.model")


# In[ ]:


pickle.dump(prob_wordvecs,open("../japanese-dataset/livedoor-news-corpus/model/prob_wordvecs_fasttext.pkl","wb"))


# In[ ]:


# gwbowv is a matrix which contains normalised document vectors.
gwbowv_fasttext = np.zeros( (train["news"].size, num_clusters*(num_features)), dtype="float32")

counter = 0

min_no = 0
max_no = 0
for review in train["news"]:
    # Get the wordlist in each news article.
    result = tokenizer.parse(review).replace("\u3000","").replace("\n","")
    result = re.sub(r'[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+', "", result)
    h = result.split(" ")
    h = filter(("").__ne__, h)
    words = list(h)
    gwbowv_fasttext[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map, word_centroid_prob_map, num_features, word_idf_dict, featurenames, num_clusters, train=True)
    counter+=1
    if counter % 1000 == 0:
        print ("Train News Covered : ",counter)

gwbowv_name = "SDV_fasttext_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"

gwbowv_fasttext_test = np.zeros( (test["news"].size, num_clusters*(num_features)), dtype="float32")

counter = 0

for review in test["news"]:
    # Get the wordlist in each news article.
    result = tokenizer.parse(review).replace("\u3000","").replace("\n","")
    result = re.sub(r'[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+', "", result)
    h = result.split(" ")
    h = filter(("").__ne__, h)
    words = list(h)
    gwbowv_fasttext_test[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map, word_centroid_prob_map, num_features, word_idf_dict, featurenames, num_clusters)
    counter+=1
    if counter % 1000 == 0:
        print ("Test News Covered : ",counter)

test_gwbowv_name = "TEST_SDV_fasttext_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"


# In[ ]:


gwbowv_fasttext[0]


# In[ ]:


print ("Making sparse...")
# Set the threshold percentage for making it sparse. 
percentage = 0.04
min_no = min_no*1.0/len(train["news"])
max_no = max_no*1.0/len(train["news"])
print ("Average min: ", min_no)
print ("Average max: ", max_no)
thres = (abs(max_no) + abs(min_no))/2
thres = thres*percentage

# Make values of matrices which are less than threshold to zero.
temp = abs(gwbowv_fasttext) < thres
gwbowv_fasttext[temp] = 0

temp = abs(gwbowv_fasttext_test) < thres
gwbowv_fasttext_test[temp] = 0

#saving gwbowv train and test matrices
np.save("../japanese-dataset/livedoor-news-corpus/model/"+gwbowv_name, gwbowv_fasttext)
np.save("../japanese-dataset/livedoor-news-corpus/model/"+test_gwbowv_name, gwbowv_fasttext_test)


# In[ ]:


## test lgb SCDV based fasttext
from sklearn.metrics import classification_report
import lightgbm as lgb

start = time.time()
clf = lgb.LGBMClassifier(objective="multiclass")
clf.fit(gwbowv_fasttext, train["class"])
Y_true, Y_pred  = test["class"], clf.predict(gwbowv_fasttext_test)
print ("Report")
print (classification_report(Y_true, Y_pred, digits=6))
print ("Accuracy: ",clf.score(gwbowv_fasttext_test,test["class"]))
print ("Time taken:", time.time() - start, "\n")

