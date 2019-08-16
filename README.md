# Text Classification with Sparse Composite Document Vectors


## Introduction
  - For text classification and information retrieval tasks, text data has to be represented as a fixed dimension vector. 
  - We propose simple feature construction technique named **Sparse Composite Document Vectors (SCDV).**
  - We demonstrate our method through experiments on multi-class classification on 20newsgroup dataset and multi-label text classification on Reuters-21578 dataset. 

## Testing
There are 2 folders named 20news and Reuters which contains code related to multi-class classification on 20Newsgroup dataset and multi-label classification on Reuters dataset.
#### 20Newsgroup
Change directory to 20news for experimenting on 20Newsgroup dataset and create train and test tsv files as follows:
```sh
$ cd 20news
$ python create_tsv.py
```
Get word vectors for all words in vocabulary:
```sh
$ python Word2Vec.py 200
# Word2Vec.py takes word vector dimension as an argument. We took it as 200.
```
Get Sparse Document Vectors (SCDV) for documents in train and test set and accuracy of prediction on test set:
```sh
$ python SCDV.py 200 60
# SCDV.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```
Get Topic coherence for documents in train set:
```sh
$ python TopicCoherence.py 200 60
# TopicCoherence.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```
#### Reuters
Change directory to Reuters for experimenting on Reuters-21578 dataset. As reuters data is in SGML format, parsing data and creating pickle file of parsed data can be done as follows:
```sh
$ python create_data.py
# We don't save train and test files locally. We split data into train and test whenever needed.
```
Get word vectors for all words in vocabulary: 
```sh
$ python Word2Vec.py 200
# Word2Vec.py takes word vector dimension as an argument. We took it as 200.
```
Get Sparse Document Vectors (SCDV) for documents in train and test set:
```sh
$ python SCDV.py 200 60
# SCDV.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```
Get performance metrics on test set:
```sh
$ python metrics.py 200 60
# metrics.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```

#### Information Retrieval
Change directory to IR for experimenting on information Retrieval task. IR Datasets mentioned in the paper can be downloaded from [TREC website](http://trec.nist.gov/data/docs_eng.html). 

You will need to run the documents and queries through a full fledged IR pipeline system like Apache Lucene or [Project Lemur](https://www.lemurproject.org/) in order to 
  - Tokenize the data, remove stop words and pass tokens through a Porter Stemmer.
  - Build inverted and forward index.
  - Build a basic language model retrieval system with Dirichlet smoothing.

Data Format
  - The IR Data folder must have a file called "queries.txt" and a folder called *raw* that has all the documents.
  - Each file in *raw* should be a single document containing space separated processed tokens. File must be named as doc_ID.txt.
  - Each line in queries.txt should be a single query containing space separated processed words.

To interpolate language model retrieval system with the query-document score obtained from SCDV:

Get word vectors for all terms in vocabulary:
```sh
$ python Word2Vec.py 300 sjm
# Word2Vec.py takes word vector dimension and folder containing IR dataset as arguments. We took 300 and sjm (San Jose Mercury).
```
Create Sparse Document Vectors (SCDV) for all documents and queries and compute similarity scores for all query-document pairs.
```sh
$ python SCDV.py 300 100 sjm
# SCDV.py takes word vector dimension, number of clusters as arguments and folder containing IR dataset as arguments. We took 300 100 and sjm.
# Change the code to store these scores in a format that can be used by the IR system.
```
Use these scores to interpolate with the language model scores with interpolation parameter 0.5.

#### Text classification of Japanese corpus with SCDV

Start docker container for jupyter.

```sh
./run_jupyter.sh
```

Open jupyter in the browser from <http://127.0.0.1:8888/?token=>*token displayed in terminal*.  
Try the following notebook to check text classification of Japanese corpus with SCDV.

- **[SCDV/ipynb/livedoor-new-corpus-test.ipynb](https://github.com/manba036/SCDV/blob/develop/ipynb/livedoor-new-corpus-test.ipynb)**

Sorry for using Japanese.  
SCDVによる日本語コーパスのテキスト分類のもろもろについては、fufufukakakaさんの記事が参考になります。  

- **[文書ベクトルをお手軽に高い精度で作れるSCDVって実際どうなのか日本語コーパスで実験した(EMNLP2017)](https://qiita.com/fufufukakaka/items/a7316273908a7c400868)**


## Requirements
Minimum requirements:
  -  Python 2.7+
  -  NumPy 1.8+
  -  Scikit-learn
  -  Pandas
  -  Gensim

For theory and explanation of SCDV, please visit https://dheeraj7596.github.io/SDV/.

    Note: You neednot download 20Newsgroup or Reuters-21578 dataset. All datasets are present in their respective directories.

[//]: # (We used SGMl parser for parsing Reuters-21578 dataset from  https://gist.github.com/herrfz/7967781)
