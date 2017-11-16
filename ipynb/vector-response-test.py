
# coding: utf-8

# In[ ]:


from gensim.models import KeyedVectors
from gensim.models.wrappers.fasttext import FastText


# In[ ]:


## load model
word2vec = KeyedVectors.load("../japanese-dataset/livedoor-news-corpus/model/vector-response-test/word2vec200.model")
word2vec_weighted = KeyedVectors.load("../japanese-dataset/livedoor-news-corpus/model/vector-response-test/word2vec_weighted.model")
fasttext = FastText.load_fasttext_format("../japanese-dataset/livedoor-news-corpus/model/vector-response-test/fasttext_model_200dim")
fasttext_weighted = KeyedVectors.load("../japanese-dataset/livedoor-news-corpus/model/vector-response-test/fasttext_weighted.model")
poincare_vec = KeyedVectors.load("../japanese-dataset/livedoor-news-corpus/model/vector-response-test/poincare_vec.model")
poincare_vec_weighted = KeyedVectors.load("../japanese-dataset/livedoor-news-corpus/model/vector-response-test/poincare_vec_weighted.model")


# In[ ]:


len(word2vec.most_similar("独身"))


# In[ ]:


model_list = [word2vec,word2vec_weighted,fasttext,fasttext_weighted,poincare_vec,poincare_vec_weighted]
column_list = ["word2vec","word2vec_weighted","fasttext","fasttext_weighted","poincare_vec","poincare_vec_weighted"]


# In[ ]:


import pandas as pd
target_words = ["女性","男性","彼氏","ダイエット","ホテル","新しく",
                "エンジニア","アップル","転職","無料","もうすぐ",
                "終わり","コンビニ","確率",
                "アンドロイド","アプリ","本田圭佑","サッカー","コピー"]
result = pd.DataFrame(target_words)
result.columns = ["target_words"]


# In[ ]:


for name,model in zip(column_list,model_list):
    res = []
    for target in target_words:
        similars = model.most_similar(target)
        sim_res = []
        for sim in similars:
            sim_res.append(sim[0])
        res.append(" ".join(sim_res))
    result[name] = res


# In[ ]:


result

