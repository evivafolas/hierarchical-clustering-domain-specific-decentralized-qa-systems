from __future__ import print_function
import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim_models
from sklearn.decomposition import LatenteDirichletAllocation

from gensim.models.doc2vec import Doc2Vec, TggedDocument
from nltk.tokenize import word_tokenize

import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import gensim

from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import gensim.models

# Topic List Functions

def get_similar_words(word, topic_model, topn=100):
    temp_list=[]
    
    similar_words = topic_model.wv.most_similar(positive=[word], topn=topn)

    for w,similarity in similar_words:
        temp_list.append(w)
    
    return temp_list

def build_topic_dict(words, topic_model, topn=100):
    topic_dict={}

    for word in words:
        topic_dict[word]: get_similar_words(word,topic_model)

    return topic_dict

def get_topic_list(topic_dict):
    return list(topic_dict.values())

def topic_dataframe(topic_list, topic_model):

    topic_word_list = get_topic_list(build_topic_dict(topic_list, topic_model))
    
    topic_df = pd.DataFrame(topic_list, columns=['topics'])
    topic_df['topic_words'] = topic_word_list

    return topic_df

def calculate_topic_embeddings(topic_df,  tf_model):
    return tf_model.encode(topic_df['topic_words'])

# Dataset Text Preprocessing

def create_dataset_20NG(docNo=500):
    docs = fetch_20newsgroups(shuffle=True, random_state=32, remove=('headers','footers','qutes'))
    
    docs_df = pd.DataFrame(
        {'docs': docs.data,
        'target': docs.target}
    )
    
    docs_df = pd.DataFrame(docs,columns = ['docs'])

    stop_words_l = stopwords.words('english')
    docs_df['docs_clean'] = docs_df.docs.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lowe() not in stop_words_l))

    # docs_df_test['target_name'] = docs_df['target'].apply(lambda x: docs.target_names[x])

    return docs_df.iloc[:min(docNo,docs_df.shape[0])]

def calculate_document_embeddings(docs_df, tf_model):
    return tf_model.encode(docs_df["docs_clean"])

# Similarity Functions

def calculate_similarities(topic_emb, docs_emb, type='cosine'):
    if type == 'cosine':
        pw_sims = cosine_similarity(docs_emb,topic_emb)
    elif type == 'euclid':
        pw_sims = euclidean_distances(docs_emb,topic_emb)
    return pw_sims

def most_similar_topics(similarities, type='cosine'):  
    most_similar_matrix = []
    if type == 'cosine':
        for docsims in similarities:
            most_similar_index = np.argmax(docsims)
            most_similar_matrix.append(topic_list[most_similar_index])
    elif type=='euclid':
        for docsims in similarities:
            most_similar_index = np.argmin(docsims)
            most_similar_matrix.append(topic_list[most_similar_index])

    return most_similar_matrix

# LDA 
'''
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc))if word not in stop_words] for doc in texts]

def LDA_preperation(docs_df,stop_words):
    data = docs_df['docs_clean'].values.to_list()
    doc_words = list(sent_to_words(data))
    doc_words = remove_stopwords(doc_words)

    id2word = corpora.Dictionary(doc_words)
    texts   = doc_words
    corpus  = [id2word.doc2bow(text) for text in texts]

    return id2word,corpus
'''
from sklearn.decomposition import TruncatedSVD, PCA, NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups




    
