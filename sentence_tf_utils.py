from __future__ import print_function
# import pyLDAvis
# import pyLDAvis.sklearn
# import pyLDAvis.gensim_models
# from sklearn.decomposition import LatenteDirichletAllocation
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from string import punctuation
import re
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from string import punctuation
import warnings
# from gensim.models.doc2vec import Doc2Vec, TggedDocument
import os
import nltk
import pandas as pd
from nltk.corpus import stopwords
from bertopic import BERTopic

# from keras.preprocessing.text import Tokenizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

# from gensim.utils import simple_preprocess
# import gensim.corpora as corpora
# import gensim.models

nltk.download('punkt')
nltk.download('stopwords')

# Topic List Functions

def get_similar_words(word, topic_model, topn=100):
    temp_list=[]

    similar_words = topic_model.most_similar(positive=[word], topn=topn)

    for w,similarity in similar_words:
        temp_list.append(w)

    return temp_list

def build_topic_dict(words, topic_model, topn=100):
    topic_dict={}

    for word in words:
        topic_dict[word] = get_similar_words(word,topic_model)

    return topic_dict

def get_topic_list(topic_dict):
    temp_list = []
    topic_words = list(topic_dict.values())

    for i,word_list in enumerate(topic_words):
        temp_list.append(' '.join(word_list))
    return temp_list

# def topic_dataframe(topic_list, topic_model):

#     topic_word_list = get_topic_list(build_topic_dict(topic_list, topic_model))

#     topic_df = pd.DataFrame(topic_list, columns=['topics'])
#     topic_df['topic_words'] = topic_word_list

#     return topic_df

def clean_text(sentence):

    # Remove Non Alphanumeric sequences

    pattern = re.compile(r'[^a-z]+')
    sentence = sentence.lower()
    sentence = pattern.sub(' ', sentence).strip()

    # Tokenize

    word_list = word_tokenize(sentence)

    # Stop Words & Punctuation

    stopwords_list = set(stopwords.words('english'))
    punct = set(punctuation)

    # Remove stop words, very small words & punctuation
    word_list = [word for word in word_list if word not in stopwords_list]
    word_list = [word for word in word_list if len(word) > 2]
    word_list = [word for word in word_list if word not in punct]

    # Stemming or Lemmatization

    '''
    stemmer = PorterStemmer()
    word_list = [stemmer.stem(word) for word in word_list]
    '''

    lemmer = WordNetLemmatizer()

    word_list = [lemmer.lemmatize(word) for word in word_list]

    sentence = ' '.join(word_list)

    return sentence


def topic_dataframe(topic_dict, topic_list):

    topic_word_list = get_topic_list(topic_dict)

    topic_df = pd.DataFrame(topic_list, columns=['topics'])
    topic_df['topic_words'] = topic_word_list

    return topic_df

def calculate_topic_embeddings(topic_df,  tf_model):
    return tf_model.encode(topic_df['topic_words'])

# Dataset Text Preprocessing

def create_dataset_20NG(docNo=500):
    tqdm.pandas()
    dataset = fetch_20newsgroups(shuffle=True, random_state=32, remove=('headers','footers','qutes'))

    docs_df = pd.DataFrame(
        {'docs': dataset.data,
        'target': dataset.target}
    )

    # docs_df = pd.DataFrame(dataset,columns = ['docs'])

    docs_df['docs_clean'] = docs_df['docs'].progress_apply(lambda x: clean_text(str(x)))

    docs_df['target_name'] = docs_df['target'].apply(lambda x: dataset.target_names[x])
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

def most_similar_topics(similarities, topic_list, type='cosine'):
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

def df_csv_save(docs_df, name):
    path = './lib'
    os.makedirs(path, exist_ok=True)
    docs_df.to_csv(path + '/' + name + '.csv')

# LDA
'''
from sklearn.decomposition import TruncatedSVD, PCA, NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
'''



