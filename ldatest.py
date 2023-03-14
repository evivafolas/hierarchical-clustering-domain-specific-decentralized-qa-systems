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

# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')

# import nltk
# nltk.download('wordnet')
# nltk.download('punkt')

# from jupytertehmes import jtplot

import umap
from sklearn.decomposition import TruncatedSVD, PCA, NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaMulticore

#######################################################
# set plot rc parameters

# jtplot.style(grid=False)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#464646'
#plt.rcParams['axes.edgecolor'] = '#FFFFFF'
plt.rcParams['figure.figsize'] = 10, 7
plt.rcParams['text.color'] = '#666666'
plt.rcParams['axes.labelcolor'] = '#666666'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.color'] = '#666666'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.color'] = '#666666'
plt.rcParams['ytick.labelsize'] = 14

# plt.rcParams['font.size'] = 16

sns.color_palette('dark')
# %matplotlib inline

#######################################################

dataset = fetch_20newsgroups(
    shuffle=True,
    random_state=32,
    remove=('headers', 'footers', 'qutes')
)


# for idx in range(10):
#     print(dataset.data[idx], '\n\n', '#'*50, '\n\n')

news_df = pd.DataFrame(
    {'News': dataset.data,
     'Target': dataset.target}
)

# print(news_df.shape)

news_df['Target_name'] = news_df['Target'].apply(lambda x: dataset.target_names[x])

"""
# plot distribution of topics in news data
fig = plt.figure(figsize=[10,7])
ax = sns.countplot(news_df['Target_name'], color=sns.xkcd_rgb['greenish cyan'])
plt.title('Distribution of Topics')
plt.xlabel('Topics')
plt.ylabel('Count of topics')
plt.xticks(rotation=90)
"""
texts=[]

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
    texts.append(word_list)
    return sentence

# Progress Monitoring
tqdm.pandas()

news_df['News'] = news_df['News'].progress_apply(lambda x: clean_text(str(x)))

print(news_df.head)

#######################################################
'''
wordcloud = WordCloud(background_color='black',
                      max_words=200).generate(str(news_df['News']))
fig = plt.figure(figsize=[16,16])
plt.title('WordCloud of News')
plt.axis('off')
plt.imshow(wordcloud)
plt.show()
'''
#######################################################

tfidf_vec = TfidfVectorizer(tokenizer=lambda x: str(x).split())
X = tfidf_vec.fit_transform(news_df['News'])

#######################################################
'''
# t-SNE Visualization 

tsne = TSNE(
    n_components=2,
    perplexity=50,
    learning_rate=300,
    n_iter=800,
    verbose=1
)

components = tsne.fit_transform(X)

def plot_embeddings(embedding, title):
    fig = plt.figure(figsie=[15,12])
    ax =sns.scatterplot(embedding[:,0], embedding[:,1], hue=news_df['Target_name'])

    plt.title(title)
    plt.xlabel('Axis 1')
    plt.ylabel('Axis 2')
    plt.legend(bbox_to_anchor=(1.05,1), loc=2)
    plt.show()
    return

plot_embeddings(components, 'Visualizing News Vectors with t-SNE')
'''
#######################################################

# texts = texts
id2word = corpora.Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]

#######################################################

# LSA Topic Model

svd_model = TruncatedSVD(
    n_components=20,
    random_state=12,
    n_iter=100,
    algorithm='randomized'
)

svd_model.fit(X)

# print(svd_model.components_.shape)

doc_topic = svd_model.fit_transform(X)

terms = tfidf_vec.get_feature_names_out()
# print(len(terms))


# LDA Topic Model 

lda_model = LatentDirichletAllocation(
    n_components=20,
    random_state=12,
    learning_method='online',
    max_iter=5,
    learning_offset=50
)

lda_model.fit(X)

doc_topic_lda = lda_model.transform(X)

'''
# Second Implemenation of LDA Topic Model (Gensim)

lda_model = LdaModel(
    corpus=corpus, 
    num_topics=20,
    id2word=id2word,
    eval_every=None,
    passes=10
)
'''
#######################################################

def map_word2topic(components, terms):
    
    words2topics = pd.Series()

    for idx, component in enumerate(components):

        term_topic = pd.Series(component, index=terms)

        term_topic.sort_values(ascending=False, inplace=True)

        words2topics['topic '+str(idx)] = list(term_topic.iloc[:10].index)
    
    return words2topics

#######################################################

words2topics = map_word2topic(svd_model.components_, terms=terms)

word2topics_lda = map_word2topic(lda_model.components_, terms)

print('LSA Topic Words \n')
print('Topics\tWords')
for idx, item in zip(words2topics.index, words2topics):
    print(idx,'\t', item)
print('\n')
print('LDA Topic Words\n')
print('Topics\t\tWords')
for idx, item in zip(word2topics_lda.index, word2topics_lda):
    print(idx,'\t',item)

#######################################################

def get_topN_topics(x, topN=10):
    
    topn = list(x.sort_values(ascending=False).head(topN).index) + list(x.sort_values(ascending=False).head(topN).values)
    
    return topn

def map_topicword2doc(model, X, topN=10):
    
    cols = ['topic_'+str(i+1)+'_name' for i in range(topN)] + ['topic_'+str(i+1)+'_prob' for i in range(topN)]
    
    doc_topic = model.fit_transform(X)
    topics = ['topic'+str(i) for i in range(20)]

    doc_topics_df = pd.DataFrame(doc_topic, columns=topics)

    outdf = doc_topics_df.progress_apply(lambda x: get_topN_topics(x, topN), axis=1)
    outdf = pd.DataFrame(dict(zip(outdf.index, outdf.values))).T
    outdf.columns = cols

    return outdf

#######################################################

top_topics = map_topicword2doc(svd_model, X)
news_topics = pd.concat([news_df, top_topics], axis=1)

news_topics= news_topics.infer_objects()

# print(news_topics.head(10))


#######################################################

# Box Plot for TopN Topic Scores
'''
cols = ['topic_1_prob','topic_2_prob','topic_3_prob']
colors = [sns.xkcd_rgb['greenish cyan'], sns.xkcd_rgb['cyan'], sns.xkcd_rgb['reddish pink']]
fig = plt.figure(figsize=[15,8])
news_topics.boxplot(
    column=cols,
    grid=False)
plt.show()
'''

#######################################################

'''
# Coherence Calculation for Gensim LDA Models

from gensim.models import CoherenceModel

coherence_model_lsa = CoherenceModel(model=svd_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lsa = coherence_model_lsa.get_coherence()

coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

print(f'Coherence Score for LDA Model: {coherence_lda}')
'''
