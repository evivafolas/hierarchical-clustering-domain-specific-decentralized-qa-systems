from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from string import punctuation
import re
import torch
import string
import numpy as np
import pandas as pd
import argparse
import yaml, json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from string import punctuation
import warnings
import os
import pickle
import gensim
import nltk
import gensim.downloader as api
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import sentence_tf_utils as utls

class Main():

    def __init__(
            self,
            topic_list: list[str] = None,
            topN_similar: int = None,
            topic_model_type: str = None,
            clustering_type: str = None,
            cluster_count: int = None,
            tf_model_path: str = None,
            multilingual: bool = False,
            nlp_model_lib: str = None,
            nlp_model_path: str = None,
            dataset: str = None,
            dataset_prunning: float = None,
            docs_df_path: str = None,
            topic_df_path: str = None            
    ):
        '''
        Initialization method.

        topic_list: A list of topics to be used in the topic model.
        topN_similar: The number of similar topic words to be generated.
        topic_model_type: The type of topic model to be used. Currently, only BERTopic is supported.
        clustering_type: The type of clustering to be used. Currently, only BERTopic is supported.
        tf_model: The gensim model used to generate the word embeddings.
        multilingual: A boolean value indicating whether the model is multilingual or not.
        nlp_model: The spacy model used to generate the word embeddings.
        dataset: The dataset to be used to train the model.
        dataset_prunning: Percentage of full dataset used.
        load_file: The file to be loaded.
        save_file: Boolean set whether tosave a file or not.
        '''

        self.topic_list = topic_list
        self.topN_similar = topN_similar
        self.topic_model_type = topic_model_type
        self.clustering_type = clustering_type
        self.cluster_count = cluster_count
        self.tf_model_path = tf_model_path
        self.multilingual = multilingual
        self.nlp_model_lib = nlp_model_lib
        self.nlp_model_path = nlp_model_path
        self.dataset = dataset
        self.dataset_prunning = dataset_prunning
        self.docs_df_path = docs_df_path
        self.topic_df_path = topic_df_path

        if docs_df_path != "None":
            self.docs_df = pd.read_csv(docs_df_path)
        else:
            self.docs_df = pd.DataFrame()
        
        if topic_df_path != "None":
            self.topic_df = pd.read_csv(topic_df_path)  
        else:        
            self.topic_df = pd.DataFrame()
    
        if self.nlp_model_lib == 'gensim':
            print(f'Loading gensim model: {self.nlp_model_path}')
            self.nlp_model = api.load(self.nlp_model_path)
            print(f'Gensim model: {self.nlp_model_path} loaded successfully.')

        print(f'Loading Sentence-Transformer model: {self.tf_model_path}')
        self.tf_model = SentenceTransformer(self.tf_model_path)
        print(f'Sentence-Transformer model: {self.tf_model_path} loaded successfully.')   

    '''
        if self.load_file_type == 'yaml':
            with open(self.load_file_path, 'r') as stream:
                try:
                    self.config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
    '''
############################################################################################################
# 0.1 Utils
############################################################################################################

    def load_csv(self, load_file):
        self.docs_df = pd.read_csv(load_file)
        return
    
    def save(self, save_file):
        if save_file:
            return
        return
    
    def df_csv_save(self, df_to_save, name):
        '''
        Saves the DataFrame as a .csv file in the ./lib directory

        docs_df: The DataFrame to be saved.
        name: The filename to be used.
        '''

        path = './lib'
        os.makedirs(path, exist_ok=True)
        df_to_save.to_csv(path + '/' + name + '.csv')

        return

############################################################################################################
# 1.1 Topic Comprehension and Preparation
############################################################################################################

    def get_similar_words(self, word):
        '''
        Generates a list of n = topn words most similar to the given word, by using a gensim model.

        word: The source word to which the words must be most similar to.
        topic_model: The gensim model used to determine the most similar words.
        topn: The number of similar words to be generated.
        '''

        temp_list=[]
        similar_words = self.nlp_model.most_similar(positive=[word], topn=self.topN_similar)
        for w,similarity in similar_words:
            temp_list.append(w)

        return temp_list

    def build_topic_dict(self):
        '''
        Returns a dictionary with the source word as a key and a list of its topn most similar words.

        words: The list of source words/topics provided by the user.
        topic_model: The gensim model used to determine the most similar words.
        topn: The number of similar words to be generated.

        '''
        topic_dict={}

        for word in self.topic_list:
            topic_dict[word] = self.get_similar_words(word)

        return topic_dict
    
    def build_topic_word_list(self, topic_dict):
        '''
        Returns a list of the similar words seperated by one space.

        topic_dict: The topic_dictionary used to get the words.

        '''
        temp_list = []
        topic_words = list(topic_dict.values())

        for i,word_list in enumerate(topic_words):
            temp_list.append(' '.join(word_list))
        return temp_list

    def build_topic_dataframe(self, topic_word_list, save=False):
        '''
        Returns a dataframe with the topic name and the list of similar words.

        topic_word_list: The list of similar words seperated by one space.
        '''
        self.topic_df = pd.DataFrame({'topic': self.topic_list, 'words': topic_word_list})

        if save:
            self.df_csv_save(self.topic_df, 'topic_df')

        return self.topic_df

############################################################################################################
# 1.2 Document Preparation and Cleaning
############################################################################################################

    def clean_text(self, sentence):
        '''
        Function that performs the basic and essential text cleaning and preproccessing for NLP, in order for the results to be fed in a TF Model and generate document embeddings. 
        Removes Non Alpha sequences - Stop words and Punctuation, tokenizes and lemmatizes the text.

        sentence: Each document or part of text to be cleaned.
        '''

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
        lemmer = WordNetLemmatizer()

        word_list = [lemmer.lemmatize(word) for word in word_list]

        sentence = ' '.join(word_list)


        return sentence

    def create_dataframe(self, save=False):
        '''
        Imports the dataset and creates a dataframe with the text, clean and the target/topic name.

        topn: The number of documents to be used from the dataset. If the dataset is smaller than the value given, it will use all the documents.
        '''

        tqdm.pandas()

        if self.dataset == '20NG':
            temp_dataset = fetch_20newsgroups(shuffle=True, random_state=32, remove=('headers', 'footers', 'quotes'))
            
            self.docs_df = pd.DataFrame(
                {'docs': temp_dataset.data,
                 'target': temp_dataset.target}
            )

            self.docs_df['docs_clean'] = self.docs_df['docs'].progress_apply(lambda x: self.clean_text(str(x)))
            self.docs_df['target_name'] = self.docs_df['target'].apply(lambda x: temp_dataset.target_names[x]) 
    
            self.docs_df = self.docs_df[self.docs_df['docs_clean'].notna()]

        else:
            print('Dataset not supported')
        
        if save:
            self.df_csv_save(self.docs_df, 'docs_df')

        return self.docs_df.iloc[:min((int)(self.dataset_prunning*self.docs_df.shape[0]), self.docs_df.shape[0])]

############################################################################################################
# 2.1 Embedding Generation and Classification
############################################################################################################

    def calcualte_embedding(self, sentence):
        '''
        Calculates the embedding of a sentence using the TF Model.

        sentence: The sentence to be embedded.
        '''

        return self.tf_model.encode(sentence)

    def generate_topic_embeddings(self, save:bool = False):
        '''
        Generates the topic embeddings using the TF Model.

        topic_df: The dataframe with the topics to be embedded.
        '''

        topic_emb = self.tf_model.encode(self.topic_df['words'])
        # self.topic_df['topic_emb'] = self.topic_df['words'].progress_apply(lambda x: self.tf_model.encode([x]))
        
        if save:
            self.df_csv_save(self.topic_df, 'topic_df_class')

        return topic_emb
    
    def generate_document_embeddings(self, save:bool = True):
        '''
        Generates the document embeddings using the TF Model.

        docs_df: The dataframe with the documents to be embedded.
        '''
        # self.docs_df['docs_emb'] = self.docs_df['docs_clean'].progress_apply(lambda x: self.tf_model.encode([x]))
        
        doc_emb = self.tf_model.encode(self.docs_df['docs_clean'])

        if save:
            self.df_csv_save(self.docs_df, 'docs_df_class')
            
            file = open('doc_emb_pickle', 'wb')
            pickle.dump(doc_emb, file)
            file.close()

        return doc_emb

    def calculate_similarities(self, v1, v2, type='euclid'):
        '''
        Calculates a pairwise similarity matrix between the two vectors given as input.

        v1: First vector of comparison.
        v2: Second vector of comparison.
        type: Type of similarity metric used, default -> euclid, alt -> cosine.
        '''

        if type == 'cosine':
            pw_sims = cosine_similarity(v1,v2)
        elif type == 'euclid':
            pw_sims = euclidean_distances(v1,v2)
        return pw_sims

    def calculate_document_similarities(self, doc_emb, topic_emb, type:str ='euclid'):
        '''
        Calculates a pairwise similarity matrix to each topic for each document used in the dataset. Two metrics available, cosine similarity and euclidean distance.

        topic_emb: List of embeddings representing each topic.
        docs_emb: List of embedding representing each document.
        type: Type of similarity metric used, default -> Euclidean Distance.
        '''

        # docs_emb = self.docs_df['docs_emb'].to_numpy()
        # print(f'{docs_emb}')
        # topic_emb = self.topic_df['topic_emb'].to_numpy()
        # print(f'{topic_emb}')

        if type == 'euclid':
            pw_sims = euclidean_distances(doc_emb, topic_emb)
        elif type == 'cosine':
            pw_sims = cosine_similarity(doc_emb, topic_emb)
        return pw_sims
    
    def most_similar_topics(self, similarities, type:str ='euclid'):
        '''
        Takes a similarity matrix and a topic list as an input and determines the most similar topic for each document.

        similarities: Similarity Matrix.
        type: The type of similarity metric used, default -> Euclidean Distance, alt. -> cosine.
        '''
        most_similar_matrix = []
        if type == 'cosine':
            for docsims in similarities:
                most_similar_index = np.argmax(docsims)
                most_similar_matrix.append(self.topic_list[most_similar_index])
        elif type=='euclid':
            for docsims in similarities:
                most_similar_index = np.argmin(docsims)
                most_similar_matrix.append(self.topic_list[most_similar_index])
        
        return most_similar_matrix  
    
    def dataset_target_name_emb(self):
        '''
        Calculates and returns the embeddings of the predetermined topics, given by the dataset, using the given transformer model.

        docs_df: The DataFrame containing all of the documents.
        tf_model: The Sentence-Transformer Model used to calculate the embeddings.
        '''
        if self.dataset == '20NG':   
            target_emb = self.tf_model.encode(self.docs_df['target_name'].apply(lambda x: self.clean_text(str(x))).unique())

        return target_emb
    
    def calculate_classification_accuracy(self, topic_emb):
        '''
        Calculates the accuracy of the Classifier, by using the classifier to predict the topic of the predetermined categories/topics themselves as given by the dataset.

        Then by summing the matches with the topic projection of the documents as predicted by the similarity matrix with the embeddings generated by he TF Model used.

        docs_df: The main pandas.DataFrame with the essential information stored. (eg. data, clean_text, projected_topic, target_name...)
        topic_list: The list of topics provided by the user.
        tf_model: The Transformer based Model used to generate the embeddings.
        dataset: The dataset used to acquire the documents

        User beware: The accuracy metric generated by this function can be considered valid to some extent, only if the initial classification of Target Names to the Topic List is accurate. 
        '''

        if(self.dataset=='20NG'):
            target_names = self.docs_df.target_name.unique().tolist()
            target_names_emb = self.dataset_target_name_emb()
            classifier_evaluation_matrix = self.most_similar_topics(self.calculate_similarities(v1=target_names_emb, v2=topic_emb), type='euclid')
            self.docs_df['target_name_projection'] = self.docs_df.target_name.apply(lambda x: classifier_evaluation_matrix[target_names.index(x)])

        n = self.docs_df.shape[0] # Number of documents in the dataset

        pos = 0 # Number of rows with matching 'Projected Topic' and 'Target Name Projection'    

        for i in range(n):
            # print(f'Target Name: {self.docs_df.target_name[i]:25} - Target-Name Projection: {self.docs_df.target_name_projection[i]:15} - Model Projected Topic: {self.docs_df.projected_topic[i]:15}')
            if self.docs_df.target_name_projection[i] == self.docs_df.projected_topic[i]:
                pos +=1

        classification_accuracy = (float)(pos / n)

        return classification_accuracy
    
############################################################################################################
# 3.1 Topic Modelling
############################################################################################################

    def build_topic_model(self, docs_df, topic_df):
        '''
        Builds the topic model using the LDA model from gensim.

        docs_df: The dataframe with the documents to be embedded.
        topic_df: The dataframe with the topics to be embedded.
        '''
        if self.topic_model_type == 'lda':

            # Create Dictionary
            id2word = corpora.Dictionary(docs_df['docs_clean'])

            # Create Corpus
            texts = docs_df['docs_clean']

            # Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in texts]

            # Build LDA model
            lda_model = gensim.models.ldamodel.LdaModel(
                corpus=corpus,
                id2word=id2word,
                num_topics=self.cluster_count,
                random_state=100,
                update_every=1,
                chunksize=100,
                passes=10,
                alpha='auto',
                per_word_topics=True)

            return lda_model
        
        elif self.topic_model_type == 'bertopic':
            
            bertopic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True, embedding_model=self.tf_model, nr_topics=self.cluster_count)
            topics, probas = bertopic_model.fit_transform(docs_df.docs_clean)

            return bertopic_model, topics, probas
        
    def build_bertopic_subtopics(self, docs_df):
        '''
        Builds the subtopics for the bertopic model.

        bertopic_model: The bertopic model to be used.
        docs_df: The dataframe with the documents to be embedded.
        '''
        for topic in self.topic_list: 
          globals()['docs_topic_%s' % topic] = docs_df.docs_clean.loc[docs_df["projected_topic"] == topic]
          globals()['embs_topic_%s' % topic] = docs_df.emb_df.iloc[docs_df.index[docs_df["projected_topic"] == topic]].to_numpy()

          globals()['subtopic_model_%s' % topic] = BERTopic()
          globals()['subtopic_model_%s_topics' % topic], globals()['subtopic_model_%s_probas' % topic] = globals()['subtopic_model_%s' % topic].fit_transform(globals()['docs_topic_%s' % topic], globals()['embs_topic_%s' % topic])
        
        return

    def calculate_BERTopic_coherence(self, bertopic_model, docus_clean, coherence_type="c_v"):
        '''
        Calculates and return the coherence metric of the BERTopic Model given as input.

        topic_model: The BERTopic model to be evaluated.
        docus_clean: The dataframe containing the preprocessed version of the documents included in the topic model.
        coherence_type: The type of the coherence metric to be used, default -> C V Coherence.
        '''

        vectorizer = bertopic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        words = vectorizer.get_feature_names_out()
        tokens = [analyzer(doc) for doc in docus_clean]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in bertopic_model.get_topic(topic)] for topic in range(len(set(self.topic_list))-1)]

        coherencemodel = CoherenceModel(
            topics = topic_words,
            texts = tokens,
            corpus = corpus,
            dictionary = dictionary,
            coherence = coherence_type
        )

        return coherencemodel.get_coherence()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file_type", type=str, default='yaml', help="The type of the config file to be used. (json, yaml)")
    parser.add_argument("--config_file_path", type=str, default='param_config.yml', help="The path of the config file to be used.")
    parser.add_argument('--process', type=str, default=None, help="The process to be executed. (topic_modelling, topic_clustering, topic_classification, topic_evaluation, topic_visualization)")

    args = parser.parse_args()

    if args.config_file_type == 'yaml':
            with open(args.config_file_path, 'r') as stream:
                try:
                    config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

    db = Main(
        topic_list=config['topic_list'],\
        topN_similar=config['topn_similar'],
        topic_model_type=config['topic_model_type'],
        clustering_type=config['clustering_type'],
        cluster_count=config['cluster_count'],
        tf_model_path=config['tf_model_path'][0],
        multilingual=config['multilingual'],
        nlp_model_lib=config['nlp_model_lib'],
        nlp_model_path=config['nlp_model_path'],
        dataset=config['dataset'],
        dataset_prunning=config['dataset_prunning'],
        docs_df_path=config['docs_df_path'],
        topic_df_path=config['topic_df_path']
    )

############################################################################################################
# n.1 Complex Processes
############################################################################################################
    
    if (args.process == 'build'):

        # Dataset Initialization

        print(f'Buiding the dataset...')

        db.topic_df = db.build_topic_dataframe(db.build_topic_word_list(db.build_topic_dict()))
        db.docs_df = db.create_dataframe()
        
        print(f'Topic & Document DataFrame loaded.')

        # Embedding Calculation

        print(f'Caclulating Topic Embeddings.')
        topic_emb = db.generate_topic_embeddings()
        
        print(f'Caclulating Document Embeddings.')
        doc_emb = db.generate_document_embeddings()

        print(f'Embeddings Calculated Successfully.')

        # Document Classification

        print(f'Predicting the Most Similar Topic for each Document.')
        db.docs_df['projected_topic'] = db.most_similar_topics(db.calculate_document_similarities(doc_emb, topic_emb, type='euclid'), type='euclid')
        print(f'Projected Topics for each document predicted.')

        # Classification Evaluation

        print(f'Classification Accuracy for {db.tf_model_path}: {db.calculate_classification_accuracy(topic_emb=topic_emb)*100}%')

        print(db.docs_df)

    elif (args.process == 'classify'):

        if config.topic_df_path == "None":
            db.topic_df = db.build_topic_dataframe(db.build_topic_word_list(db.build_topic_dict()))
        else:
            db.topic_df = db.load_csv(config.topic_df_path)

        if config.docs_df_path == "None":
            db.docs_df = db.create_dataframe()
        else: 
            db.docs_df = db.load_csv(config.docs_df_path)

    elif (args.process == 'evaluate'):
        print(f'3')
    else:
        print(f'Process {args.process} not found.')
    