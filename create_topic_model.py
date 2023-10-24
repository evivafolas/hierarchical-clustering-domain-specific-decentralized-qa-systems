import re
import yaml
import pickle
import pprint
import argparse

import numpy as np
import pandas as pd

import torch
from torch import multiprocessing as mp

from tqdm import tqdm
from string import punctuation

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bertopic import BERTopic
from transformers import pipeline
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########  
# -------------------------------------   Load Configuration File   --------------------------------------------------- #
# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### 

# Parse the arguments from the Command Line
parser = argparse.ArgumentParser()

# Add the arguments to the parser, depending on the argument name.
parser.add_argument("-type", type=str, default='yaml', help="The type of the config file to be used. (json, yaml)")
parser.add_argument("-file", type=str, default='topic_model_params.yml', help="The path of the config file to be used.")

args = parser.parse_args()

# Load and Read the Config File
if args.type == 'yaml':
  with open(args.file, 'r') as stream:
      try:
        topic_model_params = yaml.safe_load(stream)
      except yaml.YAMLError as ex:
        print(ex)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    torch.set_num_threads(mp.cpu_count() - 1)

transformer_model_name = topic_model_params['transformer_model_name'][2]

# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########  
# ----------------------------------------  Preprocessing & Import Functions  ----------------------------------------- #
# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########

# Function for text preprocessing
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
    word_list = [word for word in word_list if len(word) > 1]
    word_list = [word for word in word_list if word not in punct]

    # Stemming or Lemmatization
    lemmer = WordNetLemmatizer()

    word_list = [lemmer.lemmatize(word) for word in word_list]

    sentence = ' '.join(word_list)

    return sentence

# Create the dataset with clean text and pruning if so specified
def create_dataframe(save=False):
  tqdm.pandas()

  if topic_model_params['dataset'] == 'BBCNews':
    temp_dataset = pd.read_csv('./lib/bbc-news-data.csv', sep='\t')
    documents_df = pd.DataFrame({
        'file': temp_dataset['filename'],
        'title': temp_dataset['title'],
        'target': temp_dataset['category'],
        'docs': temp_dataset['content'],
      }
    )

  documents_df['docs_clean'] = documents_df['docs'].progress_apply(lambda x: clean_text(str(x)))

  documents_df = documents_df[documents_df['docs_clean'].notna()]
  return documents_df

# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########  
# ----------------------------------------------  Document DataFrame   ------------------------------------------------ #
# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########

# Create the documents' dataframe
documents = create_dataframe()

print(documents.head)

# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########  
# -----------------------------------------         Topic Modelling       --------------------------------------------- #
# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########

# Calculate Embeddings for all the documents
embedding_model = SentenceTransformer(transformer_model_name)
embeddings = embedding_model.encode(
  documents['docs_clean'],
  device=0,
  show_progress_bar=True
)

# Create the topic model with the existing document embeddings
topic_model = BERTopic(
  embedding_model=transformer_model_name,
  top_n_words = topic_model_params['topic_model_top_n'],
  n_gram_range = (1,5)
)

topics, probs = topic_model.fit_transform(documents['docs_clean'], embeddings)

# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########  
# --------------------------------------------------------------------------------------------------------------------- #
# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########

all_topics = topic_model.get_topics()

bertopic_keywords = []

for topic, keys in all_topics.items():
  keywords = []

  for k in keys: keywords.append(k[0])

  bertopic_keywords.append(keywords)

  print(f'Topic: {topic}, {keywords}')

# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########  
# -------------------------------------------   Topic Classification    ----------------------------------------------- #
# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########

topic_classifier_pipeline = pipeline(
   "zero-shot-classification",
   model=topic_model_params['zero_shot_classification_model'],
   device=0
)

topic_labels = topic_model_params['topic_labels']

# Word list for each BERTopic Generated topic
bertopics = []

for topic in bertopic_keywords:
  bertopics.append(' '.join(topic))

# Classification of each BERTopic Generated topic to one of the user defined topic labels
topic_clf_res = []

print(f'Now running Zero-Shot Classification for Topics')
for topic in tqdm(bertopics):
  topic_clf_res.append(topic_classifier_pipeline(topic, topic_labels, multi_label=True))

# print(topic_clf_res)

# Check for Multiple Label Classification
mutli = []

if topic_model_params['multilabel_classification']:

  for i,topic in enumerate(topic_clf_res):

    if topic['scores'][1] > topic_model_params['mutlilabel_classifition_threshold']:
      
      print(f'Topic {i-1}: Multi')
      mutli.append(True)

    else:
      mutli.append(False)

else:
  mutli = [False] * len(topic_clf_res)


# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########  
# -----------------------------------       Document Classification Accuracy     -------------------------------------- #
# ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########

pos = 0
labels = []
labels2 = []

for doc, doc_topic in tqdm(enumerate(topics)):

  labels.append(topic_clf_res[doc_topic+1]['labels'][0])
  
  if mutli[doc_topic+1] == True:

    labels2.append(topic_clf_res[doc_topic+1]['labels'][1])
    if documents.iloc[doc].target in topic_clf_res[doc_topic+1]['labels'][:topic_model_params['multilable_classification_number']]:
      pos += 1
  
  else:
    
    labels2.append(topic_clf_res[doc_topic+1]['labels'][0])
    if documents.iloc[doc].target == topic_clf_res[doc_topic+1]['labels'][0]:
      pos += 1

accu = pos / len(topics)

documents['label_clf'] = labels
documents['label_clf2']= labels2

# ZSClf Accuracy
print(f'Zero Shot Topic Classification Accuracy: {round(accu*100,2)}%')

# Add each document embedding to the dataframe
embeddings_list=[]
for i in tqdm(range(len(embeddings))):
  embeddings_list.append(embeddings[i])

documents['embeddings'] = embeddings_list

# Save dataframe to csv
documents.to_csv(f'./lib/processed_dataframe/minilm_l6_v2.csv')

for topic in topic_labels:
  
  # Create topic specific CSV documents
  globals()['%s_documents' % topic] = documents.loc[(documents.label_clf == topic) | (documents.label_clf2 == topic)]

  # Save topic specific documents
  globals()['%s_documents' % topic].to_pickle(f'./lib/classified/{topic}_document')

