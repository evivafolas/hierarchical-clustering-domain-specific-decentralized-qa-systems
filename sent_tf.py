import sys
import torch
import pandas as pd
import sentence_tf_utils as utls
import gensim.downloader as api
from sentence_transformers import SentenceTransformer

if torch.cuda.is_available():
    dev = "cuda:0"
else: 
    dev = "cpu"



topic_word_model_str = 'glove-wiki-gigaword-300'
topic_word_model = api.load(topic_word_model_str)
print("Gensim Model: " + topic_word_model_str + " loaded Successfully")

tf_model_str = 'sentence-transformers/gtr-t5-base'
tf_model = SentenceTransformer('sentence-transformers/gtr-t5-base')
print("Sentence-Transformer Model: " + tf_model_str + " loaded Successfully")

topic_list = ["computers","vehicles","sports","classifieds","politics","religion"]

# topic_df = utls.topic_dataframe(topic_list=topic_list, topic_model=topic_word_model)
topic_df = utls.topic_dataframe(utls.build_topic_dict(topic_list,topic_word_model),topic_list)
print("Topic word lists generated successfully")
top_emb = utls.calculate_topic_embeddings(topic_df=topic_df, tf_model=tf_model)
print("Topic word embeddings generated successfully")

docs_df = utls.create_dataset_20NG(docNo=1000)
print("Documents loaded and preprocessed")
doc_emb = utls.calculate_document_embeddings(docs_df=docs_df, tf_model=tf_model)
print("Document Embeddings calculated successfully")

pairwise_similarity_matrix = utls.calculate_similarities(topic_emb=top_emb, docs_emb=doc_emb, type="cosine")
print('Similarity Matrix Calculated Successfully.')

docs_df['projected_topic'] = utls.most_similar_topics(similarities=pairwise_similarity_matrix, topic_list=topic_list, type='cosine') 
utls.df_csv_save(docs_df=docs_df, name='docs_df')
print('Dcument DataFrame saved.')

emb_df = pd.DataFrame(doc_emb)
topemb_df = pd.DataFrame(top_emb)

utls.df_csv_save(emb_df,'emb_df')
utls.df_csv_save(topemb_df,'topic_emb')
print('Embedding DataFrames saved.')