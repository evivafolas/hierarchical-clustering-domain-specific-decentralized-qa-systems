import yaml
import torch
import argparse
from torch import multiprocessing as mp

import pandas as pd

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import FARMReader

from haystack.pipelines import ExtractiveQAPipeline


from haystack.utils import print_answers

import uvicorn
from fastapi import FastAPI

app=FastAPI()

# Parse the arguments from the Command Line
parser = argparse.ArgumentParser()

# Add the arguments to the parser, depending on the argument name.
parser.add_argument("--topic", type=str, default='None', help="The topic label for the QA sub-system")
# parser.add_argument("--config_file_path", type=str, default='topic_model_params.yml', help="The path of the config file to be used.")

args = parser.parse_args()

# Minimize usage when run locally, by running it on CUDA or the last CPU Thread
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    torch.set_num_threads(mp.cpu_count() - 1)

with open(f'{args.topic}_qa_params', 'r') as stream:
    try:
        qa_params = yaml.safe_load(stream)[args.topic]
    except yaml.YAMLError as exc:
        print(exc)

documents = pd.read_pickle(qa_params['dataframe_filepath'])

document_store = InMemoryDocumentStore()

for index, row in documents.iterrows():
    document_store.write_documents(
        [
            {
                "content": row["docs"],
                'embedding': row['embeddings'],
                "meta": {
                    "tags": row["label_clf"],
                    'title': row['title']
                }
            }
        ]
    )

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model=qa_params['embedding_model'],
    use_gpu=True
)

reader = FARMReader(
    model_name_or_path=qa_params['reader_model'],
    use_gpu=True
)

pipeline = ExtractiveQAPipeline(
    reader=reader,
    retriever=retriever
)

def query(
    query: str,
    retriever_top_k = 10,
    reader_top_k = 5
):
  answer = pipeline.run(
      query,
      params={
          'Retriever': {
              'top_k': retriever_top_k
          },
          'Reader': {
              'top_k': reader_top_k
          }       
        }
  )

  if answer['answers'][0].score < 0.5:
    confidence = False
  else: confidence = True
    
  return answer, confidence 

@app.post("/answer_query")
async def answer_query(input_query: str):
    print(f'Question: {input_query}')
    answer, confidence = query(input_query, retriever_top_k=qa_params['retriever_top_k'], reader_top_k=qa_params['reader_top_k'])
    return{
       'answer': answer,
       'confidence': confidence
    }

if __name__ == "__main__":

   uvicorn.run(app, host='127.0.0.1', port= qa_params['port_number'])