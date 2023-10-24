import re
import yaml
import time
import pickle
import torch
import argparse
import pandas as pd
import numpy as np 

from torch import multiprocessing as mp

import uvicorn
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from haystack.nodes import TransformersQueryClassifier
from haystack.utils import print_answers 


import create_qa_subsystems

app = FastAPI()

def ask_question(
        query: str,
        topic_labels: list(str),
        qa_config_files: list(dict),
        query_classifier
):
    query_classification_result = query_classifier.run(query)

    query_topic = topic_labels[int(query_classification_result[1].split('_')[-1])-1]

    qa_url = f'"http://127.0.0.1:{qa_config_files[query_topic]["port_number"]}/answer_query'
    data = query

    response = requests.post(qa_url, json=data)

    if response.status_code == 200:
        result = response.json()
        answer = result["result1"]
        confidence = result["result2"]

        if confidence == True:
            return answer, None 
        else: 
            alt_answers = []
            for topic in topic_labels:
                if topic == query_topic:
                    continue
                else:
                    qa_url = f'"http://127.0.0.1:{qa_config_files[topic]["port_number"]}/answer_query'
                    result = response.json()
                    answer = result["result1"]

                    alt_answers.append(
                        {
                            topic: answer
                        }
                    )
                return answer, alt_answers
    else:
        return None, None
    
if __name__ == "__main__":
    
    # uvicorn.run(app, host='127.0.0.1', port= 8000)

    with open('master_config.yml', 'r') as master_stream:
        try:
            master_params = yaml.safe_load(master_stream)
        except yaml.YAMLError as ex:
            print(ex)
    
    query_classifier = TransformersQueryClassifier(
        model_name_or_path=master_params['query_classifier_model'],
        use_gpu=True,
        task='zero-shot-classification',
        labels = master_params['topic_labels']
    )

    query_answer_pairs = {}
    for query in master_params['query_list']:
        
        print(f'Query: {query}')

        temp_answer = ask_question(query, master_params['topic_labels'], create_qa_subsystems.create_qa_subsystems(), query_classifier)
        
        if temp_answer[0]:
            print('There was a problem fetching the answer.')
            continue

        print()
        print_answers(temp_answer[0])

        if temp_answer[1] == None:
            pass    
        else:
            for ans in temp_answer[1]:
                print(f"{list(ans.keys())[0].title()} QA Sub-system:")
                print_answers(ans[list(ans.keys())[0]])

