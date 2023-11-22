import re
import yaml
import time
import pickle
import torch
import time
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
        topic_labels,
        qa_config_files,
        query_classifier
):
    query_classification_result = query_classifier.run(query)

    query_topic = topic_labels[int(query_classification_result[1].split('_')[-1])-1]

    print(f'Query classified to the topic: {query_topic}')

    qa_url = f'http://127.0.0.1:{qa_config_files[query_topic]["port_number"]}/answer_query'
    question = {
        'input_query': query
    }
    # print(question)
    response = requests.post(qa_url,params=question)

    if response.status_code == 200:
        result = response.json()

        answer = result["answer"]
        confidence = result["confidence"]

        if confidence == True:
            return answer, None 
        else: 
            alt_answers = []
            for topic in topic_labels:
                if topic == query_topic:
                    continue
                else:
                    qa_url = f'http://127.0.0.1:{qa_config_files[topic]["port_number"]}/answer_query'
                    alt_response = requests.post(qa_url,params=question)
                    alt_result = alt_response.json()
                    alt_answer = alt_result["answer"]

                    alt_answers.append(
                        {
                            topic: alt_answer
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

    qa_subsystems = create_qa_subsystems.create_qa_subsystems(master_params['topic_labels'], master_params['embedding_model'], master_params['reader_model'])
    
    subsystem_readiness = ''
    while(subsystem_readiness.lower() != 'r' ):
        subsystem_readiness = str(input('Enter R for ready when QA Systems are launched\n'))

    type_of_input = str(input('Enter T to type the questions or L for the list in the parameters file\n'))

    if type_of_input.lower() == 't':

        query= None
        while(query!=""):
            query = str(input("Type a queston (or blank to quit): \n"))
        
            if query == '': break

            print(f'Query: {query}')

            temp_answer = ask_question(query, master_params['topic_labels'], qa_subsystems, query_classifier)
            
            if temp_answer[0] == None:
                print('There was a problem fetching the answer.')
                continue

            print()
            # print(temp_answer)
            # print_answers(temp_answer[0])
            print(f'{temp_answer[0]["answers"][0]["answer"]}, {round(temp_answer[0]["answers"][0]["score"] * 100, 2)}% confident, as shown in: {temp_answer[0]["answers"][0]["context"]}')

            if temp_answer[1] == None:
                pass    
            else:
                for ans in temp_answer[1]:
                    print(f"{list(ans.keys())[0].title()} QA Sub-system:")
                    # print(ans)
                    print(f'{ans[list(ans.keys())[0]]["answers"][0]["answer"]}, {round(ans[list(ans.keys())[0]]["answers"][0]["score"] * 100, 2)}% confident, as shown in: {ans[list(ans.keys())[0]]["answers"][0]["context"]}')
                    # print_answers(ans[list(ans.keys())[0]])

    elif type_of_input.lower() == 'l':
        # query_answer_pairs = {}
        # for query in master_params['query_list']:
            
        #     print('\n########################')
        #     print(f'Query: {query}')

        #     temp_answer = ask_question(query, master_params['topic_labels'], qa_subsystems, query_classifier)
            
        #     if temp_answer[0] == None:
        #         print('There was a problem fetching the answer.')
        #         continue

        #     print()
        #     # print(temp_answer)
        #     # print_answers(temp_answer[0])
        #     print(f'{temp_answer[0]["answers"][0]["answer"]}, {round(temp_answer[0]["answers"][0]["score"] * 100, 2)}% confident, as shown in: {temp_answer[0]["answers"][0]["context"]}')

        #     if temp_answer[1] == None:
        #         pass    
        #     else:
        #         for ans in temp_answer[1]:
        #             print(f"{list(ans.keys())[0].title()} QA Sub-system:")
        #             # print(ans)
        #             print(f'{ans[list(ans.keys())[0]]["answers"][0]["answer"]}, {round(ans[list(ans.keys())[0]]["answers"][0]["score"] * 100, 2)}% confident, as shown in: {ans[list(ans.keys())[0]]["answers"][0]["context"]}')
        #             # print_answers(ans[list(ans.keys())[0]])
        
        for query in master_params['query_list']:
            
            print('\n########################')
            print(f'Query: {query["question"]}')

            temp_answer = ask_question(query["question"], master_params['topic_labels'], qa_subsystems, query_classifier)
            
            if temp_answer[0] == None:
                print('There was a problem fetching the answer.')
                continue

            print()
            # print(temp_answer)
            # print_answers(temp_answer[0])
            print(f'{temp_answer[0]["answers"][0]["answer"]}, {round(temp_answer[0]["answers"][0]["score"] * 100, 2)}% confident, as shown in: {temp_answer[0]["answers"][0]["context"]}')
            print(f'Actual Answer: {query["answer"]}')
            print(f'Question Topic: {query["topic"]}')

            if temp_answer[1] == None:
                pass    
            else:
                for ans in temp_answer[1]:
                    print(f"{list(ans.keys())[0].title()} QA Sub-system:")
                    # print(ans)
                    print(f'{ans[list(ans.keys())[0]]["answers"][0]["answer"]}, {round(ans[list(ans.keys())[0]]["answers"][0]["score"] * 100, 2)}% confident, as shown in: {ans[list(ans.keys())[0]]["answers"][0]["context"]}')
                    # print_answers(ans[list(ans.keys())[0]])
                    print("\n")

