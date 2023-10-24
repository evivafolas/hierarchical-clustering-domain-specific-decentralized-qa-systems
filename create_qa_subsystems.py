import os
import re
import yaml
import pickle
import pprint
import requests
import argparse

import numpy as np
import pandas as pd

from fastapi import FastAPI

# Parse the arguments from the Command Line
parser = argparse.ArgumentParser()

# Add the arguments to the parser, depending on the argument name.
parser.add_argument("--config_file_type", type=str, default='yaml', help="The type of the config file to be used. (json, yaml)")
parser.add_argument("--config_file_path", type=str, default='topic_model_params.yml', help="The path of the config file to be used.")

args = parser.parse_args()

# Load and Read the Config File
with open(args.config_file_path, 'r') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def create_qa_subsystems():
    topic_labels = params['topic_labels']

    qa_config_files = {}

    for i, topic in enumerate(topic_labels):
        temp = {
            topic:{
                'dataframe_filepath': f'/lib/classified/{topic}_document.csv',
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'reader_model': 'deepset/tinyroberta-squad2',
                'retriever_top_k': 10,
                'reader_top_k': 5,
                'port_number': int(9000 + i)
            }
        }
        
        qa_config_files[topic]=temp[topic]

        file = open(f'{topic}_qa_params', 'w')
        yaml.dump(temp, file)
        file.close()

        temp={}

        os.system(f'python3 qa_subsystem_init_script.py -c {topic}')

        return qa_config_files