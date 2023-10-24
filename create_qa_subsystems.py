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

def create_qa_subsystems(
        topic_labels: list[str], 
        embedding_model: str,
        reader_model: str,
        retriever_top_k: int = 10,
        reader_top_k: int = 5

):

    qa_config_files = {}

    for i, topic in enumerate(topic_labels):
        temp = {
            topic:{
                'dataframe_filepath': f'./lib/classified/{topic}_document',
                'embedding_model': embedding_model,
                'reader_model': reader_model,
                'retriever_top_k': retriever_top_k,
                'reader_top_k': reader_top_k,
                'port_number': int(9001 + i)
            }
        }
        
        qa_config_files[topic]=temp[topic]

        file = open(f'{topic}_qa_params', 'w')
        yaml.dump(temp, file)
        file.close()

        temp={}

        os.system(f'start python3 qa_subsystem_init_script.py --topic {topic}')

    return qa_config_files