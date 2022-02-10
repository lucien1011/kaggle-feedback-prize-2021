import copy
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset 
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import FeedbackDataset
from pipeline import Module

def _prepare_training_data_helper(args, tokenizer, discourse_df, train_ids, class_name='class'):
    training_samples = []
    for idx in tqdm(train_ids):

        temp_df = discourse_df[discourse_df.id==idx]
        temp_df['start_position'] = temp_df['predictionstring'].apply(lambda x: int(x.split()[0]))
        temp_df = temp_df.sort_values(by=['start_position'])

        disc_texts = temp_df.discourse_text.tolist()
        disc_types = temp_df[class_name].tolist()

        n = len(disc_texts)
        i = 1
        prev_disc_type = disc_types[0]
        prev_disc_text = disc_texts[0]
        while i < n:
            curr_disc_type = disc_types[i]
            curr_disc_text = disc_texts[i]
            if prev_disc_type != 'Claim':
                prev_disc_type,prev_disc_text = curr_disc_type,curr_disc_text
                i += 1
                continue
            encoded_text = tokenizer(
                    prev_disc_text,curr_disc_text,
                    return_offsets_mapping=True,
                    )
            label = 0
            if curr_disc_type == 'Evidence':
                label = 1
            sample = dict(
                    id=idx,
                    input_ids=encoded_text['input_ids'],
                    text=' '.join([prev_disc_text,curr_disc_text]),
                    offset_mapping=encoded_text['offset_mapping'],
                    label=label,
                    )
            training_samples.append(sample)
            prev_disc_type,prev_disc_text = curr_disc_type,curr_disc_text
            i += 1

    return training_samples

def prepare_training_data(df, tokenizer, args, num_jobs):
    training_samples = []
    train_ids = df["id"].unique()

    train_ids_splits = np.array_split(train_ids, num_jobs)

    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_training_data_helper)(args, tokenizer, df, idx) for idx in train_ids_splits
    )
    for result in results:
        training_samples.extend(result)

    return training_samples

class Preprocess(Module):

    _required_params = ['discourse_df_csv_path',]
    
    def __init__(self):
        pass

    def prepare(self,container,params):

        container.add_item('discourse_df',pd.read_csv(params['discourse_df_csv_path']),'df_csv',mode='read')

    def fit(self,container,params):
        
        df = container.discourse_df

        tokenizer = AutoTokenizer.from_pretrained(params['bert_model'])

        samples = prepare_training_data(df, tokenizer, params['args'], num_jobs=params['num_jobs'])
 
        container.add_item('samples',samples,'pickle',mode='write')

    def wrapup(self,container,params):
        container.save()
