import copy
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset 
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer

from .MultiTaskDataset import MultiTaskDataset,MultiTaskDatasetValid,Collate
from pipeline import Module

discourse_type_map = {
    "B-Lead": 0,
    "I-Lead": 1,
    "B-Position": 2,
    "I-Position": 3,
    "B-Evidence": 4,
    "I-Evidence": 5,
    "B-Claim": 6,
    "I-Claim": 7,
    "B-Concluding Statement": 8,
    "I-Concluding Statement": 9,
    "B-Counterclaim": 10,
    "I-Counterclaim": 11,
    "B-Rebuttal": 12,
    "I-Rebuttal": 13,
    "O": 14,
    "PAD": -100,
}

stance_type_map = {
    "B-Lead": 0,
    "I-Lead": 0,
    "B-Position": 1,
    "I-Position": 1,
    "B-Evidence": -100,
    "I-Evidence": -100,
    "B-Claim": 1,
    "I-Claim": 1,
    "B-Concluding Statement": 1,
    "I-Concluding Statement": 1,
    "B-Counterclaim": 2,
    "I-Counterclaim": 2,
    "B-Rebuttal": 1,
    "I-Rebuttal": 1,
    "O": -100,
    "PAD": -100,
}

class PrepareData(Module):
    
    _required_params = ['discourse_df_csv_path',]
    
    def __init__(self):
        pass

    def prepare(self,container,params):

        container.add_item('discourse_df',pd.read_csv(params['discourse_df_csv_path']),'df_csv',mode='read')

    def fit(self,container,params):
        
        df = container.discourse_df

        container.read_item_from_path('train_samples',params['train_samples_path'],'pickle')
        container.read_item_from_path('valid_samples',params['valid_samples_path'],'pickle')

        tokenizer = AutoTokenizer.from_pretrained(params['bert_model'])

        train_set = MultiTaskDataset(container.train_samples, params['max_len'], tokenizer, discourse_type_map, stance_type_map)
        train_loader = DataLoader(train_set, batch_size=params['train_bs'], shuffle=True)
        
        val_set = MultiTaskDatasetValid(container.valid_samples, params['max_len'], tokenizer)
        val_loader = DataLoader(val_set, batch_size=params['val_bs'], collate_fn=Collate(tokenizer))
        
        id_target_map = {v:k for k,v in discourse_type_map.items()} 
        container.add_item('id_target_map',id_target_map,'pickle','read')
        container.add_item('discourse_type_map',discourse_type_map,'pickle','read')
        container.add_item('stance_type_map',stance_type_map,'pickle','read')
 
        container.add_item('train_set',train_set,'torch_dataset',mode='read')
        container.add_item('train_loader',train_loader,'torch_dataloader',mode='read')

        container.add_item('val_set',val_set,'torch_dataset',mode='read')
        container.add_item('val_loader',val_loader,'torch_dataloader',mode='read')

    def wrapup(self,container,params):
        container.save()

