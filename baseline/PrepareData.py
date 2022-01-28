import copy
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset 
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import FeedbackDataset,FeedbackDatasetValid,Collate
from pipeline import Module
from .Preprocess import target_id_map

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

        train_set = FeedbackDataset(container.train_samples, params['max_len'], tokenizer, target_id_map)
        train_loader = DataLoader(train_set, batch_size=params['train_bs'], shuffle=True)
        
        val_set = FeedbackDatasetValid(container.valid_samples, params['max_len'], tokenizer)
        val_loader = DataLoader(val_set, batch_size=params['val_bs'], collate_fn=Collate(tokenizer))

        id_target_map = {v:k for k,v in target_id_map.items()} 
        container.add_item('target_id_map',target_id_map,'pickle','read')
        container.add_item('id_target_map',id_target_map,'pickle','read')
 
        container.add_item('train_set',train_set,'torch_dataset',mode='read')
        container.add_item('train_loader',train_loader,'torch_dataloader',mode='read')

        container.add_item('val_set',val_set,'torch_dataset',mode='read')
        container.add_item('val_loader',val_loader,'torch_dataloader',mode='read')

    def wrapup(self,container,params):
        container.save()

