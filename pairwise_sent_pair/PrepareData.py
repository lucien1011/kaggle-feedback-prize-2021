import copy
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset 
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer

from .PairwiseDataset import PairwiseDataset
from pipeline import Module

class PrepareData(Module):
    
    _required_params = []
    
    def __init__(self):
        pass

    def fit(self,container,params):
        
        container.read_item_from_path('train_samples',params['train_samples_path'],'pickle')
        container.read_item_from_path('valid_samples',params['valid_samples_path'],'pickle')

        tokenizer = AutoTokenizer.from_pretrained(params['bert_model'])

        train_set = PairwiseDataset(container.train_samples, params['max_len'], tokenizer)
        train_loader = DataLoader(train_set, batch_size=params['train_bs'], shuffle=True)
        
        val_set = PairwiseDataset(container.valid_samples, params['max_len'], tokenizer)
        val_loader = DataLoader(val_set, batch_size=params['val_bs'])

        container.add_item('train_set',train_set,'torch_dataset',mode='read')
        container.add_item('train_loader',train_loader,'torch_dataloader',mode='read')

        container.add_item('val_set',val_set,'torch_dataset',mode='read')
        container.add_item('val_loader',val_loader,'torch_dataloader',mode='read')

    def wrapup(self,container,params):
        container.save()

