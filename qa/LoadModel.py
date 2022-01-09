import copy
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig,AutoModelForTokenClassification,AdamW, get_linear_schedule_with_warmup

from pipeline import TorchModule
from utils import set_seed

class LoadModel(TorchModule):

    _required_params = ['type','bert_model','model_name']
    
    def prepare(self,container,params):
        if params['type'] == 'AutoModelForTokenClassification':
            config_model = AutoConfig.from_pretrained(params['bert_model'],**params.get('config_args',{})) 
            model = AutoModelForTokenClassification.from_pretrained(params['bert_model'],config=config_model)
        else:
            raise NotImplementedError

        if params['saved_model']: model.load_state_dict(torch.load(params['saved_model']))
           
        model.to(self.device)

        container.add_item(params['model_name'],model,'torch_model','read')
