import torch
import torch.nn as nn
from transformers import AutoConfig,AutoModelForTokenClassification

from model import *
from pipeline import TorchModule
from utils import set_seed

class LoadModel(TorchModule):

    _required_params = ['type','bert_model','model_name']
    
    def prepare(self,container,params):
        if 'seed' in params: set_seed(params['seed'])
        
        if params['type'] == 'AutoModelForTokenClassification':
            config_model = AutoConfig.from_pretrained(params['bert_model'],**params.get('config_args',{})) 
            model = AutoModelForTokenClassification.from_pretrained(params['bert_model'],config=config_model)
        elif params['type'] == 'CustomModel':
            config_model = AutoConfig.from_pretrained(params['bert_model'],**params.get('config_args',{})) 
            model = eval(params['custom_model'])(**params['args'])
        else:
            raise NotImplementedError

        if 'saved_model' in params and params['saved_model']: model.load_state_dict(torch.load(params['saved_model']))
           
        model.to(self.device)

        container.add_item(params['model_name'],model,'torch_model','read')
