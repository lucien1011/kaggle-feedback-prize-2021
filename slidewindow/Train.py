import copy
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AdamW 

from comp import evaluate_score_from_df
from .Infer import get_pred_df
from .PredictionString import get_predstr_df
from pipeline import TorchModule
from utils import set_seed

def model_name_in_path(fname):
    return fname.replace('/','-')

def train_one_step(ids,mask,token_type_ids,labels,model,optimizer,params):
    loss, logits = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, labels=labels)
    torch.nn.utils.clip_grad_norm_(
        parameters=model.parameters(),max_norm=params['max_grad_norm']
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss,logits

def evaluate_accuracy_one_step(labels,logits,num_labels):
    flattened_targets = labels.view(-1)
    active_logits = logits.view(-1, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1)
    active_accuracy = labels.view(-1) != -100 
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy) 
    return accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())

def evaluate_score(discourse_df,val_loader,model,ids_to_labels,device):
    with torch.no_grad():
        pred_df = get_predstr_df(get_pred_df(val_loader,model,ids_to_labels,device,add_true_class=True))
    mean_f1_score,f1s = evaluate_score_from_df(discourse_df,pred_df)
    return mean_f1_score

class Train(TorchModule):

    _header = '-'*100
    _required_params = ['model_name','seed']
    
    def prepare(self,container,params):

        if 'seed' in params: set_seed(params['seed'])

        self.model = container.get(params['model_name'])
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=params['lr'][0])

    def fit(self,container,params):
        best_score = -np.Inf
        
        for epoch in range(params['epochs']):
          
            tqdm.write(self._header)
            tqdm.write(f"### Training epoch: {epoch + 1}")
            for g in self.optimizer.param_groups: 
                g['lr'] = params['lr'][epoch]
            lr = self.optimizer.param_groups[0]['lr']
            tqdm.write(f'### LR = {lr}\n')

            self.model.train()
            
            tr_loss,tr_step = 0.,1.
            for idx, batch in enumerate(tqdm(container.train_loader)):
                batch_dataset = TensorDataset(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['labels'])
                batch_loader = DataLoader(batch_dataset, batch_size=params['sub_bs'])

                for ids,mask,token_type_ids,labels in batch_loader:

                    ids = ids.to(self.device, dtype = torch.long)
                    mask = mask.to(self.device, dtype = torch.long)
                    token_type_ids = token_type_ids.to(self.device, dtype = torch.long)
                    labels = labels.to(self.device, dtype = torch.float)

                    loss,tr_logits = train_one_step(ids,mask,token_type_ids,labels,self.model,self.optimizer,params)
                
                if idx % params['print_every']==0:
                    tqdm.write(f"Training loss after {idx:04d} training steps: {tr_loss/tr_step}")
                    tr_loss,tr_step = 0.,0.
                else:
                    tr_loss += loss.item()
                    tr_step += 1.

            best_model = copy.deepcopy(self.model)
            best_model_name = '{}_ep{}'.format(model_name_in_path(params['bert_model']), epoch)
            container.save_one_item(best_model_name,best_model.state_dict(),'torch_model',check_dir=True)

            #val_score = evaluate_score(container.discourse_df,container.val_loader,self.model,container.ids_to_labels,self.device)
            #tqdm.write(f"Validation score at this epoch: {val_score}") 
            #if val_score > best_score:
            #    tqdm.write("Best validation score improved from {} to {}".format(best_score, val_score))
            #    best_model = copy.deepcopy(self.model)
            #    best_score = val_score
            #    best_model_name = '{}_valscore{}_ep{}'.format(model_name_in_path(params['bert_model']), round(best_score, 5), epoch)
            #    container.save_one_item(best_model_name,best_model.state_dict(),'torch_model',check_dir=True) 

            torch.cuda.empty_cache()
            gc.collect()

    def wrapup(self,container,params):
        container.save() 
