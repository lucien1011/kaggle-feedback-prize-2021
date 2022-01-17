import copy
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW 

from comp import evaluate_score_from_df
from .Infer import get_pred_df
from .PredictionString import get_predstr_df
from pipeline import TorchModule
from utils import set_seed

def model_name_in_path(fname):
    return fname.replace('/','-')

def train_one_step(
        cont_ids,cont_mask,
        pos_ids,pos_mask,
        neg_ids,neg_mask,
        model,
        optimizer,params
        ):
    loss, probs = model(
            cont_ids,cont_mask,
            pos_ids,pos_mask,
            neg_ids,neg_mask,
            )
    torch.nn.utils.clip_grad_norm_(
        parameters=model.parameters(),max_norm=params['max_grad_norm']
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss,probs

def calculate_valid_loss(val_loader,model,device):
    with torch.no_grad():
        losses = []
        for idx, batch in enumerate(tqdm(val_loader)):
            pos_ids = batch['pos_input_ids'].to(device, dtype = torch.long)
            pos_mask = batch['pos_attention_mask'].to(device, dtype = torch.long)
            neg_ids = batch['neg_input_ids'].to(device, dtype = torch.long)
            neg_mask = batch['neg_attention_mask'].to(device, dtype = torch.long)
            cont_ids = batch['cont_input_ids'].to(device, dtype = torch.long)
            cont_mask = batch['cont_attention_mask'].to(device, dtype = torch.long)
        loss, probs = model(
                cont_ids,cont_mask,
                pos_ids,pos_mask,
                neg_ids,neg_mask,
                )
        losses.append(loss.item())
    return np.mean(losses)

class Train(TorchModule):

    _header = '-'*100
    _required_params = ['model_name','seed']
    
    def prepare(self,container,params):

        if 'seed' in params: set_seed(params['seed'])

        self.model = container.get(params['model_name'])
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=params['lr'][0])

    def fit(self,container,params):
        best_score = np.Inf
        
        for epoch in range(params['epochs']):
          
            tqdm.write(self._header)
            tqdm.write(f"### Training epoch: {epoch + 1}")
            for g in self.optimizer.param_groups: 
                g['lr'] = params['lr'][epoch]
            lr = self.optimizer.param_groups[0]['lr']
            tqdm.write(f'### LR = {lr}\n')

            self.model.train()
            
            tr_loss,tr_steps = 0,1
            for idx, batch in enumerate(tqdm(container.train_loader)):
 
                pos_ids = batch['pos_input_ids'].to(self.device, dtype = torch.long)
                pos_mask = batch['pos_attention_mask'].to(self.device, dtype = torch.long)
                neg_ids = batch['neg_input_ids'].to(self.device, dtype = torch.long)
                neg_mask = batch['neg_attention_mask'].to(self.device, dtype = torch.long)
                cont_ids = batch['cont_input_ids'].to(self.device, dtype = torch.long)
                cont_mask = batch['cont_attention_mask'].to(self.device, dtype = torch.long)

                loss,probs = train_one_step(
                        cont_ids,cont_mask,
                        pos_ids,pos_mask,
                        neg_ids,neg_mask,
                        self.model,self.optimizer,params
                        )
                
                if idx % params['print_every']==0:
                    tqdm.write(f"Training loss after {idx:04d} training steps: {tr_loss/tr_steps}")
                    tr_loss,tr_steps = 0,1
                else:
                    tr_loss += loss.item()
                    tr_steps += 1
            val_loss = calculate_valid_loss(container.val_loader,self.model,self.device)
            tqdm.write(f"Validation loss at this epoch: {val_loss}") 
            if val_score < best_score:
                tqdm.write("Best validation loss improved from {} to {}".format(best_score, val_score))
                best_model = copy.deepcopy(self.model)
                best_score = val_score
                best_model_name = '{}_valloss{}_ep{}'.format(model_name_in_path(params['bert_model']), round(best_score, 5), epoch)
                container.save_one_item(best_model_name,best_model.state_dict(),'torch_model',check_dir=True) 

            torch.cuda.empty_cache()
            gc.collect()

    def wrapup(self,container,params):
        container.save() 
