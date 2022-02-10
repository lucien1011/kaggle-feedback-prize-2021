import copy
import gc
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW,get_cosine_schedule_with_warmup

from comp import evaluate_score_from_df
from .PredictionString import get_pred_df
from pipeline import TorchModule
from utils import set_seed

def model_name_in_path(fname):
    return fname.replace('/','-')

def train_one_step(ids,mask,labels,model,optimizer,scheduler,scaler,params):
    out = model(input_ids=ids, attention_mask=mask, labels=labels,return_dict=True)
    loss = out['loss']
    tr_logits = out['logits']
    if params['max_grad_norm'] is not None:
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),max_norm=params['max_grad_norm']
        )
    if scaler:
        with torch.cuda.amp.autocast():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    else:
        loss.backward()
        optimizer.step()
    scheduler.step()
    model.zero_grad()
    return loss,tr_logits

def evaluate_score(dataloader,model,device):
    with torch.no_grad():
        all_preds,all_labels = [],[]
        for batch in tqdm(dataloader):
            ids = batch['input_ids'].to(device,dtype=torch.long)
            mask = batch['attention_mask'].to(device,dtype=torch.long)
            labels = batch['labels'].to(device,dtype=torch.float)
            outputs = model(input_ids=ids,attention_mask=mask,labels=labels,return_dict=True)
            preds = outputs['probs'] > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        acc = metrics.accuracy_score(np.concatenate(all_labels),np.concatenate(all_preds))
    return acc

class Train(TorchModule):

    _header = '-'*100
    _required_params = ['model_name','seed','optimizer_type',]

    def prepare_optimizer(self,container,params):
        if params['optimizer_type'] == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=params['lr'][0])
        elif params['optimizer_type'] == 'AdamW':
            param_optimizer = list(self.model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias"]
            optimizer_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay": params['wd'],
                },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(optimizer_parameters, lr=params['lr'])
        else:
            raise NotImplementedError
    
    def prepare_scheduler(self,container,params):
        if params['scheduler_type'] == 'cosine_schedule_with_warmup':
            num_train_steps = int(len(container.train_set) / container.train_loader.batch_size / params['epochs'])
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(params['warmup_frac'] * num_train_steps),
                num_training_steps=num_train_steps,
                num_cycles=1,
                last_epoch=-1,
            )
        else:
            raise NotImplementedError
    
    def prepare(self,container,params):

        if 'seed' in params: set_seed(params['seed'])
        
        self.model = container.get(params['model_name'])
        self.prepare_optimizer(container,params)
        self.prepare_scheduler(container,params)
        self.scaler = torch.cuda.amp.GradScaler() if params['fp16'] else None
        
    def fit(self,container,params):
         
        best_score = -np.Inf 
        
        for epoch in range(params['epochs']):
          
            tqdm.write(self._header)
            tqdm.write(f"### Training epoch: {epoch + 1}")
            
            if params['optimizer_type'] == 'Adam':
                for g in self.optimizer.param_groups: 
                    g['lr'] = params['lr'][epoch]
                lr = self.optimizer.param_groups[0]['lr']
                tqdm.write(f'### LR = {lr}\n')

            self.model.train()
            self.model.zero_grad()
            
            for idx, batch in enumerate(tqdm(container.train_loader)):

                ids = batch['input_ids'].to(self.device,dtype=torch.long)
                mask = batch['attention_mask'].to(self.device,dtype=torch.long)
                labels = batch['labels'].to(self.device,dtype=torch.float)

                loss,tr_logits = train_one_step(ids,mask,labels,self.model,self.optimizer,self.scheduler,self.scaler,params)
                
                if idx % params['print_every']==0:
                    tqdm.write(f"Training loss after {idx:04d} training steps: {loss.item()}")
            
            val_score = evaluate_score(container.val_loader,self.model,self.device)
            tqdm.write(f"Validation score at this epoch: {val_score}")
            save_model_name = '{}_valscore{}_ep{}'.format(model_name_in_path(params['bert_model']), round(val_score, 5), epoch)
            #save_model_name = '{}_ep{}'.format(model_name_in_path(params['bert_model']), epoch)
            container.save_one_item(save_model_name,self.model.state_dict(),'torch_model',check_dir=True) 
            if val_score > best_score:
                tqdm.write("Best validation score improved from {} to {}".format(best_score, val_score))
                best_model = copy.deepcopy(self.model)
                best_score = val_score

            torch.cuda.empty_cache()
            gc.collect()

    def wrapup(self,container,params):
        container.save() 
