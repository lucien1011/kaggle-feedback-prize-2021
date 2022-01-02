import copy
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.cuda.amp import autocast,GradScaler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import SentenceDataset
from model import SentenceClassifier
from pipeline import Module
from utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_loss(net, device, criterion, dataloader, batch_size):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it,batch in enumerate(tqdm(dataloader)):
            if it > 1: break
            batch_dataset = TensorDataset(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label'])
            batch_loader = DataLoader(batch_dataset, batch_size=batch_size)

            for seq, attn_masks, token_type_ids, labels in batch_loader:
                seq, attn_masks, token_type_ids, labels = \
                    seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                mean_loss += criterion(logits.squeeze(-1), labels)
                count += 1

    return mean_loss.item() / count

class Train(Module):
    
    def prepare(self,container,params):
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)

        self.model = SentenceClassifier(params['bert_model'], freeze_bert=params['freeze_bert'], num_class=params['num_class'])
        self.model.to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.opt = AdamW(self.model.parameters(), lr=params['lr'], weight_decay=params['wd'])
        num_warmup_steps = 0
        num_training_steps = params['epochs'] * len(container.train_loader)
        t_total = (len(container.train_loader) // params['iters_to_accumulate']) * params['epochs']
        self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.opt, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
        self.scaler = GradScaler()

    def train_one_batch(self,batch,batch_size,model,criterion,scaler,iters_to_accumulate):
        batch_dataset = TensorDataset(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label'])
        batch_loader = DataLoader(batch_dataset, batch_size=batch_size)

        for seq, attn_masks, token_type_ids, labels in batch_loader:

            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
        
            with autocast():
                logits = model(seq, attn_masks, token_type_ids)

                loss = criterion(logits.squeeze(-1), labels)
                loss = loss / iters_to_accumulate

            scaler.scale(loss).backward()
        return loss

    def fit(self,container,params):

        best_loss = np.Inf
        best_ep = 1
        nb_iterations = len(container.train_loader)
        print_every = nb_iterations // params['print_every']
        iters = []
        train_losses = []
        val_losses = []

        for ep in range(params['epochs']):

            self.model.train()
            running_loss = 0.0
            for it,batch in enumerate(tqdm(container.train_loader)):
                if it > 10: break
                loss = self.train_one_batch(batch,params['train_bs'],self.model,self.criterion,self.scaler,params['iters_to_accumulate'])

                if (it + 1) % params['iters_to_accumulate'] == 0:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()

                running_loss += loss.item()

                if (it + 1) % print_every == 0:
                    tqdm.write("[train] Iteration {}/{} of epoch {} complete. Loss : {} "
                          .format(it+1, nb_iterations, ep+1, running_loss / print_every))
                    val_loss = evaluate_loss(self.model, device, self.criterion, container.val_loader, params['val_bs'])
                    tqdm.write("[validation] Iteration {}/{} of epoch {} complete. Loss : {} "
                          .format(it+1, nb_iterations, ep+1, val_loss))

                    running_loss = 0.0

            val_loss = evaluate_loss(self.model, device, self.criterion, container.val_loader, params['val_bs'])
            tqdm.write("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

            if val_loss < best_loss:
                tqdm.write("Best validation loss improved from {} to {}".format(best_loss, val_loss))
                best_model = copy.deepcopy(self.model)
                best_loss = val_loss
                best_ep = ep + 1

        best_model_name = '{}_lr_{}_val_loss_{}_ep_{}'.format(params['bert_model'], params['lr'], round(best_loss, 5), best_ep)
        container.add_item(best_model_name,best_model.state_dict(),'torch_model',mode='write') 

        del loss
        torch.cuda.empty_cache()
    
    def wrapup(self,container,params):
        container.save()
