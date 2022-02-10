import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from transformers import AutoModelForTokenClassification

from .NERModel import NERModel

class NERLSTMModel(nn.Module):
    def __init__(self, 
            ner_model_type='google/bigbird-roberta-base',
            ner_model_weight='',
            ner_model_args={},
            freeze_ner=False, 
            stance_model_type='',
            stance_model_weight='',
            stance_model_args={},
            stance_num_labels=0,
            freeze_stance=False,
            include_hidden_state=False,
            num_labels=1, 
            ):

        super(NERLSTMModel, self).__init__()

        self.num_labels = num_labels
        self.ner_layer = self.prepare_bert_layer(ner_model_type,ner_model_weight,ner_model_args,num_labels,freeze=freeze_ner)
        self.stance_layer = self.prepare_bert_layer(stance_model_type,stance_model_weight,stance_model_args,stance_num_labels,freeze=freeze_stance) if stance_model_type else None
        
        num_hidden_state = self.num_labels
        if include_hidden_state:
            num_hidden_state += self.ner_layer.config.hidden_size
        if self.stance_layer:
            num_hidden_state += self.stance_layer.num_labels

        self.lstm = nn.LSTM(
                input_size=num_hidden_state,
                hidden_size=num_hidden_state,
                batch_first=True,
                bidirectional=True,
                )
        self.cls_layer = nn.Linear(num_hidden_state*2,self.num_labels)
        self.include_hidden_state = include_hidden_state

    def prepare_bert_layer(self,model_type,model_weight,model_args,num_labels,freeze=True):
        if 'AutoModel' in model_type:
            layer = AutoModelForTokenClassification.from_pretrained(model_weight,num_labels=num_labels)
        else:
            layer = eval(model_type)(**model_args)
            if model_weight: layer.load_state_dict(torch.load(model_weight))

        if freeze:
            for p in layer.parameters():
                p.requires_grad = False
        return layer

    def loss(self,logits,labels,attention_mask):
        loss_fct = nn.CrossEntropyLoss()

        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)
        true_labels = labels.view(-1)
        outputs = active_logits.argmax(dim=-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = loss_fct(active_logits, true_labels)
        return loss

    def forward(self, 
            input_ids,
            attention_mask,
            labels=None,
            return_dict=None,
            ):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
        '''
        outputs = self.ner_layer(input_ids, attention_mask, return_dict=True) 
        logits = outputs['logits']
        
        if self.stance_layer:
            stances = self.stance_layer(input_ids, attention_mask, return_dict=True)['logits']
            logits = torch.cat([logits,stances],dim=-1)

        if self.include_hidden_state:
            logits = torch.cat([logits,outputs['hidden_state']],dim=-1)
        
        logits,_ = self.lstm(logits)
        logits = self.cls_layer(logits)
        probs = torch.softmax(logits,dim=-1)

        loss = None
        if labels is not None:
            loss = self.loss(logits,labels,attention_mask)
 
        if return_dict:
            return dict(
                loss=loss,
                logits=logits,
                probs=probs
            )
        else:
            return loss,logits,probs

    def predict(self,
            dataloader,
            show_iters=True,
            device='cuda',
            ):
        probs = []
        batches = tqdm(dataloader) if show_iters else dataloader
        for batch in batches:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            out = self(ids,mask,return_dict=True)
            prob = out['probs'].cpu().numpy()
            probs.append(prob)
        return probs
