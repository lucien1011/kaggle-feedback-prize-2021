import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from transformers import AutoModelForTokenClassification

from .NERModel import NERModel

hidden_size_map = {
        'google/bigbird-roberta-base': 768,
        'roberta-base': 768,
        }

class NERLSTMModel(nn.Module):
    def __init__(self, 
            ner_model_type='google/bigbird-roberta-base',
            ner_model_weight='',
            ner_model_args={},
            freeze_ner=False, 
            num_labels=1, 
            ):

        super(NERLSTMModel, self).__init__()

        self.ner_model_type = ner_model_type
        self.ner_model_weight = ner_model_weight
        self.ner_model_args = ner_model_args
        if 'AutoModel' in ner_model_type:
            self.ner_layer = AutoModelForTokenClassification.from_pretrained(ner_model_weight,num_labels=num_labels)
        else:
            self.ner_layer = eval(ner_model_type)(**ner_model_args)
            self.ner_layer.load_state_dict(torch.load(ner_model_weight))

        if freeze_ner:
            for p in self.ner_layer.parameters():
                p.requires_grad = False
        
        self.num_labels = num_labels
        self.lstm = nn.LSTM(
                input_size=self.num_labels,
                hidden_size=self.num_labels,
                batch_first=True,
                bidirectional=True,
                )
        self.cls_layer = nn.Linear(self.num_labels*2,self.num_labels)

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
        logits = self.ner_layer(input_ids, attention_mask, return_dict=True)['logits']
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
