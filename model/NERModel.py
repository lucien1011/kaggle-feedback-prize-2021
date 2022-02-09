import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel,AutoConfig
from tqdm import tqdm

hidden_size_map = {
        'google/bigbird-roberta-base': 768,
        'roberta-base': 768,
        }

class NERModel(nn.Module):
    def __init__(self, 
            bert_model='google/bigbird-roberta-base',
            saved_bert_model='',
            freeze_bert=False, 
            num_labels=1,
            dropouts=0.1,
            pred_type='softmax',
            ):

        super(NERModel, self).__init__()

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        self.num_labels = num_labels
        self.dropouts = dropouts
        
        config = AutoConfig.from_pretrained(bert_model)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
            }
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  
        self.bert_layer = AutoModel.from_pretrained(bert_model,config=config)
        if saved_bert_model:
            self.bert_layer.load_state_dict(torch.load(saved_bert_model))
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.dropout_layers = nn.ModuleList([nn.Dropout(p=do) for do in self.dropouts])
        self.cls_layer = nn.Linear(config.hidden_size, self.num_labels)
        self.pred_type = pred_type

    def softmax_loss(self,logits,labels,attention_mask):
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

    def binary_loss(self,logits,labels,attention_mask):
        loss_fct = nn.BCEWithLogitsLoss()
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)
        true_labels = labels.view(-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = F.one_hot(true_labels[idxs].to(torch.long),num_classes=self.num_labels)
        loss = loss_fct(active_logits, true_labels)
        return loss


    def loss(self,logits,labels,attention_mask):
        if self.pred_type == 'softmax':
            return self.softmax_loss(logits,labels,attention_mask)
        elif self.pred_type == 'binary':
            return self.binary_loss(logits,labels,attention_mask)

    def probs(self,logits):
        if self.pred_type == 'softmax':
            return torch.softmax(logits,dim=-1)
        elif self.pred_type == 'binary':
            return torch.sigmoid(logits)

    def forward(self, 
            input_ids,
            attention_mask,
            token_type_ids=None,
            labels=None,
            return_dict=None,
            ):

        if token_type_ids:
            transformer_out = self.bert_layer(input_ids, attention_mask, token_type_ids)
        else:
            transformer_out = self.bert_layer(input_ids, attention_mask)
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        logits = [self.cls_layer(do_layer(sequence_output)) for do_layer in self.dropout_layers]

        loss = None
        if labels is not None:
            losses = [self.loss(logit,labels,attention_mask) for logit in logits]
            loss = torch.stack(losses,dim=0).sum(dim=0) / len(self.dropouts)

        logits = torch.stack(logits,dim=0).sum(dim=0) / len(self.dropouts)
        probs = self.probs(logits)
        
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
