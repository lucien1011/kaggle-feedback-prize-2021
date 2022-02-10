import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel,AutoConfig
from tqdm import tqdm

class MultiTaskNERModel(nn.Module):
    def __init__(self, 
            bert_model='google/bigbird-roberta-base',
            saved_bert_model='',
            freeze_bert=False, 
            num_labels=1,
            num_stance_labels=1,
            dropouts=0.1,
            pred_type='softmax',
            ):

        super(MultiTaskNERModel, self).__init__()

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        self.num_labels = num_labels
        self.num_stance_labels = num_stance_labels
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
        self.discourse_cls_layer = nn.Linear(config.hidden_size, self.num_labels)
        self.stance_cls_layer = nn.Linear(config.hidden_size, self.num_stance_labels)
        self.pred_type = pred_type

    def softmax_loss(self,logits,labels,attention_mask,num_labels):
        loss_fct = nn.CrossEntropyLoss()

        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, num_labels)
        true_labels = labels.view(-1)
        outputs = active_logits.argmax(dim=-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = loss_fct(active_logits, true_labels)
        return loss

    def binary_loss(self,logits,labels,attention_mask,num_labels):
        loss_fct = nn.BCEWithLogitsLoss()

        active_loss = (attention_mask.view(-1) == 1) * (labels.view(-1) != -100)
        active_logits = logits.view(-1,num_labels)
        true_labels = labels.view(-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = F.one_hot(true_labels[idxs],num_classes=num_labels).to(torch.float)
        
        loss = loss_fct(active_logits, true_labels)
        return loss

    def loss(self,logits,labels,attention_mask,num_labels):
        if self.pred_type == 'softmax':
            return self.softmax_loss(logits,labels,attention_mask,num_labels)
        elif self.pred_type == 'binary':
            return self.binary_loss(logits,labels,attention_mask,num_labels)

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

        discourse_logits = [self.discourse_cls_layer(do_layer(sequence_output)) for do_layer in self.dropout_layers]
        stance_logits = [self.stance_cls_layer(do_layer(sequence_output)) for do_layer in self.dropout_layers]

        loss = None
        if labels is not None:
            discourse_labels,stance_labels = labels
            discourse_losses = [self.loss(logit,discourse_labels,attention_mask,self.num_labels) for logit in discourse_logits]
            discourse_loss = torch.stack(discourse_losses,dim=0).sum(dim=0) / len(self.dropouts)
            stance_losses = [self.loss(logit,stance_labels,attention_mask,self.num_stance_labels) for logit in stance_logits]
            stance_loss = torch.stack(stance_losses,dim=0).sum(dim=0) / len(self.dropouts)
            loss = discourse_loss + stance_loss

        discourse_logits = torch.stack(discourse_logits,dim=0).sum(dim=0) / len(self.dropouts)
        discourse_probs = self.probs(discourse_logits)
        stance_logits = torch.stack(stance_logits,dim=0).sum(dim=0) / len(self.dropouts)
        stance_probs = self.probs(stance_logits)
        logits = torch.cat([discourse_logits,stance_logits],dim=-1)
        probs = torch.cat([discourse_probs,stance_probs],dim=-1)
        
        if return_dict:
            return dict(
                loss=loss,
                discourse_logits=discourse_logits,
                discourse_probs=discourse_probs,
                stance_logits=stance_logits,
                stance_probs=stance_probs,
                logits=logits,
                probs=probs,
            )
        else:
            return loss,logits,probs

    def predict(self,
            dataloader,
            show_iters=True,
            device='cuda',
            key='discourse_probs',
            ):
        probs = []
        batches = tqdm(dataloader) if show_iters else dataloader
        for batch in batches:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            out = self(ids,mask,return_dict=True)
            prob = out[key].cpu().numpy()
            probs.append(prob)
        return probs
