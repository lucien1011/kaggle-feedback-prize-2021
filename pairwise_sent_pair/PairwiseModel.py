import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from transformers import AutoModel,AutoConfig
from tqdm import tqdm

hidden_size_map = {
        'google/bigbird-roberta-base': 768,
        'roberta-base': 768,
        }

class PairwiseModel(nn.Module):
    def __init__(self, 
            bert_model='google/bigbird-roberta-base',
            saved_bert_model='',
            freeze_bert=False, 
            dropouts=0.1,
            ):

        super(PairwiseModel, self).__init__()

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        self.num_labels = 1 
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

    def loss(self,logits,labels):
        loss_fct = nn.BCELoss()
        loss = loss_fct(logits,labels)
        return loss

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
        sequence_output = transformer_out.pooler_output
        sequence_output = self.dropout(sequence_output)

        logits = self.cls_layer(sequence_output).squeeze()
        sigmoid = nn.Sigmoid()
        probs = sigmoid(logits)

        loss = None
        if labels is not None:
            loss = self.loss(probs,labels)
 
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
