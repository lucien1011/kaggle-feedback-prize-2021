import torch
from torch.cuda.amp import autocast
import torch.nn as nn

from transformers import AutoModelForTokenClassification

hidden_size_map = {
        'google/bigbird-roberta-base': 768,
        'roberta-base': 768,
        'allenai/longformer-base-4096': 768,
        }

class AttentionHeadModel(nn.Module):
    def __init__(self, 
            bert_model='google/bigbird-roberta-base',
            saved_model='',
            freeze_bert=False, 
            num_labels=1, layer_norm_eps=1e-5
            ):
        super(AttentionHeadModel, self).__init__()

        self.num_labels = num_labels
        self.hidden_size = hidden_size_map[bert_model]

        self.bert_layer = AutoModelForTokenClassification.from_pretrained(bert_model,num_labels=num_labels)
        if saved_model:
            self.bert_layer.load_state_dict(torch.load(saved_model))
        
        self.cls_layer = nn.Linear(self.hidden_size, self.num_labels)

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.attention_head = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    @autocast()
    def forward(self, 
            input_ids,
            attention_mask,
            labels=None,
            ):
        sequence_outputs = self.bert_layer.longformer(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                )
        sequence_outputs = self.attention_head(sequence_outputs.last_hidden_state)
        logits = self.cls_layer(self.dropout(sequence_outputs))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  
        return loss,logits
