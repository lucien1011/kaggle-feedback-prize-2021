import torch
from torch.cuda.amp import autocast
import torch.nn as nn

from transformers import AutoModelForTokenClassification

hidden_size_map = {
        'google/bigbird-roberta-base': 768,
        'roberta-base': 768,
        }

class TextSegmentationModel(nn.Module):
    def __init__(self, 
            bert_model='google/bigbird-roberta-base',
            saved_bert_model='',
            freeze_bert=False, 
            num_labels=1, 
            ):

        super(TextSegmentationModel, self).__init__()

        self.bert_layer = AutoModelForTokenClassification.from_pretrained(bert_model,num_labels=num_labels)
        if saved_bert_model:
            self.bert_layer.load_state_dict(torch.load(saved_bert_model))
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        self.num_labels = num_labels
        self.lstm = nn.LSTM(
                input_size=self.num_labels,
                hidden_size=self.num_labels,
                batch_first=True,
                bidirectional=True,
                )
        self.cls_layer = nn.Linear(self.num_labels*2,self.num_labels)

    @autocast()
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
        logits = self.bert_layer(input_ids, attention_mask).logits
        logits,_ = self.lstm(logits)
        logits = self.cls_layer(logits)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
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
