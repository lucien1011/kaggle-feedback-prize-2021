import torch
from torch.cuda.amp import autocast
import torch.nn as nn

from transformers import AutoModel

hidden_size_map = {
        'google/bigbird-roberta-base': 768,
        'roberta-base': 768,
        'allenai/longformer-base-4096': 768,
        }

class SlideWindowModel(nn.Module):
    def __init__(self, bert_model='google/bigbird-roberta-base', freeze_bert=False, num_labels=1, layer_norm_eps=1e-5):
        super(SlideWindowModel, self).__init__()
        self.bert_layer = AutoModel.from_pretrained(bert_model)
        
        self.hidden_size = hidden_size_map[bert_model]

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.num_labels = num_labels
        self.cls_layer = nn.Sequential(            
            nn.Linear(self.hidden_size, 256),            
            nn.ReLU(),                       
            nn.Linear(256, self.num_labels),
        )        
        self.dropout = nn.Dropout(p=0.1)

    @autocast()
    def forward(self, 
            input_ids,
            attention_mask,
            token_type_ids,
            labels=None,
            ):
        outputs = self.bert_layer(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
                )
        logits = self.cls_layer(self.dropout(outputs.pooler_output))
        loss = None
        if labels is not None:
            loss_fct = nn.KLDivLoss()
            norm_fct = nn.LogSoftmax()
            loss = loss_fct(norm_fct(logits),labels)
        return loss,logits
