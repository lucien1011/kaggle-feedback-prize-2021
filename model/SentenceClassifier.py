import torch
from torch.cuda.amp import autocast
import torch.nn as nn

from transformers import AutoModel

class SentenceClassifier(nn.Module):

    def __init__(self, bert_model="albert-base-v2", freeze_bert=False, num_class=1):
        super(SentenceClassifier, self).__init__()
        self.bert_layer = AutoModel.from_pretrained(bert_model)

        if bert_model == "albert-base-v2":
            hidden_size = 768
        elif bert_model == "albert-large-v2":
            hidden_size = 1024
        elif bert_model == "albert-xlarge-v2":
            hidden_size = 2048
        elif bert_model == "albert-xxlarge-v2":
            hidden_size = 4096
        elif bert_model == "bert-base-uncased":
            hidden_size = 768

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.num_class = num_class
        self.cls_layer = nn.Linear(hidden_size, self.num_class)

        self.dropout = nn.Dropout(p=0.1)

    @autocast()
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''
        outputs = self.bert_layer(input_ids, attn_masks, token_type_ids)
        pooler_output = outputs[1]
        logits = self.cls_layer(self.dropout(pooler_output))
        return logits
