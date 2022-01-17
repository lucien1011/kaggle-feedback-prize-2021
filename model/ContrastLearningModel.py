import torch
from torch.cuda.amp import autocast
import torch.nn as nn

from transformers import AutoModel

hidden_size_map = {
        'google/bigbird-roberta-base': 768,
        'roberta-base': 768,
        'allenai/longformer-base-4096': 768,
        }

class ContrastLearningModel(nn.Module):
    def __init__(self, bert_model='google/bigbird-roberta-base', freeze_bert=False):
        super(ContrastLearningModel, self).__init__()
        self.bert_layer = AutoModel.from_pretrained(bert_model)
        
        self.hidden_size = hidden_size_map[bert_model]

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(p=0.1)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.softmax = nn.Softmax(dim=-1)

    @autocast()
    def forward(self, 
            pos_input_ids,
            pos_attention_mask,
            neg_input_ids,
            neg_attention_mask,
            cont_input_ids,
            cont_attention_mask,
            ):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
        '''
        pos_outputs = self.bert_layer(pos_input_ids, pos_attention_mask).pooler_output
        neg_outputs = self.bert_layer(neg_input_ids, neg_attention_mask).pooler_output
        cont_outputs = self.bert_layer(cont_input_ids, cont_attention_mask).pooler_output

        pos_dot = self.cos(pos_outputs,cont_outputs)
        neg_dot = self.cos(neg_outputs,cont_outputs)

        probs = self.softmax(torch.cat([1.-pos_dot,neg_dot],dim=-1))
        loss = -torch.log(probs).mean()

        return loss,probs
