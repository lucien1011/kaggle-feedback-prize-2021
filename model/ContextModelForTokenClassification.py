import torch
from torch.cuda.amp import autocast
import torch.nn as nn

from transformers import AutoModel

hidden_size_map = {
        'google/bigbird-roberta-base': 768,
        'roberta-base': 768,
        }

class ContextModelForTokenClassification(nn.Module):
    def __init__(self, bert_model='google/bigbird-roberta-base', freeze_bert=False, num_labels=1):
        super(ContextModelForTokenClassification, self).__init__()
        self.bert_layer = AutoModel.from_pretrained(bert_model)
        
        self.hidden_size = hidden_size_map[bert_model]

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.num_labels = num_labels
        self.cls_layer = nn.Linear(2*self.hidden_size, self.num_labels)

        self.dropout = nn.Dropout(p=0.1)

    @autocast()
    def forward(self, 
            input_ids,
            attn_masks,
            context_input_ids,
            context_attn_masks,
            labels=None,
            ):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
        '''
        outputs = self.bert_layer(input_ids, attn_masks)
        contexts = self.bert_layer(context_input_ids, context_attn_masks)
        bs,maxlen,_ = outputs.last_hidden_state.shape
        concat_output = torch.cat(
                (
                    outputs.last_hidden_state,
                    contexts.pooler_output.unsqueeze(dim=1).expand([bs,maxlen,self.hidden_size])
                    ),
                dim=-1,
                )
        logits = self.cls_layer(self.dropout(concat_output))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attn_masks is not None:
                active_loss = attn_masks.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss,logits
