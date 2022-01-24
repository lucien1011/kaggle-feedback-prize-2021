import torch
from torch.cuda.amp import autocast
import torch.nn as nn

from transformers import AutoModel,AutoConfig

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

    @autocast()
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

        logits = torch.stack(
            [self.cls_layer(do_layer(sequence_output)) for do_layer in self.dropout_layers],
            dim=0,
            ).sum(dim=0) / len(self.dropouts)
        probs = torch.softmax(logits,dim=-1)

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
        
        if return_dict:
            return dict(
                loss=loss,
                logits=logits,
                probs=probs
            )
        else:
            return loss,logits,probs
