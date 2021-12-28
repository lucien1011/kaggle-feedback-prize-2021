from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np

class SentencePairDataset(Dataset):

    def __init__(self, data, maxlen, with_label=True, bert_model='albert-base-v2', one_hot_label=True, num_classes=None):

        self.data = data
        self.text_ids = self.data.id.unique()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen
        self.with_label = with_label
        self.one_hot_label = one_hot_label
        if self.with_label and self.one_hot_label:
            if num_classes:
                self.num_classes = num_classes
            else:
                self.num_classes = len(self.data.label.unique())

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, index):

        tid = self.text_ids[index]

        tid_bool = self.data.id==tid
        sent1 = self.data.loc[tid_bool, 'sent1'].tolist()
        sent2 = self.data.loc[tid_bool, 'sent2'].tolist()

        encoded_pair = self.tokenizer(
                sent1, sent2, 
                padding='max_length',
                truncation=True,
                max_length=self.maxlen,  
                return_tensors='pt'
                )
        
        token_ids = encoded_pair['input_ids']
        attn_masks = encoded_pair['attention_mask']
        token_type_ids = encoded_pair['token_type_ids']

        if self.with_label:
            label = self.data.loc[tid_bool, 'label'].tolist()
            if self.one_hot_label:
                label = F.one_hot(torch.tensor(label),num_classes=self.num_classes)
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids

    @staticmethod
    def collate_fn(batch):
        d = len(batch[0])
        return [torch.cat(list(map(lambda d: d[i],batch))) for i in range(d)]
