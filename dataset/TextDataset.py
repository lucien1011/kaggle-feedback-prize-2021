import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

class TextDataset(Dataset):

    def __init__(self, text_data, discourse_data, ent_pair_to_cat, cat_to_ent_pair,
        max_length=512,
        bert_model='albert-base-v2',
        with_label=True,
        one_hot_label=True,
        ): 
        self.text_data = text_data
        self.discourse_data = discourse_data
        self.ent_pair_to_cat = ent_pair_to_cat
        self.cat_to_ent_pair = cat_to_ent_pair
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.with_label = with_label
        self.one_hot_label = one_hot_label
        if self.with_label and self.one_hot_label:
            self.num_classes = len(self.ent_pair_to_cat)
        self._cache = {}
        self.return_tensors_type = 'pt'

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):

        row = self.text_data.iloc[index]
        tid = row.id
        text = row.text

        if tid in self._cache: return self._cache[tid]
        return self.encode(tid,text)

    def encode(self,tid,text):

        discourse_df = self.discourse_data[self.discourse_data['id']==tid]
        sents,labels = self.construct_sent_ent(text,discourse_df,tid)
        encoded_pair = self.tokenizer(
            sents[:-1], sents[1:], 
            padding='max_length',
            truncation=True,
            max_length=self.max_length,  
            return_tensors=self.return_tensors_type
            )
        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        
        if self.with_label:
            label = [self.ent_pair_to_cat['_'.join([labels[i][0],labels[i+1][0]])] for i in range(len(labels)-1)]
            if self.one_hot_label:
                label = F.one_hot(torch.tensor(label),num_classes=self.num_classes)
            self._cache[tid] = (token_ids, attn_masks, token_type_ids, label)
            return token_ids, attn_masks, token_type_ids, label  
        else:
            self._cache[tid] = (token_ids, attn_masks, token_type_ids)
            return token_ids, attn_masks, token_type_ids

    def construct_sent_ent(self,text,discourse_df,tid):
        words = text.split()
        n = len(words)
        if self.with_label:
            ents = ['O']*n
            for j in discourse_df.iterrows():
                discourse = j[1]['discourse_type']
                list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
                for k in list_ix: ents[k] = f"{discourse}"
            sents,labels = self.construct_sent_label(words,ents)
        else:
            labels = None
            sents = self.construct_sent(words)
        return sents,labels
        
    def construct_sent_label(self,words,ents):
        assert len(words) == len(ents)
        sents,labels = [],[]
        start,end = 0,1
        prev_ent = ents[0]
        n = len(words)
        while end < n:
            stop_punct = all([punct not in words[end] for punct in ['.','!','?']])
            same_ent = (end <= n-2) and (ents[end].split('-')[-1] == ents[end+1].split('-')[-1]) 
            if stop_punct and same_ent:
                end += 1
            else:
                end += 1
                sents.append(' '.join(words[start:end]))
                labels.append(ents[start:end])
                start = end
        return sents,labels

    def construct_sent(self,words):
        sents = []
        start,end = 0,1
        n = len(words)
        while end < n:
            stop_punct = all([punct not in words[end] for punct in ['.','!','?']])
            if stop_punct:
                end += 1
            else:
                end += 1
                sents.append(' '.join(words[start:end]))
                start = end
        return sents
   
    @staticmethod
    def collate_fn(batch):
        d = len(batch[0])
        return [torch.cat(list(map(lambda d: d[i],batch))) for i in range(d)]

    def save_cache(self,saved_path):
        for ir,row in tqdm(self.text_data.iterrows()):
            self._cache[row.id] = self.encode(row.id,row.text)
        pickle.dump(self._cache,open(saved_path,'wb'))

    def load_cache(self,saved_path):
        self._cache = pickle.load(open(saved_path,'rb'))
