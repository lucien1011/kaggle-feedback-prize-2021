import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

class TextDataset(Dataset):

    def __init__(self, text_data): 
        self.text_data = text_data

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):

        row = self.text_data.iloc[index]
        tid = row.id
        text = row.text
        return text

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

    def predictionstring(self,index):
        row = self.discourse_data.iloc[index]
        return eval(row.predictionstring)

    def text(self,index):
        row = self.text_data.iloc[index]
        return row.text
