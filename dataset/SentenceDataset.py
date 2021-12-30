import copy
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

def get_sublist_index(l1,l2):
    """
    Assume n1 > n2
    l1: input_ids as list
    l2: input_ids as list
    """
    n1 = len(l1)
    n2 = len(l2)
    i = 0
    while i < n1-n2:
        if l1[i:(i+n2)] == l2:
            return i,i+n2+1
        i += 1

class SentenceDataset(Dataset):

    def __init__(self, 
            sent_data, text_data, maxlen, 
            bert_model='albert-base-v2', 
            num_classes=None, 
            with_label=True, one_hot_label=True,
            ):

        self.sent_data = sent_data
        self.text_data = text_data[text_data.id.isin(sent_data.index.tolist())]
        self.text_ids = self.text_data.id.unique()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen
        self.num_classes = num_classes
        
        self.with_label = with_label
        self.one_hot_label = one_hot_label

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, index):

        tid = self.text_ids[index]

        sents = eval(self.sent_data.loc[tid,'sent'])
        text = ' '.join(sents)
      
        enc_text = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.maxlen,  
                )
        enc_sents = self.tokenizer(sents)

        enc_outs = dict(input_ids=[],attention_mask=[],token_type_ids=[])
        for j in range(len(sents)):
            input_ids = copy.deepcopy(enc_text['input_ids'])
            attention_mask= copy.deepcopy(enc_text['attention_mask'])
            token_type_ids = copy.deepcopy(enc_text['token_type_ids'])
            inds = get_sublist_index(input_ids,enc_sents['input_ids'][j][1:-1])
            if inds is not None:
                for i in range(*inds):
                    token_type_ids[i] = 1
            enc_outs['input_ids'].append(input_ids)
            enc_outs['attention_mask'].append(attention_mask)
            enc_outs['token_type_ids'].append(token_type_ids)
        
        enc_outs['input_ids'] = torch.tensor(enc_outs['input_ids'])
        enc_outs['attention_mask'] = torch.tensor(enc_outs['attention_mask'])
        enc_outs['token_type_ids'] = torch.tensor(enc_outs['token_type_ids'])
        if self.with_label:
            enc_outs['label'] = torch.tensor(eval(self.sent_data.loc[tid,'approx_sent_discourse_cat']))
            if self.one_hot_label:
                enc_outs['label'] = F.one_hot(enc_outs['label'],num_classes=self.num_classes)
        return enc_outs

    @staticmethod
    def collate_fn(batch):
        n = len(batch)
        keys = batch[0].keys()
        return {k: torch.cat([batch[i][k] for i in range(n)]) for k in keys}

if __name__ == "__main__":
    sent_df = pd.read_csv('../storage/output/211228_baseline/sent_df.csv',index_col=0)
    text_df = pd.read_csv('../storage/output/211228_baseline/text_df.csv',index_col=0)
    dataset = SentenceDataset(sent_df,text_df,256,bert_model='bert-base-uncased',num_classes=8)
    dataloader = DataLoader(dataset,batch_size=2,collate_fn=SentenceDataset.collate_fn)
    for obj in tqdm(dataloader):
        print(obj['input_ids'].shape)
        print(obj['attention_mask'].shape)
        print(obj['token_type_ids'].shape)
        print(obj['label'].shape)
        break
