import copy
import torch
from torch.utils.data import Dataset

class SlideWindowDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, get_labels, labels_to_ids, tokenizer_args,window_size):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_labels = get_labels
        self.labels_to_ids = labels_to_ids
        self.tokenizer_args = tokenizer_args
        self.window_size = window_size
        self.nlabel = len(labels_to_ids)

    def __getitem__(self, index):
        text = self.data.text[index]        
        word_labels = self.data.ent[index] if self.get_labels else None
        text_id = self.data.id[index]

        encoding = self.tokenizer(
                text.split(),
                max_length=self.max_len,
                **self.tokenizer_args,
                )
        word_ids = encoding.word_ids()
        
        token_type_ids,labels = [],[]
        for start_idx in range(0,len(word_ids),self.window_size):
            end_idx = min(start_idx+self.window_size,len(word_ids)-1)

            if all([word_ids[word_idx] is None for word_idx in range(start_idx,end_idx+1)]): break

            token_type_id = [0]*start_idx + [1]*(end_idx-start_idx) + [0]*(self.max_len-end_idx)
            label = [0]*self.nlabel
            for word_idx in range(start_idx,end_idx+1):
                if word_ids[word_idx] is None: continue
                label[self.labels_to_ids[word_labels[word_ids[word_idx]]]] += 1.
            nword = sum(label)
            label = [k/nword for k in label]
            
            token_type_ids.append(token_type_id)
            labels.append(label)

        n = len(labels)
        encoding['input_ids'] = [encoding['input_ids']]*n
        encoding['attention_mask'] = [encoding['attention_mask']]*n
        encoding['token_type_ids'] = token_type_ids
        encoding['labels'] = labels

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        word_ids2 = [[w if w is not None else -1 for w in word_ids]]*n
        item['wids'] = torch.as_tensor(word_ids2)
        item['id'] = [text_id]*n

        return item

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(batch):
        import itertools 
        keys = list(batch[0].keys())
        out = {}
        for k in keys:
            if k in ['id']:    
                out[k] = list(itertools.chain(*[b[k] for b in batch]))
            else:
                out[k] = torch.cat([b[k] for b in batch])
        return out
