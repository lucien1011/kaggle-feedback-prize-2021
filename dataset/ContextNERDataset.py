
import torch
from torch.utils.data import Dataset

class ContextNERDataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len, get_labels, labels_to_ids, tokenizer_args, context_labels):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_labels = get_labels
        self.labels_to_ids = labels_to_ids
        self.tokenizer_args = tokenizer_args
        self.context_labels = context_labels

  def __getitem__(self, index):
        text = self.data.text[index]        
        word_labels = self.data.ent[index] if self.get_labels else None
        text_id = self.data.id[index]

        encoding = self.tokenizer(
                text.split(),
                max_length=self.max_len,
                padding='max_length', 
                truncation=True,
                **self.tokenizer_args,
                )
        word_ids = encoding.word_ids()
        glob_attn_mask = [0]*len(word_ids)
        
        if self.get_labels:
            label_ids = []
            for i,word_idx in enumerate(word_ids):                            
                if word_idx is None:
                    label_ids.append(-100)
                elif word_labels[word_idx] in self.context_labels:
                    label_ids.append(-100)
                    glob_attn_mask[i] = 1
                else:
                    label_ids.append( self.labels_to_ids[word_labels[word_idx]] )
            encoding['labels'] = label_ids
            encoding['global_attention_mask'] = glob_attn_mask

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        word_ids2 = [w if w is not None else -1 for w in word_ids]
        item['wids'] = torch.as_tensor(word_ids2)
        item['id'] = text_id

        return item

  def __len__(self):
        return self.len

