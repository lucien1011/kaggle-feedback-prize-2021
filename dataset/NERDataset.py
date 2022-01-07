import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len, get_labels, labels_to_ids, tokenizer_args):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_labels = get_labels
        self.labels_to_ids = labels_to_ids
        self.tokenizer_args = tokenizer_args

  def __getitem__(self, index):
        text = self.data.text[index]        
        word_labels = self.data.ents[index] if self.get_labels else None

        encoding = self.tokenizer(
                text.split(),
                max_length=self.max_len,
                **self.tokenizer_args,
                )
        word_ids = encoding.word_ids()
        
        if self.get_labels:
            #previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:                            
                if word_idx is None: 
                    label_ids.append(-100)
                else:
                    label_ids.append( self.labels_to_ids[word_labels[word_idx]] )
                #elif word_idx != previous_word_idx:              
                #    label_ids.append( self.labels_to_ids[word_labels[word_idx]] )
                #else:
                #    label_ids.append( self.labels_to_ids[word_labels[word_idx]] )
                #previous_word_idx = word_idx
            encoding['labels'] = label_ids

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        word_ids2 = [w if w is not None else -1 for w in word_ids]
        item['wids'] = torch.as_tensor(word_ids2)

        return item

  def __len__(self):
        return self.len
