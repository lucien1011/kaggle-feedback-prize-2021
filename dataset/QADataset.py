import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
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
        word_types = self.data.types[index]
        text_id = self.data.id[index]

        encoding = self.tokenizer(
                text.split(),
                max_length=self.max_len,
                **self.tokenizer_args,
                )
        word_ids = encoding.word_ids()
       
        token_type_ids = []
        for i,word_idx in enumerate(word_ids):
            if word_idx is None:
                if encoding.input_ids[i] == 65:
                    token_type_ids.append(word_types[word_ids[i+1]])
                elif encoding.input_ids[i] == 66:
                    token_type_ids.append(word_types[word_ids[i-1]])
                else:
                    token_type_ids.append(0)
            else:
                token_type_ids.append(word_types[word_idx])
        encoding['token_type_ids'] = token_type_ids

        if self.get_labels:
            label_ids = []
            for word_idx in word_ids:                            
                if word_idx is None: 
                    label_ids.append(-100)
                elif word_types[word_idx] == 0:
                    label_ids.append(-100)
                else:
                    label_ids.append( self.labels_to_ids[word_labels[word_idx]] )

            encoding['labels'] = label_ids

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        word_ids2 = [w if w is not None else -1 for w in word_ids]
        item['wids'] = torch.as_tensor(word_ids2)
        item['id'] = text_id

        return item

    def __len__(self):
        return self.len
