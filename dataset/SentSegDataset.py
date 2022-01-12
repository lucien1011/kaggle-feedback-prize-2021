import torch
from torch.utils.data import Dataset

class SentSegDataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len, get_labels, labels_to_ids, tokenizer_args,max_seq_length):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_labels = get_labels
        self.labels_to_ids = labels_to_ids
        self.tokenizer_args = tokenizer_args
        self.max_seq_length = max_seq_length

  def __getitem__(self, index):
        text = self.data.text[index]        
        word_labels = self.data.ents[index] if self.get_labels else None
        text_id = self.data.id[index]

        encoding = self.tokenizer(
                text.split(),
                max_length=self.max_len,
                **self.tokenizer_args,
                )
        word_ids = encoding.word_ids()

        item['sent_index_str'] = ["_".join(str(s),str(e)) for s,e in self.data.sent_index[index]]
        encoding['sent_seg_atention_mask'] = + [1]*n + [0]*(self.max_seq_length-n)
        if self.get_labels:
            n = len(self.data.sent_discourse[index])
            encoding['sent_seg_labels'] = self.data.sent_discourse[index] + [0]*(self.max_seq_length-n))

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        word_ids2 = [w if w is not None else -1 for w in word_ids]
        item['wids'] = torch.as_tensor(word_ids2)
        item['id'] = text_id

        return item

  def __len__(self):
        return self.len
