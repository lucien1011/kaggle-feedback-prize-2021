import torch
from torch.utils.data import Dataset

class ContrastLearningDataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len, get_labels, tokenizer_args):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.maxlen = max_len
        self.get_labels = get_labels
        self.tokenizer_args = tokenizer_args

  def __getitem__(self, index):
        text_id = self.data.id[index]
        
        pos_texts = self.data[self.data.id==text_id].pos_text.tolist()
        neg_texts = self.data[self.data.id==text_id].neg_text.tolist()
        cont_texts = self.data[self.data.id==text_id].cont_text.tolist()

        pos_enc = self.tokenizer(pos_texts,max_length=self.maxlen,**self.tokenizer_args)
        neg_enc = self.tokenizer(neg_texts,max_length=self.maxlen,**self.tokenizer_args)
        cont_enc = self.tokenizer(cont_texts,max_length=self.maxlen,**self.tokenizer_args)

        encoding = {}
        for attr in ['input_ids','attention_mask']:
            encoding['pos_'+attr] = pos_enc[attr]
            encoding['neg_'+attr] = neg_enc[attr]
            encoding['cont_'+attr] = cont_enc[attr]

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['id'] = text_id

        return item

  def __len__(self):
        return self.len
