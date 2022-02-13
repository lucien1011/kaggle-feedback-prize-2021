import torch
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    def __init__(self, 
            samples, max_len, tokenizer, 
            discourse_type_map, stance_type_map,
            ):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)
        self.discourse_type_map = discourse_type_map
        self.stance_type_map = stance_type_map

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_labels = self.samples[idx]["input_labels"]
        
        discourse_other_label_id = self.discourse_type_map["O"]
        discourse_padding_label_id = self.discourse_type_map["PAD"] 
        discourse_labels = [self.discourse_type_map[x] for x in input_labels]

        stance_other_label_id = self.stance_type_map["O"]
        stance_padding_label_id = self.stance_type_map["PAD"] 
        stance_labels = [self.stance_type_map[x] for x in input_labels]

        # add start token id to the input_ids
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        discourse_labels = [discourse_other_label_id] + discourse_labels
        stance_labels = [stance_other_label_id] + stance_labels

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]
            discourse_labels = discourse_labels[: self.max_len - 1]
            stance_labels = stance_labels[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        discourse_labels = discourse_labels + [discourse_other_label_id] 
        stance_labels = stance_labels + [stance_other_label_id]

        attention_mask = [1] * len(input_ids)
 
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            if self.tokenizer.padding_side == "right":
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                discourse_labels = discourse_labels + [discourse_padding_label_id] * padding_length
                stance_labels = stance_labels + [stance_padding_label_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            else:
                input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                discourse_labels = [discourse_padding_label_id] * padding_length + discourse_labels
                stance_labels = [stance_padding_label_id] * padding_length + stance_labels
                attention_mask = [0] * padding_length + attention_mask

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "discourse_labels": torch.tensor(discourse_labels, dtype=torch.long),
            "stance_labels": torch.tensor(stance_labels, dtype=torch.long),
        }

class MultiTaskDatasetValid:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_ids = [self.tokenizer.cls_token_id] + input_ids

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(input_ids) for input_ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)

        return output
