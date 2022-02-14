from bisect import bisect_left
from collections import Counter
import numpy as np
import pandas as pd
import pickle
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset 
from tqdm import tqdm
from transformers import AutoTokenizer

from baseline.Preprocess import target_id_map
from dataset import FeedbackDataset,FeedbackDatasetValid,Collate
from model import NERModel

fold = 0
target_id_map={
    "B-Lead": 0,
    "I-Lead": 1,
    "B-Position": 2,
    "I-Position": 3,
    "B-Evidence": 4,
    "I-Evidence": 5,
    "B-Claim": 6,
    "I-Claim": 7,
    "B-Concluding Statement": 8,
    "I-Concluding Statement": 9,
    "B-Counterclaim": 10,
    "I-Counterclaim": 11,
    "B-Rebuttal": 12,
    "I-Rebuttal": 13,
    "O": 14,
    "PAD": -100,
}

disc_id_map={
    'Lead': 1,
    'Position': 2,
    'Evidence': 3,
    'Claim': 4,
    'Concluding Statement': 5,
    'Counterclaim': 6,
    'Rebuttal': 7,
}

conf = dict(

        device='cuda',

        PrepareData=dict(
            train_samples_path='storage/output/220126_baseline_preprocess_bi_mskfold/NERPreprocessKFold/train_samples_fold{:d}.p'.format(fold),
            valid_samples_path='storage/output/220126_baseline_preprocess_bi_mskfold/NERPreprocessKFold/valid_samples_fold{:d}.p'.format(fold),
            bert_model="allenai/longformer-base-4096",
            max_len=1536,
            train_bs=32,
            val_bs=4,
            target_id_map=target_id_map,
        ),
    
        PrepareModel=dict(
            model_args=dict(
                bert_model="allenai/longformer-base-4096",
                num_labels=15,
                freeze_bert=False,
                dropouts=[0.1,0.2,0.3,0.4,0.5],
                ),
            device='cuda',
            saved_model_path='storage/output/220128_baseline+cvfold0_longformer/NERTrain/allenai-longformer-base-4096_valscore0.64357_ep5.pt',
            ),

        PrepareSeqDataset=dict(
            disc_path='storage/train_folds.csv',
            fold=fold,
            max_seq_len=20,
            disc_id_map=disc_id_map,
            num_labels=7,
            ),

        PrepareLSTM=dict(
            num_input_state=15,
            num_hidden_state=15,
            num_labels=7,
            ),
        
        )

def PrepareData(train_samples_path,valid_samples_path,bert_model,max_len,train_bs,val_bs,target_id_map):
    train_samples = pickle.load(open(train_samples_path,'rb'))
    valid_samples = pickle.load(open(valid_samples_path,'rb'))

    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    train_set = FeedbackDataset(train_samples, max_len, tokenizer, target_id_map)
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True)
    
    valid_set = FeedbackDataset(valid_samples, max_len, tokenizer, target_id_map)
    valid_loader = DataLoader(valid_set, batch_size=val_bs, collate_fn=Collate(tokenizer))

    return dict(
        train_samples=train_samples,
        valid_samples=valid_samples,
        tokenizer=tokenizer,
        train_set=train_set,
        train_loader=train_loader,
        valid_set=valid_set,
        valid_loader=valid_loader,
        )

def PrepareModel(model_args,saved_model_path,device):
    model = NERModel(**model_args)
    model.load_state_dict(torch.load(saved_model_path))
    model.to(device)
    return model

def Infer(dataloader,model,device,show_iters=True):
    out = dict(
            probs=[],
            id=[],
            )
    with torch.no_grad():
        probs = []
        batches = tqdm(dataloader) if show_iters else dataloader
        idx = 0
        for batch in batches:
            if idx > 10: break
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            model_output = model(ids,mask,return_dict=True)
            prob = model_output['probs'].cpu().numpy().tolist()
            out['probs'].extend(prob)
            out['id'].extend(batch['id'])
            idx += 1
    return out

class SequenceDataset(Dataset):
    def __init__(self,probs,tids,disc_df,max_seq_len,disc_id_map,num_labels):
        self.probs = probs
        self.tids = tids
        self.disc_df = disc_df
        self.max_seq_len = max_seq_len
        self.disc_id_map = disc_id_map
        self.num_labels = num_labels

    def __len__(self):
        return len(self.probs)

    def __getitem__(self,idx):
        tid = self.tids[idx]
        tprobs = self.probs[idx]

        num_words = len(tprobs)

        gt_idx = set()
        gt_arr = np.zeros(num_words, dtype=int)
        disc_gt = self.disc_df.loc[self.disc_df.id == tid]
        
        for row_i, row in enumerate(disc_gt.iterrows()):
            splt = row[1]['predictionstring'].split()
            start, end = int(splt[0]), int(splt[-1]) + 1
            gt_idx.add((start, end))
            gt_arr[start:end] = disc_id_map[row['discourse_type']]

        probs,targets,masks = [],[],[]
        for pred_start in range(num_words):
            for pred_end in range(pred_start+1,min(num_words+1,pred_start+self.max_seq_len)):
                prob = tprobs[pred_start:pred_end+1]
                if len(prob) < self.max_seq_len:
                    prob += [[0. for _ in range(len(target_id_map)-1)]] * (self.max_seq_len-len(prob))
                target = np.bincount(gt_arr[pred_start:pred_end+1],minlength=self.num_labels)
                target = target / target.sum()
                mask = np.zeros(self.max_seq_len)
                mask[pred_start:pred_end+1] = 1
                probs.append(prob)
                targets.append(target)
                masks.append(mask)

        return dict(
                prob=torch.tensor(probs,dtype=torch.float),
                target=torch.tensor(targets,dtype=torch.float),
                mask=torch.tensor(masks,dtype=torch.float),
                )

def PrepareSequenceDataset(disc_path,fold,out,max_seq_len,disc_id_map,num_labels):
    disc_df = pd.read_csv(disc_path)
    disc_df = disc_df[disc_df['kfold']==fold]
    seq_dataset = SequenceDataset(out['probs'],out['id'],disc_df,max_seq_len,disc_id_map,num_labels)
    return seq_dataset 

class SequenceModel(nn.Module):
    def __init__(self,num_input_state,num_hidden_state,num_labels):
        super(SequenceModel, self).__init__()

        self.num_input_state = num_input_state
        self.num_hidden_state = num_hidden_state
        self.num_labels = num_labels
        self.lstm = nn.LSTM(
                input_size=num_input_state,
                hidden_size=num_hidden_state,
                batch_first=True,
                bidirectional=True,
                )
        self.cls_layer = nn.Linear(num_hidden_state*2,num_labels)

    def loss(self,logit,target):
        loss_fct = nn.BCELoss()
        loss = loss_fct(logit,target)
        return loss

    def forward(
            self,
            prob,
            mask,
            target=None,
            ):

        sequence_out,_ = self.lstm(prob)
        logit = self.cls_layer(sequence_out).mean(dim=1)
        prob = torch.softmax(logit,dim=-1)
        
        if target is not None:
            loss = self.loss(prob,target)

        return dict(
                loss=loss,
                logit=logit,
                prob=prob,
                )

def PrepareLSTM(num_input_state,num_hidden_state,num_labels):
    model = SequenceModel(num_input_state,num_hidden_state,num_labels)
    return model

def PrepareOptimizer(model,lr=1e-3):
    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=lr)
    return optimizer

def CollateBatch(batch):
    return dict(
            prob=torch.cat([b['prob'] for b in batch],dim=0),
            target=torch.cat([b['target'] for b in batch],dim=0),
            mask=torch.cat([b['mask'] for b in batch],dim=0),
            )

def Train(dataset,model,optimizer,batch_size=4,print_every=1):
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=CollateBatch)
    for idx,batch in enumerate(tqdm(train_loader)):
        out = model(**batch)
        loss = out['loss']
        loss.backward()
        optimizer.step()
        model.zero_grad()
        if idx % print_every==0:
            tqdm.write(f"Training loss after {idx:04d} training steps: {loss.item()}")

def run():
    data = PrepareData(**conf['PrepareData'])
    ner_model = PrepareModel(**conf['PrepareModel'])
    out = Infer(data['train_loader'],ner_model,conf['device'])

    conf['PrepareSeqDataset']['out'] = out
    dataset = PrepareSequenceDataset(**conf['PrepareSeqDataset'])
    seq_model = PrepareLSTM(**conf['PrepareLSTM'])
    opt = PrepareOptimizer(seq_model)
    Train(dataset,seq_model,opt)

if __name__ == '__main__':
    run()
