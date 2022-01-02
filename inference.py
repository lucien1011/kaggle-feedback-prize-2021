import copy
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.cuda.amp import autocast,GradScaler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from comp import score_feedback_comp,cat_to_ent
from dataset import SentenceDataset
from model import SentenceClassifier
from utils import mkdir_p,read_attr_conf
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using ',device)

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('conf',action='store')
    return parser.parse_args()

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_test_split(df):
    from sklearn.model_selection import GroupShuffleSplit
    train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7).split(df, groups=df.index))
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]
    return train,test

def compute_local_comp_score(true_df,pred_df,class_label,true_label='discourse_type'):
    f1s = []
    CLASSES = pred_df[class_label].unique()
    print()
    for c in CLASSES:
        p_df = pred_df.loc[pred_df[class_label]==c].copy()
        gt_df = true_df.loc[true_df[true_label]==c].copy()
        f1 = score_feedback_comp(p_df, gt_df, class_label)
        print('class {:s}: {:4.4f}'.format(c,f1))
        if c not in ['O']: f1s.append(f1)
    print()
    print('Overall',np.mean(f1s))
    print()

def inference():
    set_seed(1)

    print("Reading dataframe...")
    sent_df = pd.read_csv(os.path.join(io_config['base_dir'],'sent_df.csv'),index_col=0)
    text_df = pd.read_csv(os.path.join(io_config['base_dir'],'text_df.csv'),index_col=0)
    discourse_df = pd.read_csv('storage/train.csv')

    _,val_df = train_test_split(sent_df)
    num_class = io_config['num_class']

    #print("Reading training data...")
    #train_set = SentenceDataset(train_df,text_df,io_config['maxlen'],bert_model=io_config['bert_model'],num_classes=num_class,one_hot_label=False)
    print("Reading validation data...")
    val_set = SentenceDataset(val_df,text_df,io_config['maxlen'],bert_model=io_config['bert_model'],num_classes=num_class,one_hot_label=False)
    
    #train_loader = DataLoader(train_set, batch_size=io_config['train_textbs'],collate_fn=SentenceDataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=1,collate_fn=SentenceDataset.collate_fn)
    
    net = SentenceClassifier(io_config['bert_model'], freeze_bert=io_config['freeze_bert'], num_class=num_class)
    net.load_state_dict(torch.load(os.path.join(io_config['base_dir'],io_config['model_path'])))
    net.eval()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    
    out_data = {
        'id': [],
        'discourse_type': [],
        'predictionstring': [],
    }
    net.to(device)
    with torch.no_grad():
        for it,batch in enumerate(tqdm(val_loader)):
            if io_config['stopping_it'] != -1 and it > io_config['stopping_it']: break
            batch_dataset = TensorDataset(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label'])
            batch_loader = DataLoader(batch_dataset, batch_size=io_config['sent_bs'])

            pred_cat = []
            for seq, attn_masks, token_type_ids, labels in batch_loader:
                seq, attn_masks, token_type_ids, labels = \
                    seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                pred_cat.extend(torch.argmax(logits,axis=1).cpu().detach().numpy().tolist())
            
            tid = val_loader.dataset.get_id(it)
            sent_wids = eval(val_loader.dataset.get_sent_column(it,'sent_wid'))
            n = len(sent_wids)

            out_data['id'].append(tid)
            out_data['discourse_type'].append(pred_cat[0])
            out_data['predictionstring'].append(" ".join(map(str,range(sent_wids[0][0],sent_wids[0][1]+1))))
            i = 1
            while i < n:
                prev_wid1 = int(out_data['predictionstring'][-1].split()[-1])
                wid0,wid1 = sent_wids[i]
                predictionstring = " ".join(map(str,range(wid0,wid1+1)))
                if out_data['discourse_type'][-1] == pred_cat[i] and prev_wid1+1 == wid0:
                    out_data['predictionstring'][-1] += " " + predictionstring
                else:
                    out_data['id'].append(tid)
                    out_data['discourse_type'].append(pred_cat[i])
                    out_data['predictionstring'].append(predictionstring)
                i += 1
    
    pred_df = pd.DataFrame(out_data)
    pred_df['discourse_type'] = pred_df['discourse_type'].apply(lambda x: cat_to_ent[x])
    mkdir_p(os.path.join(io_config['base_dir'],io_config['rev_dir']))
    pred_df.to_csv(os.path.join(io_config['base_dir'],io_config['rev_dir'],'pred_df.csv'))

    ids = pred_df['id'].unique().tolist()
    compute_local_comp_score(discourse_df[discourse_df['id'].isin(ids)],pred_df,'discourse_type')

if __name__ == "__main__":
    args = parse_arguments()
    global io_config
    io_config = read_attr_conf(args.conf,'eval_conf')
    inference()
