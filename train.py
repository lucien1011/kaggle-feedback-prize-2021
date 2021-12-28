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

from dataset import SentencePairDataset
from model import SentencePairClassifier
from utils import mkdir_p,read_attr_conf,set_seed
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using ',device)

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('conf',action='store')
    return parser.parse_args()

def evaluate_loss(net, device, criterion, dataloader):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float())
            count += 1

    return mean_loss / count

def evaluate_metric(net, device, criterion, dataloader):
    net.eval()
    with torch.no_grad():
        for it in tqdm(range(len(dataloader))):
            seq, attn_masks, token_type_ids, labels = dataloader[it]
            predstr = dataloader.dataset.predictionstring[it]
            text = dataloader.dataset.text[it]
            pred = net(seq,attn_masks,token_type_ids)

def train_sent_pair(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, pred_loader, epochs, iters_to_accumulate):

    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // io_config['print_every']
    iters = []
    train_losses = []
    val_losses = []

    scaler = GradScaler()

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        for it, (batch_seq, batch_attn_masks, batch_token_type_ids, batch_labels) in enumerate(tqdm(train_loader)):
            batch_dataset = TensorDataset(batch_seq, batch_attn_masks, batch_token_type_ids, batch_labels)
            batch_loader = DataLoader(batch_dataset, batch_size=io_config['train_bs'])

            for seq, attn_masks, token_type_ids, labels in batch_loader:

                seq, attn_masks, token_type_ids, labels = \
                    seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
    
                with autocast():
                    logits = net(seq, attn_masks, token_type_ids)

                    loss = criterion(logits.squeeze(-1), labels.float())
                    loss = loss / iters_to_accumulate

                scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                scaler.step(opti)
                scaler.update()
                lr_scheduler.step()
                opti.zero_grad()

            running_loss += loss.item()

            if (it + 1) % print_every == 0:
                tqdm.write("[train] Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, running_loss / print_every))
                val_loss = evaluate_loss(net, device, criterion, val_loader)
                tqdm.write("[validation] Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, val_loss.item()))
                val_metric = evaluate_metric(net, device, criterion, pred_loader)
                tqdm.write("[validation] Iteration {}/{} of epoch {} complete. metric : {} "
                      .format(it+1, nb_iterations, ep+1, val_metric.item()))

                running_loss = 0.0

        val_loss = evaluate_loss(net, device, criterion, val_loader)
        tqdm.write("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

        if val_loss < best_loss:
            tqdm.write("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            best_net = copy.deepcopy(net)
            best_loss = val_loss
            best_ep = ep + 1

    out_dir = os.path.join(io_config['base_dir'],io_config['rev_dir'])
    mkdir_p(out_dir)
    path_to_model=os.path.join(out_dir,'{}_lr_{}_val_loss_{}_ep_{}.pt'.format(io_config['bert_model'], lr, round(best_loss, 5), best_ep))
    torch.save(best_net.state_dict(), path_to_model)
    tqdm.write("The model has been saved in {}".format(path_to_model))

    del loss
    torch.cuda.empty_cache()

def train_test_split(df):
    from sklearn.model_selection import GroupShuffleSplit
    train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7).split(df, groups=df['id']))
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]
    return train,test

def run_sent_pair_classification():
    set_seed(1)

    print("Reading dataframe...")
    df = pd.read_csv(os.path.join(io_config['base_dir'],'sent_df.csv'),index_col=0)
    discourse_df = pd.read_csv(io_config['discourse_df_path'])
    text_df = pd.read_csv(os.path.join(io_config['base_dir'],'text_df.csv'),index_col=0)
    train_df,val_df = train_test_split(df)
    num_class = len(df.label.unique())

    print("Reading training data...")
    train_set = SentencePairDataset(train_df,io_config['maxlen'],bert_model=io_config['bert_model'],num_classes=num_class)
    print("Reading validation data...")
    val_set = SentencePairDataset(val_df,io_config['maxlen'],bert_model=io_config['bert_model'],num_classes=num_class)
    pred_set = SentencePairDataset(val_df,io_config['maxlen'],bert_model=io_config['bert_model'],num_classes=num_class,discourse_df=discourse_df,text_df=text_df)
    
    train_loader = DataLoader(train_set, batch_size=io_config['train_textbs'],collate_fn=TextDataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=io_config['val_textbs'],collate_fn=TextDataset.collate_fn)
    pred_loader = DataLoader(pred_set, batch_size=1,collate_fn=TextDataset.collate_fn)
    
    net = SentencePairClassifier(io_config['bert_model'], freeze_bert=io_config['freeze_bert'], num_class=num_class)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    
    net.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    opti = AdamW(net.parameters(), lr=io_config['lr'], weight_decay=io_config['wd'])
    num_warmup_steps = 0
    num_training_steps = io_config['epochs'] * len(train_loader)
    t_total = (len(train_loader) // io_config['iters_to_accumulate']) * io_config['epochs']
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    
    train_sent_pair(net, criterion, opti, io_config['lr'], lr_scheduler, train_loader, val_loader, pred_loader, io_config['epochs'], io_config['iters_to_accumulate'])

if __name__ == '__main__':
    args = parse_arguments()
    global io_config
    io_config = read_attr_conf(args.conf,'train_conf')
    run_sent_pair_classification()
    
