import os
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

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

def train_test_split(df):
    from sklearn.model_selection import GroupShuffleSplit
    train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7).split(df, groups=df['id']))
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]
    return train,test

def construct_sent(words):
    sents,indices = [],[]
    start,end = 0,1
    n = len(words)
    while end < n:
        stop_punct = all([punct not in words[end] for punct in ['.','!','?']])
        if stop_punct:
            end += 1
        else:
            indices.append((start,end))
            end += 1
            sents.append(' '.join(words[start:end]))
            start = end
    return sents,indices

def encode_text_to_sent_pair(tokenizer,text):
    sents,indices = construct_sent(text.split()) 
    encoded_pair = tokenizer(
            sents[:-1], sents[1:], 
            padding='max_length',
            truncation=True,
            max_length=128,  
            return_tensors='pt'
            )
    token_ids = encoded_pair['input_ids']
    attn_masks = encoded_pair['attention_mask']
    token_type_ids = encoded_pair['token_type_ids']
    return token_ids, attn_masks, token_type_ids, sents,indices

def predict(model,token_ids,attn_masks,token_type_ids):
    logits = model(token_ids,attn_masks,token_type_ids)
    return logits

def shorten(si,di):
    assert len(si) == len(di)
    n = len(si)
    so,do = [],[]
    for i in range(1,n):
        if si[i][0] == si[i-1][-1]:
            so[-1].extend(si[i])
        else:
            so.append(si[i])
            do.append(di[i])
    return so,do

def convert_logits_to_predictionstring(logits,sent_indices,cat_to_ent_pair):
    cats = torch.argmax(logits,axis=1).squeeze().tolist()
    ent_pairs = [cat_to_ent_pair[cat] for cat in cats]
    ent1s = [ent_pair.split('_')[0] for ent_pair in ent_pairs]
    ent2s = [ent_pair.split('_')[1] for ent_pair in ent_pairs]
    matched = [(sent_indices[0],ent1s[0])] + [(sent_indices[i+1],ent1s[i+1]) for i in range(len(ent_pairs)-1) if ent1s[i+1] == ent2s[i]]
    predictionstring = [list(range(inds[0],inds[1]+1)) for inds,ent in matched]
    discourse_type = [ent for x,ent in matched]
    return shorten(predictionstring,discourse_type)

def run():
    set_seed(1)

    print("Reading dataframe...")
    df = pd.read_csv(os.path.join(io_config['base_dir'],'sent_df.csv'),index_col=0)
    discourse_df = pd.read_csv(io_config['discourse_df_path'])
    text_df = pd.read_csv(os.path.join(io_config['base_dir'],'text_df.csv'),index_col=0)
    train_df,val_df = train_test_split(df)
    num_class = len(df.label.unique())

    print("Reading validation data...")
    pred_set = SentencePairDataset(val_df,io_config['maxlen'],bert_model=io_config['bert_model'],num_classes=num_class,discourse_df=discourse_df,text_df=text_df)
    dataloader = DataLoader(pred_set, batch_size=1,collate_fn=SentencePairDataset.collate_fn)

    print("Reading saved model...")
    model = SentencePairClassifier('bert-base-uncased',num_class=num_class)
    model.load_state_dict(torch.load('storage/output/211225_baseline/train_rev_01/bert-base-uncased_lr_2e-05_val_loss_0.04246_ep_1.pt'))
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    cat_to_ent_pair = pickle.load(open('storage/output/211225_baseline/cat_to_ent_pair.p','rb'))

    model.eval()
    with torch.no_grad():
        for it,_ in enumerate(tqdm(dataloader)):
            true_string = dataloader.dataset.predictionstring(it).tolist()
            true_discourse = dataloader.dataset.discourse_type(it).tolist()
            text = dataloader.dataset.text(it)
            token_ids, attn_masks, token_type_ids, sents, sent_indices = encode_text_to_sent_pair(tokenizer,text)
            logits = predict(model,token_ids, attn_masks, token_type_ids)
            pred_string,pred_discourse = convert_logits_to_predictionstring(logits,sent_indices,cat_to_ent_pair)
            
            print(true_string,pred_string)
            print(true_discourse,pred_discourse)
            break

if __name__ == '__main__':
    args = parse_arguments()
    global io_config
    io_config = read_attr_conf(args.conf,'train_conf')
    run()
    

