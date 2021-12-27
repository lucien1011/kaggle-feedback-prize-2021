import os
import nltk
import pandas as pd
import pickle
from tqdm import tqdm

from utils import mkdir_p,read_attr_conf,Timer

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('conf',action='store')
    return parser.parse_args()

def construct_text_df(input_text_dir):
    tqdm.write('-'*100)
    tqdm.write('Construct text dataframe')
    ids, texts = [], []
    for f in tqdm(list(os.listdir(input_text_dir))):
        ids.append(f.replace('.txt', ''))
        texts.append(open(os.path.join(input_text_dir,f),'r').read())
    text_df = pd.DataFrame({'id': ids, 'text': texts})
    tqdm.write('-'*100)
    return text_df

def construct_ner_df(discourse_df,text_df):
    tqdm.write('-'*100)
    tqdm.write('Construct ner dataframe')
    all_entities = []
    for i in tqdm(text_df.iterrows()):
        total = i[1]['text'].split().__len__()
        entities = ["O"]*total
        for j in discourse_df[discourse_df['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            entities[list_ix[0]] = f"B-{discourse}"
            for k in list_ix[1:]: entities[k] = f"I-{discourse}"
        all_entities.append(entities)
    text_df['entities'] = all_entities

    output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']
    labels_to_ids = {v:k for k,v in enumerate(output_labels)}
    ids_to_labels = {k:v for k,v in enumerate(output_labels)}
    tqdm.write('-'*100)
    return text_df,labels_to_ids,ids_to_labels

def construct_sent_label(words,ents):
    assert len(words) == len(ents)
    sents,labels = [],[]
    start,end = 0,1
    prev_ent = ents[0]
    n = len(words)
    while end < n:
        stop_punct = all([punct not in words[end] for punct in ['.','!','?']])
        same_ent = (end <= n-2) and (ents[end].split('-')[-1] == ents[end+1].split('-')[-1]) 
        if stop_punct and same_ent:
            end += 1
        else:
            end += 1
            sents.append(' '.join(words[start:end]))
            labels.append(ents[start:end])
            start = end
    return sents,labels 

def construct_sent_pair_df(ner_df):
    
    def clean_ner_label(s):
        return s.split('-')[-1] if '-' in s else s

    tqdm.write('-'*100)
    tqdm.write('Construct sentence pair dataframe')
    entries = []
    for irow,row in tqdm(ner_df.iterrows()):
        words = row.text.split()
        ents = eval(row.entities)
        sents,labels = construct_sent_label(words,ents)
        entries.append(([row.id]*len(sents),sents,labels))
    sent_df = pd.DataFrame(entries,columns=['id','sent','ent'])
    sent_df['id'] = sent_df['id'].apply(lambda x: x[:-1])
    sent_df['sent1'] = sent_df['sent'].apply(lambda x: x[:-1])
    sent_df['sent2'] = sent_df['sent'].apply(lambda x: x[1:])
    sent_df['ent_pair'] = sent_df['ent'].apply(lambda x: ["_".join([clean_ner_label(x[i][0]),clean_ner_label(x[i+1][0])]) for i in range(len(x)-1)])
    sent_df = sent_df.explode(['id','sent1','sent2','ent_pair']).reset_index()
    sent_df['label'] = sent_df['ent_pair'].astype('category').cat.codes
    sent_df = sent_df[['id','sent1','sent2','ent_pair','label']].dropna()
    cat_to_ent_pair = dict( zip( sent_df['label'],sent_df['ent_pair'] ) )
    ent_pair_to_cat = dict( zip( sent_df['ent_pair'],sent_df['label'] ) )
    return sent_df,ent_pair_to_cat,cat_to_ent_pair

def save(obj,fname,mode='df'):
    mkdir_p(io_config['base_dir'])
    if mode == 'df':
        obj.to_csv(os.path.join(io_config['base_dir'],fname))
    elif mode == 'pickle':
        pickle.dump(obj,open(os.path.join(io_config['base_dir'],fname),'wb'))
    else:
        raise RuntimeError

def run_ner():
    discourse_df = pd.read_csv(io_config['input_train_csv_path'])
    text_df = construct_text_df(io_config['input_train_text_dir'])
    save(text_df,'text_df.csv')
    df,labels_to_ids,ids_to_labels = construct_ner_df(discourse_df,text_df)
    save(df,'ner_df.csv')
    save(labels_to_ids,'labels_to_ids.csv',mode='pickle')
    save(ids_to_labels,'ids_to_labels.csv',mode='pickle')

def run_sent_pair():
    ner_df_path = os.path.join(io_config['base_dir'],'ner_df.csv')
    if not os.path.exists(ner_df_path):
        run_ner()
    else:
        ner_df = pd.read_csv(ner_df_path,index_col=0)
        sent_df,ent_pair_to_cat,cat_to_ent_pair = construct_sent_pair_df(ner_df)
        save(sent_df,'sent_df.csv')
        save(ent_pair_to_cat,'ent_pair_to_cat.p',mode='pickle')
        save(cat_to_ent_pair,'cat_to_ent_pair.p',mode='pickle')


def run():
    if io_config['type'] == 'ner':
        run_ner()
    elif io_config['type'] == 'sent_pair':
        run_sent_pair()

if __name__ == "__main__":
    args = parse_arguments()
    global io_config
    io_config = read_attr_conf(args.conf,'preprocess_conf')
    run()
