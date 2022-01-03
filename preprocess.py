from collections import Counter
import os
import nltk
import pandas as pd
import pickle
from tqdm import tqdm

from utils import mkdir_p,read_attr_conf,Timer

ent_to_cat = {
        'Lead': 0,
        'Position': 1,
        'Claim': 2,
        'Counterclaim': 3,
        'Rebuttal' : 4,
        'Evidence' : 5,
        'Concluding Statement' : 6,
        'O': 7,
        }

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('conf',action='store')
    return parser.parse_args()

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
        words = i[1]['text'].split()
        sents,sent_indices = construct_sent(words)
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

def construct_sent_df(discourse_df,text_df):
    tqdm.write('-'*100)
    tqdm.write('Construct sentence dataframe')
    data = []
    for i in tqdm(text_df.iterrows()):
        tid = i[1]['id']
        text = i[1]['text']
        words = text.split()
        sents,sent_indices = construct_sent(words)
        disc_df_id = discourse_df[discourse_df['id'] == tid]
        discourses = [(j[1]['discourse_type'],list(map(int,j[1]['predictionstring'].split(' ')))) for j in disc_df_id.iterrows()] 
        entities = ["O"]*len(words)
        for j in disc_df_id.iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            for k in list_ix: entities[k] = f"{discourse}"
        if sents:
            approx_sent_discourses = [entities[sent_index[0]] for sent_index in sent_indices]
            sent_discourses = [dict(Counter(entities[sent_index[0]:sent_index[1]+1])) for sent_index in sent_indices]
            data.append([tid,sents,sent_indices,approx_sent_discourses,sent_discourses])
    sent_df = pd.DataFrame(data,columns=('id','sent','sent_wid','approx_sent_discourse','sent_discourse')).set_index('id')
    sent_df['approx_sent_discourse_cat'] = sent_df['approx_sent_discourse'].apply(lambda x: [ent_to_cat[i] for i in x])
    tqdm.write('-'*100)
    return sent_df 

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

def run_sent_classification():
    discourse_df = pd.read_csv(io_config['input_train_csv_path'])
    if not os.path.exists(os.path.join(io_config['base_dir'],'text_df.csv')):
        text_df = construct_text_df(io_config['input_train_text_dir'])
        save(text_df,'text_df.csv')
    else:
        text_df = pd.read_csv(os.path.join(io_config['base_dir'],'text_df.csv'),index_col=0)
    sent_df = construct_sent_df(discourse_df,text_df)
    save(sent_df,'sent_df.csv')

def run():
    if io_config['type'] == 'ner':
        run_ner()
    elif io_config['type'] == 'sent_classification':
        run_sent_classification()

if __name__ == "__main__":
    args = parse_arguments()
    global io_config
    io_config = read_attr_conf(args.conf,'preprocess_conf')
    run()
