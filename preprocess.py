import os
import pandas as pd
import pickle
from tqdm import tqdm

from utils import mkdir_p,read_attr_conf

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

    output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
              'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']
    labels_to_ids = {v:k for k,v in enumerate(output_labels)}
    ids_to_labels = {k:v for k,v in enumerate(output_labels)}
    tqdm.write('-'*100)
    return text_df,labels_to_ids,ids_to_labels

def save(obj,fname,mode='df'):
    mkdir_p(io_config['base_dir'])
    if mode == 'df':
        obj.to_csv(os.path.join(io_config['base_dir'],fname))
    elif mode == 'pickle':
        pickle.dump(obj,open(os.path.join(io_config['base_dir'],fname),'wb'))
    else:
        raise RuntimeError

def run():
    discourse_df = pd.read_csv(io_config['input_train_csv_path'])
    text_df = construct_text_df(io_config['input_train_text_dir'])
    save(text_df,'text_df.csv')
    df,labels_to_ids,ids_to_labels = construct_ner_df(discourse_df,text_df)
    save(df,'df.csv')
    save(labels_to_ids,'labels_to_ids.csv',mode='pickle')
    save(ids_to_labels,'ids_to_labels.csv',mode='pickle')

if __name__ == "__main__":
    args = parse_arguments()
    global io_config
    io_config = read_attr_conf(args.conf,'preprocess_conf')
    run()
