from collections import Counter
import os
import pandas as pd
from tqdm import tqdm

from comp import ent_to_cat
from pipeline import Module

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

class Preprocess(Module):

    _required_params = ['discourse_df_csv_path','text_df_csv_fname','text_dir']
    
    def __init__(self):
        pass

    def prepare(self,container,params):

        self.check_params(params)
        container.add_item('discourse_df',pd.read_csv(params['discourse_df_csv_path']),'df_csv',mode='read')

    def fit(self,container,params):
        
        if not container.file_exists(params['text_df_csv_fname']):
            container.add_item('text_df',construct_text_df(params['text_dir']),'df_csv',mode='write')
        else:
            container.read_item_from_dir('text_df','df_csv',args=dict(index_col=0))
        
        sent_df = construct_sent_df(container.discourse_df,container.text_df)
        container.add_item('sent_df',sent_df,'df_csv',mode='write')

    def wrapup(self,container,params):
        container.save()

    def check_params(self,params):
        assert all([p in params for p in self._required_params])
