import os
import pandas as pd
from tqdm import tqdm

from pipeline import Module

def construct_contextner_df(discourse_df,text_df):
    all_entities = []
    all_contexts = []
    all_ignore_words = []
    for ii,i in tqdm(enumerate(text_df.iterrows())):
        total = i[1]['text'].split().__len__()
        entities = ["O"]*total
        ignore_words = [0]*total
        context_text = ""
        for j in discourse_df[discourse_df['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            discourse_text = j[1]['discourse_text']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            if discourse in ['Lead', 'Position','Concluding Statement']:
                context_text += discourse_text
                for k in list_ix: ignore_words[k] = 1
            elif discourse in ['Claim','Counterclaim','Rebuttal','Evidence']:
                entities[list_ix[0]] = f"B-{discourse}"
                for k in list_ix[1:]: entities[k] = f"I-{discourse}"
        all_entities.append(entities)
        all_contexts.append(context_text)
        all_ignore_words.append(ignore_words)
    text_df['ent'] = all_entities
    text_df['context'] = all_contexts
    text_df['ignore_words'] = all_ignore_words
    return text_df

class Preprocess(Module):

    _required_params = ['discourse_df_csv_path','text_df_csv_path',]
    
    def __init__(self):
        pass

    def prepare(self,container,params):

        container.add_item('discourse_df',pd.read_csv(params['discourse_df_csv_path']),'df_csv',mode='read')
        container.add_item('text_df',pd.read_csv(params['text_df_csv_path']),'df_csv',mode='read')

    def fit(self,container,params):

        ner_df = construct_contextner_df(container.discourse_df,container.text_df)
        container.add_item('ner_df',ner_df,'df_csv',mode='write')

    def wrapup(self,container,params):
        container.save()
