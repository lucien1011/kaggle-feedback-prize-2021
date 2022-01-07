import os
import pandas as pd
from tqdm import tqdm

from pipeline import Module

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

def construct_qa_df(discourse_df,text_df):
    all_entities = []
    for ii,i in tqdm(enumerate(text_df.iterrows())):
        total = i[1]['text'].split().__len__()
        entities = ["O"]*total
        for j in discourse_df[discourse_df['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            entities[list_ix[0]] = f"B-{discourse}"
            entities[list_ix[-1]] = f"E-{discourse}"
        all_entities.append(entities)
    text_df['ents'] = all_entities
    return text_df

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
        
        qa_df = construct_qa_df(container.discourse_df,container.text_df)
        container.add_item('qa_df',qa_df,'df_csv',mode='write')

    def wrapup(self,container,params):
        container.save()

    def check_params(self,params):
        assert all([p in params for p in self._required_params])
