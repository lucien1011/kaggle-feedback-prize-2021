import os
import pandas as pd
from tqdm import tqdm

from pipeline import Module

def construct_df(discourse_df):
    text_ids = list(discourse_df.id.unique())
    data = dict(
        id=[],
        cont_text=[],
        pos_text=[],
        neg_text=[],
        )
    for text_id in tqdm(text_ids):
        df = discourse_df[discourse_df.id==text_id]
        counterclaims = df[df.discourse_type=='Counterclaim'].discourse_text.tolist()
        claims = df[df.discourse_type=='Claim'].discourse_text.tolist()
        positions = df[df.discourse_type=='Position'].discourse_text.tolist()
        if len(df[df.discourse_type=='Counterclaim']) == 0: continue 
        for position in positions:
            for claim in claims:
                for counterclaim in counterclaims:
                    data['id'].append(text_id)
                    data['cont_text'].append(position)
                    data['pos_text'].append(claim)
                    data['neg_text'].append(counterclaim)
    df = pd.DataFrame(data)
    return df

class Preprocess(Module):

    _required_params = ['discourse_df_csv_path',]
    
    def __init__(self):
        pass

    def prepare(self,container,params):

        container.add_item('discourse_df',pd.read_csv(params['discourse_df_csv_path']),'df_csv',mode='read')

    def fit(self,container,params):

        df = construct_df(container.discourse_df)
        container.add_item('df',df,'df_csv',mode='write')

    def wrapup(self,container,params):
        container.save()
