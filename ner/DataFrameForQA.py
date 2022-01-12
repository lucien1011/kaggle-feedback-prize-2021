import pandas as pd

from comp import score_feedback_comp
from pipeline import Module

class DataFrameForQA(Module):
    
    _required_params = ['discourse_df_name','submission_df_name','text_df_name','out_discourse_df_name','out_text_df_name',]

    def prepare(self,container,params):

        try:
            self.discourse_df = container.get(params['discourse_df_name'])
        except AttributeError:
            self.discourse_df = pd.read_csv(params['discourse_df_name'],index_col=0)
        try:
            self.submission_df = container.get(params['submission_df_name'])
        except AttributeError:
            self.submission_df = pd.read_csv(params['submission_df_name'],index_col=0)

        try:
            self.text_df = container.get(params['text_df_name'])
        except AttributeError:
            self.text_df = pd.read_csv(params['text_df_name'],index_col=0)

    def fit(self,container,params):
        
        true_df = self.discourse_df.loc[self.discourse_df['discourse_type'].isin(['Claim','Counterclaim','Rebuttal','Evidence'])]
        true_df = true_df[['discourse_type','predictionstring']].reset_index()
        
        pred_df = self.submission_df.loc[self.submission_df['class'].isin(['Lead','Position','Concluding Statement'])]
        pred_df = pred_df[['id','class','predictionstring']].rename(columns={'class': 'discourse_type'})
        
        true_df = true_df.loc[true_df['id'].isin(pred_df.id.unique().tolist())]
        text_df = self.text_df.loc[self.text_df['id'].isin(pred_df.id.unique().tolist())]

        df = pd.concat([true_df,pred_df])
        container.add_item(params['out_discourse_df_name'],df,'df_csv','write')
        container.add_item(params['out_text_df_name'],text_df,'df_csv','write')

    def wrapup(self,container,params):
        container.save()
