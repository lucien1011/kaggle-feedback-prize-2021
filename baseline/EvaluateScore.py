import numpy as np
import pandas as pd

from comp import score_feedback_comp
from pipeline import Module

def evaluate_score_from_df(discourse_df,submission_df):
    if not len(submission_df): return 0.
    f1s = []
    CLASSES = submission_df['class'].unique()
    print()
    for c in CLASSES:
        pred_df = submission_df.loc[submission_df['class']==c].copy()
        gt_df = discourse_df.loc[discourse_df['discourse_type']==c].copy()
        f1,tp,fp,fn = score_feedback_comp(pred_df, gt_df, 'class')
        print(c,f1,tp,fp,fn)
        f1s.append(f1)
    mean_f1_score = np.mean(f1s)
    print()
    print('Overall',np.mean(f1s))
    print()
    return mean_f1_score

class EvaluateScore(Module):
    
    _required_params = ['discourse_df_name','submission_df_name']

    def prepare(self,container,params):

        try:
            self.discourse_df = container.get(params['discourse_df_name'])
        except AttributeError:
            self.discourse_df = pd.read_csv(params['discourse_df_name'])
        try:
            self.submission_df = container.get(params['submission_df_name'])
        except AttributeError:
            self.submission_df = pd.read_csv(params['submission_df_name'])
            for c in self.submission_df.columns:
                if c in ['id']: continue
                self.submission_df[c] = self.submission_df[c].apply(lambda x: eval(x))

    def fit(self,container,params):
        valid = self.discourse_df.loc[self.discourse_df['id'].isin(self.submission_df.id.tolist())]
        evaluate_score_from_df(valid,self.submission_df) 
