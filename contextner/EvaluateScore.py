import numpy as np

from comp import score_feedback_comp
from pipeline import Module

class EvaluateScore(Module):
    
    _required_params = ['discourse_df_name','submission_df_name']

    def prepare(self,container,params):

        self.discourse_df = container.get(params['discourse_df_name'])
        self.submission_df = container.get(params['submission_df_name'])

    def fit(self,container,params):
        if not len(self.submission_df): return 0.
        valid = self.discourse_df.loc[self.discourse_df['id'].isin(self.submission_df.id.tolist())]
        f1s = []
        CLASSES = self.submission_df['class'].unique()
        print()
        for c in CLASSES:
            pred_df = self.submission_df.loc[self.submission_df['class']==c].copy()
            gt_df = valid.loc[valid['discourse_type']==c].copy()
            f1 = score_feedback_comp(pred_df, gt_df, 'class')
            print(c,f1)
            f1s.append(f1)
        mean_f1_score = np.mean(f1s)
        print()
        print('Overall',np.mean(f1s))
        print()
        return mean_f1_score
