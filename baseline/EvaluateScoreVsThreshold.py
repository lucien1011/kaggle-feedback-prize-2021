import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from comp import score_feedback_comp
from pipeline import Module
from .PredictionString import get_predstr_df

class EvaluateScoreVsThreshold(Module):
    
    _required_params = ['pred_df_name','discourse_df_name','prob_thresholds','len_thresholds','classes']

    def prepare(self,container,params):

        try:
            self.pred_df = container.get(params['pred_df_name'])
        except AttributeError:
            self.pred_df = pd.read_csv(params['pred_df_name'])
            for c in self.pred_df.columns:
                if c in ['id']: continue
                self.pred_df[c] = self.pred_df[c].apply(lambda x: eval(x))
        
        try:
            self.discourse_df = container.get(params['discourse_df_name'])
        except AttributeError:
            self.discourse_df = pd.read_csv(params['discourse_df_name'])

    def fit(self,container,params):

        for c in params['classes']:
            print('-'*100)
            print(c)
            f1s = []
            for thresh in params['prob_thresholds']:
                get_predstr_df_args = copy.deepcopy(params.get('get_predstr_df_args',{}))
                get_predstr_df_args['get_string_args']['min_probs_thresh'][c] = thresh                
                predstr_df = get_predstr_df(self.pred_df,**get_predstr_df_args)

                valid = self.discourse_df.loc[self.discourse_df['id'].isin(predstr_df.id.tolist())]
                pred_df = predstr_df.loc[predstr_df['class']==c].copy()
                gt_df = valid.loc[valid['discourse_type']==c].copy()
                f1,tp,fp,fn = score_feedback_comp(pred_df, gt_df, 'class')
                f1s.append(f1)

            if params['prob_thresholds']:
                fig,ax = plt.subplots()
                ax.plot(params['prob_thresholds'],f1s)
                container.add_item('ScoreVsProbThresh_'+c,(fig,ax),'matplotlib_fig+ax',mode='write')
            
            f1s = []
            for thresh in params['len_thresholds']:
                get_predstr_df_args = copy.deepcopy(params.get('get_predstr_df_args',{}))
                get_predstr_df_args['get_string_args']['min_words_thresh'][c] = thresh                
                predstr_df = get_predstr_df(self.pred_df,**get_predstr_df_args)

                valid = self.discourse_df.loc[self.discourse_df['id'].isin(predstr_df.id.tolist())]
                pred_df = predstr_df.loc[predstr_df['class']==c].copy()
                gt_df = valid.loc[valid['discourse_type']==c].copy()
                f1,tp,fp,fn = score_feedback_comp(pred_df, gt_df, 'class')
                f1s.append(f1)
            if params['len_thresholds']:
                fig,ax = plt.subplots()
                ax.plot(params['len_thresholds'],f1s)
                container.add_item('ScoreVsLenThresh_'+c,(fig,ax),'matplotlib_fig+ax',mode='write')


    def wrapup(self,container,params):
        container.save()
