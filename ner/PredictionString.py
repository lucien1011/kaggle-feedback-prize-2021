import pandas as pd
from tqdm import tqdm

from pipeline import Module

def get_string_bi(idx,pred,minword=7):
    preds = []
    j = 0
    while j < len(pred):
        cls = pred[j]
        if cls == 'O': j += 1
        else: cls = cls.replace('B-','I-').replace('E-','I-') # spans start with B
        end = j + 1
        while end < len(pred) and pred[end] == cls:
            end += 1
        
        if cls != 'O' and cls != '' and end - j > minword:
            preds.append((idx, cls.replace('I-',''),
                                 ' '.join(map(str, list(range(j, end))))))
        j = end
    return preds

def get_string_be(idx,pred,minword=7):
    preds = []
    j = 0
    while j < len(pred):
        cls = pred[j]
        if cls.startswith('B-'):
            end_cls = cls.replace('B-','E-')
            end = j + 1
            while end < len(pred) and pred[end] != end_cls:
                end += 1 
            if end < len(pred) and pred[end] == end_cls and end - j > 7:
                preds.append((idx, cls.replace('B-',''),' '.join(map(str, list(range(j, end))))))
                j = end + 1
            else:
                j += 1
        else:
            j += 1
    return preds

def get_predstr_df(pred_df):
    n = len(pred_df)
    preds = []
    for i in tqdm(range(n)):
        idx = pred_df.id.values[i]
        pred = pred_df.pred_class.values[i]
        preds.extend(get_string_bi(idx,pred))
    df = pd.DataFrame(preds)
    if preds: df.columns = ['id','class','predictionstring']
    return df

class PredictionString(Module):
    
    _required_params = ['pred_df_name','submission_df_name']

    def prepare(self,container,params):

        try:
            self.pred_df = container.get(params['pred_df_name'])
        except AttributeError:
            self.pred_df = pd.read_csv(params['pred_df_name'],index_col=0)
            self.pred_df['predictionstring'] = self.pred_df['predictionstring'].apply(lambda x: eval(x))

    def fit(self,container,params):
        df = get_predstr_df(self.pred_df)
        container.add_item(params['submission_df_name'],df,'df_csv',mode='write')

    def wrapup(self,container,params):
        container.save()
