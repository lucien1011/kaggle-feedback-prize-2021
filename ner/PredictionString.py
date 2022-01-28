import pandas as pd
from tqdm import tqdm

from pipeline import Module

def get_string_bi(
        idx,pred,score,
        minword=7,
        minprob=0.5,
        min_words_thresh = {
            "Lead": 9,
            "Position": 5,
            "Evidence": 14,
            "Claim": 3,
            "Concluding Statement": 11,
            "Counterclaim": 6,
            "Rebuttal": 4,
        },
        min_probs_thresh = {
            "Lead": 9,
            "Position": 5,
            "Evidence": 14,
            "Claim": 3,
            "Concluding Statement": 11,
            "Counterclaim": 6,
            "Rebuttal": 4, 
        },
        ):
    preds = []
    j = 0
    while j < len(pred):
        cls = pred[j]
        tot_score = 0.

        if cls == 'O' or cls.startswith('I-'):
            j += 1
            continue
        elif cls.startswith('B-'): 
            cls = cls.replace('B-','')
        
        end = j + 1
        
        while end < len(pred) and pred[end].replace('I-','') == cls:
            tot_score += score[j]
            end += 1

        if cls != 'O' and cls != '':
            min_word_thresh = minword if not min_words_thresh else min_words_thresh[cls]
            min_prob_thresh = minprob if not min_probs_thresh else min_probs_thresh[cls]
            if end - j > min_word_thresh and tot_score / (end-j) > min_prob_thresh:
                preds.append(
                    (idx, cls,' '.join(map(str, list(range(j, end)))))
                    )
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

def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])

def link_evidence(oof):
    thresh = 1
    idu = oof['id'].unique()
    idc = idu[1]
    eoof = oof[oof['class'] == "Evidence"]
    neoof = oof[oof['class'] != "Evidence"]
    for thresh2 in range(20,21, 1):
        retval = []
        for idv in tqdm(idu): #
            for c in  ['Lead', 'Position', 'Evidence', 'Claim', 'Concluding Statement',
                   'Counterclaim', 'Rebuttal']:
                q = eoof[(eoof['id'] == idv) & (eoof['class'] == c)]
                if len(q) == 0:
                    continue
                pst = []
                for i,r in q.iterrows():
                    pst = pst +[-1] + [int(x) for x in r['predictionstring'].split()]
                start = 1
                end = 1
                for i in range(2,len(pst)):
                    cur = pst[i]
                    end = i
                    if (cur == -1 and c != 'Evidence') or ((cur == -1) and ((pst[i+1] > pst[end-1] + thresh) or (pst[i+1] - pst[start] > thresh2))):
                        retval.append((idv, c, jn(pst, start, end)))
                        start = i + 1
                v = (idv, c, jn(pst, start, end+1))
                retval.append(v)
        roof = pd.DataFrame(retval, columns = ['id', 'class', 'predictionstring']) 
        roof = roof.merge(neoof, how='outer')
        return roof

def get_predstr_df(pred_df,ner_type='bi',get_string_args={}):
    n = len(pred_df)
    preds = []
    for i in tqdm(range(n)):
        idx = pred_df.id.values[i]
        pred = pred_df.pred_class.values[i]
        score = pred_df.score.values[i]
        if ner_type == 'bi':
            preds.extend(get_string_bi(idx,pred,score,**get_string_args))
        elif ner_type == 'be':
            preds.extend(get_string_be(idx,pred,**get_string_args))
    df = pd.DataFrame(preds)
    if preds: df.columns = ['id','class','predictionstring']
    #df = link_evidence(df)
    return df

class PredictionString(Module):
    
    _required_params = ['pred_df_name','submission_df_name']

    def prepare(self,container,params):

        try:
            self.pred_df = container.get(params['pred_df_name'])
        except AttributeError:
            self.pred_df = pd.read_csv(params['pred_df_name'])
            for c in self.pred_df.columns:
                if c in ['id']: continue
                self.pred_df[c] = self.pred_df[c].apply(lambda x: eval(x))

    def fit(self,container,params):
        get_predstr_df_args = params.get('get_predstr_df_args',{})
        df = get_predstr_df(self.pred_df,**get_predstr_df_args)
        container.add_item(params['submission_df_name'],df,'df_csv',mode='write')

    def wrapup(self,container,params):
        container.save()
