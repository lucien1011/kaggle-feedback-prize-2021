import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline import Module

def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])

def link_evidence(oof):
    thresh = 1
    idu = oof['id'].unique()
    idc = idu[1]
    eoof = oof[oof['class'] == "Evidence"]
    neoof = oof[oof['class'] != "Evidence"]
    for thresh2 in range(26,27, 1):
        retval = []
        for idv in idu:
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
                    #if pst[start] == 205:
                    #   print(cur, pst[start], cur - pst[start])
                    if (cur == -1 and c != 'Evidence') or ((cur == -1) and ((pst[i+1] > pst[end-1] + thresh) or (pst[i+1] - pst[start] > thresh2))):
                        retval.append((idv, c, jn(pst, start, end)))
                        start = i + 1
                v = (idv, c, jn(pst, start, end+1))
                #print(v)
                retval.append(v)
        roof = pd.DataFrame(retval, columns = ['id', 'class', 'predictionstring']) 
        roof = roof.merge(neoof, how='outer')
        return roof

def get_pred_df(
    probs,samples,ids_to_labels,
    proba_thresh = {
        "Lead": 0.7,
        "Position": 0.55,
        "Evidence": 0.65,
        "Claim": 0.55,
        "Concluding Statement": 0.7,
        "Counterclaim": 0.5,
        "Rebuttal": 0.55,
    },
    min_thresh = {
        "Lead": 9,
        "Position": 5,
        "Evidence": 14,
        "Claim": 3,
        "Concluding Statement": 11,
        "Counterclaim": 6,
        "Rebuttal": 4,
    },
    ):
   
    final_preds = []
    final_scores = []
    for preds in probs:
        pred_class = np.argmax(preds, axis=2)
        pred_scrs = np.max(preds, axis=2)
        for pred, pred_scr in zip(pred_class, pred_scrs):
            final_preds.append(pred.tolist())
            final_scores.append(pred_scr.tolist())

    for j in range(len(samples)):
        tt = [ids_to_labels[p] for p in final_preds[j][1:]]
        tt_score = final_scores[j][1:]
        samples[j]["preds"] = tt
        samples[j]["pred_scores"] = tt_score
        
    submission = []
    for sample_idx, sample in enumerate(samples):
        preds = sample["preds"]
        offset_mapping = sample["offset_mapping"]
        sample_id = sample["id"]
        sample_text = sample["text"]
        sample_input_ids = sample["input_ids"]
        sample_pred_scores = sample["pred_scores"]
        sample_preds = []
    
        if len(preds) < len(offset_mapping):
            preds = preds + ["O"] * (len(offset_mapping) - len(preds))
            sample_pred_scores = sample_pred_scores + [0] * (len(offset_mapping) - len(sample_pred_scores))
        
        idx = 0
        phrase_preds = []
        while idx < len(offset_mapping):
            start, _ = offset_mapping[idx]
            label = preds[idx]
            phrase_scores = []
            phrase_scores.append(sample_pred_scores[idx])
            idx += 1
            while idx < len(offset_mapping):
                matching_label = f"{label}"
                if preds[idx] == matching_label:
                    _, end = offset_mapping[idx]
                    phrase_scores.append(sample_pred_scores[idx])
                    idx += 1
                else:
                    break
            if "end" in locals():
                phrase = sample_text[start:end]
                phrase_preds.append((phrase, start, end, label, phrase_scores))
    
        temp_df = []
        for phrase_idx, (phrase, start, end, label, phrase_scores) in enumerate(phrase_preds):
            word_start = len(sample_text[:start].split())
            word_end = word_start + len(sample_text[start:end].split())
            word_end = min(word_end, len(sample_text.split()))
            ps = " ".join([str(x) for x in range(word_start, word_end)])
            if label != "O":
                if sum(phrase_scores) / len(phrase_scores) >= proba_thresh[label]:
                    if len(ps.split()) >= min_thresh[label]:
                        temp_df.append((sample_id, label, ps))
        
        temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])
        submission.append(temp_df)
    
    df = pd.concat(submission).reset_index(drop=True)
    #df = link_evidence(df)
    return df

class PredictionString(Module):
    
    _required_params = ['probs_name','submission_df_name']

    def prepare(self,container,params):

        self.loader = container.get(params['dataloader'])
        self.ids_to_labels = container.id_target_map
        try:
            self.probs = container.get(params['probs_name'])
        except AttributeError:
            container.read_item_from_path('probs',params['probs_name'],'np_arr')

    def fit(self,container,params):
        df = get_pred_df(container.probs,self.loader.dataset.samples,self.ids_to_labels,params['proba_thresh'],params['min_thresh'])
        container.add_item(params['submission_df_name'],df,'df_csv',mode='write')

    def wrapup(self,container,params):
        container.save()
