import copy
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig,AutoModelForTokenClassification

from comp import score_feedback_comp
from pipeline import Module
from utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(batch,model,ids_to_labels):
                
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    outputs = model(ids, attention_mask=mask, return_dict=False)
    all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy() 
    predictions = []
    for k,text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]

        prediction = []
        word_ids = batch['wids'][k].numpy()  
        previous_word_idx = -1
        for idx,word_idx in enumerate(word_ids):                            
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:              
                prediction.append(token_preds[idx])
                previous_word_idx = word_idx
        predictions.append(prediction)
    return predictions

def get_predictions(val_set,val_loader,model,ids_to_labels):
    model.eval()  
    y_pred2 = []
    i = 0
    for batch in tqdm(val_loader):
        labels = inference(batch,model,ids_to_labels)
        y_pred2.extend(labels)
    final_preds2 = []
    for i in tqdm(range(len(y_pred2))):
        idx = val_set.id.values[i]
        pred = y_pred2[i] # Leave "B" and "I"
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
                    final_preds2.append((idx, cls.replace('B-',''),' '.join(map(str, list(range(j, end))))))
                    j = end + 1
                else:
                    j += 1
            else:
                j += 1
    oof = pd.DataFrame(final_preds2)
    if final_preds2: oof.columns = ['id','class','predictionstring']
    return oof

def evaluate_score(discourse_df,pred_df):
    with torch.no_grad():
        f1s = []
        CLASSES = pred_df['class'].unique()
        print()
        for c in CLASSES:
            pred_df_per_class = pred_df.loc[pred_df['class']==c].copy()
            gt_df = discourse_df.loc[discourse_df['discourse_type']==c].copy()
            f1 = score_feedback_comp(pred_df_per_class, gt_df, 'class')
            print(c,f1)
            f1s.append(f1)
        mean_f1_score = np.mean(f1s)
        print()
        print('Overall',np.mean(f1s))
        print()
    return mean_f1_score

class CrossValidate(Module):
    
    def prepare(self,container,params):
        if 'seed' in params: set_seed(params['seed'])
        
        config_model = AutoConfig.from_pretrained(params['bert_model'],**params['config_args']) 
        self.model = AutoModelForTokenClassification.from_pretrained(params['bert_model'],config=config_model)
        self.model.to(device)
        self.model.load_state_dict(torch.load(params['saved_model']))

    def fit(self,container,params):
       
        with torch.no_grad():
            pred_df = get_predictions(container.val_set.data,container.val_loader,self.model,container.ids_to_labels)
        container.add_item('pred_df',pred_df,'df_csv','write')
        true_df = container.discourse_df.loc[container.discourse_df['id'].isin(container.val_set.data.id.tolist())]
        container.add_item('true_df',true_df,'df_csv','write')

        evaluate_score(true_df,pred_df)

    def wrapup(self,container,params):
        container.save()
