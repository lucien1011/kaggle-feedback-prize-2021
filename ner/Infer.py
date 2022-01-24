import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from pipeline import TorchModule

def inference(batch,tokenizer,model,ids_to_labels,device,true_labels=None):
                
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    out = model(ids, attention_mask=mask, return_dict=True)
    logits = out['logits']
    probs = torch.softmax(logits,dim=-1)
    predictions,scores,tokens,labels = [],[],[],[]
    for k,prob in enumerate(probs):
        score_preds = np.max(prob.cpu().numpy(),axis=-1)
        class_preds = np.argmax(prob.cpu().numpy(),axis=-1)
        token_preds = [ids_to_labels[i] for i in class_preds]
        if true_labels is not None: token_trues = [ids_to_labels[i] if i != -100 else 'NA' for i in true_labels[k]]
        token_strs = tokenizer.convert_ids_to_tokens(ids[k])
        prediction,score,token,label = [],[],[],[]
        word_ids = batch['wids'][k].numpy()  
        previous_word_idx = -1
        for idx,word_idx in enumerate(word_ids):                            
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:              
                prediction.append(token_preds[idx])
                token.append(token_strs[idx])
                score.append(score_preds[idx])
                if true_labels is not None: label.append(token_trues[idx])
                previous_word_idx = word_idx
            else:
                token[-1] += token_strs[idx]
        predictions.append(prediction)
        tokens.append(token)
        scores.append(score)
        if true_labels is not None: labels.append(label)
    return predictions,scores,tokens,labels

def get_pred_df(dataloader,model,ids_to_labels,device,add_true_class=True):
    data = {'id':[], 'token':[], 'pred_class':[], 'score':[]}
    if add_true_class: data['true_class'] = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader):
            preds,scores,tokens,trues = inference(batch,dataloader.dataset.tokenizer,model,ids_to_labels,device,batch['labels'].tolist() if add_true_class else None)
            data['token'].extend(tokens)
            data['pred_class'].extend(preds)
            data['score'].extend(scores)
            data['id'].extend(batch['id'])
            if add_true_class: data['true_class'].extend(trues)
    df = pd.DataFrame(data)
    return df

class Infer(TorchModule):

    _required_params = ['model_name','dataloader','add_true_class','pred_df_name',]

    def prepare(self,container,params):

        self.model = container.get(params['model_name'])
        self.loader = container.get(params['dataloader'])
        self.ids_to_labels = container.ids_to_labels
    
    def fit(self,container,params):

        with torch.no_grad():
            self.model.eval()
            df = get_pred_df(self.loader,self.model,self.ids_to_labels,self.device,params['add_true_class'])
        container.add_item(params['pred_df_name'],df,'df_csv',mode='write')

    def wrapup(self,container,params):
        container.save()
