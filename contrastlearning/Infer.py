import pandas as pd
import torch
from tqdm import tqdm

from pipeline import TorchModule

def inference(batch,tokenizer,model,ids_to_labels,device,true_labels=None):
                
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    outputs = model(ids, attention_mask=mask, return_dict=False)
    all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy() 
    predictions,tokens,labels = [],[],[]
    for k,text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]
        if true_labels is not None: token_trues = [ids_to_labels[i] if i != -100 else 'NA' for i in true_labels[k]]
        token_strs = tokenizer.convert_ids_to_tokens(ids[k])

        prediction,token,label = [],[],[]
        word_ids = batch['wids'][k].numpy()  
        previous_word_idx = -1
        for idx,word_idx in enumerate(word_ids):                            
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:              
                prediction.append(token_preds[idx])
                token.append(token_strs[idx])
                if true_labels is not None: label.append(token_trues[idx])
                previous_word_idx = word_idx
            else:
                token[-1] += token_strs[idx]
        predictions.append(prediction)
        tokens.append(token)
        if true_labels is not None: labels.append(label)
    return predictions,tokens,labels

def get_pred_df(dataloader,model,ids_to_labels,device,add_true_class=True):
    data = {'id':[], 'token':[], 'pred_class':[]}
    if add_true_class: data['true_class'] = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            preds,tokens,trues = inference(batch,dataloader.dataset.tokenizer,model,ids_to_labels,device,batch['labels'].tolist() if add_true_class else None)
            data['token'].extend(tokens)
            data['pred_class'].extend(preds)
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
