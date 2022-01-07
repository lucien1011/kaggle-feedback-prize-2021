import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset 
from tqdm import tqdm
from transformers import AutoConfig,AutoModelForTokenClassification

from pipeline import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(batch,tokenizer,model,ids_to_labels,true_labels):
                
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    outputs = model(ids, attention_mask=mask, return_dict=False)
    all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy() 
    predictions,tokens,labels = [],[],[]
    for k,text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]
        token_trues = [ids_to_labels[i] if i != -100 else 'NA' for i in true_labels[k]]
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
                label.append(token_trues[idx])
                previous_word_idx = word_idx
            else:
                token[-1] += token_strs[idx]
        predictions.append(prediction)
        tokens.append(token)
        labels.append(label)
    return predictions,tokens,labels

class Infer(Module):

    _header = '-'*100

    def prepare(self,container,params):
        config_model = AutoConfig.from_pretrained(params['bert_model'],**params['config_args']) 
        self.model = AutoModelForTokenClassification.from_pretrained(params['bert_model'],config=config_model)
        self.model.to(device)
        self.model.load_state_dict(torch.load(params['saved_model']))
    
    def fit(self,container,params):

        with torch.no_grad():
            self.model.eval()  
            data = {'token':[], 'pred_class':[], 'true_class':[]}
            for batch in tqdm(container.val_loader):
                preds,tokens,trues = inference(batch,container.val_loader.dataset.tokenizer,self.model,container.ids_to_labels,batch['labels'].tolist())
                data['token'].extend(tokens)
                data['pred_class'].extend(preds)
                data['true_class'].extend(trues)
        
        df = pd.DataFrame(data)
        container.add_item('df',df,'df_csv',mode='write')

    def wrapup(self,container,params):
        container.save()
