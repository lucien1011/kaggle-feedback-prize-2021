import pandas as pd
import numpy as np
from sklearn import metrics
import torch
from tqdm import tqdm

from pipeline import TorchModule

def evaluate_score(dataloader,model,device):
    with torch.no_grad():
        all_preds,all_labels = [],[]
        for batch in tqdm(dataloader):
            ids = batch['input_ids'].to(device,dtype=torch.long)
            mask = batch['attention_mask'].to(device,dtype=torch.long)
            labels = batch['labels'].to(device,dtype=torch.float)
            outputs = model(input_ids=ids,attention_mask=mask,labels=labels,return_dict=True)
            preds = outputs['probs'] > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        acc = metrics.accuracy_score(np.concatenate(all_labels),np.concatenate(all_preds))
    return acc

class EvaluateScore(TorchModule):

    _required_params = ['model_name','dataloader','add_true_class','pred_df_name',]

    def prepare(self,container,params):

        self.model = container.get(params['model_name'])
        self.loader = container.get(params['dataloader'])
    
    def fit(self,container,params):

        score = evaluate_score(self.loader,self.model,self.device)
        print(score)

    def wrapup(self,container,params):
        container.save()
