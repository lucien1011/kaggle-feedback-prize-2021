import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from pipeline import TorchModule

class Infer(TorchModule):

    _required_params = ['model_name','dataloader','add_true_class','pred_df_name',]

    def prepare(self,container,params):

        self.model = container.get(params['model_name'])
        self.loader = container.get(params['dataloader'])
        self.ids_to_labels = container.id_target_map
    
    def fit(self,container,params):

        with torch.no_grad():
            all_preds,all_labels = [],[]
            for batch in tqdm(self.loader):
                ids = batch['input_ids'].to(self.device,dtype=torch.long)
                mask = batch['attention_mask'].to(self.device,dtype=torch.long)
                labels = batch['labels'].to(self.device,dtype=torch.long)
                outputs = model(input_ids=ids,attention_mask=mask,labels=labels,return_dict=True)
                logits = outputs['logits']
                all_preds.append(outputs['probs'].argmax(dim=-1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            f1 = evaluate_f1_per_batch(logits,labels,mask,model.num_labels)

    def wrapup(self,container,params):
        container.save()
