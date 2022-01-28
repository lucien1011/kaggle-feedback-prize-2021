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
            self.model.eval()
            probs = self.model.predict(self.loader,device=self.device)
        container.add_item('probs',probs,'np_arr',mode='read')

    def wrapup(self,container,params):
        container.save()
