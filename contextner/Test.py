import torch
from tqdm import tqdm

from pipeline import TorchModule
from utils import set_seed

class TestModel(TorchModule):
    
    def prepare(self,container,params):
 
        self.model = container.get(params['model_name'])
        self.model.to(self.device)

    def fit(self,container,params):
        for idx, batch in enumerate(tqdm(container.train_loader)):
            ids = batch['input_ids'].to(self.device, dtype = torch.long)
            mask = batch['attention_mask'].to(self.device, dtype = torch.long)
            context_ids = batch['context_input_ids'].to(self.device, dtype = torch.long)
            context_mask = batch['context_attention_mask'].to(self.device, dtype = torch.long)

            labels = batch['labels'].to(self.device, dtype = torch.long)
            self.model(ids,mask,context_ids,context_mask)


    def wrapup(self,container,params):
        container.save()
