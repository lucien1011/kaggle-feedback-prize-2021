import copy
from torch.utils.data import DataLoader,Dataset 
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import ContrastLearningDataset
from pipeline import Module
from utils import set_seed

class PrepareData(Module):
    
    def prepare(self,container,params):
        
        if 'seed' in params: set_seed(params['seed'])
         
        container.read_item_from_dir(params['train_df'],'df_csv',args=dict(),mod_name=params['input_mod'],newkey='train_df')
        container.read_item_from_dir(params['test_df'],'df_csv',args=dict(),mod_name=params['input_mod'],newkey='test_df')
        
        tokenizer = AutoTokenizer.from_pretrained(params['bert_model'],**params['tokenizer_instant_args'])
        container.add_item('tokenizer',tokenizer,'huggingface_tokenizer','read')
        
        container.train_df = container.train_df.dropna().reset_index()
        container.test_df = container.test_df.dropna().reset_index()
 
        print("Reading training data...")
        train_set = ContrastLearningDataset(container.train_df,tokenizer,params['maxlen'],True,params['tokenizer_args'])
        train_loader = DataLoader(train_set, batch_size=params['train_bs'])
        container.add_item('train_set',train_set,'torch_dataset',mode='read')
        container.add_item('train_loader',train_loader,'torch_dataloader',mode='read')
        
        print("Reading validation data...")
        val_set = ContrastLearningDataset(container.test_df,tokenizer,params['maxlen'],True,params['tokenizer_args'])
        val_loader = DataLoader(val_set, batch_size=params['val_bs'])
        container.add_item('val_set',val_set,'torch_dataset',mode='read')
        container.add_item('val_loader',val_loader,'torch_dataloader',mode='read')

