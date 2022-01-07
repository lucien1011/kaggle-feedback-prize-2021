import copy
from torch.utils.data import DataLoader,Dataset 
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import NERDataset
from pipeline import Module
from utils import set_seed

class PrepareData(Module):
    
    def prepare(self,container,params):
        
        if 'seed' in params: set_seed(params['seed'])
         
        container.read_item_from_path('discourse_df',params['discourse_df_path'],'df_csv')
        container.read_item_from_dir(params['train_qa_df'],'df_csv',args=dict(index_col=0),mod_name=params['input_mod'],newkey='train_qa_df')
        container.read_item_from_dir(params['test_qa_df'],'df_csv',args=dict(index_col=0),mod_name=params['input_mod'],newkey='test_qa_df')
        
        container.train_qa_df['ents'] = container.train_qa_df['ents'].apply(lambda x: eval(x))
        container.test_qa_df['ents'] = container.test_qa_df['ents'].apply(lambda x: eval(x))

        tokenizer = AutoTokenizer.from_pretrained(params['bert_model'],**params['tokenizer_instant_args'])
        container.add_item('tokenizer',tokenizer,'huggingface_tokenizer','read')
        labels_to_ids = {v:k for k,v in enumerate(params['labels'])}
        ids_to_labels = {k:v for k,v in enumerate(params['labels'])}
        container.add_item('labels_to_ids',labels_to_ids,'pickle','read')
        container.add_item('ids_to_labels',ids_to_labels,'pickle','read')

        print("Reading training data...")
        train_set = NERDataset(container.train_qa_df,tokenizer,params['maxlen'],True,labels_to_ids,params['tokenizer_args'])
        train_loader = DataLoader(train_set, batch_size=params['train_bs'])
        container.add_item('train_set',train_set,'torch_dataset',mode='read')
        container.add_item('train_loader',train_loader,'torch_dataloader',mode='read')
        
        print("Reading validation data...")
        val_set = NERDataset(container.test_qa_df,tokenizer,params['maxlen'],True,labels_to_ids,params['tokenizer_args'])
        val_loader = DataLoader(val_set, batch_size=params['val_bs'])
        container.add_item('val_set',val_set,'torch_dataset',mode='read')
        container.add_item('val_loader',val_loader,'torch_dataloader',mode='read')

