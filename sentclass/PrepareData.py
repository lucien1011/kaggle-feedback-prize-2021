import copy
from torch.utils.data import DataLoader,Dataset 
from tqdm import tqdm

from dataset import SentenceDataset
from pipeline import Module
from utils import set_seed

class PrepareData(Module):

    @staticmethod
    def train_test_split(
            df,
            split_args=dict(test_size=0.20,n_splits=2,random_state=7),
        ):
        from sklearn.model_selection import GroupShuffleSplit
        train_inds, test_inds = next(GroupShuffleSplit(**split_args).split(df, groups=df.index))
        train = df.iloc[train_inds]
        test = df.iloc[test_inds]
        return train,test
    
    def prepare(self,container,params):
         
        container.read_item_from_dir('sent_df','df_csv',args=dict(index_col=0),mod_name=params['input_mod'])
        container.read_item_from_dir('text_df','df_csv',args=dict(index_col=0),mod_name=params['input_mod'])
        train_df,val_df = self.train_test_split(container.sent_df)

        print("Reading training data...")
        train_set = SentenceDataset(train_df,container.text_df,params['maxlen'],bert_model=params['bert_model'],num_classes=params['num_class'],one_hot_label=False)
        train_loader = DataLoader(train_set, batch_size=params['train_textbs'],collate_fn=SentenceDataset.collate_fn)
        container.add_item('train_set',train_set,'torch_dataset',mode='read')
        container.add_item('train_loader',train_loader,'torch_dataloader',mode='read')
        
        print("Reading validation data...")
        val_set = SentenceDataset(val_df,container.text_df,params['maxlen'],bert_model=params['bert_model'],num_classes=params['num_class'],one_hot_label=False)
        val_loader = DataLoader(val_set, batch_size=params['val_textbs'],collate_fn=SentenceDataset.collate_fn)
        container.add_item('val_set',val_set,'torch_dataset',mode='read')
        container.add_item('val_loader',val_loader,'torch_dataloader',mode='read')

