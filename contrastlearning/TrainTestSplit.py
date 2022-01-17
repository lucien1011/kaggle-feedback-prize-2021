from sklearn.model_selection import GroupKFold

from pipeline import Module
from utils import set_seed

class TrainTestSplit(Module):
    
    def prepare(self,container,params):
        
        if 'seed' in params: set_seed(params['seed'])
         
        container.read_item_from_dir('df','df_csv',args=dict(),mod_name=params['input_mod'])

        group_kfold = GroupKFold(**params['split_args'])
        for i,(train_inds, test_inds) in enumerate(group_kfold.split(container.df, groups=container.df['id'])):
            train_df = container.df.iloc[train_inds].reset_index()
            test_df = container.df.iloc[test_inds].reset_index()
            container.add_item('fold{:d}_train_df'.format(i),train_df,'df_csv',mode='write')
            container.add_item('fold{:d}_test_df'.format(i),test_df,'df_csv',mode='write')

    def wrapup(self,container,params):
        container.save()
