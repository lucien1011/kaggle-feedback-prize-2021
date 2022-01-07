from sklearn.model_selection import GroupKFold

from pipeline import Module
from utils import set_seed

class TrainTestSplit(Module):
    
    def prepare(self,container,params):
        
        if 'seed' in params: set_seed(params['seed'])
         
        container.read_item_from_path('discourse_df',params['discourse_df_path'],'df_csv')
        container.read_item_from_dir('qa_df','df_csv',args=dict(index_col=0),mod_name=params['input_mod'])
        container.qa_df['ents'] = container.qa_df['ents'].apply(lambda x: eval(x))

        group_kfold = GroupKFold(**params['split_args'])
        for i,(train_inds, test_inds) in enumerate(group_kfold.split(container.qa_df, groups=container.qa_df['id'])):
            train_df = container.qa_df.iloc[train_inds].reset_index()
            test_df = container.qa_df.iloc[test_inds].reset_index()
            container.add_item('fold{:d}_train_df'.format(i),train_df,'df_csv',mode='write')
            container.add_item('fold{:d}_test_df'.format(i),test_df,'df_csv',mode='write')

    def wrapup(self,container,params):
        container.save()
