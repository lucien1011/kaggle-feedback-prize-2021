from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
from sklearn.model_selection import GroupKFold

from pipeline import Module
from utils import set_seed

class TrainTestSplit(Module):
    
    _required_params = ['seed','discourse_df_path','input_mod','split_type']
    
    def prepare(self,container,params):
        
        if 'seed' in params: set_seed(params['seed'])
         
        container.read_item_from_path('discourse_df',params['discourse_df_path'],'df_csv')
        container.read_item_from_dir('ner_df','df_csv',args=dict(index_col=0),mod_name=params['input_mod'])

        container.ner_df['ents'] = container.ner_df['ents'].apply(lambda x: eval(x))

        if params['split_type'] == 'GroupKFold':
            self.GroupKFold(container.ner_df,container.ner_df['id'],params['split_args'],container)
        elif params['split_type'] == 'MultilabelStratifiedKFold':
            self.MultilabelStratifiedKFold(container.ner_df,container.discourse_df,params['split_args'],container)
        else:
            raise NotImplementedError
    
    def wrapup(self,container,params):
        container.save()

    @staticmethod
    def GroupKFold(df,groups,split_args,container):
        group_kfold = GroupKFold(**split_args)
        for i,(train_inds, test_inds) in enumerate(group_kfold.split(df,groups=groups)):
            train_df = container.ner_df.iloc[train_inds].reset_index()
            test_df = container.ner_df.iloc[test_inds].reset_index()
            container.add_item('fold{:d}_train_df'.format(i),train_df,'df_csv',mode='write')
            container.add_item('fold{:d}_test_df'.format(i),test_df,'df_csv',mode='write')

    @staticmethod
    def MultilabelStratifiedKFold(ner_df,discourse_df,split_args,container):
        dfx = pd.get_dummies(discourse_df, columns=["discourse_type"]).groupby(["id"], as_index=False).sum()
        cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
        dfx = dfx[cols]
        
        mskf = MultilabelStratifiedKFold(**split_args)
        labels = [c for c in dfx.columns if c != "id"]
        dfx_labels = dfx[labels]
        dfx["kfold"] = -1
        
        for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
            dfx.loc[val_, "kfold"] = fold
        
        ner_df = ner_df.merge(dfx[["id", "kfold"]], on="id", how="left")

        for i in ner_df.kfold.unique():
            train_df = ner_df[ner_df['kfold']!=i]
            test_df = ner_df[ner_df['kfold']==i]
            container.add_item('fold{:d}_train_df'.format(i),train_df,'df_csv',mode='write')
            container.add_item('fold{:d}_test_df'.format(i),test_df,'df_csv',mode='write')

