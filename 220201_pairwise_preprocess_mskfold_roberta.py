from pipeline import BasePipeline
import pairwise_sent_pair as mod
from utils import read_attr_conf

name = '220201_pairwise_preprocess_mskfold_roberta'

conf = dict(
    base_dir='storage/output/'+name+'/',

    slurm=dict(
        fname='slurm.cfg',
        name=name,
        memory='32gb',
        email='kin.ho.lo@cern.ch',
        time='04:00:00',
        gpu='a100',
        ncore='1',
        ntasks='1',
        commands='python3 '+name+'py',
    ),

    PreprocessKFold=dict(
        discourse_df_csv_path='storage/train_folds.csv',
        fold=0,
        bert_model="roberta-base",
        args=dict(
            input='storage/',
            ),
        num_jobs=1,
        max_len=512,
        train_bs=4,
        val_bs=128,
    ),
)

pipelines = [
        ('PreprocessKFold',mod.PreprocessKFold()),
        ]

if __name__ == "__main__":
    
    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
