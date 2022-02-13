from pipeline import BasePipeline
import baseline
from utils import read_attr_conf

name = '220205_baseline+aug_preprocess_bi_mskfold'

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

    NERPreprocessKFold=dict(
        discourse_df_csv_path='storage/output/220205_aug_discourse_df/train.csv',
        fold=0,
        bert_model="allenai/longformer-base-4096",
        args=dict(
            input='storage/output/220205_aug_discourse_df/',
            ),
        num_jobs=1,
        max_len=1536,
        train_bs=4,
        val_bs=128,
    ),
)

pipelines = [
        ('NERPreprocessKFold',baseline.PreprocessKFold()),
        ]

if __name__ == "__main__":
    
    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
