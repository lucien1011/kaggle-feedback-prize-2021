from pipeline import BasePipeline
import ner
import qa
from utils import read_attr_conf

name = '220125_ner_preprocess_bi_mskfold'

conf = dict(
    base_dir='storage/output/{:s}/'.format(name),

    slurm=dict(
        fname='slurm.cfg',
        name=name,
        memory='32gb',
        email='kin.ho.lo@cern.ch',
        time='04:00:00',
        gpu='a100',
        ncore='1',
        ntasks='1',
        commands='python3 {:s}.py'.format(name),
    ),

    NERPreprocess=dict(
        discourse_df_csv_path='storage/train.csv',
        text_df_csv_path='storage/text.csv',
        ner_scheme='bi',
        ),
    
    NERTrainTestSplit=dict(
        discourse_df_path='storage/train.csv',
        input_mod='NERPreprocess/',
        split_type='MultilabelStratifiedKFold',
        split_args=dict(n_splits=5,shuffle=True,random_state=42),
        seed=42,
        ),

)

pipelines = [
        ('NERPreprocess',ner.Preprocess()),
        ('NERTrainTestSplit',ner.TrainTestSplit()),
        ]

if __name__ == "__main__":

    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
