from pipeline import BasePipeline
import stance as mod
from utils import read_attr_conf

fold = 0
name = '220207_stance_cvfold{:d}_longformer'.format(fold)

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
        commands='python3 '+name+'.py',
    ),

    NERPrepareData=dict(
        discourse_df_csv_path='storage/train_folds.csv',
        train_samples_path='storage/output/220126_baseline_preprocess_bi_mskfold/NERPreprocessKFold/train_samples_fold{:d}.p'.format(fold),
        valid_samples_path='storage/output/220126_baseline_preprocess_bi_mskfold/NERPreprocessKFold/valid_samples_fold{:d}.p'.format(fold),
        bert_model="allenai/longformer-base-4096",
        num_jobs=1,
        max_len=1536,
        train_bs=4,
        val_bs=32,
        seed=42,
    ),

    NERLoadModel=dict(
        type='CustomModel',
        custom_model='NERModel',
        args=dict(
            bert_model="allenai/longformer-base-4096",
            num_labels=3,
            freeze_bert=False,
            dropouts=[0.1,0.2,0.3,0.4,0.5],
            ),
        bert_model="allenai/longformer-base-4096",
        model_name='model',
        seed=42,
        ),

    NERTrain=dict(
        model_name='model',
        optimizer_type='AdamW',
        lr=2.5e-5,
        wd=0.01,
        scheduler_type='cosine_schedule_with_warmup',
        warmup_frac=0.1,
        epochs=10,
        print_every=200,
        eval_every=1500,
        max_grad_norm=None,
        seed=42,
        bert_model="allenai/longformer-base-4096",
        fp16=True,
        ),
)

pipelines = [
        ('NERPrepareData',mod.PrepareData()),
        ('NERLoadModel',mod.LoadModel()),
        ('NERTrain',mod.Train()),
        ]

if __name__ == "__main__":
    
    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
