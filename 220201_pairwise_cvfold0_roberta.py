from pipeline import BasePipeline
import pairwise_sent_pair as mod 
from utils import read_attr_conf

fold = 0
name = '220201_pairwise_cvfold{:d}_roberta'.format(fold)

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

    PrepareData=dict(
        discourse_df_csv_path='storage/train_folds.csv',
        train_samples_path='storage/output/220201_pairwise_preprocess_mskfold_roberta/PreprocessKFold/train_samples_fold{:d}.p'.format(fold),
        valid_samples_path='storage/output/220201_pairwise_preprocess_mskfold_roberta/PreprocessKFold/valid_samples_fold{:d}.p'.format(fold),
        bert_model="roberta-base",
        num_jobs=1,
        max_len=512,
        train_bs=8,
        val_bs=128,
        seed=42,
    ),

    LoadModel=dict(
        type='CustomModel',
        custom_model='PairwiseModel',
        args=dict(
            bert_model="roberta-base",
            freeze_bert=False,
            dropouts=[0.1,0.2,0.3,0.4,0.5],
            ),
        bert_model="roberta-base",
        model_name='model',
        seed=42,
        ),

    Train=dict(
        model_name='model',
        optimizer_type='AdamW',
        lr=1e-5,
        wd=0.01,
        scheduler_type='cosine_schedule_with_warmup',
        warmup_frac=0.1,
        epochs=10,
        print_every=200,
        max_grad_norm=None,
        seed=42,
        bert_model="roberta-base",
        fp16=True,
        ),

)

pipelines = [
        ('PrepareData',mod.PrepareData()),
        ('LoadModel',mod.LoadModel()),
        ('Train',mod.Train()),
        #('Infer',mod.Infer()),
        #('PredictionString',mod.PredictionString()),
        #('EvaluateScore',mod.EvaluateScore()),
        #('EvaluateScoreVsThreshold',mod.EvaluateScoreVsThreshold()),
        ]

if __name__ == "__main__":
    
    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
