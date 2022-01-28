from pipeline import BasePipeline
import baseline
from utils import read_attr_conf

fold = 0
name = '220125_baseline+cvfold{:d}_longformer-large'.format(fold)

conf = dict(
    base_dir='storage/output/'+name+'/',

    slurm=dict(
        fname='slurm.cfg',
        name=name,
        memory='32gb',
        email='kin.ho.lo@cern.ch',
        time='10:00:00',
        gpu='a100',
        ncore='1',
        ntasks='1',
        commands='python3 '+name+'.py',
    ),

    NERPreprocess=dict(
        discourse_df_csv_path='storage/train_folds.csv',
        fold=0,
        bert_model="allenai/longformer-large-4096",
        args=dict(
            input='storage/',
            ),
        num_jobs=12,
        max_len=1536,
        train_bs=4,
        val_bs=128,
    ),

    NERLoadModel=dict(
        type='CustomModel',
        custom_model='NERModel',
        args=dict(
            bert_model="allenai/longformer-large-4096",
            num_labels=15,
            freeze_bert=False,
            dropouts=[0.1,0.2,0.3,0.4,0.5],
            ),
        bert_model="allenai/longformer-large-4096",
        model_name='model',
        saved_model='storage/output/220125_baseline+cvfold0_longformer-large/NERTrain/allenai-longformer-large-4096_ep9.pt',
        ),

    NERTrain=dict(
        model_name='model',
        optimizer_type='AdamW',
        lr=1e-5,
        wd=0.01,
        #optimizer_type='Adam',
        #lr=[2.5e-5, 2.5e-5, 2.5e-5, 2.5e-5, 2.5e-5],
        scheduler_type='cosine_schedule_with_warmup',
        warmup_frac=0.1,
        epochs=10,
        print_every=200,
        max_grad_norm=10.,
        seed=42,
        bert_model="allenai/longformer-large-4096",
        ),

    NERInfer=dict(
        model_name='model',
        dataloader='val_loader',
        add_true_class=True,
        pred_df_name='pred_df',
        ),

    NEREvaluateScore=dict(
        discourse_df_name='storage/train.csv',
        submission_df_name='submission_df',
        ),
)

pipelines = [
        ('NERPreprocess',baseline.Preprocess()),
        ('NERLoadModel',baseline.LoadModel()),
        #('NERTrain',baseline.Train()),
        ('NERInfer',baseline.Infer()),
        ('NEREvaluateScore',baseline.EvaluateScore()),
        ]

if __name__ == "__main__":
    
    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
