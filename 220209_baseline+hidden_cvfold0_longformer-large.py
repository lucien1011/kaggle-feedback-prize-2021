from pipeline import BasePipeline
import baseline
from utils import read_attr_conf

fold = 0
name = '220209_baseline+hidden_cvfold{:d}_longformer-large'.format(fold)

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
        bert_model="allenai/longformer-large-4096",
        num_jobs=1,
        max_len=1536,
        train_bs=4,
        val_bs=4,
        seed=42,
    ),

    NERLoadModel=dict(
        type='CustomModel',
        custom_model='NERModel',
        args=dict(
            bert_model="allenai/longformer-large-4096",
            num_labels=15,
            freeze_bert=False,
            dropouts=[0.1,0.2,0.3,0.4,0.5],
            all_hidden_state=True,
            ),
        bert_model="allenai/longformer-large-4096",
        model_name='model',
        seed=42,
        ),

    NERTrain=dict(
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
        bert_model="allenai/longformer-large-4096",
        fp16=True,
        ),

    NERInfer=dict(
        model_name='model',
        dataloader='val_loader',
        add_true_class=True,
        pred_df_name='pred_df',
        ),

    NERPredictionString=dict(
        dataloader='val_loader',
        probs_name='probs',
        proba_thresh = {
            "Lead": 0.7,
            "Position": 0.55,
            "Evidence": 0.65,
            "Claim": 0.55,
            "Concluding Statement": 0.7,
            "Counterclaim": 0.5,
            "Rebuttal": 0.55,
        },
        min_thresh = {
            "Lead": 9,
            "Position": 5,
            "Evidence": 14,
            "Claim": 3,
            "Concluding Statement": 11,
            "Counterclaim": 6,
            "Rebuttal": 4,
        },
        submission_df_name='submission_df',
        ),

    NEREvaluateScore=dict(
        discourse_df_name='storage/train.csv',
        submission_df_name='submission_df',
        ),

    NEREvaluateScoreVsThreshold=dict(
        discourse_df_name='storage/train.csv',
        dataloader='val_loader',
        probs_name='probs',
        classes=['Lead','Position','Evidence','Claim','Concluding Statement','Counterclaim','Rebuttal',],
        prob_thresholds=[0.1*i for i in range(1,10)],
        len_thresholds=list(range(2,20)),
        get_predstr_df_args=dict(
            min_thresh={
                "Lead": 7,
                "Position": 7,
                "Evidence": 7,
                "Claim": 7,
                "Concluding Statement": 7,
                "Counterclaim": 7,
                "Rebuttal": 7,
            },
            proba_thresh={
                "Lead": 0.5,
                "Position": 0.5,
                "Evidence": 0.5,
                "Claim": 0.5,
                "Concluding Statement": 0.5,
                "Counterclaim": 0.5,
                "Rebuttal": 0.5,
                },
            ),
        ),
)

pipelines = [
        ('NERPrepareData',baseline.PrepareData()),
        ('NERLoadModel',baseline.LoadModel()),
        ('NERTrain',baseline.Train()),
        ('NERInfer',baseline.Infer()),
        ('NERPredictionString',baseline.PredictionString()),
        ('NEREvaluateScore',baseline.EvaluateScore()),
        #('NEREvaluateScoreVsThreshold',baseline.EvaluateScoreVsThreshold()),
        ]

if __name__ == "__main__":
    
    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
