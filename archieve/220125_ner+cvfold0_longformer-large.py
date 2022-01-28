from pipeline import BasePipeline
import ner
from utils import read_attr_conf

fold = 0
name = '220125_ner+cvfold{:d}_longformer-large'.format(fold)
preprocess_dir = 'storage/output/220125_ner_preprocess_bi_mskfold/NERTrainTestSplit/'

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

    NERPrepareData=dict(
        discourse_df_path='storage/train.csv',
        train_ner_df=preprocess_dir+'fold{:d}_train_df.csv'.format(fold),
        test_ner_df=preprocess_dir+'fold{:d}_test_df.csv'.format(fold),
        tokenizer_instant_args=dict(
            add_prefix_space=True,
        ),
        tokenizer_args=dict(
            is_split_into_words=True,
            padding='max_length', 
            truncation=True, 
        ),
        maxlen=1536,
        bert_model="allenai/longformer-large-4096",
        train_bs=4,
        val_bs=128,
        labels=['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement'],
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

    NERPredictionString=dict(
        pred_df_name='storage/output/220125_ner+cvfold0_longformer/NERInfer/pred_df.csv',
        submission_df_name='submission_df',
        get_predstr_df_args=dict(
            ner_type='bi',
            get_string_args=dict(
                min_words_thresh={
                    "Lead": 5,
                    "Position": 2,
                    "Evidence": 20,
                    "Claim": 2,
                    "Concluding Statement": 11,
                    "Counterclaim": 6,
                    "Rebuttal": 2,
                },
                min_probs_thresh={
                    "Lead": 0.2,
                    "Position": 0.2,
                    "Evidence": 0.2,
                    "Claim": 0.2,
                    "Concluding Statement": 0.2,
                    "Counterclaim": 0.2,
                    "Rebuttal": 0.2,
                    },
                ),
            ),
        ),

    NEREvaluateScore=dict(
        discourse_df_name='storage/train.csv',
        submission_df_name='submission_df',
        ),

    NEREvaluateScoreVsThreshold=dict(
        discourse_df_name='storage/train.csv',
        pred_df_name='storage/output/220125_ner+cvfold0_longformer/NERInfer/pred_df.csv',
        classes=['Lead','Position','Evidence','Claim','Concluding Statement','Counterclaim','Rebuttal',],
        prob_thresholds=[0.1*i for i in range(1,10)],
        len_thresholds=list(range(2,20)),
        get_predstr_df_args=dict(
            ner_type='bi',
            get_string_args=dict(
                min_words_thresh={
                    "Lead": 7,
                    "Position": 7,
                    "Evidence": 7,
                    "Claim": 7,
                    "Concluding Statement": 7,
                    "Counterclaim": 7,
                    "Rebuttal": 7,
                },
                min_probs_thresh={
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
        ),
)

pipelines = [
        ('NERPrepareData',ner.PrepareData()),
        ('NERLoadModel',ner.LoadModel()),
        ('NERTrain',ner.Train()),
        ('NERInfer',ner.Infer()),
        #('NERPredictionString',ner.PredictionString()),
        #('NEREvaluateScore',ner.EvaluateScore()),
        #('NEREvaluateScoreVsThreshold',ner.EvaluateScoreVsThreshold()),
        ]

if __name__ == "__main__":
    
    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
