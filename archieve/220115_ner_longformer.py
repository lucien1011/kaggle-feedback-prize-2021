from pipeline import BasePipeline
import ner
import qa
from utils import read_attr_conf

conf = dict(
    base_dir='storage/output/220115_ner_longformer/',

    slurm=dict(
        fname='slurm.cfg',
        pyscript='run_qa.py',
        name='220115_ner_longformer-base',
        memory='32gb',
        email='kin.ho.lo@cern.ch',
        time='04:00:00',
        gpu='a100',
        ncore='1',
        ntasks='1',
        commands='python3 220115_ner_longformer.py',
    ),
 
    NERPrepareData=dict(
        discourse_df_path='storage/train.csv',
        input_mod='NERTrainTestSplit/',
        train_ner_df='fold0_train_df',
        test_ner_df='fold0_test_df',
        tokenizer_instant_args=dict(
            add_prefix_space=True,
        ),
        tokenizer_args=dict(
            is_split_into_words=True,
            padding='max_length', 
            truncation=True, 
        ),
        maxlen=2048,
        bert_model="allenai/longformer-base-4096",
        train_bs=4,
        val_bs=128,
        labels=['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement'],
        ),

    NERLoadModel=dict(
        type='AutoModelForTokenClassification',
        bert_model="allenai/longformer-base-4096",
        config_args = dict(num_labels=15),
        model_name='model',
        saved_model='storage/output/220115_ner_longformer/NERTrain/allenai-longformer-base-4096_valscore0.59712_ep0.pt',
        ),

    NERTrain=dict(
        model_name='model',
        #lr= [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
        lr= [2.5e-6, 2.5e-6, 2.5e-6, 2.5e-6, 2.5e-7],
        epochs=3,
        print_every=200,
        max_grad_norm=10.,
        seed=42,
        bert_model="allenai/longformer-base-4096",
        ),

    NERInfer=dict(
        model_name='model',
        dataloader='val_loader',
        add_true_class=True,
        pred_df_name='pred_df',
        ),

    NERPredictionString=dict(
        pred_df_name='storage/output/220115_ner_longformer/NERInfer/pred_df.csv',
        submission_df_name='submission_df',
        get_predstr_df_args=dict(
            ner_type='bi',
            get_string_args=dict(
                min_words_thresh={
                    "Lead": 7,
                    "Position": 2,
                    "Evidence": 17,
                    "Claim": 2,
                    "Concluding Statement": 17,
                    "Counterclaim": 6,
                    "Rebuttal": 2,
                },
                min_probs_thresh={
                    "Lead": 0.6,
                    "Position": 0.4,
                    "Evidence": 0.4,
                    "Claim": 0.4,
                    "Concluding Statement": 0.4,
                    "Counterclaim": 0.4,
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
        pred_df_name='storage/output/220115_ner_longformer/NERInfer/pred_df.csv',
        classes=['Lead','Position','Evidence','Claim','Concluding Statement','Counterclaim','Rebuttal',],
        prob_thresholds=[0.1*i for i in range(0,10)],
        #prob_thresholds=[],
        #len_thresholds=list(range(2,30)),
        len_thresholds=[],
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
        #('NERPrepareData',ner.PrepareData()),
        #('NERLoadModel',ner.LoadModel()),
        #('NERTrain',ner.Train()),
        #('NERInfer',ner.Infer()),
        #('NERPredictionString',ner.PredictionString()),
        #('NEREvaluateScore',ner.EvaluateScore()),
        ('NEREvaluateScoreVsThreshold',ner.EvaluateScoreVsThreshold()),
        ]

if __name__ == "__main__":

    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
