from pipeline import BasePipeline
import ner
import qa
from utils import read_attr_conf

conf = dict(
    base_dir='storage/output/220112_ner_google-bigbird-roberta-base-atthead/',

    slurm=dict(
        fname='slurm.cfg',
        pyscript='run_ner.py',
        name='220112_ner_google-bigbird-roberta-base-atthead',
        memory='32gb',
        email='kin.ho.lo@cern.ch',
        time='04:00:00',
        gpu='a100',
        ncore='1',
        ntasks='1',
        commands='python3 220112_ner_google-bigbird-roberta-base-atthead.py',
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
        maxlen=1024,
        bert_model="google/bigbird-roberta-base",
        train_bs=4,
        val_bs=128,
        labels=['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement'],
        ),

    NERLoadModel=dict(
        type='CustomModel',
        bert_model="google/bigbird-roberta-base",
        config_args = dict(num_labels=15),
        model_name='model',
        custom_model='BigBirdAttHeadForTokenClassification',
        ),

    NERTrain=dict(
        model_name='model',
        lr= [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
        epochs=5,
        print_every=200,
        max_grad_norm=10.,
        seed=42,
        bert_model="google/bigbird-roberta-base",
        ),

    NERInfer=dict(
        model_name='model',
        dataloader='val_loader',
        add_true_class=True,
        pred_df_name='pred_df',
        ),

    NERPredictionString=dict(
        pred_df_name='pred_df',
        submission_df_name='submission_df',
        ),

    NEREvaluateScore=dict(
        discourse_df_name='discourse_df',
        submission_df_name='submission_df',
        ),
)

pipelines = [
        ('NERPrepareData',ner.PrepareData()),
        ('NERLoadModel',ner.LoadModel()),
        ('NERTrain',ner.Train()),
        ('NERInfer',ner.Infer()),
        #('NERPredictionString',ner.PredictionString()),
        #('NEREvaluateScore',ner.EvaluateScore()),
        ]

if __name__ == "__main__":

    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
