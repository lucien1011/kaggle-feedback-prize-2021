from pipeline import BasePipeline
import atthead
from utils import read_attr_conf

conf = dict(
    base_dir='storage/output/220115_atthead_longformer/',

    slurm=dict(
        fname='slurm.cfg',
        pyscript='run_qa.py',
        name='220110_atthead_longformer',
        memory='32gb',
        email='kin.ho.lo@cern.ch',
        time='04:00:00',
        gpu='a100',
        ncore='1',
        ntasks='1',
        commands='python3 220115_atthead_longformer.py',
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
        bert_model="allenai/longformer-base-4096",
        train_bs=4,
        val_bs=128,
        labels=['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement'],
        ),

    NERLoadModel=dict(
        type='CustomModel',
        custom_model='AttentionHeadModel',
        bert_model="allenai/longformer-base-4096",
        args = dict(
            bert_model="allenai/longformer-base-4096",
            saved_model='storage/output/220110_ner_longformer/NERTrain/allenai-longformer-base-4096_valscore0.59635_ep2.pt',
            num_labels=15,
            freeze_bert=False,
            ),
        model_name='model',
        ),

    NERTrain=dict(
        model_name='model',
        lr= [2.5e-6, 2.5e-6, 2.5e-6, 2.5e-6, 2.5e-6],
        epochs=5,
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
        pred_df_name='pred_df',
        submission_df_name='submission_df',
        ),

    NEREvaluateScore=dict(
        discourse_df_name='discourse_df',
        submission_df_name='submission_df',
        ),
)

pipelines = [
        ('NERPrepareData',atthead.PrepareData()),
        ('NERLoadModel',atthead.LoadModel()),
        ('NERTrain',atthead.Train()),
        ('NERInfer',atthead.Infer()),
        #('NERPredictionString',atthead.PredictionString()),
        #('NEREvaluateScore',atthead.EvaluateScore()),
        ]

if __name__ == "__main__":

    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
