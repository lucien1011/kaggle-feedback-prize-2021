from pipeline import BasePipeline
import contrastlearning as m
from utils import read_attr_conf

conf = dict(
    base_dir='storage/output/220115_contrastlearning_longformer/',

    slurm=dict(
        fname='slurm.cfg',
        pyscript='run_qa.py',
        name='220115_contrastlearning_longformer',
        memory='32gb',
        email='kin.ho.lo@cern.ch',
        time='04:00:00',
        gpu='a100',
        ncore='1',
        ntasks='1',
        commands='python3 220115_contrastlearning_longformer.py',
    ),

    Preprocess=dict(
        discourse_df_csv_path='storage/train.csv',
        ),

    TrainTestSplit=dict(
        seed=42,
        input_mod='Preprocess',
        split_args=dict(n_splits=5),
        ),
 
    PrepareData=dict(
        discourse_df_path='storage/train.csv',
        input_mod='TrainTestSplit/',
        train_df='fold0_train_df',
        test_df='fold0_test_df',
        tokenizer_instant_args=dict(
            add_prefix_space=True,
        ),
        tokenizer_args=dict(
            is_split_into_words=True,
            padding='max_length', 
            truncation=True, 
        ),
        maxlen=128,
        bert_model="allenai/longformer-base-4096",
        train_bs=4,
        val_bs=128,
        ),

    LoadModel=dict(
        type='CustomModel',
        custom_model='ContrastLearningModel',
        args=dict(
            bert_model='allenai/longformer-base-4096',
            ),
        bert_model="allenai/longformer-base-4096",
        model_name='model',
        ),

    Train=dict(
        model_name='model',
        lr= [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
        epochs=3,
        print_every=200,
        max_grad_norm=10.,
        seed=42,
        bert_model="allenai/longformer-base-4096",
        ),

    Infer=dict(
        model_name='model',
        dataloader='val_loader',
        add_true_class=True,
        pred_df_name='pred_df',
        ),

    PredictionString=dict(
        pred_df_name='pred_df',
        submission_df_name='submission_df',
        ),

    EvaluateScore=dict(
        discourse_df_name='discourse_df',
        submission_df_name='submission_df',
        ),
)

pipelines = [
        #('Preprocess',m.Preprocess()),
        #('TrainTestSplit',m.TrainTestSplit()),
        ('PrepareData',m.PrepareData()),
        ('LoadModel',m.LoadModel()),
        ('Train',m.Train()),
        #('Infer',m.Infer()),
        #('PredictionString',m.PredictionString()),
        #('EvaluateScore',m.EvaluateScore()),
        ]

if __name__ == "__main__":

    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
