from pipeline import BasePipeline
import contextner
from utils import read_attr_conf

conf = dict(
    base_dir='storage/output/220115_contextner_longformer-base/',

    slurm=dict(
        fname='slurm.cfg',
        name='220115_contextner_longformer-base',
        memory='32gb',
        email='kin.ho.lo@cern.ch',
        time='04:00:00',
        gpu='a100',
        ncore='1',
        ntasks='1',
        commands='python3 220115_contextner_longformer-base.py',
    ),

    Preprocess=dict(
        discourse_df_csv_path='storage/train.csv',
        text_df_csv_path='storage/text.csv',
        ),

    TrainTestSplit = dict(
        input_mod='Preprocess/',
        split_args=dict(n_splits=5),
        seed=42,
    ),

    PrepareData=dict(
        discourse_df_path='storage/train.csv',
        input_mod='TrainTestSplit/',
        train_ner_df='fold0_train_df',
        test_ner_df='fold0_test_df',
        tokenizer_instant_args=dict(
            add_prefix_space=True,
        ),
        tokenizer_args=dict(
            is_split_into_words=True,
        ),
        maxlen=1024,
        bert_model="allenai/longformer-base-4096",
        train_bs=4,
        val_bs=32,
        labels=['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement'],
        context_labels=['B-Position','I-Position','B-Concluding Statement', 'I-Concluding Statement',],
        ),

    LoadModel=dict(
        seed=42,
        type='AutoModelForTokenClassification',
        bert_model="allenai/longformer-base-4096",
        config_args = dict(num_labels=15),
        model_name='model',
        ),

    TestModel=dict(
        model_name='model',
        ),

    Train=dict(
        seed=42,
        model_name='model',
        bert_model="allenai/longformer-base-4096",
        lr= [2.5e-5, 2.5e-5, 2.5e-5, 2.5e-5, 2.5e-5],
        epochs=5,
        print_every=200,
        max_grad_norm=10.,
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
        #('Preprocess',contextner.Preprocess()),
        #('TrainTestSplit',contextner.TrainTestSplit()),
        ('PrepareData',contextner.PrepareData()),
        ('LoadModel',contextner.LoadModel()),
        #('TestModel',contextner.TestModel()),
        ('Train',contextner.Train()),
        #('Infer',contextner.Infer()),
        #('PredictionString',contextner.PredictionString()),
        #('EvaluateScore',contextner.EvaluateScore()),
        ]

if __name__ == "__main__":

    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
