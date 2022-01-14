from pipeline import BasePipeline
import contextner
from utils import read_attr_conf

conf = dict(
    base_dir='storage/output/220113_contextner+mlphead+freezebert_bigbird-base/',

    slurm=dict(
        fname='slurm.cfg',
        name='220113_contextner+mlphead+freezebert_bigbird-base',
        memory='32gb',
        email='kin.ho.lo@cern.ch',
        time='04:00:00',
        gpu='a100',
        ncore='1',
        ntasks='1',
        commands='python3 220113_contextner+mlphead+freezebert_bigbird-base.py',
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
        bert_model="google/bigbird-roberta-base",
        train_bs=4,
        val_bs=32,
        labels=['O','B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence',],
        ),

    LoadModel=dict(
        seed=42,
        type='CustomModel',
        bert_model="google/bigbird-roberta-base",
        custom_model='ContextModelForTokenClassification',
        args = dict(
            num_labels=9,
            bert_model="google/bigbird-roberta-base",
            freeze_bert=True,
            ),
        model_name='model',
        ),

    TestModel=dict(
        model_name='model',
        ),

    Train=dict(
        seed=42,
        model_name='model',
        bert_model="bigbird-roberta-base",
        lr= [2.5e-6, 2.5e-6, 2.5e-6, 2.5e-6, 2.5e-6],
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
