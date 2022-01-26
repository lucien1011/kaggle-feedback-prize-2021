from pipeline import BasePipeline
import slidewindow
from utils import read_attr_conf

conf = dict(
    base_dir='storage/output/220115_slidewindow_bert-uncased-base/',

    slurm=dict(
        fname='slurm.cfg',
        pyscript='run_qa.py',
        name='220115_slidewindow_bert-uncased-base',
        memory='32gb',
        email='kin.ho.lo@cern.ch',
        time='04:00:00',
        gpu='a100',
        ncore='1',
        ntasks='1',
        commands='python3 220115_slidewindow_bert-uncased-base.py',
    ),

    Preprocess=dict(
        discourse_df_csv_path='storage/train.csv',
        text_df_csv_path='storage/text.csv',
        ),

    TrainTestSplit=dict(
        discourse_df_path='storage/train.csv',
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
            padding='max_length', 
            truncation=True, 
        ),
        maxlen=512,
        bert_model="bert-uncased-base",
        train_bs=2,
        val_bs=8,
        labels=['O', 'Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal', 'Evidence', 'Concluding Statement'],
        window_size=20,
        ),

    LoadModel=dict(
        type='CustomModel',
        custom_model='SlideWindowModel',
        bert_model="google/bigbird-roberta-base",
        args = dict(
            bert_model="google/bigbird-roberta-base",
            num_labels=8,
            freeze_bert=False,
            ),
        model_name='model',
        ),

    Train=dict(
        model_name='model',
        lr= [2.5e-5, 2.5e-5, 2.5e-5, 2.5e-5, 2.5e-6],
        epochs=5,
        print_every=100,
        max_grad_norm=10.,
        seed=42,
        bert_model="google/bigbird-roberta-base",
        sub_bs=32,
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
        #('Preprocess',slidewindow.Preprocess()),
        #('TrainTestSplit',slidewindow.TrainTestSplit()),
        ('PrepareData',slidewindow.PrepareData()),
        ('LoadModel',slidewindow.LoadModel()),
        ('Train',slidewindow.Train()),
        #('Infer',slidewindow.Infer()),
        #('PredictionString',ner.PredictionString()),
        #('EvaluateScore',ner.EvaluateScore()),
        ]

if __name__ == "__main__":

    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
