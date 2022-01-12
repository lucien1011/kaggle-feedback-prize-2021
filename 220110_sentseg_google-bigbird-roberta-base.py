from pipeline import BasePipeline
import sentseg
import qa
from utils import read_attr_conf

conf = dict(
    base_dir='storage/output/220110_sentseg_google-bigbird-roberta-base/',

    slurm=dict(
        fname='slurm.cfg',
        pyscript='run_qa.py',
        name='220110_sentseg_google-bigbird-roberta-base',
        memory='32gb',
        email='kin.ho.lo@cern.ch',
        time='04:00:00',
        gpu='a100',
        ncore='1',
        ntasks='1',
        commands='python3 220110_sentseg_google-bigbird-roberta-base.py',
    ),

    Preprocess=dict(
        discourse_df_csv_path='storage/train.csv',
        text_df_csv_path='storage/text.csv',
    ),

    TrainTestSplit=dict(
        seed=42,
        input_mod='Preprocess',
        split_args=dict(n_splits=5),
    ),
 
    PrepareData=dict(
        discourse_df_path='storage/train.csv',
        input_mod='TrainTestSplit/',
        train_sent_df='fold0_train_df',
        test_sent_df='fold0_test_df',
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

    LoadModel=dict(
        type='AutoModelForTokenClassification',
        bert_model="google/bigbird-roberta-base",
        config_args = dict(num_labels=15),
        model_name='model',
        ),

    Train=dict(
        model_name='model',
        lr= [1e-5, 1e-5, 1e-6, 1e-6, 1e-7],
        epochs=5,
        print_every=200,
        max_grad_norm=10.,
        seed=42,
        bert_model="google/bigbird-roberta-base",
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
        #('Preprocess',sentseg.Preprocess())
        ('TrainTestSplit',sentseg.TrainTestSplit()),
        #('PrepareData',sentseg.PrepareData()),
        #('LoadModel',sentseg.LoadModel()),
        #('Train',sentseg.Train()),
        #('Infer',sentseg.Infer()),
        #('PredictionString',sentseg.PredictionString()),
        #('EvaluateScore',sentseg.EvaluateScore()),
        ]

if __name__ == "__main__":

    pp = BasePipeline(pipelines,base_dir=conf['base_dir'])
    pp.run(conf)
