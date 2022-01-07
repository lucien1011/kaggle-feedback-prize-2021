conf = dict(
    base_dir='storage/output/220106_qa_google-bigbird-roberta-base/',
    slurm=dict(
        fname='slurm.cfg',
        pyscript='run_qa.py',
        name='220106_qa_google-bigbird-roberta-base',
        memory='32gb',
        email='kin.ho.lo@cern.ch',
        time='04:00:00',
        gpu='quadro',
        ncore='1',
        ntasks='1',
    ),
    Preprocess = dict(
        discourse_df_csv_path='storage/train.csv',
        text_dir='storage/train/',
        text_df_csv_fname='text_df.csv',
    ),
    TrainTestSplit = dict(
        discourse_df_path='storage/train.csv',
        input_mod='Preprocess/',
        split_args=dict(n_splits=5),
        seed=42,
    ),
    PrepareData = dict(
        discourse_df_path='storage/train.csv',
        input_mod='TrainTestSplit/',
        train_qa_df='fold0_train_df',
        test_qa_df='fold0_test_df',
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
        val_bs=32,
        labels=['O', 'B-Lead', 'E-Lead', 'B-Position', 'E-Position', 'B-Claim', 'E-Claim', 'B-Counterclaim', 'E-Counterclaim', 'B-Rebuttal', 'E-Rebuttal', 'B-Evidence', 'E-Evidence', 'B-Concluding Statement', 'E-Concluding Statement'],
        ),
    Train = dict(
        bert_model="google/bigbird-roberta-base",
        config_args = dict(num_labels=15),
        lr= [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
        epochs=5,
        print_every=200,
        max_grad_norm=10.,
        seed=42,
        ),
    
    CrossValidate = dict(
        bert_model="google/bigbird-roberta-base",
        config_args = dict(num_labels=15),
        saved_model='storage/output/220106_qa_google-bigbird-roberta-base/Train/google-bigbird-roberta-base_valscore0.40137_ep2.pt',
        ),

    Infer = dict(
        bert_model="google/bigbird-roberta-base",
        config_args = dict(num_labels=15),
        saved_model='storage/output/220106_qa_google-bigbird-roberta-base/Train/google-bigbird-roberta-base_valscore0.40137_ep2.pt',
        ),
)
