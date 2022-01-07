conf = dict(
    base_dir='storage/output/220104_ner_google-bigbird-roberta-base/',
    slurm=dict(
        fname='slurm.cfg',
        pyscript='run_ner.py',
        name='220104_ner_google-bigbird-roberta-base',
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
    PrepareData = dict(
        discourse_df_path='storage/train.csv',
        input_mod='Preprocess/',
        split_args=dict(test_size=0.1),
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
        labels=['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement'],
        ),
    Train = dict(
        bert_model="google/bigbird-roberta-base",
        config_args = dict(num_labels=15),
        lr= [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
        epochs=5,
        print_every=200,
        max_grad_norm=10.,
        ),

    CrossValidate = dict(
        bert_model="google/bigbird-roberta-base",
        config_args = dict(num_labels=15),
        saved_model='storage/output/220104_ner_google-bigbird-roberta-base/Train/google-bigbird-roberta-base_valscore0.60794_ep2.pt',
        ),

    Infer = dict(
        bert_model="google/bigbird-roberta-base",
        config_args = dict(num_labels=15),
        saved_model='storage/output/220104_ner_google-bigbird-roberta-base/Train/google-bigbird-roberta-base_valscore0.60794_ep2.pt',
        ),
)
