conf = dict(
    base_dir='storage/output/220101_sentclass_bert+base+uncased/',
    Preprocess = dict(
        discourse_df_csv_path='storage/train.csv',
        text_dir='storage/train/',
        text_df_csv_fname='text_df.csv',
    ),
    PrepareData = dict(
        input_mod='Preprocess/',
        split_args=dict(test_size=0.20,n_splits=2,random_state=7),
        maxlen=512,
        bert_model = "bert-base-uncased",
        num_class=8,
        train_textbs=1,
        val_textbs = 32,
        ),
    Train = dict(
        num_class=8,
        bert_model = "bert-base-uncased",
        freeze_bert = False,
        iters_to_accumulate = 1,
        train_bs = 4,
        val_bs = 16,
        lr = 2e-5,
        wd = 1e-2,
        epochs = 1,
        print_every = 1,
        ),
)
